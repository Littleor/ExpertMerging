import fnmatch
from collections import defaultdict
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Union
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint

from merging_utils import TaskVector, move_to_device, scale_tensor_by_coeffs, get_param_names_to_merge, activate_func, inverse_activate_func

logger = logging.getLogger(__name__)

import random
import torch
from accelerate.utils import set_seed

class BalancedDataset:
    def __init__(self, original_dataset, indices):
        self.dataset = original_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class BaseParametricTaskVectorModel(nn.Module, ABC):
    """
    Abstract base class for parametric task vector models.
    Defines the common interface that all model-specific implementations should follow.
    """

    def __init__(
        self,
        base_model: nn.Module,
        teacher_models: List[nn.Module],
        exclude_param_names_regex: List[str],
        device: torch.device = None,
        weight_coeffs_init_value: Union[float, List[float]] = 0.1,
        offload_base_model: bool = False,
        use_checkpoints: bool = False,
        dropout_rates: Dict[str, float] = None,
        default_dropout_rate: float = 0,
        coeffs_size_dict: dict = {},
        task_vector_coeffs_dict: dict = {}
    ):
        """
        Initialize parametric task vector model

        Args:
            base_model: Base pretrained model
            teacher_models: List of fine-tuned teacher models
            exclude_param_names_regex: Regex patterns for parameters to exclude from merging
            device: Device to run the student model on (default: None, uses current device)
        """
        super().__init__()

        self.base_model = base_model
        self.num_teachers = len(teacher_models)
        self.exclude_param_names_regex = exclude_param_names_regex
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.offload_base_model = offload_base_model
        self.use_checkpoints = use_checkpoints

        task_vector_coeffs_dict = task_vector_coeffs_dict if task_vector_coeffs_dict is not None else {}
        coeffs_size_dict = coeffs_size_dict if coeffs_size_dict is not None else {}

        if self.offload_base_model:
            logger.warning(f"Offloading base model is enabled.")
        if self.use_checkpoints:
            logger.warning(f"Using checkpoints is enabled.")

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        for model in teacher_models:
            for param in model.parameters():
                param.requires_grad = False

        # Get parameter names to merge
        base_param_dict = dict(self.base_model.named_parameters())
        self.param_names_to_merge = get_param_names_to_merge(
            input_param_names=list(base_param_dict.keys()),
            exclude_param_names_regex=exclude_param_names_regex,
        )
        self.default_dropout_rate = default_dropout_rate
        self.dropout_rate = dropout_rates if dropout_rates is not None else {}
        self.coeffs_size_dict = {}

        logger.info(f"dropout_rate: {self.dropout_rate}")

        # Fuzzy matching
        for key in coeffs_size_dict:
            for param_name in self.param_names_to_merge:
                if fnmatch.fnmatch(param_name, key):
                    self.coeffs_size_dict[param_name] = coeffs_size_dict[key]

        logger.info(f"actual coeffs_size_dict: {self.coeffs_size_dict}")

        # Precompute and store task vectors
        logger.info("Precomputing task vectors...")
        self.task_vectors = {}
        for param_name in self.param_names_to_merge:
            self.task_vectors[param_name] = []

        task_vector_list = []
        for teacher_model in teacher_models:
            teacher_name = teacher_model.name_or_path.split("/")[-1].split("_")[-1].split("-")[-1]

            task_vector = TaskVector(
                pretrained_model=base_model,
                finetuned_model=teacher_model,
                exclude_param_names_regex=exclude_param_names_regex,
            )
            task_vector_list.append(task_vector)

        for i, task_vector in enumerate(task_vector_list):
            teacher_model = teacher_models[i]
            teacher_name = teacher_model.name_or_path.split("/")[-1].split("_")[-1]
            if teacher_name in task_vector_coeffs_dict:
                logger.info(f"Using task vector coeffs for teacher model {teacher_name}, coeffs: {task_vector_coeffs_dict[teacher_name]}")

            for param_name in self.param_names_to_merge:
                if param_name in task_vector.task_vector_param_dict:

                    vector = task_vector.task_vector_param_dict[param_name].clone().detach().cpu()
                    if teacher_name in task_vector_coeffs_dict:
                        vector = vector * task_vector_coeffs_dict[teacher_name]

                    # Store on CPU to save GPU memory
                    self.task_vectors[param_name].append(
                        vector
                    )
        self.weight_coeffs_init_value = torch.tensor(weight_coeffs_init_value if isinstance(weight_coeffs_init_value, list) else [weight_coeffs_init_value for _ in range(len(teacher_models)) ], dtype=torch.float32, device=self.device)

        # Initialize learnable parameters
        self.trainable_params_count = 0
        self._initialize_learnable_params(self.weight_coeffs_init_value)

        logger.info(f"Initialized parametric model with {self.num_teachers} teachers")

        # Move base model to specified device
        # self.base_model = self.base_model.to(self.device)
        if self.offload_base_model:
            self.base_model = self.base_model.cpu()

    def _initialize_learnable_params(self, weight_coeffs_init_value: torch.Tensor):
        """Initialize weight coefficients - start from base model (weights=0)"""
        self.weight_coeffs = nn.ParameterDict()
        self.fixed_weight_coeffs = {}

        for param_name in self.param_names_to_merge:
            if param_name in self.task_vectors:
                param_key = param_name.replace(".", "_")
                # Weight coefficients: [num_teachers], initialized to 0 to start from base model
                # We'll use activation to constrain weights to [0,1]
                # We'll use sigmoid activation to constrain weights to [0,1]
                coeff_size = self.coeffs_size_dict.get(param_name, 1)

                if coeff_size <= 0:
                    fixed_value = weight_coeffs_init_value.detach().clone().unsqueeze(-1).to(self.device)
                    buffer_name = f"_fixed_weight_coeff_{param_key}"
                    self.register_buffer(buffer_name, fixed_value)
                    self.fixed_weight_coeffs[param_key] = getattr(self, buffer_name)
                    continue

                value = inverse_activate_func(weight_coeffs_init_value.detach().clone())
                value = value.unsqueeze(-1).repeat(1, coeff_size)
                shape = value.shape
                assert shape[0] == self.num_teachers and shape[1] == coeff_size, f"Shape: {shape}"
                self.weight_coeffs[param_key] = nn.Parameter(
                    value,
                    requires_grad=True,
                )
                self.trainable_params_count += shape[0] * shape[1]

        # Move weight coefficients to the same device as the base model
        for param in self.weight_coeffs.parameters():
            param.data = param.data.to(self.device)

        logger.info(f"Initialized parametric model with {sum([value.numel() for _, value in self.weight_coeffs.items()])}({len(self.weight_coeffs.keys())}) parameters to merge")
        if self.fixed_weight_coeffs:
            logger.info(f"Initialized {len(self.fixed_weight_coeffs)} fixed coefficient groups (kept at init value)")
        logger.info(
            f"Initialized learnable weight coefficients vectors with init_value={weight_coeffs_init_value}"
        )

        self.weight_coeffs.requires_grad_ = True

    def _get_merged_params(self) -> Dict[str, torch.Tensor]:
        """Compute merged parameters using current weights with activation"""
        merged_params = {}
        if self.offload_base_model:
            self.base_model = self.base_model.cpu()
        base_param_dict = dict(self.base_model.named_parameters())
        # base_param_dict = move_to_device(base_param_dict, "cpu")

        for param_name in self.param_names_to_merge:
            if param_name in self.task_vectors:
                param_key = param_name.replace(".", "_")
                base_param = base_param_dict[param_name]

                # Get current weights and apply activate to constrain
                if param_key in self.weight_coeffs:
                    raw_weights = self.weight_coeffs[param_key]
                    weights = activate_func(raw_weights)  # Constrain weights to [0,1]
                elif param_key in self.fixed_weight_coeffs:
                    weights = self.fixed_weight_coeffs[param_key]
                else:
                    # If coefficients were never initialized for this param, skip scaling
                    weights = None

                # Compute weighted task vector combination
                combined_task_vector = torch.zeros_like(base_param, device=base_param.device)
                if weights is not None:
                    for i, task_vector in enumerate(self.task_vectors[param_name]):
                        # Apply weight (scalar) to each task vector
                        combined_task_vector += scale_tensor_by_coeffs(
                            task_vector.to(base_param.device),
                            weights[i].to(base_param.device),
                        )

                merged_params[param_name] = base_param + combined_task_vector
            else:
                # Use base parameter for excluded layers
                merged_params[param_name] = base_param_dict[param_name]

        return move_to_device(merged_params, self.device)

    def _forward(self, *args, **kwargs):
        try:
            from torch.func import functional_call
        except ImportError:
            from torch.nn.utils.stateless import functional_call

        """
        Forward with merged parameters using functional_call to retain autograd.
        """
        # 1. Get Merging Parameters
        merged_params = self._get_merged_params()

        # 2. Get All Parameters and Buffers. Only Replace the Parameters You Need to Merge, Others Use base_model's Own
        param_dict = dict(self.base_model.named_parameters())
        buffer_dict = dict(self.base_model.named_buffers())
        # Note: functional_call requires parameter names to match state_dict exactly
        merged_state = {}
        merged_state.update(param_dict)
        merged_state.update(buffer_dict)
        for param_name in self.param_names_to_merge:
            merged_state[param_name] = merged_params[param_name]

        for param_name, parm in merged_state.items():
            if isinstance(parm, torch.Tensor):
                merged_state[param_name] = parm.to(self.device)

        # 3. Call functional_call
        return functional_call(self.base_model, merged_state, args, kwargs)

    def forward(self, *args, **kwargs):
        if self.use_checkpoints:
            # Use checkpointing to save memory
            return checkpoint(
                self._forward,
                *args,
                **kwargs,
                preserve_rng_state=True,  # Preserve RNG state for reproducibility
                use_reentrant=False,  # Use non-reentrant mode for better performance
            )
        else:
            return self._forward(*args, **kwargs)
    def get_final_merged_params(self) -> Dict[str, torch.Tensor]:
        """Get final merged parameters after training"""
        return self._get_merged_params()


class BaseExpertMergingTrainer(ABC):
    """
    Abstract base class for parametric ExpertMerging trainers.
    Defines the common interface that all trainer-specific implementations should follow.
    """

    def __init__(
            self,
            base_model: nn.Module,
            teacher_models: List[nn.Module],
            temperature: float = 3.0,
            learning_rate: float = 1e-4,
            log_dir: str = "logs/expert_merging",
            exclude_param_names_regex=None,
            gradient_accumulation_steps: int = 8,
            cpu_offload_teachers: bool = True,
            mixed_precision: str = "fp16",
            seed: int = 42,
            # New regularization parameters
            sparsity_reg_weight: float = 0.1,
            max_weight_norm: float = 1.0,
            # Hidden states alignment parameters
            hidden_states_layers: List[int] = None,
            hidden_states_weight: float = 0.1,
            # Task-specific loss weights
            loss_alphas: Dict[str, float] = None,
            # Device parameters
            student_device: torch.device = None,
            teacher_device: torch.device = None,
            loss_device: torch.device = None,
            weight_coeffs_init_value: float = 0.1,
            # Weight balance parameters
            group_shuffle: bool = False
    ):
        """
        Initialize ExpertMerging trainer

        Args:
            base_model: Base pretrained model
            teacher_models: List of fine-tuned teacher models
            temperature: Temperature for logits alignment
            learning_rate: Learning rate for optimization
            log_dir: Directory for logging
            exclude_param_names_regex: Regex patterns for parameters to exclude
            gradient_accumulation_steps: Steps for gradient accumulation
            cpu_offload_teachers: Whether to offload teachers to CPU
            mixed_precision: Mixed precision setting
            seed: Random seed
            sparsity_reg_weight: Sparsity regularization weight
            max_weight_norm: Maximum weight norm for clipping
            student_device: Device to run student model on
            teacher_devices: Devices to run teacher models on (one per teacher)
            loss_device: Device to compute loss on (default: student_device)
            loss_alphas: Dictionary mapping task names to loss weights, defaults to 1.0 if not specified
        """
        from accelerate import Accelerator
        
        self.task_names = ["OCR", "VQA", "Geometry", "Chart", "Grounding"]
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.log_dir = Path(log_dir)
        self.exclude_param_names_regex = exclude_param_names_regex or []
        self.cpu_offload_teachers = cpu_offload_teachers
        self.teacher_models = teacher_models
        self.weight_coeffs_init_value = weight_coeffs_init_value

        # Task-specific loss weights
        self.loss_alphas = loss_alphas if loss_alphas is not None else {}

        # Device configuration
        self.student_device = student_device if student_device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher_device = teacher_device if teacher_device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_device = loss_device if loss_device is not None else self.student_device

        # Regularization parameters
        self.sparsity_reg_weight = sparsity_reg_weight
        self.max_weight_norm = max_weight_norm
        self.group_shuffle = group_shuffle

        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with="tensorboard",
            project_dir=self.log_dir,
        )

        self.scheduler = None

        # Set seed
        set_seed(seed)

        # Hidden states alignment setup
        self.hidden_states_layers = hidden_states_layers if hidden_states_layers else []
        self.hidden_states_weight = hidden_states_weight

        self.global_step = 0

        logger.info("Trainer Parametric ExpertMerging Trainer initialized:")
        logger.info(f"  Temperature: {self.temperature}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Log directory: {self.log_dir}")
        logger.info(f"  Exclude parameter names regex: {self.exclude_param_names_regex}")
        logger.info(f"  Gradient accumulation steps: {self.accelerator.gradient_accumulation_steps}")
        logger.info(f"  Seed: {seed}")
        logger.info(f"  Sparsity regularization weight: {self.sparsity_reg_weight}")
        logger.info(f"  Max weight norm: {self.max_weight_norm}")
        logger.info(f"  Hidden states layers: {self.hidden_states_layers}")
        logger.info(f"  Hidden states weight: {self.hidden_states_weight}")
        logger.info(f"  Task-specific loss weights: {self.loss_alphas}")
        logger.info(f"  Student device: {self.student_device}")
        logger.info(f"  Teacher device: {self.teacher_device}")
        logger.info(f"  Loss device: {self.loss_device}")
        logger.info(f"  Weight coefficients init value: {self.weight_coeffs_init_value}")
        logger.info(f"  We will use group shuffle: [bold red]{self.group_shuffle}")

    @abstractmethod
    def _prepare_inputs(self, question, response, *args, **kwargs):
        """
        Abstract method to prepare inputs for model forward pass.
        Must be implemented by subclasses.
        """
        pass

    def _get_teacher_model(self, teacher_idx: int):
        """Load teacher model to its assigned device when needed"""
        teacher_model = self.teacher_models[teacher_idx - 1]

        if self.cpu_offload_teachers:
            torch.cuda.empty_cache()
            teacher_model = teacher_model.to(self.teacher_device)

        return teacher_model

    def _offload_teacher_model(self, teacher_model, teacher_idx: int):
        """Offload teacher model back to CPU to save memory"""
        if self.cpu_offload_teachers:
            self.teacher_models[teacher_idx - 1] = teacher_model.cpu()
            torch.cuda.empty_cache()

    def compute_kldiv_loss(self, teacher_logits, student_logits):
        temp = self.temperature
        student_logits_scaled = student_logits / temp
        teacher_logits_scaled = teacher_logits / temp

        loss_kd = (
                F.kl_div(
                    F.log_softmax(student_logits_scaled, dim=-1).to(self.loss_device),
                    F.softmax(teacher_logits_scaled, dim=-1).to(self.loss_device),
                    reduction="batchmean",
                )
                * (temp**2) / student_logits.shape[1]
        )
        return loss_kd.to(student_logits.device)

    def compute_ce_loss(self, teacher_logits, student_logits):
        """Compute cross-entropy loss with soft targets from teacher logits"""
        # Move tensors to loss device if they're on different devices
        if teacher_logits.device != self.loss_device:
            teacher_logits = teacher_logits.to(self.loss_device)
        if student_logits.device != self.loss_device:
            student_logits = student_logits.to(self.loss_device)

        # Apply temperature and compute soft probabilities
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        # Cross entropy loss with soft targets
        loss = (
            -(teacher_probs * F.log_softmax(student_logits / self.temperature, dim=-1))
            .sum(dim=-1)
            .mean()
        )

        # Scale by temperature squared
        return loss * (self.temperature**2)

    def compute_regularization_loss(self, task_name: str=None):
        """Compute regularization losses for weight coefficients"""
        if self.sparsity_reg_weight <= 0:
            return 0.

        if getattr(self.student_model, "trainable_params_count", 0) == 0:
            return 0.

        reg_loss = 0.0

        for _, weights in self.student_model.weight_coeffs.items():
            # Apply actiavte func to get actual weights
            actual_weights = activate_func(weights)

            # Sparsity regularization - encourage sparse solutions
            # Use L1 norm to encourage sparsity
            reg_loss += torch.sum(torch.abs(actual_weights - self.student_model.weight_coeffs_init_value.view(-1, 1)))

        reg_loss /= (self.student_model.trainable_params_count / len(self.teacher_models))

        total_reg_loss = (
                self.sparsity_reg_weight * reg_loss
        )

        return total_reg_loss

    def compute_hidden_states_loss(self, teacher_hidden_states, student_hidden_states):
        """Compute MSE loss between teacher and student hidden states for specified layers"""
        if not self.hidden_states_layers:
            return 0.0, {}

        total_loss = 0.0
        layer_losses = {}

        for layer_idx in self.hidden_states_layers:
            if layer_idx >= len(teacher_hidden_states):
                logger.warning(f"Layer {layer_idx} is out of range for hidden states")
                continue

            # Move tensors to loss device if they're on different devices
            teacher_layer = teacher_hidden_states[layer_idx]
            student_layer = student_hidden_states[layer_idx]

            if teacher_layer.device != self.loss_device:
                teacher_layer = teacher_layer.to(self.loss_device)
            if student_layer.device != self.loss_device:
                student_layer = student_layer.to(self.loss_device)

            # Convert to float32 for loss computation
            teacher_layer = teacher_layer.float()
            student_layer = student_layer.float()

            # Compute MSE loss for this layer
            layer_loss = F.mse_loss(student_layer, teacher_layer)
            total_loss += layer_loss
            layer_losses[f"hidden_state_loss_layer_{layer_idx}"] = layer_loss.item()

        # Average loss across layers
        if layer_losses:
            total_loss = total_loss / len(layer_losses)

        return self.hidden_states_weight * total_loss.to(self.student_device), layer_losses

    @abstractmethod
    def train_step(self, batch):
        """
        Abstract method for a single training step.
        Must be implemented by subclasses.
        """
        pass

    def create_balanced_dataloader(self, dataset, batch_size=1):
        """Create a balanced dataloader that ensures equal sampling from each teacher"""
        from collections import defaultdict
        from torch.utils.data import DataLoader

        # Group samples by task_name
        task_samples = defaultdict(list)
        for i, sample in enumerate(dataset):
            task_samples[sample['task_name']].append(i)

        # Create balanced sampling indices
        balanced_indices = []
        total_size = len(dataset)

        while len(balanced_indices) != total_size:
            for task_name in task_samples.keys():
                if task_samples[task_name]:
                    balanced_indices.append(task_samples[task_name].pop())

        def group_and_shuffle_flat(lst, n):
            result = []
            for i in range(0, len(lst), n):
                group = lst[i:i + n]
                random.shuffle(group)
                result.extend(group)
            return result
        if self.group_shuffle:
            balanced_indices = group_and_shuffle_flat(balanced_indices, len(task_samples.keys()))

        assert len(balanced_indices) == total_size, "Balanced indices size mismatch"
        assert len(list(set(balanced_indices))) == total_size, "Indices are not unique"

        balanced_dataset = BalancedDataset(dataset, balanced_indices)

        return DataLoader(
            balanced_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: {
                "teacher_model_idx": [x[0]["teacher_model_idx"]],
                "question": [x[0]["question"]],
                "response": [x[0]["response"]],
                "image_paths": [x[0]["image_paths"]],
                "pixel_values": [x[0].get("pixel_values", None)],
                "task_name": [x[0]["task_name"]],
                "question_id": [x[0]["question_id"]],
            },
        )

  
    def compute_loss(self, teacher_outputs, student_outputs, task_name=None):
        alpha = self.loss_alphas.get(task_name, 1.0)

        teacher_logits = teacher_outputs.logits
        student_logits = student_outputs.logits

        # Compute logits alignment loss
        logits_align_loss = self.compute_ce_loss(teacher_logits, student_logits) * alpha

        # Compute regularization loss
        reg_loss = self.compute_regularization_loss(task_name=task_name)

        # Compute hidden states loss
        hidden_states_loss, layer_losses = self.compute_hidden_states_loss(
            teacher_outputs.hidden_states,
            student_outputs.hidden_states
        )
        hidden_states_loss *= alpha


        # Total loss
        total_loss = logits_align_loss + reg_loss + hidden_states_loss

        # Prepare return dict
        loss_dict = {
            "total_loss": total_loss,
            "logits_align_loss": logits_align_loss,
            "reg_loss": reg_loss,
            "hidden_states_loss": hidden_states_loss
        }
        # Add individual layer losses
        loss_dict.update(layer_losses)
        keys = [*list(loss_dict.keys())]
        for key in keys:
            loss_dict[f"Loss-{task_name}-Task/{key}"] = loss_dict[key]


        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                loss_dict[key] = value.item()

        return total_loss, loss_dict



    def train(self, dataset, num_epochs: int = 3):
        """Train the parametric student model with improved strategy"""
        logger.info(
            f"Starting parametric ExpertMerging training for {num_epochs} epochs"
        )
        # self._log_weight_statistics()
        # return self.accelerator.unwrap_model(self.student_model)

        # Create balanced dataloader
        dataloader = self.create_balanced_dataloader(dataset, batch_size=1)
        dataloader = self.accelerator.prepare(dataloader)

        # Track best loss for early stopping
        best_loss = float('inf')
        patience_counter = 0
        patience = 3  # Early stopping patience

        for epoch in range(num_epochs):
            total_losses = defaultdict(float, default=0.0)
            num_updates = 0

            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for step, batch in enumerate(progress_bar):
                logger.info(f"Step {step}: {batch['task_name'][0]}-{batch['question_id'][0]} {batch['image_paths'][0]}")
                with self.accelerator.accumulate(self.student_model):
                    loss_dict = self.train_step(batch)

                    # Log task metrics (this may not update)
                    log_dict = {}
                    for key in loss_dict:
                        if "/" in key:
                            log_dict[key] = loss_dict[key]
                    self.accelerator.log(log_dict, step=self.global_step)
                    del log_dict

                    if self.accelerator.sync_gradients:
                        # Gradient clipping for stability
                        self.accelerator.clip_grad_norm_(
                            self.student_model.weight_coeffs.parameters(),
                            max_norm=1.0
                        )

                        self.optimizer.step()
                        if self.scheduler is not None:
                            self.scheduler.step()
                        self.optimizer.zero_grad()
                        self.global_step += 1
                        num_updates += 1

                        # Log metrics
                        if self.accelerator.is_main_process:
                            log_dict = {
                                "Learning_Rate": self.optimizer.param_groups[0]["lr"],
                            }
                            for key in loss_dict:
                                if "/" not in key:
                                    log_dict[f"Loss-Train/{key}"] = loss_dict[key]
                            self.accelerator.log(log_dict, step=self.global_step)

                    # Accumulate losses
                    for key in loss_dict:
                        total_losses[key] += loss_dict[key]

                    postfix_dict = {
                        "total": f"{loss_dict['total_loss']:.2}",
                    }
                    if "logits_align_loss" in loss_dict:
                        postfix_dict["logits_align"] = f"{loss_dict['logits_align_loss']:.2}"
                    if "hidden_states_loss" in loss_dict:
                        postfix_dict["hidden_states"] = f"{loss_dict['hidden_states_loss']:.2}"
                    if "reg_loss" in loss_dict:
                        postfix_dict["reg"] = f"{loss_dict['reg_loss']:.2}"
                    progress_bar.set_postfix(postfix_dict)

            # Calculate average losses
            avg_losses = {key: val / len(dataloader) for key, val in total_losses.items()}

            if self.accelerator.is_main_process:
                logger.info(
                    f"Epoch {epoch+1} completed. "
                )
                for key, val in avg_losses.items():
                    logger.info(f"{key}: {val:.4f}")

                # Log epoch metrics
                epoch_log_dict = {}
                for key, val in avg_losses.items():
                    if "/" not in key:
                        epoch_log_dict[f"Loss-Epoch/{key}"] = val
                    else:
                        # `f"Loss-{task_name}-Task/{key}"`
                        task_name = key.split("/")[0].split("-")[1]
                        epoch_log_dict[f"Loss-Epoch-{task_name}/{key.split('/')[1]}"] = val

                epoch_log_dict["Updates/Epoch"] = num_updates
                self.accelerator.log(epoch_log_dict, step=epoch)

                # Early stopping check
                current_loss = avg_losses['total_loss']
                if current_loss < best_loss:
                    best_loss = current_loss
                    patience_counter = 0
                    logger.info(f"New best loss: {best_loss:.4f}")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    logger.info("Early stopping triggered!")
                    break

        if self.accelerator.is_main_process:
            logger.info("Parametric ExpertMerging training completed!")

            # Log final weight statistics
            self._log_weight_statistics()

            self.accelerator.end_training()

        return self.accelerator.unwrap_model(self.student_model)

    def _log_weight_statistics(self):
        """Log statistics about learned weights"""
        if not self.accelerator.is_main_process:
            return

        logger.info("=== Final Weight Statistics ===")

        for param_name, raw_weights in self.student_model.weight_coeffs.items():
            raw_weights = raw_weights.mean(dim=-1)
            weights = activate_func(raw_weights).detach().cpu()

            logger.info(f"Parameter: {param_name}")
            logger.info(f"  Raw weights: {raw_weights.detach().cpu().numpy()}")
            logger.info(f"  Origin weights: {weights.numpy()}")
            logger.info(f"  Mean: {weights.mean().item():.4f}")
            logger.info(f"  Std: {weights.std().item():.4f}")
            logger.info(f"  Max: {weights.max().item():.4f}")
            logger.info(f"  Min: {weights.min().item():.4f}")

            # Show which teachers are most important
            sorted_indices = torch.argsort(weights, descending=True)
            logger.info(f"  Teacher importance ranking: {sorted_indices.tolist()}")
            logger.info("---")

        for param_name, weights in getattr(self.student_model, "fixed_weight_coeffs", {}).items():
            weights_cpu = weights.mean(dim=-1).detach().cpu()
            logger.info(f"Parameter (fixed): {param_name}")
            logger.info(f"  Origin weights: {weights_cpu.numpy()}")
            logger.info(f"  Mean: {weights_cpu.mean().item():.4f}")
            logger.info(f"  Std: {weights_cpu.std().item():.4f}")
            logger.info(f"  Max: {weights_cpu.max().item():.4f}")
            logger.info(f"  Min: {weights_cpu.min().item():.4f}")
            sorted_indices = torch.argsort(weights_cpu, descending=True)
            logger.info(f"  Teacher importance ranking: {sorted_indices.tolist()}")
            logger.info("---")

        # Save the weight coefficients as JSON file
        weight_coeffs_path = os.path.join(os.path.dirname(self.log_dir), "weight_coeffs.json")
        os.makedirs(os.path.dirname(weight_coeffs_path), exist_ok=True)
        with open(weight_coeffs_path, "w") as f:
            import json
            json.dump(self.get_weight_coeffs_dict(mean=False), f, indent=4, ensure_ascii=False)


    def get_weight_coeffs_dict(self, mean=True):
        """Get the weight coefficients dictionary with JSON serializable tensors"""
        weight_coeffs_dict = {}
        for param_name, weights in self.student_model.weight_coeffs.items():
            # Add sigmoid activation to constrain weights to [0,1]
            weights = activate_func(weights)
            if mean:
                weights = weights.mean(dim=-1)
            else:
                weights = weights.squeeze(dim=-1)
            # Add activation to constrain weights to [0,1]
            weight_coeffs_dict[param_name] = weights.detach().cpu().numpy().tolist()

        for param_name, weights in getattr(self.student_model, "fixed_weight_coeffs", {}).items():
            weights_tensor = weights
            if mean:
                weights_tensor = weights_tensor.mean(dim=-1)
            else:
                weights_tensor = weights_tensor.squeeze(dim=-1)
            weight_coeffs_dict[param_name] = weights_tensor.detach().cpu().numpy().tolist()
        return weight_coeffs_dict
