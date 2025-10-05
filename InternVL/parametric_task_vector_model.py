import logging

# Import from the common base
import sys
from typing import Dict, List

import torch
import torch.nn as nn
from conversation import get_conv_template

sys.path.append("../global_utils")
from expert_merging_base import (
    BaseExpertMergingTrainer,
    BaseParametricTaskVectorModel,
)

logger = logging.getLogger(__name__)


class ParametricTaskVectorModel(BaseParametricTaskVectorModel):
    def __init__(
        self,
        base_model: nn.Module,
        teacher_models: List[nn.Module],
        exclude_param_names_regex: List[str],
        device: torch.device = None,
        weight_coeffs_init_value: float = 0.1,
        **kwargs,
    ):
        """
        Initialize parametric task vector model

        Args:
            base_model: Base pretrained model
            teacher_models: List of fine-tuned teacher models
            exclude_param_names_regex: Regex patterns for parameters to exclude from merging
            device: Device to run the student model on (default: None, uses current device)
        """
        super().__init__(
            base_model,
            teacher_models,
            exclude_param_names_regex,
            device,
            weight_coeffs_init_value,
            **kwargs,
        )

        # Store base model template for attribute access
        self.template = base_model.template if hasattr(base_model, "template") else None


class ExpertMergingTrainer(BaseExpertMergingTrainer):
    def __init__(
        self,
        base_model: nn.Module,
        teacher_models: List[nn.Module],
        tokenizer,
        temperature: float = 3.0,  # Increased temperature for better logits alignment
        learning_rate: float = 1e-4,  # Increased learning rate
        log_dir: str = "logs/expert_merging",
        exclude_param_names_regex=[
            "vision_model.*",
            ".*lm_head.*",
            ".*norm.*",
            ".*embed_tokens.*",
            ".*bias.*",
        ],
        gradient_accumulation_steps: int = 8,
        cpu_offload_teachers: bool = True,
        mixed_precision: str = "fp16",
        seed: int = 42,
        # New regularization parameters
        sparsity_reg_weight: float = 0.1,
        max_weight_norm: float = 1.0,
        hidden_states_layers: List[int] = None,  # Layers to compute hidden states loss
        hidden_states_weight: float = 0.1,  # Weight for hidden states loss
        # Task-specific loss weights
        loss_alphas: Dict[str, float] = None,
        # Device parameters
        student_device: torch.device = None,
        teacher_device: torch.device = None,
        loss_device: torch.device = None,
        weight_coeffs_init_value: float = 0.1,
        dropout_rates: Dict[str, float] = None,
        default_dropout_rate: float = 0.2,
        coeffs_size_dict: dict = {},
        task_vector_coeffs_dict: dict = {},
    ):
        super().__init__(
            base_model=base_model,
            teacher_models=teacher_models,
            temperature=temperature,
            learning_rate=learning_rate,
            log_dir=log_dir,
            exclude_param_names_regex=exclude_param_names_regex,
            gradient_accumulation_steps=gradient_accumulation_steps,
            cpu_offload_teachers=cpu_offload_teachers,
            mixed_precision=mixed_precision,
            seed=seed,
            sparsity_reg_weight=sparsity_reg_weight,
            max_weight_norm=max_weight_norm,
            hidden_states_layers=hidden_states_layers,
            hidden_states_weight=hidden_states_weight,
            loss_alphas=loss_alphas,  # Pass the loss_alphas parameter
            student_device=student_device,
            teacher_device=teacher_device,
            loss_device=loss_device,
            weight_coeffs_init_value=weight_coeffs_init_value,
        )

        self.tokenizer = tokenizer
        self.cpu_offload_teachers = cpu_offload_teachers
        self.teacher_models = teacher_models

        for model in [base_model] + teacher_models:
            # Set img_context_token_id
            IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
            img_context_token_id = self.tokenizer.convert_tokens_to_ids(
                IMG_CONTEXT_TOKEN
            )
            model.img_context_token_id = img_context_token_id

        # Create parametric student model
        self.student_model = ParametricTaskVectorModel(
            base_model=base_model,
            teacher_models=teacher_models,
            exclude_param_names_regex=exclude_param_names_regex,
            device=student_device,
            weight_coeffs_init_value=weight_coeffs_init_value,
            dropout_rates=dropout_rates,
            default_dropout_rate=default_dropout_rate,
            coeffs_size_dict=coeffs_size_dict,
            task_vector_coeffs_dict=task_vector_coeffs_dict,
        )

        # Move teacher models to CPU if offloading enabled
        if self.cpu_offload_teachers:
            for i, teacher in enumerate(teacher_models):
                self.teacher_models[i] = teacher.cpu()

        # Setup optimizer - only optimize weight coefficients
        learnable_params = list(self.student_model.weight_coeffs.parameters())

        self.optimizer = torch.optim.AdamW(
            learnable_params,
            lr=learning_rate,
            weight_decay=1e-5,  # Small weight decay for stability
            betas=(0.9, 0.999),
        )

        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2, eta_min=learning_rate * 0.1
        )

        # Prepare with accelerator
        self.student_model, self.optimizer = self.accelerator.prepare(
            self.student_model, self.optimizer
        )

        # Initialize tracking
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("parametric_expert_merging")

    def _prepare_inputs(self, question, response, pixel_values, model):
        """
        Prepare inputs for model forward pass following InternVL chat method exactly

        Args:
            question: Input question text
            response: Expected response text
            pixel_values: Image tensor
            model: Model to get template and tokens from
        """
        # Add image token to question if not present
        if "<image>" not in question:
            question = "<image>\n" + question

        # Get conversation template
        template = get_conv_template(model.template)
        template.system_message = model.system_message

        # Build conversation with question and response
        template.append_message(template.roles[0], question)
        if response is not None and len(response) > 0:
            template.append_message(template.roles[1], response)
        query = template.get_prompt()

        # Handle image tokens - follow InternVL chat method exactly
        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

        # Get number of patches
        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []

        # Replace <image> with proper image tokens
        for num_patches in num_patches_list:
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)

        # Tokenize
        model_inputs = self.tokenizer(query, return_tensors="pt")
        device = self.accelerator.device

        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)

        num_patches = len(pixel_values)
        image_flags = torch.ones(num_patches, dtype=torch.long, device=device)

        return {
            "pixel_values": (
                pixel_values.to(device) if pixel_values is not None else None
            ),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_flags": image_flags,
            "return_dict": True,
        }

    def train_step(self, batch):
        """Single training step with regularization"""
        teacher_indices = batch["teacher_model_idx"]
        questions = batch["question"]
        responses = batch["response"]
        pixel_values_list = batch["pixel_values"]
        # Get task_name from batch if available
        task_names = batch["task_name"]

        teacher_idx = teacher_indices[0]
        teacher_model = self._get_teacher_model(teacher_idx)
        question = questions[0]
        response = responses[0]
        pixel_values = pixel_values_list[0]

        # Get alpha value for this task
        task_name = task_names[0]

        # Prepare inputs
        inputs = self._prepare_inputs(question, response, pixel_values, teacher_model)

        # Get teacher logits
        teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = teacher_model(
                **{
                    key: (
                        value.to(self.teacher_device) if hasattr(value, "to") else value
                    )
                    for key, value in inputs.items()
                },
                output_attentions=None,
                output_hidden_states=True,
            )

        # Offload teacher model to save memory
        self._offload_teacher_model(teacher_model, teacher_idx)

        # Get student logits
        self.student_model.train()
        student_outputs = self.student_model(
            **inputs, use_cache=False, output_attentions=None, output_hidden_states=True
        )

        total_loss, loss_dict = self.compute_loss(
            teacher_outputs, student_outputs, task_name=task_name
        )

        # Backward pass
        self.accelerator.backward(total_loss)

        return loss_dict
    def create_balanced_dataloader(self, dataset, batch_size=1):
            """Create a balanced dataloader that ensures equal sampling from each teacher"""
            from collections import defaultdict
            from torch.utils.data import DataLoader

            # Group samples by teacher
            teacher_samples = defaultdict(list)
            for i, sample in enumerate(dataset):
                teacher_idx = sample['teacher_model_idx']
                teacher_samples[teacher_idx].append(i)

            # Create balanced sampling indices
            balanced_indices = []
            max_samples = max(len(samples) for samples in teacher_samples.values())

            for epoch_step in range(max_samples):
                for teacher_idx in sorted(teacher_samples.keys()):
                    samples = teacher_samples[teacher_idx]
                    if samples:
                        # Cycle through samples for this teacher
                        idx = samples[epoch_step % len(samples)]
                        balanced_indices.append(idx)

            # Create custom dataset with balanced indices
            class BalancedDataset:
                def __init__(self, original_dataset, indices):
                    self.dataset = original_dataset
                    self.indices = indices

                def __len__(self):
                    return len(self.indices)

                def __getitem__(self, idx):
                    return self.dataset[self.indices[idx]]

            balanced_dataset = BalancedDataset(dataset, balanced_indices)

            return DataLoader(
                balanced_dataset,
                batch_size=batch_size,
                shuffle=True,
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

