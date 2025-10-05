import argparse
import copy
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from utils import (
    TaskVector,
    build_transform,
    dynamic_preprocess,
    get_param_names_to_merge,
    load_image,
    task_arithmetic,
)

sys.path.append("../global_utils")

from config import save_config, setup_logging  # type: ignore


def ties_merging(
    merged_model: nn.Module,
    models_to_merge: list,
    exclude_param_names_regex: list,
    param_value_mask_rate: float = 0.8,
    scaling_coefficient: float = 1.0,
):
    """
    ties merging method (layer-by-layer implementation to save memory)
    :param merged_model: nn.Module, the merged model
    :param models_to_merge: list, individual models that need to be merged
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
    :param scaling_coefficient: float, scaling coefficient to merge the task vectors
    :return:
    """

    def task_vector_param_dict_to_single_vector(task_vector: TaskVector):
        """
        convert parameter dictionary in task vector to a single vector
        :param task_vector: TaskVector, task vector
        :return:
        """
        task_vector_param_dict = copy.deepcopy(task_vector.task_vector_param_dict)
        sorted_task_vector_param_dict = OrderedDict(
            sorted(task_vector_param_dict.items())
        )

        # Tensor, shape (num_total_params, )
        return nn.utils.parameters_to_vector(
            [param.flatten() for param in sorted_task_vector_param_dict.values()]
        )

    def single_vector_to_task_vector_param_dict(
        single_vector: torch.Tensor, task_vector: TaskVector
    ):
        """
        convert a single vector to parameter dictionary in task vector
        :param single_vector: Tensor, single vector that contain all parameters in task_vector.task_vector_param_dict
        :param task_vector: TaskVector, task vector
        :return:
        """
        task_vector_param_dict = copy.deepcopy(task_vector.task_vector_param_dict)
        sorted_task_vector_param_dict = OrderedDict(
            sorted(task_vector_param_dict.items())
        )

        nn.utils.vector_to_parameters(
            single_vector, sorted_task_vector_param_dict.values()
        )

        return sorted_task_vector_param_dict

    def mask_smallest_magnitude_param_values(
        flattened_models_to_merge_param: torch.Tensor,
        param_value_mask_rate: float = 0.8,
    ):
        """
        mask the smallest-magnitude parameter values (set to zeros) based on parameter value mask rate
        :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params)
        :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
        :return:
        """
        # Convert to float32 to support kthvalue operation
        flattened_models_to_merge_param = flattened_models_to_merge_param.float()

        num_mask_params = int(
            flattened_models_to_merge_param.shape[1] * param_value_mask_rate
        )

        # Calculate the threshold
        kth_values, _ = flattened_models_to_merge_param.abs().kthvalue(
            k=num_mask_params, dim=1, keepdim=True
        )

        # Create mask and apply
        mask = flattened_models_to_merge_param.abs() >= kth_values

        # Apply mask and convert back to original dtype
        return (flattened_models_to_merge_param * mask).to(
            flattened_models_to_merge_param.dtype
        )

    def get_param_signs(flattened_models_to_merge_param: torch.Tensor):
        """
        get the signs for each parameter in flattened_models_to_merge_param, computed over individual models that need to be merged
        :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
        :return:
        """
        # Tensor, shape (num_total_params, ), the signs of parameters aggregated across individual models that need to be merged
        param_signs = torch.sign(flattened_models_to_merge_param.sum(dim=0))
        # Tensor, shape (, ), a scalar, replace 0 in param_signs to the major sign in param_signs
        majority_sign = torch.sign(param_signs.sum(dim=0))
        param_signs[param_signs == 0] = majority_sign
        return param_signs

    def disjoint_merge(
        flattened_models_to_merge_param: torch.Tensor, param_signs: torch.Tensor
    ):
        """
        disjoint merge that only keeps the parameter values in individual models whose signs are the same as the param_signs, and calculates the averaged parameters.
        :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
        :param param_signs: Tensor, shape (num_total_params, ), the signs of parameters aggregated across individual models that need to be merged
        :return:
        """
        # Tensor, shape (num_models_to_merge, num_total_params), where True is for parameters that we want to preserve
        param_to_preserve_mask = (
            (param_signs.unsqueeze(dim=0) > 0) & (flattened_models_to_merge_param > 0)
        ) | ((param_signs.unsqueeze(dim=0) < 0) & (flattened_models_to_merge_param < 0))
        # Tensor, shape (num_models_to_merge, num_total_params), the preserved parameters
        param_to_preserve = flattened_models_to_merge_param * param_to_preserve_mask

        # Tensor, shape (num_total_params, ), the number of models whose parameters can be preserved
        num_models_param_preserved = (param_to_preserve != 0).sum(dim=0).float()
        # Tensor, shape (num_total_params, ), the averaged flattened parameters
        merged_flattened_param = torch.sum(param_to_preserve, dim=0) / torch.clamp(
            num_models_param_preserved, min=1.0
        )

        return merged_flattened_param

    assert isinstance(
        scaling_coefficient, float
    ), "wrong type of scaling_coefficient, should be float!"

    models_to_merge_task_vectors = [
        TaskVector(
            pretrained_model=merged_model,
            finetuned_model=model_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
        )
        for model_to_merge in models_to_merge
    ]

    flattened_models_to_merge_param = [
        task_vector_param_dict_to_single_vector(task_vector=task_vector)
        for task_vector in models_to_merge_task_vectors
    ]
    # Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
    flattened_models_to_merge_param = torch.vstack(flattened_models_to_merge_param)

    with torch.no_grad():
        # Tensor, shape (num_models_to_merge, num_total_params), mask the smallest-magnitude parameter values using param_value_mask_rate
        flattened_models_to_merge_param = mask_smallest_magnitude_param_values(
            flattened_models_to_merge_param=flattened_models_to_merge_param,
            param_value_mask_rate=param_value_mask_rate,
        )

        # Tensor, shape (num_total_params, ), get the signs for each parameter in flattened_models_to_merge_param
        param_signs = get_param_signs(
            flattened_models_to_merge_param=flattened_models_to_merge_param
        )

        # Tensor, shape (num_total_params, ), disjoint merge
        merged_flattened_param = disjoint_merge(
            flattened_models_to_merge_param=flattened_models_to_merge_param,
            param_signs=param_signs,
        )

        # merged parameter dictionary
        merged_task_vector_param_dict = single_vector_to_task_vector_param_dict(
            single_vector=merged_flattened_param,
            task_vector=models_to_merge_task_vectors[0],
        )
        merged_task_vector = TaskVector(
            task_vector_param_dict=merged_task_vector_param_dict
        )
        # combine with parameters of the merged model based on scaling coefficient
        merged_params = merged_task_vector.combine_with_pretrained_model(
            pretrained_model=merged_model, scaling_coefficient=scaling_coefficient
        )

    return merged_params


def copy_params_to_model(params: dict, model: nn.Module):
    """
    copy parameters in "params" to the model
    :param params: dict, dictionary of parameters
    :param model: nn.Module, model that needs to copy parameters
    :return:
    """
    for param_name, param_value in model.named_parameters():
        if param_name in params:
            param_value.data.copy_(params[param_name])


def weight_average(
    merged_model: nn.Module,
    models_to_merge: list,
    exclude_param_names_regex: list,
    scaling_coefficient: float = 1.0,
):
    """
    weight average method that directly averages the weights of finetuned models
    :param merged_model: nn.Module, the merged model
    :param models_to_merge: list, individual models that need to be merged
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :param scaling_coefficient: float, scaling coefficient to merge the weights (default: 1.0)
    :return:
    """
    assert isinstance(
        scaling_coefficient, float
    ), "wrong type of scaling_coefficient, should be float!"
    logging.info(f"Weight averaging scaling coefficient: {scaling_coefficient}")

    # Get parameter names to merge
    pretrained_param_dict = {
        param_name: param_value
        for param_name, param_value in merged_model.named_parameters()
    }
    param_names_to_merge = get_param_names_to_merge(
        input_param_names=list(pretrained_param_dict.keys()),
        exclude_param_names_regex=exclude_param_names_regex,
    )

    # Initialize merged parameters with base model parameters
    merged_params = {}
    for param_name in param_names_to_merge:
        merged_params[param_name] = pretrained_param_dict[param_name].clone()

    # Calculate the average of finetuned model parameters
    with torch.no_grad():
        for param_name in tqdm(param_names_to_merge, desc="Averaging model parameters"):
            # Sum up parameters from all models to merge
            param_sum = None
            for model in models_to_merge:
                if param_sum is None:
                    param_sum = model.state_dict()[param_name].clone()
                else:
                    param_sum += model.state_dict()[param_name]

            # Calculate average and apply scaling coefficient
            param_avg = param_sum / len(models_to_merge)
            merged_params[param_name] = pretrained_param_dict[
                param_name
            ] + scaling_coefficient * (param_avg - pretrained_param_dict[param_name])

    return merged_params


def mask_input_with_mask_rate(
    input_tensor: torch.Tensor, mask_rate: float, use_rescale: bool, mask_strategy: str
):
    """
    mask the input with mask rate
    :param input_tensor: Tensor, input tensor
    :param mask_rate: float, mask rate
    :param use_rescale: boolean, whether to rescale the input by 1 / (1 - mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    """
    assert (
        0.0 <= mask_rate <= 1.0
    ), f"wrong range of mask_rate {mask_rate}, should be [0.0, 1.0]!"
    original_dtype = input_tensor.dtype
    input_tensor = input_tensor.float()
    if mask_strategy == "random":
        mask = torch.bernoulli(
            torch.full_like(input=input_tensor, fill_value=mask_rate)
        ).to(input_tensor.device)
        masked_input_tensor = input_tensor * (1 - mask)
    else:
        assert (
            mask_strategy == "magnitude"
        ), f"wrong setting for mask_strategy {mask_strategy}!"
        original_shape = input_tensor.shape
        input_tensor = input_tensor.flatten()
        num_mask_params = int(len(input_tensor) * mask_rate)
        # Tensor, shape (1, ), find the num_mask_params-th smallest magnitude element of all the parameters in the model
        kth_values, _ = input_tensor.abs().kthvalue(
            k=num_mask_params, dim=0, keepdim=True
        )
        # Tensor, shape (num_total_params, ), where True is for parameters that we want to perform mask
        mask = input_tensor.abs() <= kth_values
        masked_input_tensor = input_tensor * (~mask)
        masked_input_tensor = masked_input_tensor.reshape(original_shape)
    if use_rescale and mask_rate != 1.0:
        masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate)
    return masked_input_tensor.to(original_dtype)


def mask_model_weights(
    finetuned_model: nn.Module,
    pretrained_model: nn.Module,
    exclude_param_names_regex: list,
    weight_format: str,
    weight_mask_rate: float,
    use_weight_rescale: bool,
    mask_strategy: str,
):
    """
    mask model weights
    :param finetuned_model: nn.Module, the finetuned model
    :param pretrained_model: nn.Module, the pretrained model
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :param weight_format: str, the format of weights to be masked, can be "finetuned_weight" and "delta_weight"
    :param weight_mask_rate: float, weight mask rate
    :param use_weight_rescale: boolean, whether to rescale the weight by 1 / (1 - weight_mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    """
    # get weights that need to be masked
    if weight_format == "finetuned_weight":
        param_dict = {
            param_name: param_value
            for param_name, param_value in finetuned_model.named_parameters()
        }
        # exclude parameter whose name matches element in exclude_param_names_regex
        param_names_to_merge = get_param_names_to_merge(
            input_param_names=list(param_dict.keys()),
            exclude_param_names_regex=exclude_param_names_regex,
        )
        model_param_dict = {
            param_name: param_dict[param_name] for param_name in param_names_to_merge
        }
    else:
        assert (
            weight_format == "delta_weight"
        ), f"wrong setting for weight_format {weight_format}!"
        task_vector = TaskVector(
            pretrained_model=pretrained_model,
            finetuned_model=finetuned_model,
            exclude_param_names_regex=exclude_param_names_regex,
        )
        model_param_dict = task_vector.task_vector_param_dict

    with torch.no_grad():
        masked_param_dict = {}
        for param_name, param_value in tqdm(model_param_dict.items()):
            masked_param_dict[param_name] = mask_input_with_mask_rate(
                input_tensor=param_value,
                mask_rate=weight_mask_rate,
                use_rescale=use_weight_rescale,
                mask_strategy=mask_strategy,
            )

        if weight_format == "delta_weight":
            new_task_vector = TaskVector(task_vector_param_dict=masked_param_dict)
            # combine with parameters of the merged model based on scaling coefficient
            masked_param_dict = new_task_vector.combine_with_pretrained_model(
                pretrained_model=pretrained_model, scaling_coefficient=1.0
            )

    return masked_param_dict


def svd_intermediate_processing(
    merged_model: nn.Module,
    models_to_merge: list,
    exclude_param_names_regex: list,
    tokenizer,
    output_path: str,
    run_name: str,
):
    """
    SVD intermediate processing method that computes TaskVector for each model to merge,
    performs SVD on each model, and saves each processed model separately.

    Args:
        merged_model: nn.Module, the base model
        models_to_merge: list, individual models that need to be merged
        exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        output_path: str, base output directory
        run_name: str, name for this run
        scaling_coefficient: float, scaling coefficient to merge the task vectors

    Returns:
        None (models are saved to disk)
    """
    scaling_coefficient = 1.0
    assert isinstance(
        scaling_coefficient, float
    ), "wrong type of scaling_coefficient, should be float!"

    # Create output directory structure: output_path/method/run_name/models
    final_output_base_path = (
        Path(output_path) / "svd_intermediate" / run_name / "models"
    )
    final_output_base_path.mkdir(parents=True, exist_ok=True)

    print("Computing task vectors and performing SVD for each model...")

    # Process each model to merge
    for idx, model_to_merge in enumerate(
        tqdm(models_to_merge, desc="Processing models")
    ):
        model_name = model_to_merge.name_or_path
        model_name = model_name.split("/")[-1]
        # Create task vector for this model
        task_vector = TaskVector(
            pretrained_model=merged_model,
            finetuned_model=model_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
        )

        # Create a new model state dict with the base model parameters
        processed_model_state_dict = {}
        for param_name, param_value in merged_model.state_dict().items():
            processed_model_state_dict[param_name] = param_value.clone()

        # Apply SVD processing to 2D parameters
        with torch.no_grad():
            for param_name, param_value in tqdm(
                task_vector.task_vector_param_dict.items(),
                desc=f"Processing parameters for model {idx}",
                leave=False,
            ):
                param_shape = param_value.shape

                if len(param_shape) == 2:
                    # For 2D parameters, perform SVD
                    original_dtype = param_value.dtype
                    param_value = param_value.cuda().to(torch.float32)
                    u, s, v = torch.linalg.svd(param_value, full_matrices=False)

                    # Reconstruct parameter with SVD
                    processed_param = (
                        torch.linalg.multi_dot([u, torch.diag(s), v])
                        .to(original_dtype)
                        .cpu()
                    )

                    # Apply scaling coefficient and add to base model parameter
                    processed_model_state_dict[param_name] = (
                        merged_model.state_dict()[param_name].cpu()
                        + scaling_coefficient * processed_param
                    )
                else:
                    logging.warning(
                        f"Skipping parameter {param_name} with shape {param_shape}"
                    )
                    # For non-2D parameters, use the original task vector value
                    processed_model_state_dict[param_name] = (
                        merged_model.state_dict()[param_name].cpu()
                        + scaling_coefficient * param_value.cpu()
                    )

        # Create a temporary model with processed parameters
        processed_model = copy.deepcopy(merged_model).cpu()
        processed_model.load_state_dict(processed_model_state_dict)

        # Save this processed model
        model_output_path = final_output_base_path / f"{model_name}"
        model_output_path.mkdir(parents=True, exist_ok=True)
        processed_model.save_pretrained(model_output_path)
        tokenizer.save_pretrained(model_output_path)

        print(f"Saved processed model {model_name} ({idx})  to {model_output_path}")

    print("SVD intermediate processing completed!")


def svd_merging(
    merged_model: nn.Module,
    models_to_merge: list,
    exclude_param_names_regex: list,
    scaling_coefficient: float = 1.0,
):
    """
    SVD merging method that uses Singular Value Decomposition to merge models.
    Args:
        merged_model: nn.Module, the base model to merge into
        models_to_merge: list, individual models that need to be merged
        exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        scaling_coefficient: float, scaling coefficient to merge the task vectors
    Returns:
        dict: merged parameters dictionary
    """
    assert isinstance(
        scaling_coefficient, float
    ), "wrong type of scaling_coefficient, should be float!"

    # Get the parameter names to merge
    pretrained_param_dict = {
        param_name: param_value
        for param_name, param_value in merged_model.named_parameters()
    }
    param_names_to_merge = get_param_names_to_merge(
        input_param_names=list(pretrained_param_dict.keys()),
        exclude_param_names_regex=exclude_param_names_regex,
    )

    # Compute task vectors
    print("Computing task vectors...")
    models_to_merge_task_vectors = []
    for model_to_merge in models_to_merge:
        task_vector_dict = {}
        for param_name in param_names_to_merge:
            # Compute difference as task vector
            task_vector_dict[param_name] = (
                model_to_merge.state_dict()[param_name]
                - merged_model.state_dict()[param_name]
            )
        models_to_merge_task_vectors.append(task_vector_dict)

    sv_reduction = 1.0 / len(models_to_merge)
    device = torch.device("cuda")
    first_param_name = list(models_to_merge_task_vectors[0].keys())[0]
    original_dtype = models_to_merge_task_vectors[0][first_param_name].dtype
    print("Computing SVD merging...")

    with torch.no_grad():
        merged_task_vector_dict = {}
        # Process each parameter
        for param_name in tqdm(
            param_names_to_merge, desc="Processing model parameters"
        ):
            # Clear CUDA cache to free memory
            torch.cuda.empty_cache()

            # Check parameter shape
            param_shape = models_to_merge_task_vectors[0][param_name].shape

            if len(param_shape) == 2 and param_name == "lm_head.weight":
                print(f"Processing parameter {param_name}, shape: {param_shape}")
                # Apply SVD merging for 2D tensors

                # Create temporary variables to store merged results
                sum_u = None
                sum_s = None
                sum_v = None

                # Process each model's task vector
                for i, task_vector_dict in enumerate(models_to_merge_task_vectors):
                    # Move parameter to GPU for computation
                    vec = task_vector_dict[param_name].to(device).float()

                    # Compute SVD
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    # Compute reduced index
                    reduced_index_s = int(s.shape[0] * sv_reduction)

                    # Initialize and prepare storage for the first model
                    if i == 0:
                        sum_u = torch.zeros_like(u, device=device)
                        sum_s = torch.zeros_like(s, device=device)
                        sum_v = torch.zeros_like(v, device=device)

                    # Store important components for each model
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                        :, :reduced_index_s
                    ]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                        :reduced_index_s
                    ]
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                        :reduced_index_s, :
                    ]

                # Compute final merged parameter
                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

                # Compute merged result and move back to CPU
                merged_param = (
                    torch.linalg.multi_dot([u_u, v_u, torch.diag(sum_s), u_v, v_v])
                    .to(original_dtype)
                    .cpu()
                )

                # Store merged parameter
                merged_task_vector_dict[param_name] = merged_param

            else:
                # Use simple averaging for non-2D tensors
                merged_param = models_to_merge_task_vectors[0][param_name].clone()
                for i, task_vector_dict in enumerate(
                    models_to_merge_task_vectors[1:], 1
                ):
                    vec = task_vector_dict[param_name]
                    merged_param += (vec - merged_param) / (i + 1)
                merged_task_vector_dict[param_name] = merged_param

        # Create merged task vector and combine with base model
        merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
        merged_params = merged_task_vector.combine_with_pretrained_model(
            pretrained_model=merged_model, scaling_coefficient=scaling_coefficient
        )

    return merged_params


def iso_merging(
    merged_model: nn.Module,
    models_to_merge: list,
    exclude_param_names_regex: list,
    scaling_coefficient: float = 1.0,
):
    """
    ISO merging method, uses SVD and equalizes singular values to reduce interference between task vectors

    Args:
        merged_model: nn.Module, the base model to merge into
        models_to_merge: list, models to be merged
        exclude_param_names_regex: list, regex patterns for parameter names to exclude
        scaling_coefficient: float, scaling coefficient for merging task vectors
    Returns:
        dict: merged parameter dictionary
    """
    assert isinstance(
        scaling_coefficient, float
    ), "wrong type of scaling_coefficient, should be float!"

    models_to_merge_task_vectors = [
        TaskVector(
            pretrained_model=merged_model,
            finetuned_model=model_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
        )
        for model_to_merge in models_to_merge
    ]

    merged_task_vector_dict = {}

    # Process each parameter
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict:
        # Get parameter shape from the first task vector
        param_shape = (
            models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape
        )

        if len(param_shape) == 2:
            # For 2D parameters, perform SVD merging
            with torch.no_grad():
                merged_param_value = (
                    models_to_merge_task_vectors[0]
                    .task_vector_param_dict[param_name]
                    .clone()
                )
                for index in range(1, len(models_to_merge_task_vectors)):
                    merged_param_value = (
                        merged_param_value
                        + models_to_merge_task_vectors[index].task_vector_param_dict[
                            param_name
                        ]
                    )

            # SVD and equalize singular values
            original_dtype = merged_param_value.dtype
            merged_param_value = merged_param_value.cuda().to(torch.float32)
            u, s, v = torch.linalg.svd(merged_param_value, full_matrices=False)
            avg_singular_value = torch.mean(s)
            avg_s = torch.diag(torch.full_like(s, avg_singular_value))

            merged_param = torch.linalg.multi_dot([u, avg_s, v]).to(original_dtype)

            # Store merged parameter
            merged_task_vector_dict[param_name] = merged_param
        else:
            # For non-2D parameters, compute the average of all task vectors
            print(param_name)
            with torch.no_grad():
                merged_param = (
                    models_to_merge_task_vectors[0]
                    .task_vector_param_dict[param_name]
                    .clone()
                )
                for i, task_vector in enumerate(models_to_merge_task_vectors[1:], 1):
                    vec = task_vector.task_vector_param_dict[param_name]
                    merged_param += (vec - merged_param) / (i + 1)

                merged_task_vector_dict[param_name] = merged_param

    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model, scaling_coefficient=scaling_coefficient
    )
    return merged_params


def wudi_merging(
    merged_model: nn.Module,
    models_to_merge: list,
    exclude_param_names_regex: list,
    scaling_coefficient: float = 1.0,
):
    """
    Wudi merging method that optimizes a merging vector to minimize interference between task vectors

    Args:
        merged_model: nn.Module, the base model to merge into
        models_to_merge: list, individual models that need to be merged
        exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        scaling_coefficient: float, scaling coefficient to apply to the final merged vector
    Returns:
        dict: merged parameters dictionary
    """
    assert isinstance(
        scaling_coefficient, float
    ), "wrong type of scaling_coefficient, should be float!"
    models_to_merge_task_vectors = [
        TaskVector(
            pretrained_model=merged_model,
            finetuned_model=model_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
        )
        for model_to_merge in models_to_merge
    ]

    def get_redundant_task_vector(param_name, vectors, iter_num=300):
        """
        Optimize a merging vector to minimize interference between task vectors

        Args:
            param_name: str, name of the parameter
            vectors: torch.Tensor, stacked task vectors to merge
            iter_num: int, number of optimization iterations
        Returns:
            torch.Tensor: optimized merging vector
        """
        original_dtype = vectors.dtype
        vectors = vectors.to(torch.float32).cuda()

        # Initialize with sum of vectors as starting point
        merging_vector = torch.nn.Parameter(torch.sum(vectors, dim=0))

        # Setup optimizer
        optimizer = torch.optim.Adam([merging_vector], lr=1e-5)

        # Compute L2 norms for normalization
        l2_norms = torch.square(
            torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=-1)
        )

        # Optimization loop
        for i in tqdm(range(iter_num), desc=f"Optimizing {param_name}", leave=False):
            # Calculate disturbing vectors
            disturbing_vectors = merging_vector.unsqueeze(0) - vectors
            # Calculate inner products
            inner_product = torch.matmul(disturbing_vectors, vectors.transpose(1, 2))
            # Calculate loss
            loss = torch.sum(
                torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1)
            )
            print(f"Step {i}, loss: {loss.item()}")
            # Gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return merging_vector.data.detach().to(original_dtype)  # .cpu()

    merged_task_vector_dict = {}

    # Process each parameter
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict:
        if (
            len(
                models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape
            )
            == 2
            and "lm_head" not in param_name
        ):
            print(
                f"Processing {param_name} with shape {models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape}"
            )

            # Stack task vectors for this parameter
            values = torch.stack(
                [
                    task_vector.task_vector_param_dict[param_name]
                    for task_vector in models_to_merge_task_vectors
                ]
            )

            # Get optimized merging vector
            merging_vector = get_redundant_task_vector(param_name, values, iter_num=300)
            merged_task_vector_dict[param_name] = merging_vector

    # Handle non-attention weights using simple averaging for completeness
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            print(f"Using simple averaging for {param_name}")
            merged_param = (
                models_to_merge_task_vectors[0]
                .task_vector_param_dict[param_name]
                .clone()
            )
            for i, task_vector in enumerate(models_to_merge_task_vectors[1:], 1):
                vec = task_vector.task_vector_param_dict[param_name]
                merged_param += (vec - merged_param) / (i + 1)
            merged_task_vector_dict[param_name] = merged_param

    # Create merged task vector and combine with base model
    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model, scaling_coefficient=scaling_coefficient
    )

    return merged_params


def wudi_merging2(
    merged_model: nn.Module,
    models_to_merge: list,
    exclude_param_names_regex: list,
    scaling_coefficient: float = 1.0,
):
    """
    Wudi merging2 method that optimizes a merging vector to minimize interference between task vectors

    Args:
        merged_model: nn.Module, the base model to merge into
        models_to_merge: list, individual models that need to be merged
        exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        scaling_coefficient: float, scaling coefficient to apply to the final merged vector
    Returns:
        dict: merged parameters dictionary
    """
    assert isinstance(
        scaling_coefficient, float
    ), "wrong type of scaling_coefficient, should be float!"
    models_to_merge_task_vectors = [
        TaskVector(
            pretrained_model=merged_model,
            finetuned_model=model_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
        )
        for model_to_merge in models_to_merge
    ]

    def get_redundant_task_vector(param_name, vectors, iter_num=300):
        """
        Optimize a merging vector to minimize interference between task vectors

        Args:
            param_name: str, name of the parameter
            vectors: torch.Tensor, stacked task vectors to merge
            iter_num: int, number of optimization iterations
        Returns:
            torch.Tensor: optimized merging vector
        """
        original_dtype = vectors.dtype
        vectors = vectors.to(torch.float32)

        average_vector = vectors.mean(dim=0)
        low_rank_list = []
        taskvector_list = []
        for i in range(vectors.shape[0]):
            vector = vectors[i]
            u, s, v = torch.linalg.svd(vector, full_matrices=True)
            u2, s2, v2 = torch.linalg.svd(vector - average_vector, full_matrices=False)
            reduced_index_s = int(s.shape[0] / vectors.shape[0])
            u2 = u2[:, :reduced_index_s]
            s2 = s2[:reduced_index_s]
            v2 = v2[:reduced_index_s, :]
            s_mask = torch.zeros_like(s)
            s_mask[:reduced_index_s] = 1
            s = s * s_mask
            v_mask = torch.zeros_like(v)
            v_mask[:reduced_index_s, :] = 1
            v = v * v_mask  # (n, n)
            S_matrix = torch.zeros(
                vector.shape[0], vector.shape[1], device=s.device
            )  # m x n
            min_dim = min(vector.shape)
            S_matrix[:min_dim, :min_dim] = torch.diag_embed(s)
            low_rank_list.append(S_matrix @ v)
            taskvector_list.append(u2 @ torch.diag_embed(s2) @ v2 + average_vector)
        low_rank = torch.stack(low_rank_list)
        taskvector = torch.stack(taskvector_list)

        # Initialize with sum of vectors as starting point
        merging_vector = torch.nn.Parameter(torch.sum(vectors, dim=0))

        # Setup optimizer
        optimizer = torch.optim.Adam([merging_vector], lr=1e-6)

        # Compute L2 norms for normalization
        l2_norms = torch.square(
            torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=-1)
        )

        # Optimization loop
        for i in tqdm(range(iter_num), desc=f"Optimizing {param_name}", leave=False):
            # Calculate disturbing vectors
            disturbing_vectors = merging_vector.unsqueeze(0) - taskvector
            # Calculate inner products
            inner_product = torch.matmul(disturbing_vectors, low_rank.transpose(1, 2))
            # Calculate loss
            loss = torch.sum(
                torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1)
            )
            if i % 10 == 0:
                print(f"Step {i}, loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return merging_vector.data.detach().to(original_dtype)

    merged_task_vector_dict = {}

    # Process each parameter
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict:
        if (
            len(
                models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape
            )
            == 2
            and "lm_head" not in param_name
        ):
            print(
                f"Processing {param_name} with shape {models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape}"
            )

            # Stack task vectors for this parameter
            values = torch.stack(
                [
                    task_vector.task_vector_param_dict[param_name]
                    for task_vector in models_to_merge_task_vectors
                ]
            )

            # Get optimized merging vector
            merging_vector = get_redundant_task_vector(param_name, values, iter_num=300)
            merged_task_vector_dict[param_name] = merging_vector

    # Handle non-attention weights using simple averaging for completeness
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            print(f"Using simple averaging for {param_name}")
            merged_param = (
                models_to_merge_task_vectors[0]
                .task_vector_param_dict[param_name]
                .clone()
            )
            for i, task_vector in enumerate(models_to_merge_task_vectors[1:], 1):
                vec = task_vector.task_vector_param_dict[param_name]
                merged_param += (vec - merged_param) / (i + 1)
            merged_task_vector_dict[param_name] = merged_param

    # Create merged task vector and combine with base model
    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model, scaling_coefficient=scaling_coefficient
    )

    return merged_params


def load_models(
    cache_dir,
    merged_model_dtype=torch.float16,
    models_to_merge_dtype=torch.float16,
    base_model_path: str = None,
):
    """
    Load models with fixed structure: base model + 5 specialized models

    Args:
        cache_dir: Base dir path where models are stored

    Returns:
        list: List containing loaded models and tokenizer
    """
    # Fixed model structure
    # model_suffixes = ['', '_OCR', '_VQA', '_Geometry', '_Chart', '_Grounding']
    # model_keys = ['a', 'b', 'c', 'd', 'e', 'f']
    model_names = [
        "InternVL2_5-1B" if base_model_path is None else base_model_path,
        "InternVL2_5-1B_OCR",
        "InternVL2_5-1B_VQA",
        "InternVL2_5-1B_Geometry",
        "InternVL2_5-1B_Chart",
        "InternVL2_5-1B_Grounding",
    ]
    if base_model_path is not None:
        logging.warning(
            f"Using base model {base_model_path} instead of InternVL2_5-1B!!!"
        )
        time.sleep(3)
    tokenizer = None

    models = []
    for i, model_name in enumerate(model_names):
        if "/" in model_name:
            model_path = model_name
        else:
            model_path = cache_dir + "/" + model_name
        print(f"Loading model {model_name} from {model_path}")
        models.append(
            AutoModel.from_pretrained(
                model_path,
                torch_dtype=merged_model_dtype if i == 0 else models_to_merge_dtype,
                trust_remote_code=True,
            ).eval()
        )

        if tokenizer is None:
            print(f"Loading tokenizer from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, use_fast=False
            )

    return models, tokenizer


def expert_merging_merging(
    merged_model: nn.Module,
    models_to_merge: list,
    tokenizer,
    logger,
    exclude_param_names_regex: list,
    data_dir: str,
    samples_per_task=None,
    default_samples_per_task: int = 100,
    num_epochs: int = 3,
    temperature: float = 1.0,
    learning_rate: float = 1e-6,
    log_dir: str = "logs/expert_merging",
    mixed_precision: str = "fp16",
    gradient_accumulation_steps: int = 8,
    seed: int = 42,
    sparsity_reg_weight: float = 0.1,
    hidden_states_layers: List[int] = None,
    hidden_states_weight: float = 0.1,
    weight_coeffs_init_value=0.1,
    loss_alphas: Dict[str, float] = None,
    coeffs_size_dict: dict = {},
    task_vector_coeffs_dict: dict = {},
):
    """
    Expert merging model merging using learnable weight and direction vectors.
    This method is more memory-efficient as it doesn't require loading all teacher models simultaneously.

    Args:
        merged_model: Base model to merge into
        models_to_merge: List of models to merge
        tokenizer: Tokenizer for the models
        logger: Logger instance
        exclude_param_names_regex: Regex patterns for parameters to exclude from merging
        data_dir: Directory containing JSON files
        samples_per_task: Number of samples to use per task
        default_samples_per_task: Default number of samples per task if not specified
        num_epochs: Number of training epochs
        temperature: Temperature for logits alignment
        learning_rate: Learning rate for optimization
        log_dir: Directory for TensorBoard logs
        mixed_precision: Mixed precision mode ("no", "fp16", "bf16")
        gradient_accumulation_steps: Number of gradient accumulation steps
        seed: Random seed for reproducibility
        sparsity_reg_weight: Sparsity regularization weight
        hidden_states_layers: Layers to use for hidden states alignment
        hidden_states_weight: Weight for hidden states loss component
        weight_coeffs_init_value: float or List[float], initial value for weight coefficients

    Returns:
        Dictionary of merged parameters compatible with existing merge_models interface
    """
    from dataset import ExpertMergingDataset
    from parametric_task_vector_model import ExpertMergingTrainer

    logger.info("Starting Expert Merging...")
    logger.info(f"Using merging parameters:")
    logger.info(f"  Temperature: {temperature}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Number of epochs: {num_epochs}")
    logger.info(f"  Sparsity regularization weight: {sparsity_reg_weight}")
    logger.info(f"  Loss alphas: {loss_alphas}")
    logger.info(f"  Weight Coeffs Init Value: {weight_coeffs_init_value}")

    if hidden_states_layers:
        logger.info(f"  Hidden states layers: {hidden_states_layers}")
        logger.info(f"  Hidden states weight: {hidden_states_weight}")

    # Create dataset
    dataset = ExpertMergingDataset(
        data_dir=data_dir,
        samples_per_task=samples_per_task,
        default_samples_per_task=default_samples_per_task,
        image_size=448,
    )

    # Create parametric trainer
    trainer = ExpertMergingTrainer(
        base_model=merged_model,
        teacher_models=models_to_merge,
        tokenizer=tokenizer,
        temperature=temperature,
        learning_rate=learning_rate,
        log_dir=log_dir,
        exclude_param_names_regex=exclude_param_names_regex,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        seed=seed,
        sparsity_reg_weight=sparsity_reg_weight,
        hidden_states_layers=hidden_states_layers,
        hidden_states_weight=hidden_states_weight,
        loss_alphas=loss_alphas,
        weight_coeffs_init_value=weight_coeffs_init_value,
        coeffs_size_dict=coeffs_size_dict,
        task_vector_coeffs_dict=task_vector_coeffs_dict,
    )

    # Train parametric student model
    trained_student = trainer.train(dataset=dataset, num_epochs=num_epochs)

    # Extract final merged parameters
    merged_params = trained_student.get_final_merged_params()

    logger.info("Expert Merging completed successfully!")
    return merged_params

def merge_models(
    args,
    cache_dir,
    merge_method="wudi2",
    scaling_coefficient=0.1,
    output_path="merged_model",
    run_name="default",
    exclude_param_names_regex=None,
    logger=None,
    # ExpertMerging-specific parameters
    data_dir="../dataset",
    samples_per_task=None,
    default_samples_per_task: int = 100,
    num_epochs=1,
    temperature=1.0,
    learning_rate=1e-6,
    log_dir="logs/expert_merging",
    mixed_precision="fp16",
    gradient_accumulation_steps=8,
    seed=42,
    base_model_path=None,
    sparsity_reg_weight=0.1,
    # Hidden states alignment parameters
    hidden_states_layers=None,
    hidden_states_weight=0.1,
    # Task-specific loss weights
    loss_alphas=None,
    **kwargs,
):
    """
    Merge multiple models using specified method

    Args:
        cache_dir: Base path where models are stored (will load base + 5 specialized models)
        merge_method: Merging method to use
        scaling_coefficient: Scaling coefficient for merging
        output_path: Base output directory
        run_name: Name for this run
        exclude_param_names_regex: List of regex patterns for parameters to exclude
        logger: Logger instance
    """
    if exclude_param_names_regex is None:
        exclude_param_names_regex = [
            "vision_model.*",
            ".*lm_head.*",
            ".*norm.*",
            ".*embed_tokens.*",
            ".*bias.*",
        ]

    # Load models with fixed structure
    models, tokenizer = load_models(cache_dir, base_model_path=base_model_path)

    logger.info("Start merging models...")

    if (
        merge_method != "ties"
        and merge_method != "dare_ties"
    ):
        models = [m.cuda() for m in models]

    base_model = models[0]
    base_state_dict = base_model.state_dict()

    models_to_merge = [m for m in models[1:]]

    logger.info(f"Start merging models using {merge_method}...")
    base_state_dict = base_model.state_dict()

    if merge_method == "task_arithmetic":
        logger.info("Running task_arithmetic...")
        merged_params = task_arithmetic(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient,
        )
    elif merge_method == "ties":
        logger.info("Running ties_merging...")
        merged_params = ties_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            param_value_mask_rate=0.8,
            scaling_coefficient=scaling_coefficient,
        )
    elif merge_method == "dare_ta":
        logger.info("Running Dare task_arithmetic...")
        weight_mask_rates = [0.2 for _ in range(len(models_to_merge))]
        with torch.no_grad():
            new_models_to_merge = models_to_merge
            for new_model_to_merge, weight_mask_rate in zip(
                new_models_to_merge, weight_mask_rates
            ):
                # for each individual model, mask its weight
                masked_param_dict = mask_model_weights(
                    finetuned_model=new_model_to_merge,
                    pretrained_model=base_model,
                    exclude_param_names_regex=exclude_param_names_regex,
                    weight_format="delta_weight",
                    weight_mask_rate=weight_mask_rate,
                    use_weight_rescale=True,
                    mask_strategy="random",
                )
                copy_params_to_model(params=masked_param_dict, model=new_model_to_merge)

        merged_params = task_arithmetic(
            merged_model=base_model,
            models_to_merge=new_models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient,
        )
    elif merge_method == "dare_ties":
        logger.info("Running Dare ties_merging...")
        weight_mask_rates = [0.2 for _ in range(len(models_to_merge))]
        with torch.no_grad():
            new_models_to_merge = models_to_merge
            for new_model_to_merge, weight_mask_rate in zip(
                new_models_to_merge, weight_mask_rates
            ):
                # for each individual model, mask its weight
                masked_param_dict = mask_model_weights(
                    finetuned_model=new_model_to_merge,
                    pretrained_model=base_model,
                    exclude_param_names_regex=exclude_param_names_regex,
                    weight_format="delta_weight",
                    weight_mask_rate=weight_mask_rate,
                    use_weight_rescale=True,
                    mask_strategy="random",
                )
                copy_params_to_model(params=masked_param_dict, model=new_model_to_merge)

        merged_params = ties_merging(
            merged_model=base_model,
            models_to_merge=new_models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            param_value_mask_rate=0.8,
            scaling_coefficient=scaling_coefficient,
        )
    elif merge_method == "svd":
        logger.info("Running svd_merging...")
        merged_params = svd_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient,
        )
    elif merge_method == "iso":
        logger.info("Running iso_merging...")
        merged_params = iso_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient,
        )
    elif merge_method == "wudi":
        logger.info("Running wudi_merging...")
        merged_params = wudi_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient,
        )
    elif merge_method == "wudi2":
        logger.info("Running wudi v2...")
        merged_params = wudi_merging2(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient,
        )
    elif merge_method == "weight_average":
        logger.info("Running weight average...")
        merged_params = weight_average(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient,
        )
    elif merge_method == "svd_intermediate":
        logger.info("Running SVD intermediate processing...")
        # This method saves processed models directly, so we don't need merged_params
        svd_intermediate_processing(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            output_path=output_path,
            tokenizer=tokenizer,
            run_name=run_name,
            # scaling_coefficient=scaling_coefficient
        )
        # Return None to indicate that no merged model needs to be saved
        return None
    elif merge_method == "expert_merging":
        logger.info("Running Expert Merging...")
        if data_dir is None:
            raise ValueError(
                "data_dir must be provided for expert merging method"
            )

        # Use expert merging - returns merged parameters
        merged_params = expert_merging_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            tokenizer=tokenizer,
            logger=logger,
            exclude_param_names_regex=exclude_param_names_regex,
            data_dir=data_dir,
            samples_per_task=samples_per_task,
            default_samples_per_task=default_samples_per_task,
            num_epochs=num_epochs,
            temperature=temperature,
            learning_rate=learning_rate,
            log_dir=log_dir,
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            seed=seed,
            sparsity_reg_weight=sparsity_reg_weight,
            hidden_states_layers=hidden_states_layers,
            hidden_states_weight=hidden_states_weight,
            loss_alphas=loss_alphas,
            weight_coeffs_init_value=scaling_coefficient,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown merge_method: {merge_method}")

    # Apply merged parameters to base model (skip for ExpertMerging method)
    if merged_params is not None:
        for key in merged_params:
            if key in base_state_dict:
                base_state_dict[key] = merged_params[key]
        base_model.load_state_dict(base_state_dict)

    # Create output directory structure: output_path/method/run_name/model
    final_output_path = Path(output_path) / merge_method / run_name / "model"
    final_output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving merged model to {final_output_path}")
    base_model.save_pretrained(final_output_path)
    tokenizer.save_pretrained(final_output_path)

    return base_model


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ]
    )
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    try:
        from decord import VideoReader, cpu
    except ImportError:
        print("Warning: decord not installed, video tests will be skipped")
        return None, None

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(
        bound, fps, max_frame, first_idx=0, num_segments=num_segments
    )
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(
            img, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def run_test_cases(model, tokenizer):
    """Run comprehensive test cases for the merged model"""
    print("\n" + "=" * 50)
    print("Running test cases...")
    print("=" * 50)

    generation_config = dict(max_new_tokens=1024, do_sample=False)

    # Test cases list
    test_cases = [
        {
            "name": "Pure text conversation",
            "tests": ["Hello, who are you?", "Can you tell me a story?"],
        }
    ]

    # Pure-text conversation tests
    print("\n1. Pure-text conversation tests:")
    history = None
    for i, question in enumerate(test_cases[0]["tests"]):
        print(f"\nTest {i+1}: {question}")
        try:
            response, history = model.chat(
                tokenizer,
                None,
                question,
                generation_config,
                history=history,
                return_history=True,
            )
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {str(e)}")

    # Image-based tests (if image files exist)
    image_tests = ["./examples/image1.jpg", "./examples/image2.jpg"]

    available_images = [img for img in image_tests if os.path.exists(img)]

    if available_images:
        print(f"\n2. Image-based tests (found {len(available_images)} images):")

        # Single image test
        if len(available_images) >= 1:
            try:
                pixel_values = (
                    load_image(available_images[0], max_num=12).to(torch.float16).cuda()
                )

                # Single-image single-round conversation
                question = "<image>\nPlease describe the image shortly."
                print(f"\nSingle image test: {question}")
                response = model.chat(
                    tokenizer, pixel_values, question, generation_config
                )
                print(f"Response: {response}")

                # Single-image multi-round conversation
                question = "<image>\nPlease describe the image in detail."
                print(f"\nMulti-round test 1: {question}")
                response, history = model.chat(
                    tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    history=None,
                    return_history=True,
                )
                print(f"Response: {response}")

                question = "Please write a poem according to the image."
                print(f"\nMulti-round test 2: {question}")
                response, history = model.chat(
                    tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    history=history,
                    return_history=True,
                )
                print(f"Response: {response}")

            except Exception as e:
                print(f"Error in single image test: {str(e)}")

        # Multi-image test
        if len(available_images) >= 2:
            try:
                pixel_values1 = (
                    load_image(available_images[0], max_num=12).to(torch.float16).cuda()
                )
                pixel_values2 = (
                    load_image(available_images[1], max_num=12).to(torch.float16).cuda()
                )
                pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

                # Combined images test
                question = "<image>\nDescribe the two images in detail."
                print(f"\nMulti-image test 1: {question}")
                response, history = model.chat(
                    tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    history=None,
                    return_history=True,
                )
                print(f"Response: {response}")

                question = "What are the similarities and differences between these two images."
                print(f"\nMulti-image test 2: {question}")
                response, history = model.chat(
                    tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    history=history,
                    return_history=True,
                )
                print(f"Response: {response}")

                # Separate images test
                num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
                question = "Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail."
                print(f"\nSeparate images test: {question}")
                response, history = model.chat(
                    tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True,
                )
                print(f"Response: {response}")

                # Batch inference test
                questions = ["<image>\nDescribe the image in detail."] * len(
                    num_patches_list
                )
                print(f"\nBatch inference test:")
                responses = model.batch_chat(
                    tokenizer,
                    pixel_values,
                    num_patches_list=num_patches_list,
                    questions=questions,
                    generation_config=generation_config,
                )
                for i, (question, response) in enumerate(zip(questions, responses)):
                    print(f"Batch {i+1} - Response: {response}")

            except Exception as e:
                print(f"Error in multi-image test: {str(e)}")
    else:
        print("\n2. Image-based tests: Skipped (no image files found)")

    # Video test (if video file exists)
    video_path = "./examples/red-panda.mp4"
    if os.path.exists(video_path):
        print(f"\n3. Video-based test:")
        try:
            pixel_values, num_patches_list = load_video(
                video_path, num_segments=8, max_num=1
            )
            if pixel_values is not None:
                pixel_values = pixel_values.to(torch.float16).cuda()
                video_prefix = "".join(
                    [f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))]
                )
                question = video_prefix + "What is the red panda doing?"
                print(f"\nVideo test 1: {question}")
                response, history = model.chat(
                    tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True,
                )
                print(f"Response: {response}")

                question = "Describe this video in detail."
                print(f"\nVideo test 2: {question}")
                response, history = model.chat(
                    tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    num_patches_list=num_patches_list,
                    history=history,
                    return_history=True,
                )
                print(f"Response: {response}")
            else:
                print("Video test skipped (decord not available)")
        except Exception as e:
            print(f"Error in video test: {str(e)}")
    else:
        print("\n3. Video-based test: Skipped (no video file found)")

    print("\n" + "=" * 50)
    print("Test cases completed!")
    print("=" * 50)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Model Merging Tool")

    # Required arguments
    parser.add_argument(
        "--method",
        type=str,
        default="expert_merging",
        choices=[
            "task_arithmetic",
            "ties",
            "dare_ta",
            "dare_ties",
            "svd",
            "iso",
            "wudi",
            "wudi2",
            "expert_merging",
            "svd_intermediate",
            "weight_average",
        ],
        help="Merging method to use",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=False,
        default=None,
        help="Cache directory for models",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default="results/logs",
        help="Base output directory",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=False,
        default=None,
        help="Name for this run (default: timestamp)",
    )

    # Optional arguments
    parser.add_argument(
        "--scaling_coefficient",
        type=str,
        default="0.1",
        help="Scaling coefficient for merging (default: 0.1). Can be a float or a list of floats, e.g., '1.0' or '[1.0, 2.0, 3.0]'",
    )
    parser.add_argument(
        "--exclude_params",
        type=str,
        nargs="*",
        default=[
            "vision_model.*",
            ".*lm_head.*",
            ".*norm.*",
            ".*embed_tokens.*",
            ".*bias.*",
        ],
        help="Regex patterns for parameters to exclude from merging",
    )
    parser.add_argument(
        "--run_tests", action="store_true", help="Run test cases after merging"
    )

    # ExpertMerging arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../dataset",
        help="Directory containing JSON files for ExpertMerging",
    )
    parser.add_argument(
        "--samples_per_task",
        type=int,
        default=5,
        help="Number of samples to use per task for ExpertMerging (default value for all tasks)",
    )
    parser.add_argument(
        "--samples_per_task_dict",
        type=str,
        default=None,
        help='JSON string specifying per-task sample counts, e.g., \'{"VizWiz": 5, "GQA": 19}\'',
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs for ExpertMerging",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for logits alignment"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-1,
        help="Learning rate for ExpertMerging training",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Path to base model to use instead of InternVL2_5-1B",
    )
    parser.add_argument(
        "--sparsity_reg_weight",
        type=float,
        default=5,
        help="Sparsity regularization weight",
    )

    # Hidden states alignment arguments
    parser.add_argument(
        "--hidden_states_layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to compute hidden states loss, e.g., '0,5,10,15,20,25'",
    )
    parser.add_argument(
        "--hidden_states_weight",
        type=float,
        default=1.0,
        help="Weight for hidden states loss",
    )

    # Task-specific loss weights
    parser.add_argument(
        "--loss_alphas",
        type=str,
        default=None,
        help='JSON string specifying per-task loss weights, e.g., \'{"VizWiz": 1.2, "GQA": 0.8}\'',
    )
    parser.add_argument(
        "--task_vector_coeffs_dict",
        type=str,
        default=None,
        help='JSON string specifying per-task loss weights, e.g., \'{"VizWiz": 1.2, "GQA": 0.8}\'',
    )
    parser.add_argument(
        "--coeffs_size_dict",
        type=str,
        default=None,
        help='JSON string specifying per-task loss weights, e.g., \'{"VizWiz": 1.2, "GQA": 0.8}\'',
    )
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    if args.run_name is None:
        args.run_name = f"{time.strftime('%m%d-%H%M%S')}"

    # Setup logging
    logger = setup_logging(args.output_path, args.method, args.run_name)

    if args.cache_dir is None:
        from dotenv import load_dotenv

        dotenv_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", ".env")
        )
        load_dotenv(override=True, dotenv_path=dotenv_path)
        logger.info(f".env path: {dotenv_path}")
        args.cache_dir = os.getenv("CACHE_DIR")

    from transformers import set_seed
    if args.seed < 0:
        import random
        args.seed = random.randint(0, 100000)
    set_seed(args.seed)
    save_config(args, args.output_path)

    # Parse scaling_coefficient to support both float and List[float]
    try:
        # Try to parse as JSON (for list format)
        args.scaling_coefficient = json.loads(args.scaling_coefficient)
    except json.JSONDecodeError:
        # If that fails, try to parse as a single float
        try:
            args.scaling_coefficient = float(args.scaling_coefficient)
        except ValueError:
            logger.error(
                f"Invalid scaling_coefficient format: {args.scaling_coefficient}"
            )
            raise ValueError(
                f"scaling_coefficient must be a float or a list of floats, got: {args.scaling_coefficient}"
            )

    logger.info(f"Starting model merging with method: {args.method}")
    logger.info(f"Cache directory: {args.cache_dir}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Run name: {args.run_name}")
    logger.info(f"Scaling coefficient: {args.scaling_coefficient}")


    # Process samples_per_task_dict if provided
    samples_per_task = args.samples_per_task
    if args.samples_per_task_dict is not None:
        try:
            samples_per_task = json.loads(args.samples_per_task_dict)
            logger.info(f"Using per-task sample counts: {samples_per_task}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing samples_per_task_dict: {e}")
            logger.info("Falling back to samples_per_task for all tasks")

    # Process hidden_states_layers if provided
    hidden_states_layers = None
    if args.hidden_states_layers is not None:
        try:
            hidden_states_layers = [
                int(layer)
                for layer in args.hidden_states_layers.replace(" ", "").split(",")
            ]
            logger.info(f"Using hidden states layers: {hidden_states_layers}")
        except ValueError as e:
            logger.error(f"Error parsing hidden_states_layers: {e}")
            logger.info("Hidden states loss will be disabled")

    args.loss_alphas = json.loads(args.loss_alphas) if args.loss_alphas else None
    args.coeffs_size_dict = (
        json.loads(args.coeffs_size_dict) if args.coeffs_size_dict else None
    )
    args.task_vector_coeffs_dict = (
        json.loads(args.task_vector_coeffs_dict)
        if args.task_vector_coeffs_dict
        else None
    )

    # Perform model merging
    merged_model = merge_models(
        args=args,
        cache_dir=args.cache_dir,
        merge_method=args.method,
        scaling_coefficient=args.scaling_coefficient,
        output_path=args.output_path,
        run_name=args.run_name,
        exclude_param_names_regex=args.exclude_params,
        logger=logger,
        log_dir=Path(args.output_path) / args.method / args.run_name / "board",
        # ExpertMerging parameters
        data_dir=args.data_dir,
        samples_per_task=samples_per_task,
        default_samples_per_task=args.samples_per_task,
        num_epochs=args.num_epochs,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        base_model_path=args.base_model_path,
        sparsity_reg_weight=args.sparsity_reg_weight,
        # Hidden states alignment parameters
        hidden_states_layers=hidden_states_layers,
        hidden_states_weight=args.hidden_states_weight,
        loss_alphas=args.loss_alphas,
        task_vector_coeffs_dict=args.task_vector_coeffs_dict,
        coeffs_size_dict=args.coeffs_size_dict,
    )

    logger.info("Model merging completed successfully!")

    # Run tests if requested
    if args.run_tests:
        logger.info("Running test cases...")
        # Load tokenizer for testing
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(args.cache_dir, "InternVL2_5-1B"),
            trust_remote_code=True,
            use_fast=False,
        )
        run_test_cases(merged_model, tokenizer)
        logger.info("Test cases completed!")


if __name__ == "__main__":
    main()
