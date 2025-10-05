import re
from typing import Any, Dict, List

import torch
import torch.nn as nn


def get_param_names_to_merge(input_param_names: list, exclude_param_names_regex: list):
    """
    get the names of parameters that need to be merged
    :param input_param_names: list, names of input parameters
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :return:
    """
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any(
            [
                re.match(exclude_pattern, param_name)
                for exclude_pattern in exclude_param_names_regex
            ]
        )
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge


class TaskVector:
    def __init__(
        self,
        pretrained_model: nn.Module = None,
        finetuned_model: nn.Module = None,
        exclude_param_names_regex: list = None,
        task_vector_param_dict: dict = None,
    ):
        """
        Task vector. Initialize the task vector from a pretrained model and a finetuned model, or
        directly passing the task_vector_param_dict dictionary.
        :param pretrained_model: nn.Module, pretrained model
        :param finetuned_model: nn.Module, finetuned model
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param task_vector_param_dict: dict, task vector to initialize self.task_vector_param_dict
        """
        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
            self.finetuned_model_name = None
        else:
            self.task_vector_param_dict = {}
            self.finetuned_model_name = finetuned_model.name_or_path
            pretrained_param_dict = {
                param_name: param_value
                for param_name, param_value in pretrained_model.named_parameters()
            }
            finetuned_param_dict = {
                param_name: param_value
                for param_name, param_value in finetuned_model.named_parameters()
            }
            param_names_to_merge = get_param_names_to_merge(
                input_param_names=list(pretrained_param_dict.keys()),
                exclude_param_names_regex=exclude_param_names_regex,
            )
            with torch.no_grad():
                for param_name in param_names_to_merge:
                    self.task_vector_param_dict[param_name] = (
                        finetuned_param_dict[param_name]
                        - pretrained_param_dict[param_name]
                    )


def move_to_device(data: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Move data to the specified device.
    :param data: Dictionary containing tensors or other data types
    :param device: Target device (e.g., 'cuda', 'cpu')
    :return: Data moved to the specified device
    """
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
    }


def scale_tensor_by_coeffs(
    param_tensor: torch.Tensor, coeff_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Splits a parameter tensor into K parts and scales each part by a corresponding coefficient.

    Args:
        param_tensor (torch.Tensor): The input parameter tensor. Can be 1D or 2D.
        coeff_tensor (torch.Tensor): A 1D tensor of coefficients. The number of
                                     elements (K) determines how many parts
                                     the param_tensor is split into.

    Returns:
        torch.Tensor: A new tensor with the same shape as param_tensor,
                      where each part has been scaled.
    """
    # --- 0. Input Validation ---
    if not isinstance(param_tensor, torch.Tensor) or not isinstance(
        coeff_tensor, torch.Tensor
    ):
        raise TypeError("Both inputs must be torch.Tensors.")
    if coeff_tensor.dim() != 1:
        raise ValueError("coeff_tensor must be a 1D tensor.")
    if coeff_tensor.numel() == 0:
        raise ValueError("coeff_tensor cannot be empty.")

    # --- 1. Store original shape and flatten the parameter tensor ---
    original_shape = param_tensor.shape
    flat_tensor = param_tensor.flatten()
    k = coeff_tensor.numel()

    if k == 1:
        # Speedup for single coefficient case
        return param_tensor * coeff_tensor[0]

    # --- 2. Split the flattened tensor into K chunks ---
    # torch.chunk will handle cases where the tensor size is not perfectly divisible by k.
    chunks: List[torch.Tensor] = torch.chunk(flat_tensor, chunks=k, dim=0)

    # --- 3. Scale each chunk by the corresponding coefficient ---
    # Use a list comprehension for a concise and efficient implementation.
    scaled_chunks = [chunk * coeff for chunk, coeff in zip(chunks, coeff_tensor)]

    # --- 4. Concatenate the scaled chunks back into a single flat tensor ---
    result_flat = torch.cat(scaled_chunks, dim=0)

    # --- 5. Reshape the result to the original shape ---
    result_tensor = result_flat.reshape(original_shape)

    return result_tensor

def activate_func(x: torch.Tensor):
    return torch.sigmoid(x)

def inverse_activate_func(x: torch.Tensor):
    return torch.logit(x)
