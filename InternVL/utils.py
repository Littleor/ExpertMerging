import re

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


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
        else:
            self.task_vector_param_dict = {}
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

    def __add__(self, other):
        """
        add task vector
        :param other: TaskVector to add, at right side
        :return:
        """
        assert isinstance(
            other, TaskVector
        ), "addition of TaskVector can only be done with another TaskVector!"
        new_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name in self.task_vector_param_dict:
                assert (
                    param_name in other.task_vector_param_dict.keys()
                ), f"param_name {param_name} is not contained in both task vectors!"
                new_task_vector_param_dict[param_name] = (
                    self.task_vector_param_dict[param_name]
                    + other.task_vector_param_dict[param_name]
                )
        return TaskVector(task_vector_param_dict=new_task_vector_param_dict)

    def __radd__(self, other):
        """
        other + self = self + other
        :param other: TaskVector to add, at left side
        :return:
        """
        return self.__add__(other)

    def combine_with_pretrained_model(
        self, pretrained_model: nn.Module, scaling_coefficient: float = 1.0
    ):
        """
        combine the task vector with pretrained model
        :param pretrained_model: nn.Module, pretrained model
        :param scaling_coefficient: float, scaling coefficient to merge the task vector
        :return:
        """
        pretrained_param_dict = {
            param_name: param_value
            for param_name, param_value in pretrained_model.named_parameters()
        }

        with torch.no_grad():
            merged_params = {}
            for param_name in self.task_vector_param_dict:
                merged_params[param_name] = (
                    pretrained_param_dict[param_name]
                    + scaling_coefficient * self.task_vector_param_dict[param_name]
                )

        return merged_params


def task_arithmetic(
    merged_model: nn.Module,
    models_to_merge: list,
    exclude_param_names_regex: list,
    scaling_coefficient: float = 1.0,
):
    """
    task arithmetic method
    :param merged_model: nn.Module, the merged model
    :param models_to_merge: list, individual models that need to be merged
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :param scaling_coefficient: float, scaling coefficient to merge the task vectors
    :return:
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

    # iterate each individual model that needs to be merged
    with torch.no_grad():
        # sum up the task vectors
        merged_task_vector = (
            models_to_merge_task_vectors[0] + models_to_merge_task_vectors[1]
        )
        for index in range(2, len(models_to_merge_task_vectors)):
            merged_task_vector = (
                merged_task_vector + models_to_merge_task_vectors[index]
            )
        # combine with parameters of the merged model based on scaling coefficient
        merged_params = merged_task_vector.combine_with_pretrained_model(
            pretrained_model=merged_model, scaling_coefficient=scaling_coefficient
        )

    return merged_params
