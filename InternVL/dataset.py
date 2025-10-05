import json
import logging
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
import time

from utils import load_image

logger = logging.getLogger(__name__)

TASK_MODEL_ID = {
    # 1. VQA
    "VizWiz": 2,
    "GQA": 2,
    # 2. Geometry
    "MathVista": 3,
    "MathVision": 3,
    # 3. Chart
    "Chart": 4,
    # 4. OCR
    "OCRVQA": 1,
    "TextVQA": 1,
    # 5. Grounding
    "Grounding": 5,
}

IMAGE_TOKEN_MAX_LENGTH = {
    # 1. VQA
    "VizWiz": 6,
    "GQA": 6,
    # 2. Geometry
    "MathVista": 6,
    "MathVision": 6,
    # 3. Chart
    "Chart": 12,
    # 4. OCR
    "OCRVQA": 12,
    "TextVQA": 6,
    # 5. Grounding
    "Grounding": 12,
}


class ExpertMergingDataset(Dataset):
    """Dataset for ExpertMerging training using annotated samples"""

    def __init__(
        self, data_dir: str, samples_per_task=None, default_samples_per_task: int = 100, image_size: int = 448
    ):
        """
        Initialize ExpertMerging dataset

        Args:
            data_dir: Directory containing annotated JSON files
            samples_per_task: Number of samples to use per task (int for all tasks, 
                            dict for per-task samples, None for all samples)
            default_samples_per_task: Default number of samples per task when not specified
            image_size: Input image size for preprocessing
        """
        self.data_dir = Path(data_dir)
        self.samples_per_task = samples_per_task
        self.default_samples_per_task = default_samples_per_task
        self.image_size = image_size

        # Task mapping
        self.task_to_model_idx = TASK_MODEL_ID

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        """Load and sample data from JSON files"""
        for task_name in self.task_to_model_idx.keys():
            json_file = self.data_dir / f"{task_name}.json"
            if not json_file.exists():
                logger.warning(f"{json_file} not found, skipping {task_name}")
                continue

            with open(json_file, "r") as f:
                task_data = json.load(f)

            # Determine number of samples for this task
            if self.samples_per_task is None:
                # Use all samples
                sampled_data = task_data
                logger.warning(f"Using all {len(task_data)} samples for {task_name}")
                time.sleep(5)
            elif isinstance(self.samples_per_task, int):
                # Use the same number of samples for all tasks
                num_samples = min(len(task_data), self.samples_per_task)
                sampled_data = task_data[:num_samples]
                logger.info(f"Using {num_samples} samples for {task_name}")
            elif isinstance(self.samples_per_task, dict):
                # Use per-task sample count
                if task_name in self.samples_per_task:
                    num_samples = min(len(task_data), self.samples_per_task[task_name])
                    sampled_data = task_data[:num_samples]
                    logger.info(f"Using {num_samples} samples for {task_name}")
                else:
                    # Use default sample count
                    num_samples = min(len(task_data), self.default_samples_per_task)
                    sampled_data = task_data[:num_samples]
                    logger.info(f"Using {num_samples} samples for {task_name} (default)")
            else:
                # Fallback to all samples
                sampled_data = task_data
                logger.warning(f"Invalid samples_per_task type, using all samples for {task_name}")
                time.sleep(5)

            # Add task info to each sample
            for sample in sampled_data:
                sample["task_name"] = task_name
                sample["teacher_model_idx"] = self.task_to_model_idx[task_name]
                self.samples.append(sample)

        logger.info(f"Loaded {len(self.samples)} total samples for ExpertMerging")
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load and preprocess image
        image_path = sample["image_path"]
        if not os.path.isabs(image_path):
            # Handle relative paths - assume they're relative to project root
            image_path = os.path.join(image_path)

        pixel_values = load_image(
            image_path,
            input_size=self.image_size,
            max_num=IMAGE_TOKEN_MAX_LENGTH[sample["task_name"]],
        ).to(torch.float16)

        return {
            "pixel_values": pixel_values,
            "question": sample["question"],
            "response": sample["response"],
            "question_id": sample["question_id"],
            "task_name": sample["task_name"],
            "teacher_model_idx": sample["teacher_model_idx"],
            "image_paths": image_path
        }
