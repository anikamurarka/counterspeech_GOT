import json
from datetime import datetime
from pathlib import Path

import torch


def create_output_directory(args):
    """Create a uniquely named directory for saving model outputs."""
    # Format model name to be filesystem-friendly
    formatted_model_name = args.model.replace("/", "-")
    # Count available GPUs for batch size calculation
    available_gpus = torch.cuda.device_count()
    # Generate directory name with hyperparameters and timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
    output_path = f"{args.output_dir}/{formatted_model_name}_lr{args.lr}_bs{args.bs * available_gpus}_op{args.output_len}_ep{args.epoch}_{timestamp}"
    # Ensure directory exists
    ensure_directory_exists(output_path)
    print(output_path)

    return output_path


def ensure_directory_exists(directory_path):
    """Create directory if it doesn't exist."""
    path_obj = Path(directory_path)
    path_obj.mkdir(parents=True, exist_ok=True)


def load_data(args, partition):
    """Load dataset from JSON file for specified partition."""
    file_path = Path(args.data_root) / Path(args.dataset) / f"{partition}.json"
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    return data


def load_dialogue_datasets(args, console):
    """Load train, validation and test datasets with logging."""
    # Log loading process
    console.log(f"""[Model]: Loading {args.model}...\n""")
    console.log(f"[Data]: Reading data...\n")

    # Load datasets for different partitions
    train_data = read_json_dataset(args, 'train')
    validation_data = read_json_dataset(args, 'dev')
    test_data = read_json_dataset(args, 'test')  # TODO change for real

    # Log dataset sizes
    console.log(f"number of train problems: {len(train_data)}\n")
    console.log(f"number of val problems: {len(validation_data)}\n")
    console.log(f"number of test problems: {len(test_data)}\n")

    return train_data, validation_data, test_data
