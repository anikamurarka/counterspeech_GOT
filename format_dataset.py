import pandas as pd
import json
from pathlib import Path
import os
from sklearn.model_selection import train_test_split


def create_formatted_dialogue(dialogue):
    # Exclude the final hate speech and counter-narrative entries
    dialogue_content = dialogue[:-2]

    # Create formatted utterances and combine them
    formatted_utterances = [f"{entry['type']}: {entry['text']}" for idx, entry in dialogue_content.iterrows()]
    combined_dialogue = "\n".join(formatted_utterances)

    return combined_dialogue


def write_json_file(data, filepath):
    """
    Write data to a JSON file with proper formatting.
    
    Args:
        data: The content to save (typically a list of dictionaries)
        filepath: Destination path for the JSON file
    """
    with open(filepath, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=4)


def organize_dialogue_data(grouped_dialogues):
    organized_data = []
    for dialogue_id, dialogue_content in grouped_dialogues:
        entry = {
            "dialogue_id": dialogue_id[0],
            "target": dialogue_content["TARGET"].unique()[0],
            "hate_speech": dialogue_content["text"].values[-2],
            "counter_narrative": dialogue_content["text"].values[-1],
            "dialogue_history": create_formatted_dialogue(dialogue_content)
        }
        organized_data.append(entry)

    return organized_data


def main():
    # Import the dataset
    dialogue_data = pd.read_csv("data/DIALOCONAN/dialoconan.csv").sort_values(by=["dialogue_id", "turn_id"])

    # Group dialogues by their ID
    grouped_dialogues = dialogue_data.groupby(["dialogue_id"])

    # Transform data into required format
    organized_data = organize_dialogue_data(grouped_dialogues)

    # Create train/dev/test splits
    [train_set, test_set] = train_test_split(organized_data, test_size=0.10, random_state=42)
    [train_set, dev_set] = train_test_split(train_set, test_size=0.1111, random_state=42)
    print(f'Train size: {len(train_set)}, Dev size: {len(dev_set)}, Test size: {len(test_set)}')

    # Create output directory and save files
    output_directory = Path("data/DIALOCONAN")
    output_directory.mkdir(parents=True, exist_ok=True)
    
    write_json_file(organized_data, filepath=output_directory / "full.json")
    write_json_file(train_set, filepath=output_directory / "train.json")
    write_json_file(dev_set, filepath=output_directory / "dev.json")
    write_json_file(test_set, filepath=output_directory / "test.json")

    ######################## Split train into 5 ########################
    # with open("./../data/DIALOCONAN/train.json") as f:
    #     full = json.load(f)
    
    # parts = []
    # bgn = 0
    # end = 500
    # for i in range(5):
    #     part = full[bgn:end]
    #     parts.append(part)
    
    #     bgn += 500
    #     end += 500
    
    #     with open(f"./../data/DIALOCONAN/train_{i}.json", "w") as f:
    #         json.dump(part, f)

    print("DONE")


if __name__ == '__main__':
    main()