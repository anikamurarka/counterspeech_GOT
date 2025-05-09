'''
Utility functions for prompt generation and text processing for dialogue-based counter-narrative generation
'''

from dataclasses import dataclass
from typing import List, Optional

import nltk


def build_train_pair_dialoconan(problems, exclude_context=False):
    examples = []
    # Format the input prompt based on whether context should be included
    if exclude_context:
        text_example = f"Hate Speech:\n{problems['hate_speech']}\n\n" \
                       f"Counter-narrative:\n"
    else:
        text_example = f"Hate Speech:\n{problems['hate_speech']}\n\n" \
                       f"Dialogue History:\n{problems['dialogue_history']}\n\n" \
                       f"Counter-narrative:\n"

    target = problems['counter_narrative']

    examples.append(text_example)
    prompt_input = '\n\n'.join(examples)

    return prompt_input, target


def postprocess_text(preds, labels):
    processed_preds = []
    for pred in preds:
        pred = pred.strip()
        try:
            # Split prediction text into sentences using nltk
            processed_pred = "\n".join(nltk.sent_tokenize(pred))
        except IndexError:
            # Handle potential errors with long text
            print(f"IndexError occurred with text: {pred}")
            processed_pred = pred
        processed_preds.append(processed_pred)

    processed_labels = []
    for label in labels:
        label = label.strip()
        try:
            # Split label text into sentences using nltk
            processed_label = "\n".join(nltk.sent_tokenize(label))
        except IndexError:
            print(f"IndexError occurred with text: {label}")
            processed_label = label
        processed_labels.append(processed_label)

    return processed_preds, processed_labels


@dataclass(frozen=True)
class InputFeatures:
    """
    Data structure for storing features of input data.
    Field names correspond to the inputs required by the model.
    """

    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    le_input_ids: List[List[int]]
    le_attention_mask: Optional[List[List[int]]]
    le_token_type_ids: Optional[List[List[int]]]
    label: Optional[int]
