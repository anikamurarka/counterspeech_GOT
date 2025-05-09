import os
import pickle

import torch
from torch.utils.data import Dataset

from train_utils.utils_prompt import build_train_pair_dialoconan


class DialoconanDatasetWithGraph(Dataset):
    """
    A custom dataset implementation for dialogue counter-narrative generation with graph data.
    Handles loading and preprocessing of text and graph data for model training.
    """

    def __init__(self, examples, split, tokenizer, source_len, target_len, args):
        self.tokenizer = tokenizer
        self.data = examples
        self.source_len = source_len
        self.target_len = target_len
        self.target_text = []
        self.source_text = []

        # Load graph data from pickle files
        graph_dir = os.path.join(args.data_root, args.dataset, args.got_root, split)
        with open(os.path.join(graph_dir, 'mc_input_text.pkl'), 'rb') as f:
            self.graph_node_text = pickle.load(f)
        with open(os.path.join(graph_dir, 'mc_adj_matrix.pkl'), 'rb') as f:
            self.graph_adjacency = pickle.load(f)

        # Process each example to create source-target pairs
        for idx, example in enumerate(self.data):
            prompt, target = build_train_pair_dialoconan(example, args.exclude_context)
            self.source_text.append(prompt)
            self.target_text.append(target)

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        # Get text data for the current example
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # Normalize whitespace
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        # Get graph data for the current example
        graph_nodes = self.graph_node_text[index]
        graph_matrix = torch.tensor(self.graph_adjacency[index])

        # Tokenize source text
        source_encoding = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Tokenize target text
        target_encoding = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.target_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Tokenize graph node text
        graph_encoding = self.tokenizer.batch_encode_plus(
            graph_nodes,
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Extract tensors from encodings
        source_ids = source_encoding["input_ids"].squeeze()
        source_mask = source_encoding["attention_mask"].squeeze()
        target_ids = target_encoding["input_ids"].squeeze().tolist()
        graph_ids = graph_encoding["input_ids"].squeeze()
        graph_mask = graph_encoding["attention_mask"].squeeze()

        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": target_ids,
            "got_adj_matrix": graph_matrix,
            "got_input_ids": graph_ids,
            "got_mask": graph_mask,
        }


class DialoconanDatasetNoGraph(Dataset):
    """
    A simplified dataset implementation for dialogue counter-narrative generation
    without graph data. Used for baseline models or when graph data is unavailable.
    """

    def __init__(self, examples, tokenizer, source_len, target_len, exclude_context):
        self.tokenizer = tokenizer
        self.data = examples
        self.source_len = source_len
        self.target_len = target_len
        self.target_text = []
        self.source_text = []

        # Process each example to create source-target pairs
        for idx, example in enumerate(self.data):
            prompt, target = build_train_pair_dialoconan(example, exclude_context)
            self.source_text.append(prompt)
            self.target_text.append(target)

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        # Get text data for the current example
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # Normalize whitespace
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        # Tokenize source text
        source_encoding = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Tokenize target text
        target_encoding = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.target_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Extract tensors from encodings
        source_ids = source_encoding["input_ids"].squeeze()
        source_mask = source_encoding["attention_mask"].squeeze()
        target_ids = target_encoding["input_ids"].squeeze().tolist()

        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": target_ids
        }
