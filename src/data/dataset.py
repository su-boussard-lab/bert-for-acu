import torch
from torch.utils.data import Dataset
from typing import List
import numpy as np
from src.utils.config import config

ordinal_regression = config.ordinal_regression


class CustomTextDataset(Dataset):
    """Custom text dataset for pytorch, which takes two dataframes (text and labels) and returns them.
    Args:
        input_dicts (List): list of dicts of text features
        labels (List): list of labels
    Returns:
        CustomTextDataset
    """

    def __init__(
        self,
        input_dicts: List,
        labels: List,
        chunks_per_note: List,
    ):
        self.input_dicts = input_dicts
        self.labels = labels
        self.chunks_per_note = chunks_per_note

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        chunk = self.chunks_per_note[idx]
        current_idx = self.chunks_per_note[:idx].sum().item()
        start_idx = int(current_idx)
        end_idx = int(current_idx + chunk)
        dicts = {k: v[start_idx:end_idx] for k, v in self.input_dicts.items()}
        
        return dicts, label, chunk


class MultiModalDataset(CustomTextDataset):
    """Multimodal dataset that loads the tabular and language data at the same time
    Args:
        input_dicts (List): list of dicts of text features
        tabular_data (np.ndarray): np array with tabular data
        labels (List): list of labels
    Returns:
        MutliModalDataset
    """

    def __init__(
        self,
        input_dicts: List,
        tabular_data: np.ndarray,
        labels: List,
        chunks_per_note: List,
    ):
        super().__init__(input_dicts, labels, chunks_per_note)
        self.tabular_data = tabular_data

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        label = self.labels[idx]
        chunk = self.chunks_per_note[idx]
        current_idx = self.chunks_per_note[:idx].sum().item()
        start_idx = int(current_idx)
        end_idx = int(current_idx + chunk)
        dicts = {k: v[start_idx:end_idx] for k, v in self.input_dicts.items()}
        dicts["tabular_x"] = torch.Tensor(self.tabular_data[idx])
        return dicts, label, chunk


def custom_collate(batch: List) -> tuple:
    """
    custom collate function for text and labels of variable length
    Input Shape: [B, network_inputs + labels, network_inputs, chunks, max_length]
    Args:
        batch: is the next batch which should be processed
    Returns:
        input_dicts (list): list of the input dicts
        labels (torch.Tensor): tensor of the labels
    """
    input_dicts = [data[0] for data in batch]
    input_dicts = {
        k: torch.cat([dic[k] for dic in input_dicts]) if k != "tabular_x" else torch.stack([dic[k] for dic in input_dicts]) for k in input_dicts[0] 
    }
    labels = torch.stack([data[1] for data in batch])
    chunks = torch.stack([data[2] for data in batch])
    if ordinal_regression:
        labels = labels.unsqueeze(1).to(torch.long)
    else:
        labels = labels.to(torch.float)
    return input_dicts, labels, chunks
