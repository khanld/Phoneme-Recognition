import sys
sys.path.append("../")
import torch
import yaml

from utils.feature import load_wav
from typing import Dict

class SimpleCustomBatch:
    def __init__(self, batch):
        self.input_values = batch.input_values
        self.labels = batch.labels

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.input_values = self.input_values.pin_memory()
        self.labels = self.labels.pin_memory()
        return self

class DefaultCollate:
    def __init__(self, feature_extracture, sr) -> None:
        self.feature_extracture = feature_extracture
        self.sr = sr

    def __call__(self, inputs) -> Dict[str, torch.tensor]:
        features, labels = zip(*inputs)
        features, labels = list(features), list(labels)
        batch = self.feature_extracture(features, sampling_rate=16000, padding="longest", return_tensors="pt")
        batch["labels"] = torch.nn.utils.rnn.pad_sequence(labels, batch_first = True, padding_value = 74)
        batch["labels"] = batch["labels"].masked_fill(batch["labels"] == 74, -100)
        return SimpleCustomBatch(batch)

class Dataset:
    def __init__(self, data, sr, preload_data, transform = None):
        self.data = data
        self.sr = sr
        self.transform = transform
        self.preload_data = preload_data
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx) -> tuple:
        item = self.data.iloc[idx]
        if not self.preload_data:
            feature = load_wav(item['path'], sr = self.sr)
        else:
            feature = item['wav']
        return feature, torch.tensor(item['label'])

