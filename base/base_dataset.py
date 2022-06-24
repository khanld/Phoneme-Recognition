import pandas as pd
import sys
import re
import librosa
import numpy as np
from typing import Dict, List

# For testing 
sys.path.append('..')

from sklearn.model_selection import train_test_split
from utils.feature import load_wav
from tqdm import tqdm
from torch.utils.data import Dataset
from dataloader.dataset import Dataset as InstanceDataset
from pandarallel import pandarallel
from multiprocessing import Pool
from g2p_en import G2p
tqdm.pandas()


class BaseDataset(Dataset):
    def __init__(self, rank, dist, path, sr, delimiter, min_duration = -np.inf, max_duration = np.inf, preload_data = False, transform = None, nb_workers = 4):
        self.rank = rank
        self.dist = dist
        self.sr = sr
        # Special characters to remove in your data 
        self.chars_to_ignore = r'[,?.!\-;:"“%\'�]'
        self.transform = transform
        self.preload_data = preload_data
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.df = self.load_data(path, delimiter)
        self.nb_workers = nb_workers
        self.g2p = G2p()

        if min_duration != -np.inf or max_duration != np.inf: 
            if self.rank == 0 and 'duration' not in self.df.columns:
                self.df['duration'] = self.df['path'].progress_apply(lambda filename: librosa.get_duration(filename=filename))       
                self.df.to_csv(path, index = False, sep = delimiter)
            self.dist.barrier()
            self.df = self.load_data(path, delimiter)
            mask = (self.df['duration'] <= self.max_duration) & (self.df['duration'] >= self.min_duration)
            self.df = self.df[mask]

        if self.preload_data:
            if self.rank == 0:
                print(f"\n*****Preloading {len(self.df)} data*****")
            self.df['wav'] = self.df['path'].progress_apply(lambda filepath: load_wav(filepath, sr = self.sr))

        if 'label' not in self.df.columns:
            if self.rank == 0:
                print("---Tokenize transcript-----")
                self.df['label'] = self.df['transcript'].progress_apply(self.tokenize)
                self.df.to_pickle(path.replace('txt', 'pkl'))
            self.dist.barrier()
            self.df = self.load_data(path.replace('txt', 'pkl'), delimiter)

        assert 'label' in self.df.columns, "Label not in df"
    
    def get_id(self, phoneme):
        if phoneme == " ":
            return 0
        elif phoneme in self.g2p.p2idx:
            return self.g2p.p2idx[phoneme]
        else:
            return self.g2p.p2idx["<unk>"]


    def tokenize(self, text):
        phonemes = ["<s>"] + self.g2p(text) + ["</s>"]
        idx = [self.get_id(phoneme) for phoneme in phonemes]
        return idx
    

    def load_data(self, path, delimiter) -> pd.DataFrame:
        if path.split('.')[-1] == 'txt' or path.split('.')[-1] == 'csv':
            df = pd.read_csv(path, delimiter = delimiter)
        else:
            df = pd.read_pickle(path)
        return df

    def get_data(self) -> Dataset:
        ds = InstanceDataset(self.df, self.sr, self.preload_data, self.transform)
        return ds


if __name__ == '__main__':
    ds = BaseDataset(
        path = '/content/drive/MyDrive/ASR Finetune/dataset/vivos/test.csv', 
        sr = 16000, 
        preload_data = False, 
        val_size = None, 
        transform = None)
    
    vocab_dict = ds.get_vocab_dict()
    for k, v in vocab_dict.items():
        print(f'{k} - {v}')