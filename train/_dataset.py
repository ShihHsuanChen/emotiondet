
import pandas as pd
import numpy as np
import torch


class Dataset:
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        text = row['text']
        label = int(row['label'])
        token = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length)
        token = {k: np.array(v) for k, v in token.items()}
        # print(token['input_ids'])
        label = np.array(label, dtype=np.int64)
        return token, label


def collate_fn(batch):
    tokens, labels = zip(*batch)
    keys = tokens[0].keys()
    data = {k: torch.from_numpy(np.stack([token[k] for token in tokens])) for k in keys}
    data['labels'] = torch.from_numpy(np.stack(labels))
    return data
