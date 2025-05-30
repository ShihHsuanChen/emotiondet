
import os
import json
import pandas as pd
from typing import Optional

def read_jsonl(path, st: Optional[int] = None, ed: Optional[int] = None) -> pd.DataFrame:
    record = []
    with open(path, 'r') as fp:
        for i, line in enumerate(fp.readlines()):
            if st is not None and i < st:
                continue
            if ed is not None and i > ed:
                continue
            row = json.loads(line)
            record.append(row)
    return pd.DataFrame.from_records(record)


def drop_duplicates(df: pd.DataFrame, df_ref: Optional[pd.DataFrame] = None):
    df = df.drop_duplicates('text')
    if df_ref is not None:
        index = pd.merge(df[['text']], df_ref[['text']], on='text', how='inner')['text']
        df = df.set_index('text').drop(index=index).reset_index()
    return df


def read_data(
    train_path: Optional[str] = None,
    valid_path: Optional[str] = None,
    test_path: Optional[str] = None,
):
    if train_path is not None:
        df_train = drop_duplicates(read_jsonl(train_path))
    else:
        df_train = None
    if valid_path is not None:
        df_valid = drop_duplicates(read_jsonl(valid_path), df_train)
    else:
        df_valid = None
    if test_path is not None:
        df_test = drop_duplicates(read_jsonl(test_path), df_train)
    else:
        df_valid = None
    return {'train': df_train, 'valid': df_valid, 'test': df_test}
