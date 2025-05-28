import marimo

__generated_with = "0.13.13"
app = marimo.App()


@app.cell
def _():
    import os
    import json
    from typing import List, Dict, Optional, Union, Tuple

    import pandas as pd
    import matplotlib.pyplot as plt

    try:
        import marimo as mo
    except:
        pass
    return Dict, Optional, json, mo, os, pd, plt


@app.cell
def _(mo):
    mo.md(r"""# Environments""")
    return


@app.cell
def _(os):
    KERNEL_TYPE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE','').upper()
    IS_KAGGLE = KERNEL_TYPE != ''
    print('================================================')
    if IS_KAGGLE:
        print(f'Notebook is running on Kaggle ({KERNEL_TYPE} mode)')
    else:
        print(f'Notebook is running locally')
    print('================================================')
    return IS_KAGGLE, KERNEL_TYPE


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Parameters and Conatants""")
    return


@app.cell
def _(IS_KAGGLE, os):
    if IS_KAGGLE:
        DATADIR = '/kaggle/input/setfit-emotion'
    else:
        DATADIR = '../setfit-emotion/'
    TRAIN_DATA_PATH = os.path.join(DATADIR, 'train.jsonl')
    VALID_DATA_PATH = os.path.join(DATADIR, 'validation.jsonl')
    TEST_DATA_PATH = os.path.join(DATADIR, 'test.jsonl')
    return TEST_DATA_PATH, TRAIN_DATA_PATH, VALID_DATA_PATH


@app.cell
def _(KERNEL_TYPE):
    DEBUG = KERNEL_TYPE != 'BATCH'
    DEBUG
    return (DEBUG,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Utilities""")
    return


@app.cell
def _(DEBUG, IS_KAGGLE, Optional, TEST_DATA_PATH, json, pd):
    if not IS_KAGGLE:
        display = print
    
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


    if DEBUG:
        df = read_jsonl(TEST_DATA_PATH)
        display(df)
    return df, display, read_jsonl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Data Visualization""")
    return


@app.cell
def _(DEBUG, df):
    if DEBUG:
        labels = df[['label', 'label_text']].groupby('label').first().sort_index()
        print(labels)
    return


@app.cell
def _(Optional, pd, plt):
    def get_labels(df: pd.DataFrame):
        labels = df[['label', 'label_text']].groupby('label').first().sort_index()
        return labels


    def plot_label_counts(df: pd.DataFrame, name: str = '', ax: Optional = None):
        if ax is None:
            fig, ax = plt.subplots()
        vc = df['label_text'].value_counts()
        ax.bar(vc.keys(), vc.values)
        ax.set_ylabel('# Records')
        ax.set_title(name)
        ax.grid()
        return ax


    def plot_label_text_length(df: pd.DataFrame, name: str = '', ax: Optional = None):
        if ax is None:
            fig, ax = plt.subplots()
        for label_text, _df in df.groupby('label_text'):
            textlen = _df['text'].apply(len)
            ax.hist(textlen, alpha=0.5, density=True, label=label_text)
        ax.set_title(name)
        ax.set_xlabel('Text length')
        ax.set_ylabel('% Records')
        ax.legend()
        ax.grid()
    return get_labels, plot_label_counts, plot_label_text_length


@app.cell
def _(
    DEBUG,
    Dict,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    VALID_DATA_PATH,
    display,
    get_labels,
    pd,
    plot_label_counts,
    plot_label_text_length,
    plt,
    read_jsonl,
):
    def visualization(**dfs: Dict[str, pd.DataFrame]):
        labels = None
        for name, df in dfs.items():
            if labels is None:
                labels = get_labels(df)
            else:
                labels = pd.merge(labels, get_labels(df))
        display(labels)

        fig, axs = plt.subplots(2, len(dfs), figsize=(4*len(dfs), 5*2))
        for i, (name, df) in enumerate(dfs.items()):
            # label counts
            plot_label_counts(df, ax=axs[0,i], name=name)

            # label-text length
            plot_label_text_length(df, ax=axs[1,i], name=name)
        plt.show()

    if DEBUG:
        df_train = read_jsonl(TRAIN_DATA_PATH)
        df_valid = read_jsonl(VALID_DATA_PATH)
        df_test  = read_jsonl(TEST_DATA_PATH)
        visualization(
            train=df_train,
            valid=df_valid,
            test =df_test,
        )
    return df_test, df_train, df_valid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Data Check""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Duplication Check
    To check if there is any text content appears in multiple records.
    """
    )
    return


@app.cell
def _(DEBUG, df_train, pd):
    def check_duplication(df: pd.DataFrame):
        for text, _df in df.groupby('text'):
            if len(_df) == 1:
                continue
            labels = _df['label_text'].unique()
            print(f'TEXT: "{text}"\nCOUNTS: {len(_df)} LABELS: {labels}\n')

    if DEBUG:
        # check_duplication(df_train)
        print(df_train[df_train['text'] == 'i am not amazing or great at photography but i feel passionate about it'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Leakage Check
    To check if there are records in the **validation**/**test** dataset **similar**/**indentical** to the **train** dataset
    """
    )
    return


@app.cell
def _(df_test, df_train, df_valid, display, pd):
    def find_leakage(df_ref, *dfs, ref_name='ref', names=()):
        """ find records of `df` that has text also appears in `df_ref` reocrds"""

        assert len(dfs) == len(names)

        df_ref = df_ref.groupby('text')['label_text'].agg(['unique'])

        df_merge = []
        for df, name in zip(dfs, names):
            df = df.groupby('text')['label_text'].agg(['unique'])
            _df = pd.merge(df_ref, df, on='text', how='inner', suffixes=('_'+ref_name, '_'+name))
            df_merge.append(_df)
        df_merge = pd.concat(df_merge, ignore_index=False, sort=False)
        display(df_merge)


    find_leakage(df_train, df_valid, df_test, ref_name='train', names=('valid', 'test'))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
