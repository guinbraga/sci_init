---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: md,ipynb
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
---

# processamentoBioreator

## Imports and configs

```python
import os
from numpy import absolute
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_asv_file(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep="\t", index_col=0)
        # df = pd.read_csv(file_path, sep='\t', index_col=0)
        if df.empty:
            raise ValueError(
                f"Erro: O arquivo ASV foi carregado, mas está vazio: {file_path}"
            )

        # Transpor a tabela (inversão de linhas e colunas)
        df_transposed = df.T

        return df_transposed
    else:
        raise FileNotFoundError(f"Erro: Arquivo ASV não encontrado: {file_path}")


def transform_sample_counts(df):
    # df.sum(axis=1):
    # Calcula a soma das contagens de todas as espécies (ou bins) para cada amostra
    # (cada linha), ou seja, a soma das colunas para cada linha.
    return df.div(df.sum(axis=1), axis=0)
    # df.div(df.sum(axis=1), axis=0): Divide cada valor de uma linha (amostra)
    #  pelo total de contagens daquela linha (soma dos bins para aquela amostra)


TRAIN_PATH = "../drive/Bettle_experiments/06_train_HRT_2Class.csv"
TEST_PATH = "../drive/Bettle_experiments/06_test_HRT_2Class.csv"
meta_all_csv_PATH = "../drive/Bettle_experiments/04_metadata_bin_ML_abs.csv"
```

## 01 Loading and exploring tables

### Loading absolute hits table:

```python
file_absolute = "../drive/Bettle_experiments/01_map_complete_absolute_n_hits_table.tsv"

absolute_hits = load_asv_file(file_absolute)

absolute_hits.head()
```

```python
absolute_hits.info()
```

```python
absolute_hits.isna().sum().sum()
```

### Loading relative abundance

```python
rel_file = (
    "../drive/Bettle_experiments/01_map_complete_relative_abundance_table.tsv"
)
rel_abundance = load_asv_file(rel_file)
rel_abundance.sum(axis=1)
```

This is weird. You'd expect a relative abundance table to sum 
to 1, but this one doesn't.

```python
rel_abundance.info()
```

```python
rel_abundance.isna().sum().sum()
```

#### Relative from absolute

This intends to use the `transform_sample_counts` function
defined by the author to create a relative abundance table and
verify it's horizontal sum:

```python
rel_from_abs = transform_sample_counts(absolute_hits)
rel_from_abs.sum(axis=1)
```

As we can see, this is more mathematically sound. We'll
further explore the author's code below.

### Loading Metadata

```python
meta_all_file = "../drive/Bettle_experiments/01_metadata_productivity.txt"
meta_df = pd.read_csv(meta_all_file, sep="\t", index_col=0)
meta_df.info()
```

```python
meta_df[['Category', 'Category2']]
```


### Copy absolute hits table

```python
abs_hits_copy = absolute_hits.copy()
```

### SUMMARY SO FAR

We have 3 Datasets loaded:

`absolute_hits` has the absolute values for each ASV per sample
`rel_abundance` has the weird relative abundance per sample
`meta_df` has the metadata of the experiment

Below, a few sanity checks to make sure these datasets are compatible
for merging and further exploration

#### Equal number of rows

```python
len(meta_df.index)
```

```python
len(absolute_hits.index)
```

```python
len(rel_abundance.index)
```

#### Same index values and columns

```python
(rel_abundance.index == absolute_hits.index).sum()
```

```python
len(rel_abundance.columns)
```

```python
(rel_abundance.columns == absolute_hits.columns).sum()
```

```python
set(meta_df.index) == set(absolute_hits.index) == set(rel_abundance.index)
```

## 02 Preparing tables

### Merging

The author does two merges: One between the absolute counts
and the metadata `Experiment` column (to use it as target label),
and another between the weird relative abundance
table and the metadata `Experiment` column. Just in case,
I'll also do a merge with the calculated relative abundance table:

```python
full_abs = pd.merge(absolute_hits, meta_df[['Experiment']],
                    left_index=True, right_index=True)
full_rel_raw = pd.merge(rel_abundance, meta_df[['Experiment']],
                    left_index=True, right_index=True)
full_rel_calc = pd.merge(rel_from_abs,meta_df[['Experiment']],
                    left_index=True, right_index=True)

full_abs.rename(columns={"Experiment": "y"}, inplace=True)
full_rel_raw.rename(columns={"Experiment": "y"}, inplace=True)
full_rel_calc.rename(columns={"Experiment": "y"}, inplace=True)

```

```python
len(full_rel_calc.index) == len(full_rel_raw) == len(full_rel_calc)
```

### Do we have ASVs with count zero in all samples?

```python
full_abs.columns[(full_abs==0).all()]
```

### Train - Test split

#### The split

```python
train_abs, test_abs = train_test_split(
    full_abs,
    test_size=0.25,
    stratify=full_abs['y'],
    random_state=42
)

print(f"train shape: {train_abs.shape}")
print(f"test shape: {test_abs.shape}")
```
#### Encoding target feature


```python
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_abs['y'])
y_test = label_encoder.transform(test_abs['y'])
X_train = train_abs.drop(columns=['y']).reset_index(drop=True)
X_test = test_abs.drop(columns=['y']).reset_index(drop=True)
```


```python
class_mapping = dict(
    zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_)
)
print(f"Mapeamento de classes: {class_mapping}")
```

```python
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
```


