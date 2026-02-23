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
      jupytext_version: 1.19.1
---

# 05-hrt_processing

## Imports and configs

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
```

## Preparing and Merging datasets

```python
metadata = pd.read_csv('../drive/C6-C8_productivity_experiment/01_metadata_productivity.txt')
metadata.info()
```

```python
metadata['HRT'].value_counts()
```

```python
abs_hits = pd.read_csv('../drive/C6-C8_productivity_experiment/01_map_complete_relative_abundance_table.csv', index_col=0)
abs_hits.info()
```

```python
abs_hits.columns.name = 'sample'
abs_hits.index.name = 'ASV'
len(abs_hits.columns)
```

```python
print(abs_hits.isna().sum().sum())
print(metadata.isna().sum().sum())
```

```python
merge = pd.merge(abs_hits.T, metadata[['Sample', 'HRT']], left_index=True,
                 right_on='Sample', how='inner').set_index('Sample')
merge.info()
```

```python
merge = merge[(merge['HRT'] == 2) | (merge['HRT'] == 8)]
merge.info()
```


## Train Test split

```python
y = merge['HRT']
X = merge.drop('HRT', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=42,
                                                    stratify=y)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
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

