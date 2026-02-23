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

# 01-HRT_preprocessing

Imports

```python
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
```

## Configs

### File loading function 

```python
def load_asv_file(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep=",", index_col=0)
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

```

### Paths

```python
TRAIN_PATH = "../drive/C6-C8_productivity_experiment/AB/06_train_HRT_2Class.csv"
TEST_PATH = "../drive/C6-C8_productivity_experiment/AB/06_test_HRT_2 Class.csv"
HRT_all_ra_ML_PATH = "../drive/C6-C8_productivity_experiment/AB/05_asv_bin_ML_ra.csv"
HRT_all_abs_ML_PATH = "../drive/C6-C8_productivity_experiment/AB/05_asv_bin_ML_ra_amerge.csv"
```

## Work

```python
# Carregando arquivos ASV
asv_all_file = "../drive/C6-C8_productivity_experiment/02_ASV_table_rarefied_13518_all.csv"

asv_all_csv = load_asv_file(asv_all_file)
asv_all_csv.head()
```
