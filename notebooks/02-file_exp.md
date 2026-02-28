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

# file_exp

## Imports

```python
import pandas as pd
import numpy as np
import seaborn as sns
```

## BCG_experiment

### 01_map_complete_absolute_n_hits_table.tsv

```python
abs_hit = pd.read_csv('../drive/BCG_experiment/01_map_complete_absolute_n_hits_table.tsv', sep='\t')
```

```python
abs_hit.head()
```

```python
abs_hit.columns
```

```python
abs_hit.info()
```
