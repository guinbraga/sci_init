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

# 03-best_model_exploration

```python
import pandas as pd
```

```python
best_model = pd.read_pickle('../outputs/Bettle/Div1/best_models/best_K-Neighbors_model.pkl')
best_model
```

```python
gscv_results = pd.read_excel('../outputs/Bettle/Div1/gridsearch_results3/gridsearch_K-Neighbors_results.xlsx')
gscv_results
```
