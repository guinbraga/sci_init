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

# 03-model_evaluation

This notebook intends to go through the model evaluation 
steps defined by the author with the intention to debug it's
program.

## Imports and configs


```python
import sys
import os

# 1. Get the current directory of the notebook
# 2. Go up one level to 'root' (..)
# 3. Go down into 'scripts'
module_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'scripts'))

# Check if the path is already in sys.path to avoid duplicates
if module_path not in sys.path:
    sys.path.append(module_path)
```

```python
import pandas as pd
import numpy as np
from sklearn.metrics import(
    make_scorer,
    roc_auc_score,
    precision_recall_curve,
    auc,
    roc_curve
)
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    GridSearchCV,
    LeaveOneOut,
    RepeatedStratifiedKFold
)
from imblearn.metrics import geometric_mean_score
from bioreactorProcessing import X_train, y_train
from configs import pipelines, params
```

```python
def auc_prec_score(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)

def g_mean_score(y_true, y_pred):
    return geometric_mean_score(y_true, y_pred)

def evaluate_model2(pipeline, param, X_train, y_train):
    # Definir scorers
    loo = LeaveOneOut()
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    # Scorers para múltiplas métricas
    # Scorer para AUC da curva Precision-Recall
    precision_recall_auc_scorer = make_scorer(auc_prec_score, response_method='predict_proba')

    # Scorer para G-Mean
    g_mean_scorer = make_scorer(g_mean_score)

    # Outros scorers
    # roc_auc_scorer = make_scorer(roc_auc_score_fixed, needs_proba=True)
    roc_auc_scorer = make_scorer(roc_auc_score, response_method='predict_proba')
    # Configuração do GridSearchCV com verbose para monitorar o progresso
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param,
        scoring={
            "roc_auc": roc_auc_scorer,
            "g_mean": g_mean_scorer,
            "auc_prec": precision_recall_auc_scorer,
        },
        refit="g_mean",  # Refitar com base no G-Mean
        cv=sss,
        n_jobs=-1,  # Usar todos os processadores para correr trabalhos em paralelo
        return_train_score=True,  # Habilita o cálculo do score no conjunto de treinamento
        verbose=0,  # Não mostra o progresso detalhado para cada iteração e fold
    )

    # added to diagnose
    print(f"Unique classes in y_train: {np.unique(y_train)}")
    print(f"Counts: {np.bincount(y_train)}")

    # Treinamento do modelo com validação cruzada
    grid_search.fit(X_train, y_train)

    return grid_search

```

```python
model_name = 'Logistic Regression'
evaluation = evaluate_model2(pipelines[model_name], params[model_name], X_train, y_train)
```

```python
evaluation.best_estimator_
```

```python
evaluation.cv_results_
```

```python
evaluation.best_params_
```

