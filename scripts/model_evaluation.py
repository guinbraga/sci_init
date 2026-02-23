from configs import pipelines, params
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    StandardScaler,
    QuantileTransformer,
    Binarizer,
    PowerTransformer,
    FunctionTransformer,
    Normalizer,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
import xgboost as xgb
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer
from sklearn.svm import SVC
from sklearn.metrics import (
    make_scorer,
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score,
    classification_report,
)
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
    LeaveOneOut,
)
import logging
from sklearn.metrics import classification_report
import os
from utils import ensure_directory_exists


# Set global plot configurations
plt.rc("font", family="Times New Roman", size=12)

# Directories for saving plots
# SAVE_DIR_BOXPLOTS = "../outputs/Bettle/Div1/plots/boxplotsexluir3/"
# SAVE_DIR_HISTOGRAMS = "../outputs/Bettle/Div1/plots/histogramsexluir3/"
# ROC_PRECISION_RECALL_CURVE = (
#     "../outputs/Bettle/Div1/plots/roc_precision_recall_curveexluir3/"
# )

SAVE_DIR_BOXPLOTS = "../outputs/HRT/AB-Analysis/plots/boxplots/"
SAVE_DIR_HISTOGRAMS = "../outputs/HRT/AB-Analysis/plots/histograms/"
ROC_PRECISION_RECALL_CURVE = (
    "../outputs/HRT/AB-Analysis/plots/roc_precision_recall_curve/"
)


def auc_prec_score(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)


# Função para calcular o G-Mean
def g_mean_score(y_true, y_pred):
    return geometric_mean_score(y_true, y_pred)


# ======================= DIAGNOSIS INSERTIONS =======================================
# Create a wrapper that selects the probability of the positive class (column 1)
def roc_auc_score_fixed(y_true, y_pred_proba):
    # If the input is a matrix (n_samples, 2), take the second column
    if y_pred_proba.ndim == 2:
        y_pred_proba = y_pred_proba[:, 1]
    return roc_auc_score(y_true, y_pred_proba)


def evaluate_model2(pipeline, param, X_train, y_train):
    # Definir scorers
    loo = LeaveOneOut()
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    # Scorers para múltiplas métricas
    # Scorer para AUC da curva Precision-Recall
    precision_recall_auc_scorer = make_scorer(auc_prec_score)

    # Scorer para G-Mean
    g_mean_scorer = make_scorer(g_mean_score)

    # Outros scorers
    roc_auc_scorer = make_scorer(roc_auc_score_fixed)
    # roc_auc_scorer = make_scorer(debug_scorer, needs_proba=True)
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


def print_validation_evaluation(
    grid_search,
    SAVE_DIR_BOXPLOTS=SAVE_DIR_BOXPLOTS,
    SAVE_DIR_HISTOGRAMS=SAVE_DIR_HISTOGRAMS,
):
    # Verificar e criar os diretórios, se necessário
    ensure_directory_exists(SAVE_DIR_BOXPLOTS)
    ensure_directory_exists(SAVE_DIR_HISTOGRAMS)
    # ============================
    # Melhor resultado geral
    # ============================
    # Melhor resultado do GridSearchCV
    cv_results = grid_search.cv_results_
    # print(cv_results)
    best_index = grid_search.best_index_
    best_model = grid_search.best_estimator_
    model_name = best_model.named_steps["classifier"].__class__.__name__
    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best G-Mean score: {grid_search.best_score_:.4f}")

    # Melhor resultado do GridSearchCV
    print(f"Best parameters for RandomForest: {grid_search.best_params_}")
    print(f"Best cross-validation G-Mean score: {grid_search.best_score_:.4f}")

    # ============================
    # Calcular os erros:
    # ============================

    # 1. Erro de treinamento no k-fold (1 - G-Mean)
    train_g_mean_mean = grid_search.cv_results_["mean_train_g_mean"][best_index]
    train_g_mean_std = grid_search.cv_results_["std_train_g_mean"][best_index]
    train_error_kfold = 1 - train_g_mean_mean
    print(
        f"Erro de treinamento no k-fold (G-Mean) {train_g_mean_mean:.4f}: {train_error_kfold:.4f} (desvio padrão: {train_g_mean_std:.4f})"
    )

    # 2. Erro de validação no k-fold (1 - G-Mean)
    val_g_mean_mean = grid_search.cv_results_["mean_test_g_mean"][best_index]
    val_g_mean_std = grid_search.cv_results_["std_test_g_mean"][best_index]
    val_error_kfold = 1 - val_g_mean_mean
    print(
        f"Erro de validação no k-fold (G-Mean) {val_g_mean_mean:.4f}: {val_error_kfold:.4f} (desvio padrão: {val_g_mean_std:.4f})"
    )

    # ============================
    # Métricas adicionais: ROC-AUC e AUC Precision-Recall
    # ============================

    # 4. Erro de treinamento no k-fold (1 - ROC-AUC)
    train_roc_auc_mean = grid_search.cv_results_["mean_train_roc_auc"][best_index]
    train_roc_auc_std = grid_search.cv_results_["std_train_roc_auc"][best_index]
    train_roc_auc_error_kfold = 1 - train_roc_auc_mean
    print(
        f"Erro de treinamento no k-fold (ROC-AUC) {train_roc_auc_mean:.4f}: {train_roc_auc_error_kfold:.4f} (desvio padrão: {train_roc_auc_std:.4f})"
    )

    # 5. Erro de validação no k-fold (1 - ROC-AUC)
    val_roc_auc_mean = grid_search.cv_results_["mean_test_roc_auc"][best_index]
    val_roc_auc_std = grid_search.cv_results_["std_test_roc_auc"][best_index]
    val_roc_auc_error_kfold = 1 - val_roc_auc_mean
    print(
        f"Erro de validação no k-fold (ROC-AUC) {val_roc_auc_mean:.4f}: {val_roc_auc_error_kfold:.4f} (desvio padrão: {val_roc_auc_std:.4f})"
    )

    # 7. Erro de treinamento no k-fold (1 - AUC Precision-Recall)
    train_auc_prec_mean = grid_search.cv_results_["mean_train_auc_prec"][best_index]
    train_auc_prec_std = grid_search.cv_results_["std_train_auc_prec"][best_index]
    train_auc_prec_error_kfold = 1 - train_auc_prec_mean
    print(
        f"Erro de treinamento no k-fold (AUC Precision-Recall) {train_auc_prec_mean:.4f}: {train_auc_prec_error_kfold:.4f} (desvio padrão: {train_auc_prec_std:.4f})"
    )

    # 8. Erro de validação no k-fold (1 - AUC Precision-Recall)
    val_auc_prec_mean = grid_search.cv_results_["mean_test_auc_prec"][best_index]
    val_auc_prec_std = grid_search.cv_results_["std_test_auc_prec"][best_index]
    val_auc_prec_error_kfold = 1 - val_auc_prec_mean
    print(
        f"Erro de validação no k-fold (AUC Precision-Recall) {val_auc_prec_mean:.4f}: {val_auc_prec_error_kfold:.4f} (desvio padrão: {val_auc_prec_std:.4f})"
    )

    # ============================
    # Calcular os erros - Versão Boxplots:
    # ============================

    # Pegando o número de folds
    num_folds = len(
        [key for key in cv_results if key.startswith("split") and "test_g_mean" in key]
    )

    # Extraindo as métricas para cada fold (para G-Mean e ROC-AUC)
    g_mean_folds_train = []
    roc_auc_folds_train = []
    auc_prec_folds_train = []

    g_mean_folds = []
    roc_auc_folds = []
    auc_prec_folds = []

    for fold in range(num_folds):
        g_mean_folds.append(cv_results[f"split{fold}_test_g_mean"][best_index])
        roc_auc_folds.append(cv_results[f"split{fold}_test_roc_auc"][best_index])
        auc_prec_folds.append(cv_results[f"split{fold}_test_auc_prec"][best_index])
        g_mean_folds_train.append(cv_results[f"split{fold}_train_g_mean"][best_index])
        roc_auc_folds_train.append(cv_results[f"split{fold}_train_roc_auc"][best_index])
        auc_prec_folds_train.append(
            cv_results[f"split{fold}_train_auc_prec"][best_index]
        )

    # Exibir os resultados por fold
    print("==========Train=============")
    for fold in range(num_folds):
        print(f"Fold {fold + 1}:")
        print(f"G-Mean: {g_mean_folds_train[fold]:.4f}")
        print(f"ROC-AUC: {roc_auc_folds_train[fold]:.4f}")
        print(f"AUC Precision-Recall: {auc_prec_folds_train[fold]:.4f}")
        print("-" * 30)

    print("==========Test=============")
    for fold in range(num_folds):
        print(f"Fold {fold + 1}:")
        print(f"G-Mean: {g_mean_folds[fold]:.4f}")
        print(f"ROC-AUC: {roc_auc_folds[fold]:.4f}")
        print(f"AUC Precision-Recall: {auc_prec_folds[fold]:.4f}")
        print("-" * 30)

    # Plotando o boxplot para cada métrica
    plt.figure(figsize=(12, 6))

    # G-Mean
    plt.subplot(1, 3, 1)
    plt.boxplot(g_mean_folds)
    plt.title("Distribuição do G-Mean por Fold")
    plt.xlabel("G-Mean")
    plt.ylabel("Frequência")

    # ROC-AUC
    plt.subplot(1, 3, 2)
    plt.boxplot(roc_auc_folds)
    plt.title("Distribuição do ROC-AUC por Fold")
    plt.xlabel("ROC-AUC")
    plt.ylabel("Frequência")

    # AUC Precision-Recall
    plt.subplot(1, 3, 3)
    plt.boxplot(auc_prec_folds)
    plt.title("Distribuição do AUC Precision-Recall por Fold")
    plt.xlabel("AUC Precision-Recall")
    plt.ylabel("Frequência")

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            SAVE_DIR_BOXPLOTS, f"boxplots_g_mean_roc_auc_auc_prec_{model_name}.png"
        )
    )
    plt.show()
    plt.close()

    # Plotando histogramas para G-Mean e ROC-AUC por fold
    plt.figure(figsize=(12, 6))

    # Histograma para G-Mean
    plt.subplot(1, 3, 1)
    plt.hist(g_mean_folds, bins=5, alpha=0.7, color="b")
    plt.title("Distribuição do G-Mean por Fold")
    plt.xlabel("G-Mean")
    plt.ylabel("Frequência")

    # Histograma para ROC-AUC
    plt.subplot(1, 3, 2)
    plt.hist(roc_auc_folds, bins=5, alpha=0.7, color="b")
    plt.title("Distribuição do ROC-AUC por Fold")
    plt.xlabel("ROC-AUC")
    plt.ylabel("Frequência")

    # Histograma para ROC-AUC
    plt.subplot(1, 3, 3)
    plt.hist(auc_prec_folds, bins=5, alpha=0.7, color="b")
    plt.title("Distribuição do AUC-PREC por Fold")
    plt.xlabel("ROC-AUC")
    plt.ylabel("Frequência")

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            SAVE_DIR_HISTOGRAMS, f"histograms_g_mean_roc_auc_auc_prec_{model_name}.png"
        )
    )
    plt.show()
    plt.close()


def print_test_evaluation(
    best_model, X_test, y_test, ROC_PRECISION_RECALL_CURVE=ROC_PRECISION_RECALL_CURVE
):
    ensure_directory_exists(ROC_PRECISION_RECALL_CURVE)
    # Avaliar o modelo treinado no conjunto de teste
    # best_model = grid_search.best_estimator_
    model_name = best_model.named_steps[
        "classifier"
    ].__class__.__name__  # extrai o nome da classe do classificador
    # Prever as probabilidades no conjunto de teste
    y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Calcular o ROC-AUC no conjunto de teste
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_proba)
    roc_auc_test = auc(fpr_test, tpr_test)

    # Calcular Precision-Recall no conjunto de teste
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred_proba)
    auc_prec_test = auc(recall_test, precision_test)

    # Calcular G-Mean no conjunto de teste
    y_test_pred = best_model.predict(X_test)
    test_g_mean = geometric_mean_score(y_test, y_test_pred)

    # Exibir os resultados finais
    print(f"AUC-ROC on test data: {roc_auc_test:.4f}")
    print(f"AUCPREC on test data: {auc_prec_test:.4f}")
    print(f"G-Mean on test data: {test_g_mean:.4f}")

    # 9. Erro no conjunto de teste (1 - AUC Precision-Recall)
    test_error_auc_prec = 1 - auc_prec_test
    print(
        f"Erro no conjunto de teste (AUC Precision-Recall): {test_error_auc_prec:.4f}"
    )

    # 3. Erro no conjunto de teste (1 - G-Mean)
    test_error_g_mean = 1 - test_g_mean
    print(f"Erro no conjunto de teste (G-Mean): {test_error_g_mean:.4f}")
    # Plotando a curva ROC-AUC para o conjunto de teste
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(
        fpr_test, tpr_test, label=f"Curva ROC (AUC = {roc_auc_test:.2f})", color="b"
    )
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
    plt.xlabel("Taxa de Falso Positivo")
    plt.ylabel("Taxa de Falso Negativo")
    plt.title("Curva ROC (Test)")
    plt.legend(loc="best")

    # Plotando a curva Precision-Recall para o conjunto de teste
    plt.subplot(1, 2, 2)
    plt.plot(
        recall_test,
        precision_test,
        label=f"Curva de Precisão-Revocação Test (AUC = {auc_prec_test:.2f})",
        color="b",
    )
    plt.xlabel("Revocação")
    plt.ylabel("Precisão")
    plt.title("Curva de Precisão-Revocação (Test)")
    plt.legend(loc="best")
    plt.savefig(
        os.path.join(
            ROC_PRECISION_RECALL_CURVE,
            f"histograms_g_mean_roc_auc_auc_prec_{model_name}.png",
        )
    )
    # Exibir os gráficos
    plt.tight_layout()
    plt.show()
    plt.close()
