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
from sklearn.svm import SVC
# from imblearn.pipeline import ImbalancedPipeline

SCALERS = [
    "None",
    "MinMaxScaler",
    "MaxAbsScaler",
    "RobustScaler",
    "StandardScaler",
    "QuantileTransformer",
    "Binarizer",
    "PowerTransformer",
    "FunctionTransformer",
]
SHAP_CONFIGS = {
    "tree_based": [
        "Random Forest",
        "Decision Tree",
        "Gradient Boosting",
        "Extra Trees",
        "LightGBM",
        "XGBoost",
    ],
    "kernel_based": ["Logistic Regression", "AdaBoost", "K-Neighbors"],
    "logistic_regression": ["SVM", "SGD"],
}
scaler_mapping = {
    "None": None,
    "MinMaxScaler": MinMaxScaler,
    "MaxAbsScaler": MaxAbsScaler,
    "RobustScaler": RobustScaler,
    "StandardScaler": StandardScaler,
    "QuantileTransformer": QuantileTransformer,
    "Binarizer": Binarizer,
    "PowerTransformer": PowerTransformer,
    "FunctionTransformer": FunctionTransformer,
}

pipelines = {
    "Logistic Regression": Pipeline(
        [
            ("scaler", "passthrough"),  # Escalonamento
            ("classifier", LogisticRegression(random_state=42)),
        ]
    ),
    "Naive Bayes": Pipeline(
        [
            (
                "scaler",
                "passthrough",
            ),  # Placeholder for scaler   - calcula as probabilidades para todas as classes e seleciona a classe com a maior probabilidade, usa da regra de Bayes e assume independencia entre as classes (probabilidade de uma classe K dado que temos a sample X)
            ("classifier", GaussianNB()),
        ]
    ),
    "Decision Tree": Pipeline(
        [
            ("scaler", "passthrough"),
            ("classifier", DecisionTreeClassifier(random_state=42)),
        ]
    ),
    "Random Forest": Pipeline(
        [
            ("scaler", "passthrough"),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    ),
    "Bagging": Pipeline(
        [("scaler", "passthrough"), ("classifier", BaggingClassifier(random_state=42))]
    ),
    "AdaBoost": Pipeline(
        [("scaler", "passthrough"), ("classifier", AdaBoostClassifier(random_state=42))]
    ),
    "Gradient Boosting": Pipeline(
        [
            ("scaler", "passthrough"),
            ("classifier", GradientBoostingClassifier(random_state=42)),
        ]
    ),
    "Extra Trees": Pipeline(
        [
            ("scaler", "passthrough"),
            ("classifier", ExtraTreesClassifier(random_state=42)),
        ]
    ),
    "K-Neighbors": Pipeline(
        [("scaler", "passthrough"), ("classifier", KNeighborsClassifier())]
    ),
    "SGD": Pipeline(
        [("scaler", "passthrough"), ("classifier", SGDClassifier(random_state=42))]
    ),
    "SVM": Pipeline(
        [
            ("scaler", "passthrough"),
            (
                "classifier",
                SVC(probability=True, random_state=42),
            ),  # Definir probability=True para permitir predict_proba
        ]
    ),
    # "XGBoost": Pipeline([
    # ('scaler', 'passthrough'),  # Utilize o escalador conforme necessário
    # ('classifier', xgb.XGBClassifier(
    #     eval_metric='logloss',
    #     objective='binary:logistic',  # Para early stopping
    #     n_jobs=-1
    # ))
    # ])
}


params = {
    "Logistic Regression": {
        "scaler": [
            None,
            MinMaxScaler(),
            MaxAbsScaler(),
            RobustScaler(),
            StandardScaler(),
            QuantileTransformer(n_quantiles=32),
            Binarizer(),
            PowerTransformer(),
            FunctionTransformer(),
        ],
        "classifier__C": [0.001, 0.01, 0.1, 1],
        # "classifier__penalt kill-pane --pane-id 13
        # m": ["elasticnet"],  # Usando elasticnet
        "classifier__l1_ratio": [0, 0.3, 0.5, 0.7],  # Ajuste da proporção entre l1 e l2
        "classifier__solver": ["saga"],  # 'saga' pode ser útil com elasticnet
        "classifier__max_iter": [
            1000,
            2000,
            3000,
            4000,
        ],  # Para garantir a convergência
    },
    "Naive Bayes": {
        "scaler": [
            None,
            MinMaxScaler(),
            MaxAbsScaler(),
            RobustScaler(),
            StandardScaler(),
            PowerTransformer(),
            FunctionTransformer(),
        ],
        "classifier__var_smoothing": [1e-09, 1e-08, 1e-07, 1e-05, 1e-04, 1e-03],
    },
    "Decision Tree": {
        "scaler": [None],
        "classifier__max_depth": [None, 10, 20, 30],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__criterion": ["gini", "entropy"],
        "classifier__splitter": ["best", "random"],
    },
    "Random Forest": {
        "scaler": [None],
        "classifier__n_estimators": [200, 100, 30, 50],  # Número de árvores
        "classifier__max_depth": [
            None,
            2,
            3,
            5,
        ],  # Limite a profundidade para evitar overfitting
        "classifier__min_samples_split": [
            2,
            5,
            10,
        ],  # The minimum number of samples required to split an internal node
        "classifier__class_weight": ["balanced"],  # Lidando com desbalanceamento
        "classifier__criterion": ["gini", "entropy"],
        "classifier__min_samples_leaf": [
            1,
            2,
            4,
        ],  # The minimum number of samples required to be at a leaf node.
        # A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches.
        "classifier__bootstrap": [True, False],
    },
    "Bagging": {
        "scaler": [None],
        "classifier__n_estimators": [30, 20, 10],  # Reduzindo o número de estimadores
        "classifier__bootstrap": [True],  # Usar Bootstrap para aumentar a robustez
        "classifier__bootstrap_features": [
            False
        ],  # Evitar usar Bootstrap nas features, pois pode ser desnecessário em datasets pequenos
    },
    "AdaBoost": {
        "scaler": [None],
        "classifier__n_estimators": [50, 75],  # Reduzindo o número de estimadores
        "classifier__learning_rate": [
            0.1,
            0.01,
            0.05,
        ],  # Reduzindo as taxas de aprendizado
        # "classifier__algorithm": [
        #     "SAMME.R",
        #     "SAMME",
        # ],  # Usando o algoritmo SAMME.R (mais eficiente em muitos casos)
    },
    "Gradient Boosting": {
        "scaler": [None],
        "classifier__n_estimators": [200, 100, 50],  # Menor número de árvores
        "classifier__learning_rate": [
            0.01,
            0.05,
            0.1,
        ],  # Ajustando as taxas de aprendizado
        "classifier__max_depth": [5, 4, 3],  # Limitando a profundidade das árvores
        "classifier__min_samples_split": [10, 5],  # Evitando divisões excessivas
        "classifier__min_samples_leaf": [
            4,
            2,
        ],  # Maior número mínimo de amostras por folha
        "classifier__subsample": [
            0.8,
            1.0,
        ],  # Ajuste de subsample para evitar overfitting
        "classifier__max_features": [
            "sqrt",
            "log2",
        ],  # Limitar o número de features consideradas por árvore
    },
    # Valid parameters are: ['memory', 'steps', 'verbose'].
    "Extra Trees": {
        "scaler": [None],
        "classifier__n_estimators": [
            200,
            100,
            10,
            30,
            50,
        ],  # Reduzindo o número de estimadores
        "classifier__max_depth": [
            15,
            None,
            2,
            5,
            10,
            15,
        ],  # Limitando a profundidade das árvores
        "classifier__min_samples_split": [
            5,
            10,
        ],  # Aumentando o mínimo de amostras para uma divisão
        "classifier__min_samples_leaf": [
            2,
            4,
            6,
        ],  # Aumentando o mínimo de amostras por folha
        "classifier__criterion": [
            "gini",
            "entropy",
        ],  # Mantendo o critério de decisão padrão
        "classifier__bootstrap": [True],  # Usando bootstrap para robustez
        "classifier__class_weight": [
            "balanced"
        ],  # Ajustando o peso das classes para lidar com o desbalanceamento - Da mais peso as classes minoritarias
    },
    "K-Neighbors": {
        "scaler": [
            None,
            PowerTransformer(method="yeo-johnson", standardize=False),
            StandardScaler(),
            MinMaxScaler(),
            RobustScaler(),
            QuantileTransformer(n_quantiles=32, output_distribution="normal"),
            Normalizer(),
        ],
        "classifier__n_neighbors": [2, 3, 5, 7, 9],
        "classifier__weights": ["uniform", "distance"],
        "classifier__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "classifier__leaf_size": [10, 30, 50],
        "classifier__p": [1, 2],
        "classifier__metric": ["minkowski"],
        "classifier__metric_params": [None],  # Mantendo o parâmetro padrão
    },
    #     "XGBoost": {
    #     'scaler':[None],
    #     'classifier__n_estimators': [200, 100, 75],  # Reduzir o número de estimadores para evitar overfitting em datasets pequenos
    #     'classifier__learning_rate': [0.01, 0.1, 0.2],  # Mantendo taxas de aprendizado baixas para permitir melhor ajuste
    #     'classifier__max_depth': [3, 4, 5],  # Limitar a profundidade das árvores para evitar overfitting
    #     'classifier__min_child_weight': [1, 3],  # Controle de regularização, mais alto é mais conservador
    #     'classifier__gamma': [0, 0.1, 0.2],  # Reduz o risco de overfitting ao exigir uma maior redução de perda
    #     'classifier__subsample': [0.7, 0.8],  # Usar uma amostragem menor pode ajudar a reduzir overfitting
    #     'classifier__colsample_bytree': [0.7, 0.8],  # Amostragem de colunas para regularização adicional
    #     'classifier__objective': ['binary:logistic'],  # Definido para problemas de classificação binária
    #     'classifier__reg_alpha': [0, 0.1, 0.5, 1],  # Regularização L1 para reduzir coeficientes não importantes
    #     'classifier__reg_lambda': [0.5, 1, 10],  # Regularização L2 para evitar grandes coeficientes
    #     'classifier__scale_pos_weight': [1, 0.5, 2, 3],  # Compensação para desbalanceamento de classes
    #     'classifier__verbosity': [0]  # Silencia logs desnecessários
    # },
    "SGD": {
        "scaler": [None, StandardScaler(), MinMaxScaler()],  # Priorizar StandardScaler
        "classifier__alpha": [
            0.001,
            0.01,
            0.1,
        ],  # Valores mais altos para evitar overfitting
        "classifier__max_iter": [
            1000,
            1500,
        ],  # Reduzir para cenários com dados pequenos
        "classifier__penalty": [
            "l2",
            "elasticnet",
        ],  # Priorizar L2 e elasticnet para maior estabilidade
        "classifier__loss": [
            "log_loss",
            "modified_huber",
        ],  # Dar mais ênfase ao log e modified_huber
        "classifier__tol": [
            0.001,
            0.0001,
        ],  # Mantendo os valores padrões, mas com uma leve ênfase em valores maiores
    },
    "SVM": {
        "scaler": [
            None,
            StandardScaler(),
            MinMaxScaler(),
            RobustScaler(),
            MaxAbsScaler(),
            QuantileTransformer(n_quantiles=32),
        ],  # Reduzir as opções de escaladores
        "classifier__C": [
            0.01,
            0.1,
            1,
        ],  # Manter valores mais baixos para evitar overfitting
        "classifier__kernel": ["linear", "rbf"],  # Priorizar kernels mais simples
        "classifier__gamma": ["scale"],  # Focar em 'scale' para dados pequenos
        "classifier__class_weight": [
            "balanced"
        ],  # Usar 'balanced' para lidar com desbalanceamento
    },
}
