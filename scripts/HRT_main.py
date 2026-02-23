# Importing
# pip install -r requirements.txt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# from processamentoBioreactor import X_train, X_test, y_train, y_test
from processamento_HRT import X_train, X_test, y_train, y_test
from model_evaluation import evaluate_model2
from model_evaluation import print_validation_evaluation
from model_evaluation import print_test_evaluation
from utils import save_model, load_model
from utils import ensure_directory_exists
from utils import exportar_csv
from configs import pipelines  # Hiperparâmetros
from configs import params
from configs import SCALERS
from sklearn.preprocessing import LabelEncoder
from shap_plots import generate_shap_plots
from limeplots import generate_lime_explanation
from configs import pipelines, params
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
import os

# Configs
warnings.filterwarnings("ignore")
plt.rc("font", family="Times New Roman", size=12)
# OUTPUT_MATRIZ = "../outputs/Bettle/Div1/plots/Confusion_matriz"
OUTPUT_MATRIZ = "../outputs/HRT/AB-Analysis/plots/confusion_matrix"
ensure_directory_exists(OUTPUT_MATRIZ)

if __name__ == "__main__":
    # Iteração sobre os modelos
    best_models = {}
    for model_name in list(pipelines.keys()):
        print(
            f"=================================================================================================="
        )
        print(f"Running GridSearch for {model_name}")

        # Rodar o GridSearchCV para encontrar o melhor modelo
        model_search = evaluate_model2(
            pipelines[model_name], params[model_name], X_train, y_train
        )
        # Salvando o modelo
        save_model(
            model_search.best_estimator_, model_name
        )  # salva o melhor modelo treinado após o GridSearchCV.

        # Carregar o melhor modelo salvo
        best_model = load_model(model_name)
        # Recupera o melhor estimador e refaz o treinamento com todos os dados de treino
        # best_model = model_search.best_estimator_
        best_model.fit(X_train, y_train)

        if best_model.named_steps["scaler"] is not None:
            X_test_transformed1 = best_model.named_steps["scaler"].transform(X_test)
            X_test_transformed = pd.DataFrame(
                X_test_transformed1, columns=X_train.columns
            )

        else:
            X_test_transformed = X_test  # Sem transformação se não houver scaler

        # Armazenar o melhor modelo
        best_models[model_name] = best_model
        print(
            f"Best parameters for {model_name}: {best_model.get_params}"
        )  # best_params_
        print(
            f"Best cross-validation accuracy for {model_name}: {model_search.best_score_}"
        )  # best_score_

        exportar_csv(model_search, model_name, pipelines, SCALERS)
        print_validation_evaluation(model_search)
        print_test_evaluation(best_model, X_test_transformed, y_test)

        # Plotar a matriz de confusão
        CM = confusion_matrix(y_test, best_model.predict(X_test_transformed))
        disp = ConfusionMatrixDisplay(confusion_matrix=CM)
        disp.plot(cmap="Blues", colorbar=True)
        plt.tight_layout()  # Ajusta automaticamente o layout para evitar corte
        # Salvar o gráfico como PNG
        plt.savefig(
            os.path.join(OUTPUT_MATRIZ, f"matriz_confusao_{model_name}.png"),
            bbox_inches="tight",
        )

        # Fechar a figura após o salvamento
        plt.close()

        # Avaliar no conjunto de teste
        print(classification_report(y_test, best_model.predict(X_test_transformed)))

        # Gerar gráficos SHAP
        generate_shap_plots(
            best_model.named_steps["classifier"],
            model_name,
            X_train,
            X_test_transformed,
        )

        # # Gerar explicação LIME
        # generate_lime_explanation(best_model, X_train, X_test)
