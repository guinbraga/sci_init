from sklearn.model_selection import GridSearchCV
from configs import pipelines, params, scaler_mapping, SCALERS
import numpy as np
import pandas as pd
import joblib
import os

# Directories for saving plots

# OUTPUT_DIR = "../outputs/Bettle/Div1/gridsearch_results3/"
# OUTPUT_MODEL_DIR = "../outputs/Bettle/Div1/best_models/"

OUTPUT_DIR = "../outputs/HRT/Model-optimazation/gridsearch_results/"
OUTPUT_MODEL_DIR = "../outputs/HRT/Model-optimazation/best_models/"


# Função para garantir que o diretório exista
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def exportar_csv(
    model_search,
    model_name,
    pipelines,
    scalers=SCALERS,
    scaler_mapping=scaler_mapping,
    OUTPUT_DIR=OUTPUT_DIR,
):
    # Checar se a pipeline tem um scaler
    results = []
    cv_results = model_search.cv_results_
    if "scaler" in pipelines[model_name].named_steps:
        print(f"\nChecking scalers for {model_name}...")
        # Verificar os parâmetros do escalador
        # for i, scaler in enumerate(model_search.cv_results_['param_scaler']):
        #     print(f"Índice: {i}, Valor: {scaler}, Tipo: {type(scaler)}")
        if "param_scaler" in model_search.cv_results_:
            for scaler_name in scalers:
                #     # Identificar os índices para o scaler específico
                #     indices_scaler = [i for i, param in enumerate(model_search.cv_results_['param_scaler']) if str(param) == scaler_name]
                # Pegar a classe correspondente ao nome do scaler
                scaler_class = scaler_mapping[scaler_name]

                # # Verificar se o tipo do parâmetro coincide com o tipo do scaler
                # indices_scaler = [
                #     i for i, param in enumerate(model_search.cv_results_['param_scaler'])
                #     if (scaler_class is None and param is None) or isinstance(param, scaler_class)]
                # Verificar se é None ou uma classe válida para comparar
                indices_scaler = [
                    i
                    for i, param in enumerate(model_search.cv_results_["param_scaler"])
                    if (param is None and scaler_class is None)
                    or (scaler_class is not None and isinstance(param, scaler_class))
                ]

                if indices_scaler:
                    # Melhor índice para o scaler específico
                    best_index_scaler = indices_scaler[
                        np.argmax(
                            model_search.cv_results_["mean_test_g_mean"][indices_scaler]
                        )
                    ]
                    best_score_scaler = model_search.cv_results_["mean_test_g_mean"][
                        best_index_scaler
                    ]
                    std_score_scaler = model_search.cv_results_["std_test_g_mean"][
                        best_index_scaler
                    ]  # Desvio padrão para a métrica de teste
                    best_score_scaler_train = model_search.cv_results_[
                        "mean_train_g_mean"
                    ][best_index_scaler]
                    std_score_scaler_train = model_search.cv_results_[
                        "std_train_g_mean"
                    ][best_index_scaler]

                    # Armazenar os resultados no formato adequado para exportação
                    results.append(
                        {
                            "Model": model_name,
                            "Scaler": scaler_name,
                            "Best Params": model_search.cv_results_["params"][
                                best_index_scaler
                            ],
                            "Mean g_mean test": best_score_scaler,
                            "Std Dev g_mean test": std_score_scaler,
                            "Mean g_mean train": best_score_scaler_train,
                            "Std Dev g_mean trai": std_score_scaler_train,
                        }
                    )

                    print(f"\nMelhores parâmetros para {scaler_name}:")
                    print(model_search.cv_results_["params"][best_index_scaler])
                    print(f"Melhor score para {scaler_name}: {best_score_scaler}")
                    print(f"Desvio padrão: {std_score_scaler}")
                else:
                    print(f"Nenhum resultado encontrado para {scaler_name}")
                    # Criar um DataFrame com os resultados
            results_df = pd.DataFrame(results)
            # Exportar para CSV
            results_df.to_excel(
                os.path.join(OUTPUT_DIR, f"gridsearch_{model_name}_results.xlsx"),
                index=False,
            )
            print(f"Resultados exportados para 'gridsearch_{model_name}_results.xlsx'")

    else:
        print(f"Scaler não encontrado nos parâmetros do GridSearch para {model_name}.")


def save_model(model, model_name, OUTPUT_MODEL_DIR=OUTPUT_MODEL_DIR):
    joblib.dump(model, os.path.join(OUTPUT_MODEL_DIR, f"best_{model_name}_model.pkl"))


def load_model(model_name, OUTPUT_MODEL_DIR=OUTPUT_MODEL_DIR):
    return joblib.load(os.path.join(OUTPUT_MODEL_DIR, f"best_{model_name}_model.pkl"))
