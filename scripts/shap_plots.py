import matplotlib
import matplotlib.pyplot as plt
import shap
from configs import SHAP_CONFIGS
from utils import ensure_directory_exists
import os
from HRTProcessing import class_mapping

# Set global plot configurations
plt.rc("font", family="Times New Roman", size=12)
matplotlib.use("Agg")  # Use backend for non-interactive plot generation

# Directories for saving plots
# SAVE_DIR_SUMMARY = "../outputs/Bettle/Div1/plots/SummaryPlots/"
# SAVE_DIR_WATERFALL = "../outputs/Bettle/Div1/plots/WaterfallPlots/"
SAVE_DIR_SUMMARY = "../outputs/HRT/AB-Analysis/plots/SummaryPlots4/"
SAVE_DIR_WATERFALL = "../outputs/HRT/AB-Analysis/plots/WaterfallPlots4/"
# Verificar e criar os diretórios, se necessário
ensure_directory_exists(SAVE_DIR_SUMMARY)
ensure_directory_exists(SAVE_DIR_WATERFALL)


def generate_shap_plots(model, model_name, X_train, X_test, SHAP_CONFIGS=SHAP_CONFIGS):
    """
    Generate SHAP plots based on the model type.

    Parameters:
    - model: The model to explain.
    - model_name: Name of the model.
    - X_train: Training data.
    - X_test: Test data.
    """
    if model_name in SHAP_CONFIGS["tree_based"]:
        plot_shap_tree_based(model, model_name, X_test)
    elif model_name in SHAP_CONFIGS["kernel_based"]:
        plot_shap_kernel(model, model_name, X_train, X_test)
    # elif model_name in SHAP_CONFIGS['gradient_based']:
    #     plot_shap_gradient_based(model, model_name, X_train, X_test)
    elif model_name in SHAP_CONFIGS["logistic_regression"]:
        plot_shap_linear_based_model(model, model_name, X_train, X_test)


def plot_shap_tree_based(
    model,
    model_name,
    X_test,
    class_mapping=class_mapping,
    SAVE_DIR_SUMMARY=SAVE_DIR_SUMMARY,
    SAVE_DIR_WATERFALL=SAVE_DIR_WATERFALL,
):
    """Plot SHAP values for tree-based models."""

    # Cria o explicador SHAP para modelos baseados em arvores
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if (
        len(shap_values.shape) == 3
    ):  # Ele devolve uma matriz onde a terceira dimensão é a classe
        # Para cada classe
        for i in range(shap_values.shape[2]):
            # Usar o mapeamento para obter o nome da classe original
            class_label = class_mapping.get(i, f"Classe {i}")
            plt.figure(figsize=(12, 6))
            plt.title(
                f"SHAP Summary Plot for {model_name} - {class_label}", fontsize=12
            )
            print("X_test")
            print(X_test)
            print("=============")
            shap.summary_plot(
                shap_values[:, :, i],
                X_test,
                feature_names=X_test.columns,
                max_display=20,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    SAVE_DIR_SUMMARY, f"{model_name}_shap_summary_{class_label}.png"
                ),
                bbox_inches="tight",
            )
            print(f"Gráfico salvo: {model_name}_shap_summary_{class_label}.png")
            plt.close()

            # Exibir Waterfall Plot para uma amostra (primeira amostra)
            shap_exp = shap.Explanation(
                values=shap_values[0, :, i],  # Primeira amostra, classe i
                base_values=explainer.expected_value[i],
                data=X_test.iloc[0],
            )
            plt.figure(figsize=(10, 6))
            plt.title(
                f"SHAP Waterfall Plot for {model_name} - {class_label} (Primeira Amostra)"
            )
            shap.plots.waterfall(shap_exp, max_display=10)  # Limitar a 10 atributos
            plt.savefig(
                os.path.join(
                    SAVE_DIR_WATERFALL, f"{model_name}_waterfall_{class_label}.png"
                ),
                bbox_inches="tight",
            )
            print(f"Gráfico salvo: {model_name}_waterfall_{class_label}.png")
            plt.close()


def plot_shap_kernel(
    model,
    model_name,
    X_train,
    X_test,
    class_mapping=class_mapping,
    SAVE_DIR_SUMMARY=SAVE_DIR_SUMMARY,
    SAVE_DIR_WATERFALL=SAVE_DIR_WATERFALL,
):
    # Cria o explicador SHAP para modelos genéricos usando KernelExplainer
    explainer = shap.KernelExplainer(
        model.predict_proba, X_train
    )  # Usar os dados de treino como baseline
    shap_values = explainer.shap_values(X_test)

    if (
        len(shap_values.shape) == 3
    ):  # Ele devolve uma matriz onde a terceira coluna é a classe
        # Para cada classe
        for i in range(shap_values.shape[2]):
            class_label = class_mapping.get(i, f"Classe {i}")
            plt.figure(figsize=(12, 6))
            plt.title(
                f"SHAP Summary Plot for {model_name} - {class_label}", fontsize=12
            )
            shap.summary_plot(
                shap_values[:, :, i],
                X_test,
                feature_names=X_test.columns,
                max_display=10,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    SAVE_DIR_SUMMARY, f"{model_name}_shap_summary_{class_label}.png"
                ),
                bbox_inches="tight",
            )
            print(f"Gráfico salvo: {model_name}_shap_summary_{class_label}.png")
            plt.close()

            # Exibir Waterfall Plot para uma amostra (primeira amostra)
            shap_exp = shap.Explanation(
                values=shap_values[0, :, i],  # Primeira amostra, classe i
                base_values=explainer.expected_value[i],
                data=X_test.iloc[0],
            )
            plt.figure(figsize=(10, 6))
            plt.title(
                f"SHAP Waterfall Plot for {model_name} - {class_label} (Primeira Amostra)",
                fontsize=12,
            )
            shap.plots.waterfall(shap_exp, max_display=10)
            plt.savefig(
                os.path.join(
                    SAVE_DIR_WATERFALL, f"{model_name}_waterfall_{class_label}.png"
                ),
                bbox_inches="tight",
            )
            print(f"Gráfico salvo: {model_name}_waterfall_{class_label}.png")
            plt.close()

    else:
        # Caso o modelo seja binário (shap_values será uma matriz 2D)
        plt.figure(figsize=(12, 6))
        plt.title(f"SHAP Summary Plot for {model_name}", fontsize=12)
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=X_test.columns,
            max_display=20,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(SAVE_DIR_SUMMARY, f"{model_name}_shap_summary.png"),
            bbox_inches="tight",
        )
        print(f"Gráfico salvo: {model_name}_shap_summary.png")
        plt.close()

        # Exibir Waterfall Plot para uma amostra (primeira amostra)
        shap_exp = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_test.iloc[0],
        )
        plt.figure(figsize=(10, 6))
        plt.title(
            f"SHAP Waterfall Plot for {model_name} - Primeira Amostra", fontsize=12
        )
        shap.plots.waterfall(shap_exp, max_display=10)
        plt.savefig(
            os.path.join(
                SAVE_DIR_WATERFALL, f"{model_name}_waterfall_primeira_amostra.png"
            ),
            bbox_inches="tight",
        )
        print(f"Gráfico salvo: {model_name}_waterfall_primeira_amostra.png")
        plt.close()


def plot_shap_linear_based_model(
    model,
    model_name,
    X_train,
    X_test,
    class_mapping=class_mapping,
    SAVE_DIR_SUMMARY=SAVE_DIR_SUMMARY,
    SAVE_DIR_WATERFALL=SAVE_DIR_WATERFALL,
):
    # Cria o explicador SHAP para modelos lineares
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)

    # Exibir o Summary Plot
    plt.figure(figsize=(12, 6))
    plt.title(f"SHAP Summary Plot for {model_name}", fontsize=12)
    shap.summary_plot(
        shap_values, X_test, feature_names=X_test.columns, max_display=20, show=False
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(SAVE_DIR_SUMMARY, f"{model_name}_shap_summary.png"),
        bbox_inches="tight",
    )
    print(f"Gráfico salvo: {model_name}_shap_summary.png")
    plt.close()

    # Exibir o Waterfall Plot para a primeira amostra
    shap_exp = shap.Explanation(
        values=shap_values[0], base_values=explainer.expected_value, data=X_test.iloc[0]
    )
    plt.figure(figsize=(10, 6))
    plt.title(f"SHAP Waterfall Plot for {model_name} - First Sample", fontsize=12)
    shap.plots.waterfall(shap_exp, max_display=10)
    plt.tight_layout()
    plt.savefig(
        os.path.join(SAVE_DIR_WATERFALL, f"{model_name}_waterfall_first_sample.png"),
        bbox_inches="tight",
    )
    print(f"Gráfico salvo: {model_name}_waterfall_first_sample.png")
    plt.close()
