import joblib
import os

OUTPUT_MODEL_DIR = "../outputs/HRT/Mean/best_models/"


# Função para carregar e inspecionar o modelo
def inspect_model(model_name, output_model_dir=OUTPUT_MODEL_DIR):
    # Carregar o modelo salvo
    model_path = os.path.join(output_model_dir, f"best_{model_name}_model.pkl")
    model = joblib.load(model_path)

    # Exibir detalhes sobre o modelo carregado
    print(f"Modelo Carregado: {model_name}")
    print(f"Tipo de modelo: {type(model)}")

    # Se for um pipeline, verificar os passos
    if hasattr(model, "named_steps"):
        print(f"Pipeline steps: {model.named_steps}")

        # Verificar o classificador final
        if "classifier" in model.named_steps:
            print(f"Classificador Final: {model.named_steps['classifier']}")
            print(
                f"Hiperparâmetros do Classificador: {model.named_steps['classifier'].get_params()}"
            )
        else:
            print("Nenhum classificador encontrado no pipeline.")
    else:
        print("O modelo carregado não é um pipeline.")

    return model


# Exemplo de uso
model_name = "Random Forest"  # Substituir pelo nome correto do modelo
modelo_carregado = inspect_model(model_name)
