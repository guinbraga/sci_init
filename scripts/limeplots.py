from lime.lime_tabular import LimeTabularExplainer


# Função para gerar explicação LIME
def generate_lime_explanation(best_model, X_train, X_test, i=0):
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns,
        class_names=["Classe 0", "Classe 1"],
        mode="classification",
    )
    exp = explainer.explain_instance(
        X_test.iloc[i], best_model.predict_proba, num_features=5
    )
    exp.show_in_notebook(show_table=True)

