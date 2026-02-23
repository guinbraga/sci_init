# Distribuição das classes no conjunto de treinamento
print("Distribuição das classes em y no conjunto de treinamento:")
print(train_df['y'].value_counts())

# Distribuição das classes no conjunto de teste
print("\nDistribuição das classes em y no conjunto de teste:")
print(test_df['y'].value_counts())

print("y=================")
print(y_train)
print(y_test)
#Consulta
print("Consulta===================")
print(train_df.shape)
print(test_df.shape)
print(train_df['y'].unique())
# train_df.dtypes
print("types=================")
print(test_df.dtypes)


# Verificação dos formatos
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")