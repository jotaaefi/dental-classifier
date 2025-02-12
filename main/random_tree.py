import pandas as pd
import cv2
import numpy as np
import os
import pickle  # Importação para salvar o modelo
from sklearn.ensemble import ExtraTreesClassifier  # Random Tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

def carregar_imagem(caminho_imagem, tamanho=(64, 64)):
    try:
        imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
        if imagem is None:
            raise FileNotFoundError(f"Imagem não encontrada: {caminho_imagem}")

        imagem = cv2.resize(imagem, tamanho)
        imagem = imagem.astype('float32') / 255.0
        return imagem.flatten()

    except Exception as e:
        print(f"Erro ao carregar a imagem {caminho_imagem}: {e}")
        return None

def preparar_dados(df, caminho_base='Images/'):
    caracteristicas_conjunto = []
    rotulos_conjunto = []

    for _, linha in df.iterrows():
        caminho_imagem = os.path.join(caminho_base, linha['filename'])
        classe = linha['class']

        imagem_caracteristicas = carregar_imagem(caminho_imagem)
        if imagem_caracteristicas is not None:
            caracteristicas_conjunto.append(imagem_caracteristicas)
            rotulos_conjunto.append(classe)

    if len(caracteristicas_conjunto) == 0:
        raise ValueError("Não há imagens suficientes para testar o modelo.")

    return np.array(caracteristicas_conjunto), np.array(rotulos_conjunto)

def treinar_modelo_random_tree(df, n_estimators=100, max_depth=None, salvar_modelo=True):
    X, y = preparar_dados(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo Random Tree (Extra Trees) no conjunto de teste: {accuracy:.2f}")

    matriz_confusao = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão - Random Tree')
    plt.show()

    # Salvar o modelo treinado automaticamente
    if salvar_modelo:
        with open("modelo_random_tree.pkl", "wb") as arquivo:
            pickle.dump(modelo, arquivo)
        print("Modelo salvo como 'modelo_random_tree.pkl'.")

    return modelo

def validar_modelo(df, modelo):
    X, y = preparar_dados(df)
    y_pred = modelo.predict(X)

    accuracy = accuracy_score(y, y_pred)
    print(f"Acurácia do modelo na validação: {accuracy:.2f}")

    matriz_confusao = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão - Validação Random Tree')
    plt.show()

def carregar_modelo(caminho_modelo="modelo_random_tree.pkl"):
    """Carrega o modelo salvo e retorna o objeto do modelo treinado."""
    try:
        with open(caminho_modelo, "rb") as arquivo:
            modelo = pickle.load(arquivo)
        print(f"Modelo carregado com sucesso de '{caminho_modelo}'.")
        return modelo
    except FileNotFoundError:
        print(f"Erro: O arquivo '{caminho_modelo}' não foi encontrado.")
        return None

df_treino = pd.read_csv(r'csv\tabela_treino.csv')
modelo_random_tree = treinar_modelo_random_tree(df_treino, n_estimators=100, max_depth=10)

df_valida = pd.read_csv(r'csv\tabela_validar.csv')
validar_modelo(df_valida, modelo_random_tree)

# Exemplo: Carregar o modelo salvo posteriormente
modelo_carregado = carregar_modelo()
