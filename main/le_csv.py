import pandas as pd
import cv2 
import numpy as np
import pickle  # Importação para salvar e carregar o modelo
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Função para carregar a imagem e pré-processá-la
def carregar_imagem(caminho_imagem, tamanho=(64, 64)):
    try:
        # Tentar carregar a imagem em escala de cinza
        imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
        
        if imagem is None:
            raise FileNotFoundError(f"Imagem não encontrada: {caminho_imagem}")
        
        # Redimensionar a imagem
        imagem = cv2.resize(imagem, tamanho)
        
        # Normalizar a imagem
        imagem = imagem.astype('float32') / 255.0
        
        # Achatar a imagem para uma lista de características (vetor)
        imagem = imagem.flatten()
        
        return imagem
    
    except Exception as e:
        print(f"Erro ao carregar a imagem {caminho_imagem}: {e}")
        return None  # Retorna None se houver erro

def modelo(df, salvar_modelo=True):
    caracteristicas_conjunto = []
    rotulos_conjunto = []

    for i, linha in df.iterrows():
        caminho_imagem = 'Images/' + linha['filename']
        classe = linha['class']
        
        imagem_caracteristicas = carregar_imagem(caminho_imagem)
        
        if imagem_caracteristicas is not None:
            caracteristicas_conjunto.append(imagem_caracteristicas)
            rotulos_conjunto.append(classe)

    if len(caracteristicas_conjunto) == 0:
        raise ValueError("Não há imagens suficientes para testar o modelo.")

    X_conjunto = np.array(caracteristicas_conjunto)
    y_conjunto = np.array(rotulos_conjunto)

    # Criar e treinar o modelo Random Forest usando todos os dados de treino
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_conjunto, y_conjunto)

    y_pred = modelo.predict(X_conjunto)

    accuracy = accuracy_score(y_conjunto, y_pred)
    print(f"Acurácia do modelo no treino: {accuracy:.2f}")

    matriz_confusao = confusion_matrix(y_conjunto, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_conjunto), yticklabels=np.unique(y_conjunto))
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.show()

    # Salvar o modelo treinado automaticamente
    if salvar_modelo:
        with open("modelo_random_forest.pkl", "wb") as arquivo:
            pickle.dump(modelo, arquivo)
        print("✅ Modelo salvo como 'modelo_random_forest.pkl'.")

    return modelo

def executar(df, modelo):
    caracteristicas_conjunto = []
    rotulos_conjunto = []

    for i, linha in df.iterrows():
        caminho_imagem = 'Images/' + linha['filename']
        classe = linha['class']
        
        imagem_caracteristicas = carregar_imagem(caminho_imagem)
        
        if imagem_caracteristicas is not None:
            caracteristicas_conjunto.append(imagem_caracteristicas)
            rotulos_conjunto.append(classe)

    if len(caracteristicas_conjunto) == 0:
        raise ValueError("Não há imagens suficientes para testar o modelo.")

    X_conjunto = np.array(caracteristicas_conjunto)
    y_conjunto = np.array(rotulos_conjunto)

    y_pred = modelo.predict(X_conjunto)

    accuracy = accuracy_score(y_conjunto, y_pred)
    print(f"Acurácia do modelo na validação: {accuracy:.2f}")

    matriz_confusao = confusion_matrix(y_conjunto, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_conjunto), yticklabels=np.unique(y_conjunto))
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.show()

def carregar_modelo(caminho_modelo="modelo_random_forest.pkl"):
    """Carrega o modelo salvo e retorna o objeto do modelo treinado."""
    try:
        with open(caminho_modelo, "rb") as arquivo:
            modelo = pickle.load(arquivo)
        print(f"✅ Modelo carregado com sucesso de '{caminho_modelo}'.")
        return modelo
    except FileNotFoundError:
        print(f"❌ Erro: O arquivo '{caminho_modelo}' não foi encontrado.")
        return None

df_treino = pd.read_csv(r'csv\tabela_treino.csv')
modelo_random_forest = modelo(df_treino, salvar_modelo=True)  # Treinar e salvar

df_valida = pd.read_csv(r'csv\tabela_validar.csv')
executar(df_valida, modelo_random_forest)

# Exemplo: Carregar o modelo salvo posteriormente
modelo_carregado = carregar_modelo()
