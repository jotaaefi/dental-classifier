import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def carregar_imagem_teste(caminho_imagem, tamanho=(64, 64)):
    """Carrega uma única imagem e a pré-processa para ser usada pelo modelo."""
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

def testar_imagem_aleatoria(modelo, pasta_imagens, df_labels=None):
    """Seleciona uma imagem aleatória da pasta e a classifica com o modelo treinado."""
    lista_imagens = os.listdir(pasta_imagens)
    if not lista_imagens:
        raise ValueError("Nenhuma imagem encontrada na pasta.")

    imagem_aleatoria = random.choice(lista_imagens)
    caminho_imagem = os.path.join(pasta_imagens, imagem_aleatoria)

    imagem_teste = carregar_imagem_teste(caminho_imagem)
    if imagem_teste is None:
        return
    
    # Converter para formato adequado para o modelo
    X_teste = np.array([imagem_teste])
    
    # Fazer a predição
    y_pred = modelo.predict(X_teste)
    
    # Se o DataFrame de labels estiver disponível, verificamos a classe real
    classe_real = None
    if df_labels is not None:
        classe_real = df_labels[df_labels['filename'] == imagem_aleatoria]['class'].values
        if len(classe_real) > 0:
            classe_real = classe_real[0]
    
    # Mostrar a imagem escolhida
    plt.imshow(X_teste.reshape(64, 64), cmap='gray')
    plt.title(f"Predição: {y_pred[0]} | Classe Real: {classe_real if classe_real else 'Desconhecida'}")
    plt.axis("off")
    plt.show()

    # Retornar os resultados
    return y_pred[0], classe_real

# Diretório onde estão as imagens de teste
pasta_teste = 'Images/'  # Substitua pelo caminho correto

# Carregar os labels, se disponível
df_labels = pd.read_csv(r'csv\tabela_validar.csv')  # Ajuste o caminho do CSV se necessário

# Testar uma imagem aleatória com o modelo treinado
predicao, classe_real = testar_imagem_aleatoria(modelo_random_tree, pasta_teste, df_labels)

# Comparar predição e realidade, se possível
if classe_real:
    acuracia = 1.0 if predicao == classe_real else 0.0
    print(f"Acurácia para esta imagem: {acuracia:.2f}")
else:
    print("Não foi possível comparar com a classe real.")
