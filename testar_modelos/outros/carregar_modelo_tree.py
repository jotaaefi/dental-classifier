import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import pickle
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# Função para carregar o modelo treinado
def carregar_modelo(caminho_modelo="modelo_random_tree.pkl"):
    try:
        with open(caminho_modelo, "rb") as arquivo:
            modelo = pickle.load(arquivo)
        print(f"✅ Modelo carregado com sucesso de '{caminho_modelo}'.")
        return modelo
    except FileNotFoundError:
        print(f"❌ Erro: O arquivo '{caminho_modelo}' não foi encontrado.")
        return None

# Função para carregar e pré-processar a imagem
def carregar_imagem_teste(caminho_imagem, tamanho=(64, 64)):
    try:
        imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
        if imagem is None:
            raise FileNotFoundError(f"Imagem não encontrada: {caminho_imagem}")

        # Redimensionar a imagem
        imagem = cv2.resize(imagem, tamanho)

        # Normalizar a imagem (escala 0 a 1)
        imagem = imagem.astype('float32') / 255.0

        # Achatar para um vetor de características
        imagem = imagem.flatten()

        return imagem

    except Exception as e:
        print(f"❌ Erro ao carregar a imagem: {e}")
        return None

# Função para carregar a imagem via interface gráfica
def selecionar_imagem():
    global caminho_imagem
    caminho_imagem = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.jpeg *.png")])
    
    if caminho_imagem:
        img = Image.open(caminho_imagem)
        img = img.resize((200, 200))  # Redimensionar para exibição
        img = ImageTk.PhotoImage(img)
        
        label_imagem.config(image=img)
        label_imagem.image = img
        label_status.config(text="Imagem carregada. Pressione 'Rodar Predição'.")

# Função para testar a imagem carregada
def testar_imagem():
    if not caminho_imagem:
        label_status.config(text="❌ Nenhuma imagem selecionada!")
        return

    imagem_teste = carregar_imagem_teste(caminho_imagem)
    
    if imagem_teste is None:
        label_status.config(text="❌ Erro ao processar a imagem.")
        return

    # Converter para formato adequado para o modelo (array 2D)
    X_teste = np.array([imagem_teste])

    # Fazer a predição
    y_pred = modelo.predict(X_teste)

    # Atualizar a interface com o resultado da predição
    label_resultado.config(text=f"✅ Classe Predita: {y_pred[0]}")

# Carregar o modelo treinado
modelo = carregar_modelo()

# Criando a Interface
root = tk.Tk()
root.title("Classificação de Imagens")
root.geometry("400x500")

# Botão para carregar imagem
btn_selecionar = tk.Button(root, text="Selecionar Imagem", command=selecionar_imagem, font=("Arial", 12))
btn_selecionar.pack(pady=10)

# Label para exibir a imagem carregada
label_imagem = tk.Label(root)
label_imagem.pack()

# Status da seleção da imagem
label_status = tk.Label(root, text="Nenhuma imagem selecionada.", font=("Arial", 10))
label_status.pack(pady=5)

# Botão para rodar a predição
btn_predicao = tk.Button(root, text="Rodar Predição", command=testar_imagem, font=("Arial", 12))
btn_predicao.pack(pady=10)

# Label para mostrar o resultado da predição
label_resultado = tk.Label(root, text="", font=("Arial", 14, "bold"), fg="blue")
label_resultado.pack(pady=10)

# Rodar a Interface
root.mainloop()
