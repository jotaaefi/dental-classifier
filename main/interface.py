import tkinter as tk
from tkinter import filedialog, messagebox
import pickle  # Ou outro método conforme seu modelo
import pandas as pd  # Caso precise processar dados


def carregar_modelo():
    with open('modelo.pkl', 'rb') as f:
        modelo = pickle.load(f)
    return modelo

modelo = carregar_modelo()

def selecionar_arquivo():
    caminho_arquivo = filedialog.askopenfilename(filetypes=[("CSV Files", ".csv"), ("All Files", ".*")])
    if caminho_arquivo:
        analisar_arquivo(caminho_arquivo)

def analisar_arquivo(caminho_arquivo):
    try:
        dados = pd.read_csv(caminho_arquivo)  # Adapte conforme seu modelo
        resultado = modelo.predict(dados)  # Supondo que o modelo aceite esse formato
        messagebox.showinfo("Resultado da Análise", f"Predição: {resultado}")
    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro ao processar o arquivo:\n{str(e)}")

#Criando a interface gráfica
root = tk.Tk()
root.title("Analisador de Arquivos")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack(pady=20)

btn_selecionar = tk.Button(frame, text="Selecionar Arquivo", command=selecionar_arquivo)
btn_selecionar.pack()

root.mainloop()


