# dental-classifier


## 🎯 Objetivo:
Identificar o melhor modelo de machine learning para classificar, com precisão e confiabilidade, diferentes classes odontológicas, como:
* 🦷 Restaurações (Fillings)
* ⚙️ Implantes (Implant)
* 🦠 Cáries (Cavity)
* ❌ Dentes Impactados (Impacted Tooth)

## 📊 Sobre o Dataset:
* Fonte: Kaggle - Dental Radiography
* Total de imagens: 1.076 (resolução 512x256)
* Atributos: coordenadas (xmin, ymin, xmax, ymax) para delimitação dos objetos e classes identificadas

## Divisão dos Dados:
* 📚 Treinamento: 1.001 imagens
* 🧪 Teste: 500 imagens
* ✅ Validação: 200 imagens
 
## 🧠 Técnicas de Aprendizado de Máquina Utilizadas:
* 🌳 Random Forest (modelo escolhido pelo melhor desempenho)
* 🌲 Decision Tree
* 🌿 Random Tree

## 💻 Ferramentas Utilizadas:
* Python: para limpeza de dados e preparação do dataset
* Tkinter: para o desenvolvimento da interface gráfica, que realiza a leitura das imagens e exibe os resultados da classificação
