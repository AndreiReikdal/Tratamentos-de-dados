import os
import pandas as pd

# Importar bibliotecas
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Carregar o arquivo CSV em um DataFrame (substitua pelo seu caminho real)
caminho_arquivo = (r'C:\\Users\Windows 10\\Downloads\\Desafios estagio IA\\tratamento_dados\\CSVDados.csv')
dados = pd.read_csv(caminho_arquivo)

# Selecionar os atributos numéricos
atributos = dados[['Atributo 1', 'Atributo 2', 'Atributo 3', 'Atributo 4']]

# Criar e treinar o modelo K-Means
n_grupos = 3
modelo_agrupamento = KMeans(n_clusters=n_grupos, random_state=42)

# Adicionar a coluna 'Categoria Prevista' ao DataFrame
dados['Categoria Prevista'] = modelo_agrupamento.fit_predict(atributos)

# Mapear os grupos para categorias
categorias = {0: 'Categoria A', 1: 'Categoria B', 2: 'Categoria C'}
dados['Categoria Prevista'] = dados['Categoria Prevista'].map(categorias)

# Visualizar os atributos e a coluna de categorias
print("\nAtributos e Categoria Prevista:")
print(dados[['Atributo 1', 'Atributo 2', 'Atributo 3', 'Atributo 4', 'Categoria Prevista']])



# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# from statsmodels.stats.multicomp import MultiComparison
# from sklearn.model_selection import train_test_split