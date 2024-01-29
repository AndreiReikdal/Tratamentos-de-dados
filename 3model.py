# Importar bibliotecas
import pandas as pd
from sklearn.cluster import KMeans

# Carregar o arquivo CSV em um DataFrame
caminho_arquivo = 'caminho/do/seu/arquivo.csv'
dados = pd.read_csv(caminho_arquivo)

# Selecionar os atributos numéricos
atributos = dados[['Atributo 1', 'Atributo 2', 'Atributo 3', 'Atributo 4']]

# Criar e treinar o modelo K-Means
n_grupos = 3
modelo_agrupamento = KMeans(n_clusters=n_grupos, random_state=42)
dados['Grupo Previsto'] = modelo_agrupamento.fit_predict(atributos)

# Visualizar os grupos formados
print("Grupos formados:")
print(dados[['Atributo 1', 'Atributo 2', 'Atributo 3', 'Atributo 4', 'Grupo Previsto']])

# Salvar o DataFrame modificado em um novo arquivo CSV
dados.to_csv('caminho/do/seu/arquivo_modificado.csv', index=False)
