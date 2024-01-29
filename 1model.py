# Importar bibliotecas
import pandas as pd
from sklearn.cluster import KMeans

# Carregar o arquivo CSV em um DataFrame
caminho_arquivo = 'caminho/do/seu/arquivo.csv'
dados = pd.read_csv(caminho_arquivo)

# Visualizar as primeiras linhas do DataFrame
print("Visualização inicial dos dados:")
print(dados.head())

# Selecionar os atributos numéricos
atributos = dados[['Atributo 1', 'Atributo 2', 'Atributo 3', 'Atributo 4']]

# Realizar normalização (se necessário)
# Exemplo: atributos_normalizados = (atributos - atributos.mean()) / atributos.std()

# Utilizar o algoritmo de agrupamento (K-Means)
n_grupos = 3
modelo_agrupamento = KMeans(n_clusters=n_grupos, random_state=42)
dados['Grupo'] = modelo_agrupamento.fit_predict(atributos)

# Visualizar os grupos formados
print("\nGrupos formados:")
print(dados[['Atributo 1', 'Atributo 2', 'Atributo 3', 'Atributo 4', 'Grupo']])

# Salvar o DataFrame modificado em um novo arquivo CSV
dados.to_csv('caminho/do/seu/arquivo_modificado.csv', index=False)

# Realizar previsões para os registros da tabela de exemplo
registros_teste = pd.DataFrame({
    'Atributo 1': [300, 360, 410, 440, 340],
    'Atributo 2': [220, 170, 200, 140, 200],
    'Atributo 3': [100, 265, 370, 350, 150],
    'Atributo 4': [20, 80, 130, 50, 50]
})

previsoes = modelo_agrupamento.predict(registros_teste)

# Adicionar as previsões ao DataFrame de teste
registros_teste['Grupo Previsto'] = previsoes

# Visualizar os resultados da predição
print("\nResultados da Predição:")
print(registros_teste)