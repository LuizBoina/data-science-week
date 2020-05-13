#!/usr/bin/env python
# coding: utf-8

# ## **Semana de Data Science**
# 
# - Minerando Dados

# ## Aula 01

# ### Conhecendo a base de dados

# Monta o drive

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# Importando as bibliotecas básicas

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# Carregando a Base de Dados

# In[ ]:


# carrega o dataset de london
from sklearn.datasets import load_boston
boston = load_boston()


# In[ ]:


# descrição do dataset
print (boston.DESCR)


# In[ ]:


# cria um dataframe pandas
data = pd.DataFrame(boston.data, columns=boston.feature_names)


# In[ ]:


# imprime as 5 primeiras linhas do dataset
data.head()


# Conhecendo as colunas da base de dados

# **`CRIM`**: Taxa de criminalidade per capita por cidade.
# 
# **`ZN`**: Proporção de terrenos residenciais divididos por lotes com mais de 25.000 pés quadrados.
# 
# **`INDUS`**: Essa é a proporção de hectares de negócios não comerciais por cidade.
# 
# **`CHAS`**: variável fictícia Charles River (= 1 se o trecho limita o rio; 0 caso contrário)
# 
# **`NOX`**: concentração de óxido nítrico (partes por 10 milhões)
# 
# **`RM`**: Número médio de quartos entre as casas do bairro
# 
# **`IDADE`**: proporção de unidades ocupadas pelos proprietários construídas antes de 1940
# 
# **`DIS`**: distâncias ponderadas para cinco centros de emprego em Boston
# 
# **`RAD`**: Índice de acessibilidade às rodovias radiais
# 
# **`IMPOSTO`**: taxa do imposto sobre a propriedade de valor total por US $ 10.000
# 
# **`B`**: 1000 (Bk - 0,63) ², onde Bk é a proporção de pessoas de descendência afro-americana por cidade
# 
# **`PTRATIO`**: Bairros com maior proporção de alunos para professores (maior valor de 'PTRATIO')
# 
# **`LSTAT`**: porcentagem de status mais baixo da população
# 
# **`MEDV`**: valor médio de casas ocupadas pelos proprietários em US $ 1000

# Adicionando a coluna que será nossa variável alvo

# In[ ]:


# adiciona a variável MEDV
data['MEDV'] = boston.target


# In[ ]:


# imprime as 5 primeiras linhas do dataframe
data.head()


# In[ ]:


data.describe()


# ### Análise e Exploração dos Dados
# 
# 

# Nesta etapa nosso objetivo é conhecer os dados que estamos trabalhando.
# 
# Podemos a ferramenta **Pandas Profiling** para essa etapa:

# In[ ]:


# Instalando o pandas profiling
pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip


# In[ ]:


# import o ProfileReport
from pandas_profiling import ProfileReport


# In[ ]:


# executando o profile
profile = ProfileReport(data, title='Relatório - Pandas Profiling', html={'style':{'full_width':True}})


# In[ ]:


profile


# In[ ]:


# salvando o relatório no disco
profile.to_file(output_file="Relatorio01.html")


# **Observações**
# 
# *   *O coeficiente de correlação varia de `-1` a `1`. 
# Se valor é próximo de 1, isto significa que existe uma forte correlação positiva entre as variáveis. Quando esse número é próximo de -1, as variáveis tem uma forte correlação negativa.*
# 
# *   *A relatório que executamos acima nos mostra que a nossa variável alvo (**MEDV**) é fortemente correlacionada com as variáveis `LSTAT` e `RM`*
# 
# *   *`RAD` e `TAX` são fortemente correlacionadas, podemos remove-las do nosso modelo para evitar a multi-colinearidade.*
# 
# *   *O mesmo acontece com as colunas `DIS` and `AGE` a qual tem a correlação de -0.75*
# 
# *   *A coluna `ZN` possui 73% de valores zero.*

# ## Aula 02

# Obtendo informações da base de dados manualmente

# In[ ]:


# Check missing values
data.isnull().sum()


# In[ ]:


# um pouco de estatística descritiva
data.describe()


# Analisando a Correlação das colunas da base de dados

# In[ ]:


# Calcule a correlação  
correlacoes = data.corr()


# In[ ]:


# Usando o método heatmap do seaborn
plt.figure(figsize=(16, 6))
sns.heatmap(data=correlacoes, annot=True)


# Visualizando a relação entre algumas features e variável alvo

# In[ ]:


# Importando o Plot.ly
import plotly.express as px


# In[ ]:


# RM vs MEDV (Número de quartos e valor médio do imóvel)
fig = px.scatter(data, x=data.RM, y=data.MEDV)
fig.show()


# In[ ]:


# LSTAT vs MEDV (índice de status mais baixo da população e preço do imóvel)
fig = px.scatter(data, x=data.LSTAT, y=data.MEDV)
fig.show()


# In[ ]:


# PTRATIO vs MEDV (percentual de proporção de alunos para professores)
fig = px.scatter(data, x=data.PTRATIO, y=data.MEDV)
fig.show()


# #### Analisando Outliers

# In[ ]:


# estatística descritiva da variável RM
data.RM.describe()


# In[ ]:


# visualizando a distribuição da variável RM
import plotly.figure_factory as ff
labels = ['Distribuição da variável RM (número de quartos)']
fig = ff.create_distplot([data.RM], labels, bin_size=.2)
fig.show()


# In[ ]:


# Visualizando outliers na variável RM
import plotly.express as px

fig = px.box(data, y='RM')
fig.update_layout( width=800,height=800)
fig.show()


# Visualizando a distribuição da variável MEDV

# In[ ]:


# estatística descritiva da variável MEDV
data.MEDV.describe()


# In[ ]:


# visualizando a distribuição da variável MEDV
import plotly.figure_factory as ff
labels = ['Distribuição da variável MEDV (preço médio do imóvel)']
fig = ff.create_distplot([data.MEDV], labels, bin_size=.2)
fig.show()


# Analisando a simetria do dado

# In[ ]:


# carrega o método stats da scipy
from scipy import stats


# In[ ]:


# imprime o coeficiente de pearson
stats.skew(data.MEDV)


# Coeficiente de Pearson
# *   Valor entre -1 e 1 - distribuição simétrica.
# *   Valor maior que 1 - distribuição assimétrica positiva.
# *   Valor maior que -1 - distribuição assimétrica negativa.

# In[ ]:


# Histogram da variável MEDV (variável alvo)
fig = px.histogram(data, x="MEDV", nbins=50, opacity=0.50)
fig.show()


# In[ ]:


# Visualizando outliers na variável MEDV
import plotly.express as px

fig = px.box(data, y='MEDV')
fig.update_layout( width=800,height=800)
fig.show()


# In[ ]:


# estatistica descritiva das variáveis
data[['PTRATIO','LSTAT','RM']].describe()


# In[ ]:


# imprimindo os 16 maiores valores de MEDV
data[['RM','LSTAT','PTRATIO','MEDV']].nlargest(16, 'MEDV')


# In[ ]:


# filtra os top 16 maiores registro da coluna MEDV
top16 = data.nlargest(16, 'MEDV').index


# In[ ]:


# remove os valores listados em top16
data.drop(top16, inplace=True)


# In[ ]:


# visualizando a distribuição da variável MEDV
import plotly.figure_factory as ff
labels = ['Distribuição da variável MEDV (número de quartos)']
fig = ff.create_distplot([data.MEDV], labels, bin_size=.2)
fig.show()


# In[ ]:


# Histogram da variável MEDV (variável alvo)
fig = px.histogram(data, x="MEDV", nbins=50, opacity=0.50)
fig.show()


# In[ ]:


# imprime o coeficiente de pearson
stats.skew(data.MEDV)


# **Definindo um Baseline**
# 
# - `Uma baseline é importante para ter marcos no projeto`.
# - `Permite uma explicação fácil para todos os envolvidos`.
# - `É algo que sempre tentaremos ganhar na medida do possível`.

# In[ ]:


# converte os dados
data.RM = data.RM.astype(int)


# In[ ]:


data.info()


# In[ ]:


# estatística descritiva da coluna numero de quartos
data.RM.describe()


# In[ ]:


# definindo a regra para categorizar os dados
categorias = []


# In[ ]:


# alimenta a lista categorias
for i in data.RM.iteritems():
  valor = (i[1])
  if valor <= 4:
    categorias.append('Pequeno')
  elif valor < 7:
    categorias.append('Medio')
  else:
    categorias.append('Grande')


# In[ ]:


# cria a coluna categorias
data['categorias'] = categorias


# In[ ]:


# imprime a contagem de categorias
data.categorias.value_counts()


# In[ ]:


# agrupa as categorias e calcula as médias
medias_categorias = data.groupby(by='categorias')['MEDV'].mean()


# In[ ]:


# visualizando a variável medias_categorias
medias_categorias


# In[ ]:


# criando o dicionario com chaves medio, grande e pequeno e seus valores
dic_baseline = {'Grande': medias_categorias[0], 'Medio': medias_categorias[1], 'Pequeno': medias_categorias[2]}


# In[ ]:


# imprime dicionario
dic_baseline


# In[ ]:


# cria a função retorna baseline
def retorna_baseline(num_quartos):
  if num_quartos <= 4:
    return dic_baseline.get('Pequeno')
  elif num_quartos < 7:
    return dic_baseline.get('Medio')
  else:
    return dic_baseline.get('Grande')


# In[ ]:


# chama a função retorna baseline
retorna_baseline(3)


# In[ ]:


for i in data.RM.iteritems():
  n_quartos = i[1]
  print('Número de quartos é: {} , Valor médio: {}'.format(n_quartos,retorna_baseline(n_quartos)))


# In[ ]:


# imprime as 5 primeiras linhas do dataframe
data.head()

