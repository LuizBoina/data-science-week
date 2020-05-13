#!/usr/bin/env python
# coding: utf-8

# ## **Semana de Data Science**
# 
# - Minerando Dados

# ### Conhecendo a base de dados

# Importando as bibliotecas básicas

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# Carregando a Base de Dados

# In[ ]:


# carrega o dataset de boston
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


# In[ ]:


# escreve o arquivo para o disco
data.to_csv('data.csv')


# Conhecendo as colunas da base de dados

# **`CRIM`**: Taxa de criminalidade per capita por regiao.
# 
# **`ZN`**: Proporção de terrenos residenciais divididos por lotes com mais de 25.000 pés quadrados.
# 
# **`INDUS`**: Essa é a proporção de hectares de negócios não comerciais por regiao.
# 
# **`CHAS`**: variável fictícia Charles River (= 1 se o trecho limita o rio; 0 caso contrário)
# 
# **`NOX`**: concentração de óxido nítrico (partes por 10 milhões)
# 
# **`RM`**: Número médio de quartos entre as casas do bairro
# 
# **`Age`**: proporção de unidades ocupadas pelos proprietários construídas antes de 1940
# 
# **`DIS`**: distâncias ponderadas para cinco centros de emprego em Boston
# 
# **`RAD`**: Índice de acessibilidade às rodovias radiais
# 
# **`IMPOSTO`**: taxa do imposto sobre a propriedade de valor total por US $ 10.000
# 
# **`B`**: 1000 (Bk - 0,63) ², onde Bk é a proporção de pessoas de descendência afro-americana por regiao
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


# ## Análise e Exploração dos Dados



# In[ ]:


# import o ProfileReport
from pandas_profiling import ProfileReport


# In[ ]:


# executando o profile
profile = ProfileReport(data, title='Relatório - Pandas Profiling', html={'style':{'full_width':True}})


# In[ ]:


profile


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

# In[ ]:


# salvando o relatório no disco
profile.to_file(output_file="Relatorio01.html")

