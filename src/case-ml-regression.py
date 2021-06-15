#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importando as bibliotecas para o desafio:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import register_cmap
from scipy import stats
import seaborn as sns; sns.set()
import sklearn as skl
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.model_selection import train_test_split
import seaborn
import warnings
import os
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Biblioteca para mapeamento:
get_ipython().system('pip install folium')


# In[3]:


import folium
from folium import plugins


# In[4]:


#Lendo o dataset:
#CSV file

datarj = pd.read_csv('dataset-faturamento.csv',header=(0))
print("Número de linhas e colunas:", datarj.shape)
datarj.head(10)


# In[5]:


#Descrição do dataframe:
datarj.describe()


# In[6]:


datarj["rendaMedia"]


# In[7]:


#Informações sobre o arquivo:
datarj.info()


# In[8]:


#Verificando dados faltantes:
datarj.isnull().sum().sort_values(ascending=False)


# In[9]:


#Extrair a mediana de rendaMedia
rendaMedia_mediana = datarj.rendaMedia.median()
print(rendaMedia_mediana)


# In[10]:


#preencher dados faltantes de rendaMedia com a mediana
datarj.fillna(rendaMedia_mediana, inplace = True)

#checar se há valor ausente:
datarj.rendaMedia.isnull().sum()


# In[11]:


#Verificar se há linha duplicada
datarj.duplicated()


# In[12]:


#Modificando o cabeçalho nome
datarj.rename({"nome":"bairro"}, axis="columns", inplace = True)
datarj


# In[13]:


#Excluir colunas desnecessárias à análise
datarj.drop(columns = ["codigo", "cidade", "estado"], inplace = True)
datarj


# In[14]:


#Maximos para o datarj:
datarj.max()


# In[15]:


#Definindo um estilo para os gráficos:
plt.style.use("bmh")


# In[16]:


#Verificando as distribuição dos dados:
plt.figure(1 , figsize = (15,5))
n = 0 
for x in ["população" , "faturamento" , "rendaMedia"]:
    n += 1
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace = 1 , wspace = 1)
    sns.distplot(datarj[x] , bins = 50)
    plt.title('{} '.format(x))
plt.show()


# In[17]:


#Obtendo a matriz de correlação dos dados:
corr = datarj.corr()

#Mostrar matriz:
plt.figure(figsize = (10,10))

#Mostrar a imagem:
plt.imshow(corr, cmap='Greens', interpolation='none', aspect='auto')

#Mostrar barra lateral de cores:
plt.colorbar()

#Incluir o nome das variáveis:
plt.xticks(range(len(corr)), corr.columns, rotation='vertical')
plt.yticks(range(len(corr)), corr.columns);
plt.suptitle('Correlação entre as variáveis', fontsize=15, fontweight='bold')
plt.grid(False)
plt.show()


# In[18]:


#Histograma do faturamento:

datarj.hist(column = "faturamento", bins = 20)
plt.show()


# In[19]:


#PCA DataRJ:
#Scaling DataRJ

features = ["rendaMedia","população", "popAte9", "popDe10a14", "popDe15a19", "popDe20a24", "popDe25a34", "popDe35a49", "popDe50a59", "popMaisDe60"]
x = datarj.loc[:,features].values
y = datarj.loc[:,["bairro"]].values
x = StandardScaler().fit_transform(x)


# In[20]:


#PCA DataRJ:

pca = sklearnPCA(n_components = 2)
principalComponents = pca.fit_transform(x)
principalDF = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])
print(pca.explained_variance_ratio_)
principalDF


# In[21]:


targetDF = datarj[["bairro"]]
print(targetDF)


# In[22]:


#Novo DF para bairro e suas variáveis:
pcadata = pd.concat([principalDF, targetDF], axis = 1)
pcadata


# In[23]:


#Explicação por principal componente
percent_variance = np.round(pca.explained_variance_ratio_* 100, decimals =2)
columns = ['PC1', 'PC2']
percent_variance = np.round(pca.explained_variance_ratio_* 100, decimals =2)
columns = ['PC1', 'PC2']
plt.bar(x= range(1,3), height=percent_variance, tick_label=columns)
plt.ylabel('Percentate of Variance Explained')
plt.xlabel('Principal Component')
plt.title('PCA Scree Plot')
plt.show()


# In[24]:


#Plot para averiguação
plt.scatter(principalDF.PC1, principalDF.PC2)
plt.title('PC1 contra PC2')
plt.xlabel('PC1')
plt.ylabel('PC2')


# In[25]:


# Boxplot para verificar outliers:

datarj.boxplot(column="rendaMedia")
plt.show()

datarj.boxplot(column="faturamento")
plt.show()


# ### Dando continuidade à análise exploratória, observando o comportamento das variáveis com os 20 maiores valores para faturamento:

# In[26]:


#Top 20 bairros de maior faturamento:
datarj.nlargest(20, "faturamento")[["bairro","rendaMedia","faturamento","população", "popAte9", "popDe10a14", "popDe15a19", "popDe20a24", "popDe25a34", "popDe35a49", "popDe50a59", "popMaisDe60", "domiciliosA1", "domiciliosA2", "domiciliosB1", "domiciliosB2", "domiciliosC1", "domiciliosC2", "domiciliosD", "domiciliosE"]]


# In[27]:


#Novo dataframe com bairros de maior faturamento:

values=[2915612.0, 2384494.0, 2211985.0,2157079.0, 2119774.0, 1981817.0, 1962438.0, 1775547.0, 1762798.0, 1626856.0, 1596252.0,1528242.0, 1491476.0, 1448872.0, 1430429.0, 1409134.0, 1384873.0,1330747.0, 1297388.0, 1289705.0]
maiorfatu = datarj[datarj.faturamento.isin(values)]
maiorfatu


# In[28]:


maiorfatu['bairro']


# In[29]:


#Obtendo a matriz de correlação dos dados para maiorfatu:

#Mostrar matriz:
plt.figure(figsize = (10,10))

#Mostrar a imagem:
plt.imshow(corr, cmap='Blues', interpolation='none', aspect='auto')

#Mostrar barra lateral de cores:
plt.colorbar()

#Incluir o nome das variáveis:
plt.xticks(range(len(corr)), corr.columns, rotation='vertical')
plt.yticks(range(len(corr)), corr.columns);
plt.suptitle('Correlação entre as variáveis', fontsize=15, fontweight='bold')
plt.grid(False)
plt.show()


# ### Aparentemente o comportamento de consumo entre as faixas etárias está bem correlacionado entre si e com o comportamento de consumo da população total. Enquanto que menores correlações são encontradas em domicilios A1, A2, B1. Além disso rendaMédia e faturamento são os indíces menos correlacionados com os outros. Logo, a hipótese é de que que quanto maior a rendaMédia do bairro, maior é o faturamento da empresa fictícia. Tal rendaMédia por sua vez é impulsionada pelos indíces dos domicílios A1, A2 e B1. Assim, independente de faixa etária, o público alvo da empresa parece ser compradores desses domicílios, sendo uma condição para expansão dos negócios da empresa a procura por bairros com um maior número desses domicílios. A seguir outras análises exploratórias:

# In[30]:


#Histograma dos bairros com maior faturamento:

maiorfatu.hist(column = "faturamento", bins = 5)
plt.show()


# In[31]:


#Verificando a entre as variáveis:
# Renda média vs faturamento:
values2 = ["Barra Da Tijuca", "Botafogo", "Copacabana", "Flamengo", "Freguesia (Jacarepaguá)", "Gávea", "Grajaú", "Humaitá", "Ipanema", "Jardim Botânico", "Jardim Guanabara", "Lagoa", "Laranjeiras", "Leblon", "Maracanã", "Méier", "Recreio Dos Bandeirantes", "São Conrado", "Tijuca", "Vila Isabel"]
plt.figure(1 , figsize = (10 , 7))
for bairro in (values2):
    plt.scatter(x = "rendaMedia" , y = "faturamento" , data = maiorfatu[maiorfatu["bairro"] == bairro] ,
                s = 150 , alpha = 0.5 , label = bairro)
plt.xlabel("Renda Média"), plt.ylabel("Faturamento") 
plt.title("Renda média vs Faturamento")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# In[32]:


# Boxplot para verificar outliers:

maiorfatu.boxplot(column="rendaMedia")
plt.show()

maiorfatu.boxplot(column="faturamento")
plt.show()


# ### #A princípio, decidi excluir o outlier para averiguação posterior:

# In[33]:


#exclusão do outlier:
maiorfatu = maiorfatu.drop(80)
maiorfatu


# In[34]:


# Boxplot para verificar outliers:

maiorfatu.boxplot(column="rendaMedia")
plt.show()

maiorfatu.boxplot(column="faturamento")
plt.show()


# In[35]:


#Verificando a relação entre as variáveis:
# população vs faturamento:
values3 = ["Barra Da Tijuca", "Botafogo", "Copacabana", "Flamengo", "Freguesia (Jacarepaguá)", "Gávea", "Grajaú", "Humaitá", "Ipanema", "Jardim Botânico", "Jardim Guanabara", "Laranjeiras", "Leblon", "Maracanã", "Méier", "Recreio Dos Bandeirantes", "São Conrado", "Tijuca", "Vila Isabel"]
plt.figure(1 , figsize = (10 , 7))
for bairro in (values3):
    plt.scatter(x = "população" , y = "faturamento" , data = maiorfatu[maiorfatu["bairro"] == bairro] ,
                s = 150 , alpha = 0.5, label = bairro)
plt.xlabel("População"), plt.ylabel("Faturamento") 
plt.title("População vs Faturamento")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# In[36]:


#Verificando a relação entre as variáveis:
# Faixas etárias vs faturamento:
plt.figure(1 , figsize = (10 , 7))
for bairro in (values3):
    plt.scatter(x = "popAte9" , y = "faturamento" , data = maiorfatu[maiorfatu["bairro"] == bairro] ,
                s = 150 , alpha = 0.5 , label = bairro)
plt.xlabel("População até 9 anos"), plt.ylabel("Faturamento") 
plt.title("Faixa etária (até 9 anos)  vs Faturamento")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# In[37]:


#Verificando a relação entre as variáveis:
# Faixas etárias vs faturamento:
plt.figure(1 , figsize = (10, 7))
for bairro in (values3):
    plt.scatter(x = "popDe10a14" , y = "faturamento" , data = maiorfatu[maiorfatu["bairro"] == bairro] ,
                s = 150 , alpha = 0.5 , label = bairro)
plt.xlabel("População entre 10 e 14 anos"), plt.ylabel("Faturamento") 
plt.title("Faixa etária 10-14  vs Faturamento")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# In[38]:


#Verificando a relação entre as variáveis:
# Faixas etárias vs faturamento:
plt.figure(1 , figsize = (10, 7))
for bairro in (values3):
    plt.scatter(x = "popDe15a19" , y = "faturamento" , data = maiorfatu[maiorfatu["bairro"] == bairro] ,
                s = 150 , alpha = 0.5 , label = bairro)
plt.xlabel("População entre 15 e 19 anos"), plt.ylabel("Faturamento") 
plt.title("Faixa etária 15-19  vs Faturamento")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# In[39]:


#Verificando a relação entre as variáveis:
# Faixas etárias vs faturamento:
plt.figure(1 , figsize = (10, 7))
for bairro in (values3):
    plt.scatter(x = "popDe20a24" , y = "faturamento" , data = maiorfatu[maiorfatu["bairro"] == bairro] ,
                s = 150 , alpha = 0.5 , label = bairro)
plt.xlabel("População entre 20 e 24 anos"), plt.ylabel("Faturamento") 
plt.title("Faixa etária 20-24  vs Faturamento")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# In[40]:


#Verificando a relação entre as variáveis:
# Faixas etárias vs faturamento:
plt.figure(1 , figsize = (10, 7))
for bairro in (values3):
    plt.scatter(x = "popDe25a34" , y = "faturamento" , data = maiorfatu[maiorfatu["bairro"] == bairro] ,
                s = 150 , alpha = 0.5 , label = bairro)
plt.xlabel("População entre 25 e 34 anos"), plt.ylabel("Faturamento") 
plt.title("Faixa etária 25-34  vs Faturamento")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# In[41]:


#Verificando a relação entre as variáveis:
# Faixas etárias vs faturamento:
plt.figure(1 , figsize = (10, 7))
for bairro in (values3):
    plt.scatter(x = "popDe35a49" , y = "faturamento" , data = maiorfatu[maiorfatu["bairro"] == bairro] ,
                s = 150 , alpha = 0.5 , label = bairro)
plt.xlabel("População entre 35 e 49 anos"), plt.ylabel("Faturamento") 
plt.title("Faixa etária 35-49  vs Faturamento")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# In[42]:


#Verificando se há reação linear entre as variáveis:
# Faixas etárias vs faturamento:
plt.figure(1 , figsize = (10, 7))
for bairro in (values3):
    plt.scatter(x = "domiciliosA1" , y = "faturamento" , data = maiorfatu[maiorfatu["bairro"] == bairro] ,
                s = 150 , alpha = 0.5 , label = bairro)
plt.xlabel("Domicíclios A1"), plt.ylabel("Faturamento") 
plt.title("Domicíclios vs Faturamento")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# In[43]:


#Verificando se há reação linear entre as variáveis:
# Faixas etárias vs faturamento:
plt.figure(1 , figsize = (10, 7))
for bairro in (values3):
    plt.scatter(x = "domiciliosA2" , y = "faturamento" , data = maiorfatu[maiorfatu["bairro"] == bairro] ,
                s = 150 , alpha = 0.5 , label = bairro)
plt.xlabel("Domicíclios A2"), plt.ylabel("Faturamento") 
plt.title("Domicíclios vs Faturamento")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# In[44]:


#Verificando se há reação linear entre as variáveis:
# Faixas etárias vs faturamento:
plt.figure(1 , figsize = (10, 7))
for bairro in (values3):
    plt.scatter(x = "domiciliosB1" , y = "faturamento" , data = maiorfatu[maiorfatu["bairro"] == bairro] ,
                s = 150 , alpha = 0.5 , label = bairro)
plt.xlabel("Domicíclios B1"), plt.ylabel("Faturamento") 
plt.title("Domicíclios vs Faturamento")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# In[45]:


#Verificando se há reação linear entre as variáveis:
# Faixas etárias vs faturamento:
plt.figure(1 , figsize = (10, 7))
for bairro in (values3):
    plt.scatter(x = "domiciliosB2" , y = "faturamento" , data = maiorfatu[maiorfatu["bairro"] == bairro] ,
                s = 150 , alpha = 0.5 , label = bairro)
plt.xlabel("Domicíclios B2"), plt.ylabel("Faturamento") 
plt.title("Domicíclios vs Faturamento")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# In[46]:


#Verificando se há reação linear entre as variáveis:
# Faixas etárias vs faturamento:
plt.figure(1 , figsize = (10, 7))
for bairro in (values3):
    plt.scatter(x = "domiciliosC1" , y = "faturamento" , data = maiorfatu[maiorfatu["bairro"] == bairro] ,
                s = 150 , alpha = 0.5 , label = bairro)
plt.xlabel("Domicíclios C1"), plt.ylabel("Faturamento") 
plt.title("Domicíclios vs Faturamento")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# In[47]:


#Verificando se há reação linear entre as variáveis:
# Faixas etárias vs faturamento:
plt.figure(1 , figsize = (10, 7))
for bairro in (values3):
    plt.scatter(x = "domiciliosC2" , y = "faturamento" , data = maiorfatu[maiorfatu["bairro"] == bairro] ,
                s = 150 , alpha = 0.5 , label = bairro)
plt.xlabel("Domicíclios C2"), plt.ylabel("Faturamento") 
plt.title("Domicíclios vs Faturamento")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# In[48]:


#Verificando se há reação linear entre as variáveis:
# Faixas etárias vs faturamento:
plt.figure(1 , figsize = (10, 7))
for bairro in (values3):
    plt.scatter(x = "domiciliosD" , y = "faturamento" , data = maiorfatu[maiorfatu["bairro"] == bairro] ,
                s = 150 , alpha = 0.5 , label = bairro)
plt.xlabel("Domicíclios D"), plt.ylabel("Faturamento") 
plt.title("Domicíclios vs Faturamento")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# In[49]:


#Verificando se há reação linear entre as variáveis:
# Faixas etárias vs faturamento:
plt.figure(1 , figsize = (10, 7))
for bairro in (values3):
    plt.scatter(x = "domiciliosE" , y = "faturamento" , data = maiorfatu[maiorfatu["bairro"] == bairro] ,
                s = 150 , alpha = 0.5 , label = bairro)
plt.xlabel("Domicíclios E"), plt.ylabel("Faturamento") 
plt.title("Domicíclios vs Faturamento")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# ### Espacialização dos dados para auxiliar a análise exploratória:

# In[50]:


#Para espacialização dos dados, primeiro resetei o index do df maiorfatu:
maiorfatu = maiorfatu.reset_index()
maiorfatu


# In[51]:


#Inseri um .csv de coordenadas lat x long dos bairros de maior faturamento para espacialização:
geoloc = pd.read_csv("geolocation.csv")
geoloc


# In[52]:


geoloc.info()


# In[53]:


#Concatenei os dataframes para espacialização:
FatuMap = pd.concat([geoloc,maiorfatu], axis = 1)
FatuMap


# In[54]:


#Espacialização através do folium:
coordenadas = FatuMap[['lat', 'long', 'faturamento']]
geolocMap = folium.Map(width = "100%", height = "100%",
                       location=[-22.908333, -43.196388])
geolocMap.add_child(plugins.HeatMap(coordenadas))
geolocMap.save("mapa-faturamento.html")
geolocMap


# In[55]:


coordenadas = FatuMap[['lat', 'long', 'rendaMedia']]
geolocMap_renda = folium.Map(width = "100%", height = "100%",
                       location=[-22.908333, -43.196388])
geolocMap_renda.add_child(plugins.HeatMap(coordenadas))
geolocMap_renda.save("mapa-renda.html")
geolocMap_renda


# In[56]:


coordenadas = FatuMap[['lat', 'long', 'domiciliosA1']]
geolocMap_AI = folium.Map(width = "100%", height = "100%",
                       location=[-22.908333, -43.196388])
geolocMap_AI.add_child(plugins.HeatMap(coordenadas))
geolocMap_AI.save("mapa-AI.html")
geolocMap_AI


# In[57]:


coordenadas = FatuMap[['lat', 'long', 'domiciliosA2']]
geolocMap_AII = folium.Map(width = "100%", height = "100%",
                       location=[-22.908333, -43.196388])
geolocMap_AII.add_child(plugins.HeatMap(coordenadas))
geolocMap_AII.save("mapa-AII.html")
geolocMap_AII


# In[58]:


coordenadas = FatuMap[['lat', 'long', 'domiciliosB1']]
geolocMap_BI = folium.Map(width = "100%", height = "100%",
                       location=[-22.908333, -43.196388])
geolocMap_BI.add_child(plugins.HeatMap(coordenadas))
geolocMap_BI.save("mapa-BI.html")
geolocMap_BI


# In[59]:


coordenadas = FatuMap[['lat', 'long', 'população']]
geolocMap_pop = folium.Map(width = "100%", height = "100%",
                       location=[-22.908333, -43.196388])
geolocMap_pop.add_child(plugins.HeatMap(coordenadas))
geolocMap_pop.save("mapa-pop.html")
geolocMap_pop


# ### Modelo regressão para tomada de decisão sobre expansão dos negócios da empresa fictícia:

# In[60]:


#Modelo ML
#Regressão linear com datarj

X = datarj[['rendaMedia','população']]
X


# In[61]:


#Separando conjuntos de treino e teste para o modelo, com 30% de teste

x_train, x_test, y_train, y_test = train_test_split(X,datarj.faturamento,test_size = 0.3)

#Verificando o shape dos dados de treino:
x_train.shape, y_train.shape


# In[62]:


#Verificando o shape dos dados de teste:
x_test.shape, y_test.shape


# In[63]:


#Instânciando a regressão linear:
from sklearn.linear_model import LinearRegression
lreg = LinearRegression()


# In[64]:


#Treinando o modelo:

lreg.fit(x_train, y_train)


# In[65]:


#Predizendo valores para o conjunto test:
pred = lreg.predict(x_test)


# In[66]:


#Calculando MSE:
mse = np.mean((pred - y_test)**2)
mse


# In[67]:


#Calculo de coeficientes:
coeff = pd.DataFrame(x_train.columns)
coeff['Coeficientes'] = pd.Series(lreg.coef_)
coeff


# In[68]:


#Calculo de r-squared
lreg.score(x_test,y_test)


# In[77]:


#Modelo ML
#Regressão linear com datarj

X = datarj[['rendaMedia','população', 'domiciliosA1','domiciliosA2','domiciliosB1']]
X


# In[78]:


#Separando conjuntos de treino e teste para o modelo, com 30% de teste

x_train, x_test, y_train, y_test = train_test_split(X,datarj.faturamento,test_size = 0.3)

#Verificando o shape dos dados de treino:
x_train.shape, y_train.shape


# In[79]:


#Verificando o shape dos dados de teste:
x_test.shape, y_test.shape


# In[80]:


#Treinando o modelo:

lreg.fit(x_train, y_train)


# In[81]:


#Predizendo valores para o conjunto test:
pred = lreg.predict(x_test)


# In[82]:


#Calculando MSE:
mse = np.mean((pred - y_test)**2)
mse


# In[83]:


#Calculo de coeficientes:
coeff = pd.DataFrame(x_train.columns)
coeff['Coeficientes'] = pd.Series(lreg.coef_)
coeff


# In[84]:


#Calculo de r-squared
lreg.score(x_test,y_test)


# In[85]:


#Regularização
#Verificando a magnitude dos coeficientes:

predictors = x_train.columns
coef = pd.Series(lreg.coef_,predictors).sort_values()
coef.plot(kind='bar')


# In[86]:


#Treinando o modelo com Ridge from sklearn
#from sklearn.linear_model import Ridge
#ridgeReg = Ridge(alpha=0.05, normalize=True)
#ridgeReg.fit(x_train,y_train)


# In[87]:


#Predição e calculo de mse
#pred = ridgeReg.predict(x_test)
#mse = np.mean((pred-y_test)**2)
#mse


# In[88]:


#ridgeReg.score(x_test,y_test)


# ### Uma vez que atraves de Ridge o poder de explicação do modelo diminuiu um pouco, opto por utilizar o modelo linear.

# In[89]:


predicoes = pd.DataFrame(pred[:100])
y_teste2 = pd.DataFrame(y_test.values[:100])

plt.style.use('ggplot')
plt.figure(figsize = (12,8))
plt.xlabel("Variáveis")
plt.ylabel("Faturamento")
plt.title("Valores reais vs Valores preditos")

plt.plot(predicoes)
plt.plot(y_teste2)

plt.legend(['Predições','Valores reais'])
plt.show()


# ### Quanto à hipótese inicial de que quanto maior a rendaMédia do bairro, maior é o faturamento da empresa fictícia, não necessáriamente os domícilios de A1 impulsionam o faturamento da empresa fictícia, mesmo que o bairro Barra da Tijuca possua um número expressivo de domícilios A1, podendo ser caracterizado como um hotspot de domícilios A1, de acordo com a regressão linear, a variável que mais explica o faturamento é Domicílios B1. Esse resultado é corroborado quando compara-se os mapas de faturamento e dominílios.

# In[ ]:




