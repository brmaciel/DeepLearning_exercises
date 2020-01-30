import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_absolute_error


        # =====   Tratamento dos dados   ===== #
base = pd.read_csv('autos.csv', encoding='ISO-8859-1')

# Excluir variaveis que não acrescentam valor
delete_col = ['dateCrawled', 'monthOfRegistration', 'dateCreated', 'nrOfPictures', 'lastSeen']
base.drop(delete_col, axis=1, inplace=True)

# Excluir variaveis com grande desbalanceamento entre seus valores
car_names = base['name'].value_counts()
base.drop('name', axis=1, inplace=True) # nomes muito poluidos
base.drop('seller', axis=1, inplace=True) # maioria dos registros é vendedor privado
base.drop('offerType', axis=1, inplace=True)

# Tratamento de sub/super valores na classe
plt.hist(base.price, bins = 10, range=(base.price.min(), 500))
low_prices = base.loc[base.price < 200]
base.drop(low_prices.index, axis=0, inplace=True)

plt.hist(base.price, bins=10, range=(350000, base.price.max()))
super_prices = base.loc[base.price > 350000]
base.drop(super_prices.index, axis=0, inplace=True)


# Tratamento de valores faltantes
# verifica qual valor mais frequente de vehicleType para um determinado modelo
null_vehicleType = base.loc[pd.isnull(base['vehicleType'])]
models = list(null_vehicleType.model.value_counts().index)
dict_models = {}
for model in models:
    base2 = base.loc[base.model == model]
    model_type = base2.vehicleType.value_counts().index[0]
    dict_models[model] = model_type

# faz a substituição de valores nulos de vehicleType
for i in null_vehicleType.index:
    if not base['model'][i] is np.nan:
        base['vehicleType'][i] = dict_models[base['model'][i]]
        
# as demais colunas recebem o valor mais frequente nos valores faltantes
valores = {}
for col in base.columns:
    if np.nan in list(base[col].unique()):
        unique_values = base[col].value_counts()
        print(unique_values, '\n')
        valores[col] = unique_values.index[0]
base.fillna(value = valores, inplace=True)


    ###   Pre processamento dos dados   ###
prev = base.iloc[:, 1:].values
clas = base['price'].values

colunas = [0,1,3,5,7,8,9]
for col in colunas:
    prev[:,col] = LabelEncoder().fit_transform(prev[:, col])
    
onehotencoder = OneHotEncoder(categorical_features=[0, 1, 5, 7, 8])
prev = onehotencoder.fit_transform(prev).toarray()

scaler = StandardScaler()
prev = scaler.fit_transform(prev)



    ###   Divisão dos dados de treino e teste   ###
train_test_data = train_test_split(prev, clas, test_size=0.25)
prev_train = train_test_data[0]
prev_test = train_test_data[1]
clas_train = train_test_data[2]
clas_test = train_test_data[3]

    ###   Criação do modelo de Regressão   ###
model = Sequential()
model.add(Dense(units=157, activation='relu', input_dim=314))
model.add(Dense(units=157, activation='relu'))
model.add(Dense(units=1, activation='linear'))
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])

    ###   Treino e Teste do modelo de Regressão   ###
model.fit(prev_train, clas_train, batch_size=100, epochs=10)
previsoes = model.predict(prev_test)

    ###   Avaliação dos Resultados   ###
mean_error = mean_absolute_error(clas_test, previsoes)





        # =====   Regressão usando Validação Cruzada   ===== #
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor

def createNeuralNetwork():
    model = Sequential()
    model.add(Dense(units=157, activation='relu', input_dim=314))
    model.add(Dense(units=157, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
    
    return model

    ### Criação do modelo de Regressão ###
model = KerasRegressor(build_fn=createNeuralNetwork, batch_size = 100, epochs=5)

    ### Avaliação dos Resultados ###
resultado = cross_val_score(model, prev, clas, cv=10, scoring='neg_mean_absolute_error')

media = resultado.mean()
std_dev = resultado.std()
