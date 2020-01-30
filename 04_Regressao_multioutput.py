import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input
from sklearn.metrics import mean_absolute_error


        # =====   Tratamento dos dados   ===== #
base = pd.read_csv('games-sales.csv')

base.drop(['Other_Sales', 'Global_Sales', 'Developer'], axis=1, inplace=True)
base.dropna(axis=0, inplace=True) # exclui valores faltantes

base = base.loc[base['NA_Sales'] >= 1]
base = base.loc[base['EU_Sales'] >= 1]

var_names = base['Name'].value_counts() # variabilidade muito grande de nomes
base.drop(['Name'], axis=1, inplace=True)

base['User_Score'] = base['User_Score'].astype('float')

previsores = base.iloc[:, [0,1,2,3,7,8,9,10,11]].values
sales_na = base.iloc[:, 4].values
sales_eu = base.iloc[:, 5].values
sales_jp = base.iloc[:, 6].values


    ###   Pre processamento dos dados   ###
encoder = LabelEncoder()
cols = [0, 2, 3, 8]
for col in cols:
    previsores[:, col] = encoder.fit_transform(previsores[:, col])

onehotencoder = OneHotEncoder(categorical_features=cols)
previsores = onehotencoder.fit_transform(previsores).toarray()
previsores = StandardScaler().fit_transform(previsores)

    ###   Criação do modelo de Regressão   ###
input_layer = Input(shape=(61, ))
hidden_layer_1 = Dense(units = 32, activation='sigmoid')(input_layer) # informa qual a camada anterior
hidden_layer_2 = Dense(units = 32, activation='sigmoid')(hidden_layer_1)
output_layer_1 = Dense(units=1, activation='linear')(hidden_layer_2)
output_layer_2 = Dense(units=1, activation='linear')(hidden_layer_2)
output_layer_3 = Dense(units=1, activation='linear')(hidden_layer_2)

regressor = Model(inputs=input_layer, outputs=[output_layer_1, output_layer_2, output_layer_3])
regressor.compile(optimizer='adam', loss='mse')

    ###   Treino e Teste do modelo de Regressão   ###
regressor.fit(previsores, [sales_na, sales_eu, sales_jp], batch_size = 10, epochs=500)
previsao_na, previsao_eu, previsao_jp = regressor.predict(previsores)

    ### Avaliação dos Resultados ###
mean_error_na = mean_absolute_error(sales_na, previsao_na)
mean_error_eu = mean_absolute_error(sales_eu, previsao_eu)
mean_error_ja = mean_absolute_error(sales_jp, previsao_jp)
