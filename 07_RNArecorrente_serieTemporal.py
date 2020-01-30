import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def create_previsores(num_prev, dataset):
    previsores = []

    for i in range(num_prev, len(dataset)):
        previsores.append(dataset[i-num_prev:i, :])
    
    classe = dataset[num_prev:, 0]
    previsores = np.array(previsores)
    
    return previsores, classe


base = pd.read_csv('petr4-acoes.csv')
base.dropna(inplace=True)

nPrevisores = 90

base_for_train = base.iloc[:-22, :]
base_for_test = base.iloc[-(22+nPrevisores):, :]


        # =====   Previsão com 1 unico atributo   ===== #
base_train = base_for_train.iloc[:, 1:2].values
base_test = base_for_test.iloc[:, 1:2].values

    ### Preprocessamento dos dados ###
normalizador = MinMaxScaler(feature_range=(0,1))
base_train = normalizador.fit_transform(base_train)
base_test = normalizador.transform(base_test)

    ### Divisão dos dados de treino e teste ###
prev_train, clas_train = create_previsores(num_prev=nPrevisores, dataset=base_train)
prev_test, clas_test = create_previsores(num_prev=nPrevisores, dataset=base_test)

    ### Criação do modelo de Predição ###
modelo = Sequential()
modelo.add(LSTM(units=100, return_sequences=True, input_shape=(nPrevisores, 1)))
 # units: quantidade de celulas de memoria
 # return_sequences: true se houver mais de uma camada LSTM
modelo.add(Dropout(0.3))
modelo.add(LSTM(units=50, return_sequences=True))
modelo.add(Dropout(0.3))
modelo.add(LSTM(units=50, return_sequences=True))
modelo.add(Dropout(0.3))
modelo.add(LSTM(units=50))
modelo.add(Dropout(0.3))
modelo.add(Dense(units=1, activation='linear'))

modelo.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

    ### Treino e Teste do modelo de Classificação ###
modelo.fit(prev_train, clas_train, batch_size=32, epochs=100)
previsoes = modelo.predict(prev_test)
previsoes = normalizador.inverse_transform(previsoes)

    ### Avaliação dos Resultados ###
clas_test = clas_test.reshape(-1,1)
clas_test = normalizador.inverse_transform(clas_test)
mean_error = mean_absolute_error(clas_test, previsoes)

plt.plot(previsoes, color='red', label='Previsões')
plt.plot(clas_test, color='blue', label='Preço Real')
plt.title('Preço ABERTURA Petrobrás (PETR4)')
plt.xlabel('Tempo')
plt.ylabel('Valor')
plt.legend()
plt.show()





        # =====   Previsão com Múltiplos Atributos e 1 Output   ===== #
base_train = base_for_train.iloc[:, 1:5].values
base_test = base_for_test.iloc[:, 1:5].values

    ### Preprocessamento dos dados ###
normalizador = MinMaxScaler(feature_range=(0,1))
base_train = normalizador.fit_transform(base_train)
base_test = normalizador.transform(base_test)

    ### Divisão dos dados de treino e teste ###
prev_train, clas_train = create_previsores(num_prev=nPrevisores, dataset=base_train)
prev_test, clas_test = create_previsores(num_prev=nPrevisores, dataset=base_test)

    ### Criação do modelo de Predição ###
modelo = Sequential()
modelo.add(LSTM(units=100, return_sequences=True, input_shape=(nPrevisores, 4)))
modelo.add(Dropout(0.3))
modelo.add(LSTM(units=50, return_sequences=True))
modelo.add(Dropout(0.3))
modelo.add(LSTM(units=50, return_sequences=True))
modelo.add(Dropout(0.3))
modelo.add(LSTM(units=50))
modelo.add(Dropout(0.3))
modelo.add(Dense(units=1, activation='linear'))

modelo.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Adição de Funções de Callback
early_stop = EarlyStopping(monitor='loss', min_delta=1e-10, patience=10, verbose=1)
 # interrompe o treinamento quanto uma metrica monitorada para de melhorar
reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, factor=0.2, verbose=1)
 # reduz a taxa de aprendizado quando uma metrica para de melhorar
m_save = ModelCheckpoint(filepath='pesos.h5', monitor='loss', save_best_only=True, verbose=1)
 # salva os pesos do melhor modelo após cada epoca

    ### Treino e Teste do modelo de Classificação ###
modelo.fit(prev_train, clas_train, batch_size=32, epochs=100,
           callbacks=[early_stop, reduce_lr, m_save])

previsoes = modelo.predict(prev_test)
previsoes = inverse_transform(normalizador, previsoes)[0]


    ### Avaliação dos Resultados ###
clas_test = clas_test.reshape(-1,1)
clas_test = inverse_transform(normalizador, clas_test)
mean_error = mean_absolute_error(clas_test, previsoes)

plt.plot(previsoes, color='red', label='Previsões')
plt.plot(clas_test, color='blue', label='Preço Real')
plt.title('Preço ABERTURA Petrobrás (PETR4)')
plt.xlabel('Tempo')
plt.ylabel('Valor')
plt.legend()
plt.show()

def inverse_transform(scaler, data, dims=4):
    if dims - data.shape[1] > 0:
        a = np.zeros(shape=(len(data),1))
        data2 = np.concatenate((data, a), axis=1)
        for i in range(1, dims-data.shape[1]):
            data2 = np.concatenate((data2, a), axis=1)
    else:
        data2 = data
    data2 = scaler.inverse_transform(data2)
    
    return data2





        # =====   Previsão com Múltiplos Atributos e Múltiplas Previsões   ===== #
def create_previsores(num_prev, dataset, col_classe):
    previsores = []

    for i in range(num_prev, len(dataset)):
        previsores.append(dataset[i-num_prev:i, :])
    
    classe = dataset[num_prev:, col_classe]
    previsores = np.array(previsores)
    
    return previsores, classe

base_train = base_for_train.iloc[:, 1:5].values
base_test = base_for_test.iloc[:, 1:5].values

    ### Preprocessamento dos dados ###
normalizador = MinMaxScaler(feature_range=(0,1))
base_train = normalizador.fit_transform(base_train)
base_test = normalizador.transform(base_test)

    ### Divisão dos dados de treino e teste ###
prev_train, clas_train = create_previsores(num_prev=nPrevisores, dataset=base_train, col_classe=[1,2])
prev_test, clas_test = create_previsores(num_prev=nPrevisores, dataset=base_test, col_classe=[1,2])

    ### Criação do modelo de Predição ###
modelo = Sequential()
modelo.add(LSTM(units=100, return_sequences=True, input_shape=(nPrevisores, 4)))
modelo.add(Dropout(0.3))
modelo.add(LSTM(units=50, return_sequences=True))
modelo.add(Dropout(0.3))
modelo.add(LSTM(units=50, return_sequences=True))
modelo.add(Dropout(0.3))
modelo.add(LSTM(units=50))
modelo.add(Dropout(0.3))
modelo.add(Dense(units=2, activation='linear'))

modelo.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

    ### Treino e Teste do modelo de Classificação ###
modelo.fit(prev_train, clas_train, batch_size=32, epochs=100)

previsoes = modelo.predict(prev_test)
previsoes = inverse_transform(normalizador, previsoes)[:, 0:2]

    ### Avaliação dos Resultados ###
clas_test = inverse_transform(normalizador, clas_test)[:, 0:2]
mean_error_open = mean_absolute_error(clas_test[:, 0], previsoes[:, 0])
mean_error_maxi = mean_absolute_error(clas_test[:, 1], previsoes[:, 1])

plt.figure(figsize=(8,8))
plt.subplot(2,1,1)
plt.plot(previsoes[:, 0], color='red', label='Previsão Abertura')
plt.plot(clas_test[:, 0], color='blue', label='Preço Real Abertura')
plt.title('Preço ABERTURA Petrobrás (PETR4)')
plt.ylabel('Valor')
plt.legend()
plt.subplot(2,1,2)
plt.plot(previsoes[:, 1], color='red', label='Previsão Máxima')
plt.plot(clas_test[:, 1], color='blue', label='Preço Real Máxima')
plt.title('Preço MAXIMO Petrobrás (PETR4)')
plt.xlabel('Dias')
plt.legend()
plt.show()
