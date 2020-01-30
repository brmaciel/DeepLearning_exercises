import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

base = pd.read_csv('iris.csv')
prev = base.iloc[:, :4].values
clas = base['class'].values
clas = LabelEncoder().fit_transform(clas)
clas_dummy = np_utils.to_categorical(clas)
# existem n classes, então o vetor classe terá n dimnesões
# rede neural tera n neuronios na camada de saída



    ###   Divisão dos dados de treino e teste   ###
train_test_data = train_test_split(prev, clas_dummy, test_size=0.25)
prev_train = train_test_data[0]
prev_test = train_test_data[1]
clas_train = train_test_data[2]
clas_test = train_test_data[3]

    ###   Criação do modelo de Classificação   ###
model = Sequential()
model.add(Dense(units=4, activation='relu', input_dim=4))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(units=3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    ###   Treino e Teste do modelo   ###
model.fit(prev_train, clas_train, batch_size = 10, epochs=1000)
previsoes_proba = model.predict(prev_test)
previsoes = previsoes_proba.round()
previsoes = [np.argmax(x) for x in previsoes] # retorna previsão p/ vetor 1D

    ###   Avaliação dos Resultados   ###
resultado = model.evaluate(prev_test, clas_test)
matriz = confusion_matrix([np.argmax(x) for x in clas_test], previsoes)





        # =====   Classificação usando Validação Cruzada   ===== #
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def createNeuralNetwork():
    model = Sequential()
    model.add(Dense(units=4, activation='relu', input_dim=4))
    model.add(Dense(units=4, activation='relu'))
    model.add(Dense(units=3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    return model

    ### Criação do modelo de Classificação ###
model = KerasClassifier(build_fn=createNeuralNetwork, batch_size = 10, epochs=1000)

    ### Avaliação dos Resultados ###
resultado = cross_val_score(model, prev, clas, cv=10, scoring='accuracy')

media = resultado.mean()
desv_pad = resultado.std()