import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, confusion_matrix

base = pd.read_csv('breast-cancer.csv')
prev = base.iloc[:, 0:30].values
clas = base.iloc[:, 30].values



        ### Divisão dos dados de treino e teste ###
train_test_data = train_test_split(prev, clas, test_size = 0.25)
prev_train = train_test_data[0]
prev_test = train_test_data[1]
clas_train = train_test_data[2]
clas_test = train_test_data[3]

        ### Criação do modelo de Classificação ###
model = Sequential()

# Criação das Camadas ocultas e de saída
# Dense: cada neuronio será ligado a cada um dos neuronios da camada seguinte
model.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
model.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
# input_dim: quantidade de parametros na camada de entrada
model.add(Dense(units=1, activation='sigmoid')) # criação da camada de saída

# Definição de outras parâmetros da rede neural
otimizador = keras.optimizers.Adam(lr=0.01, decay=0.0001, clipvalue=0.5)
model.compile(optimizer=otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

model.summary()

        ### Treino e Teste do modelo de Classificação ###
model.fit(prev_train, clas_train, batch_size=10, epochs=100)
previsoes = model.predict(prev_test)
previsoes = previsoes.round(0)

        ### Avaliação dos Resultados ###
score = accuracy_score(clas_test, previsoes)
matriz = confusion_matrix(clas_test, previsoes)

resultado = model.evaluate(prev_test, clas_test) # avaliação equivalente ao score

        ### Visualização dos pesos do modelo ###
# Pesos e Bias
pesos0 = model.layers[0].get_weights() # entre camada de entrada e 1ª camada oculta
pesos1 = model.layers[1].get_weights() # entre camadas ocultadas
pesos2 = model.layers[2].get_weights() # entre camada 2ª oculta e camada de saída





        # =====   Classificação usando Validação Cruzada   ===== #
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def createNeuralNetwork():
    model = Sequential()

    # Criação das Camadas ocultas e de saída
    model.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
    model.add(Dropout(0.2))
        # irá zerar 20% dos dados da camada de entrada
        # tecnica pra reduzir overfitting
    model.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    
    # Definição de outras parâmetros da rede neural
    otimizador = keras.optimizers.Adam(lr=0.01, decay=0.0001, clipvalue=0.5)
    model.compile(optimizer=otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])
    
    return model

    ### Criação do modelo de Classificação ###
model = KerasClassifier(build_fn=createNeuralNetwork, batch_size=10, epochs=100)

    ### Avaliação dos Resultados ###
resultados = cross_val_score(model, prev, y=clas, cv=10, scoring='accuracy')

media = resultados.mean()
desv_pad = resultados.std() # desvio padrão para verificar over/underfitting





        # =====   Tuning dos Parametros   ===== #
# Processo pode demorar algumas horas
from sklearn.model_selection import GridSearchCV

def createNeuralNetwork(otimizador, loss_fct, kernel_init, activation, n_neurons):
    model = Sequential()

    # Criação das Camadas ocultas e de saída
    model.add(Dense(units=n_neurons, activation=activation, kernel_initializer=kernel_init, input_dim=30))
    model.add(Dropout(0.2))
        # irá zerar 20% dos dados da camada de entrada
        # tecnica pra reduzir overfitting
    model.add(Dense(units=n_neurons, activation=activation, kernel_initializer=kernel_init))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    
    # Definição de outras parâmetros da rede neural
    model.compile(optimizer=otimizador, loss=loss_fct, metrics=['binary_accuracy'])
    
    return model

    ### Criação do modelo de Classificação ###
model = KerasClassifier(build_fn=createNeuralNetwork)

    ### Teste de diferentes parâmetros para o modelo ###
parametros = {'batch_size' : [10, 30],
              'epochs' : [50, 100],
              'otimizador' : ['adam', 'SGD'],
              'loss_fct' : ['binary_crossentropy', 'hinge'],
              'kernel_init' : ['random_uniform', 'normal'],
              'activation' : ['relu', 'tanh'],
              'n_neurons' : [16, 8]}
grid_search = GridSearchCV(estimator=model, param_grid=parametros, scoring='accuracy', cv=5)
grid_search = grid_search.fit(prev, clas)

    ### Coleta dos melhores parametros ###
best_param = grid_search.best_params_
best_score = grid_search.best_score_





        # =====   Classificação de Novo Registro   ===== #
model = Sequential()
model.add(Dense(units=8, activation='relu', kernel_initializer='normal', input_dim=30))
model.add(Dropout(0.2))
model.add(Dense(units=8, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

    ### Treino do modelo ###
model.fit(prev, clas, batch_size=10, epochs=100)

    ### Previsão de Novo Registro ###
novo_registro = np.array([[15.8, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                           0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05,
                           0.015, 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                           0.84, 158, 0.363]])
previsao = model.predict(novo_registro)





        # =====   Salvar Modelo em disco   ===== #
def save_model(model, file_name):    
    model_json = model.to_json()
    with open('{}.json'.format(file_name), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('{}.h5'.format(file_name))
    
    return model_json

estrutura_rna = save_model(model, file_name = 'model_breast')




        # =====   Carregar Modelo do disco   ===== #
from keras.models import model_from_json

def load_model(file_name):
    arquivo = open('{}.json'.format(file_name), 'r')
    estrutura_rna = arquivo.read()
    arquivo.close()

    modelo = model_from_json(estrutura_rna)
    modelo.load_weights('{}.h5'.format(file_name))
    
    return modelo

    ### Carrega Modelo do arquivo ###
modelo = load_model(file_name = 'model_breast')

    ### Previsão de Novo Registro ###
previsao = modelo.predict(novo_registro)
previsao = previsao.round(0)

    ### Avaliação na Base de Dados de Teste ###
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
resultado = modelo.evaluate(prev, clas)