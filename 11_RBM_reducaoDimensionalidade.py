import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline


base = datasets.load_digits()

prev = np.array(base.data)
clas = base.target

    ### Pre processamento dos Dados ###
scaler = MinMaxScaler(feature_range=(0,1))
prev = scaler.fit_transform(prev)

    ### Divisão dos dados de treino e teste ###
train_test_data = train_test_split(prev, clas, test_size=0.2, random_state=0)
prev_train = train_test_data[0]
prev_test = train_test_data[1]
clas_train = train_test_data[2]
clas_test = train_test_data[3]

    ### Criação do modelo de Redução de Dimensionalidade ###
rbm = BernoulliRBM(n_components=50, n_iter=25, random_state=0)

    ### Criação do modelo de Classificação ###
naive_rbm = GaussianNB()

    ### Treino e Teste do modelo de Classificação ###
model_rbm = Pipeline(steps=[('rbm', rbm), ('naive', naive_rbm)])
 # realiza 2 processos: 1º Redução de Dimensionalidade. 2º Classificação
model_rbm.fit(prev_train, clas_train)


    ### Visualizar Imagens após Redução de Dimensionalidade   ###
# Imagens geradas no tamanho orignal a partir de dimensão reduzida
plt.figure(figsize=(9,9))
for i, comp in enumerate(rbm.components_):
    plt.subplot(9, 10, i+1)
    plt.imshow(comp.reshape(8,8), cmap=plt.cm.gray_r)
    plt.xticks(())
    plt.yticks(())
plt.show()

    ### Avaliação dos Resultados ###
previsoes_rbm = model_rbm.predict(prev_test)
score_rbm = metrics.accuracy_score(clas_test, previsoes_rbm)

# Comparação do Resultado sem uso de Redução de Dimensionalidade
naive_bayes = GaussianNB()
naive_bayes.fit(prev_train, clas_train)
previsoes_naive = naive_bayes.predict(prev_test)
score_naive = metrics.accuracy_score(clas_test, previsoes_naive)
