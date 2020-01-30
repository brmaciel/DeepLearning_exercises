import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import pcolor, colorbar, plot

base = pd.read_csv('credit-data.csv')

    ###   Pre processamento dos dados   ###
base.dropna(inplace=True)
base.loc[base.age < 0, 'age'] = base['age'].loc[base.age > 0].mean()

prev = base.iloc[:, 0:4].values
clas = base.iloc[:, 4].values

scaler = MinMaxScaler(feature_range=(0,1))
prev = scaler.fit_transform(prev)

    ### Criação do modelo de Agrupamento ###
som = MiniSom(x=15, y=15, input_len=(4), sigma=1.0, random_seed=0)
som.random_weights_init(prev)

    ### Treino do modelo de Agrupamento ###
som.train_random(data=prev, num_iteration=100)

    ### Avaliação dos Resultados ###
markers = ['o', 'x', '+']
colors = ['orange', 'magenta', 'r']

pcolor(som.distance_map().T)
colorbar()

for i, registro in enumerate(prev):
    w = som.winner(registro)
    plot(w[0]+0.5, w[1]+0.5, markers[clas[i]], markeredgecolor=colors[clas[i]],
         markerfacecolor='None', markersize=8, markeredgewidth=2)


    ### Identificação dos Outliers ###
mapeamento = som.win_map(prev)
posicao_outliers = [(4,5), (6,13), (7,6)]
outliers = np.zeros(shape=(1,4))
for pos in posicao_outliers:
    outliers = np.concatenate((outliers, mapeamento[pos]), axis=0)
outliers = np.delete(outliers, obj=0, axis=0)

# Visualizaão Gráfica
pcolor(som.distance_map().T)
colorbar()
for i, registro in enumerate(outliers):
    w = som.winner(registro)
    plot(w[0]+0.5, w[1]+0.5, markers[clas[i]], markeredgecolor=colors[clas[i]],
         markerfacecolor='None', markersize=8, markeredgewidth=2)


# Identificação se esses registros tiveram credito aprovado
outliers = scaler.inverse_transform(outliers)

classe_outliers = []
for i in range(len(base)):
    for j in range(len(outliers)):
        if base.iloc[i, 0] == int(round(outliers[j, 0])):
            classe_outliers.append(base.iloc[i, 4])
classe_outliers = np.array(classe)

outliers_final = np.column_stack((outliers, classe_outliers))
outliers_final = outliers_final[outliers_final[:, 4].argsort()]
