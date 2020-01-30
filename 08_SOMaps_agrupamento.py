import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import pcolor, colorbar, plot



base = pd.read_csv('wines.csv')

prev = base.iloc[:, 1:].values
clas = base.iloc[:, 0].values

    ###   Pre processamento dos dados   ###
scaler = MinMaxScaler(feature_range=(0,1))
prev = scaler.fit_transform(prev)

    ### Criação do modelo de Agrupamento ###
som = MiniSom(x=8, y=8, input_len=13, sigma=1.0, learning_rate=0.5, random_seed=2)
 # 5*sqrt(nRegistros) = 66 = 8x8
 # sigma: raio de alcance do neurônio BMU
som.random_weights_init(prev)

    ### Treino do modelo de Agrupamento ###
som.train_random(data=prev, num_iteration=100)
 # num_iterarion = epochs

    ### Avaliação dos Resultados ###
weights = som._weights
BMUs = som._activation_map
nBMU = som.activation_response(prev) # quantas vezes cada neuronio foi selecionado como BMU


# Visualização gráfica
markers = ['o', 'x', '+']
colors = ['r', 'orange', 'magenta']

pcolor(som.distance_map().T)
 # calcula o MID (Mean Interneuron Distance)
 # media da distancia entre o neuronio e os neuronios a sua volta
colorbar()

for i, registro in enumerate(prev):
    w = som.winner(registro) # identifica com o neuronio classificado como BMU para cada registro
    plot(w[0]+0.5, w[1]+0.5, markers[clas[i]-1], markeredgecolor=colors[clas[i]-1],
         markerfacecolor='None', markersize=10, markeredgewidth=2)

    