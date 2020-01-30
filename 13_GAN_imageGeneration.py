import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Flatten, Reshape
from keras.regularizers import L1L2
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
# biblioteca GAN: github.com/bstriner/keras-adversarial

# pip install keras==2.1.2
# pip install tensorflow==1.14
# pip install --upgrade keras
# pip install --upgrade tensorflow

(prev_train, _), (_, _) = mnist.load_data()

    ### Pre processamento dos Dados ###
prev_train = prev_train.astype('float32') / 255


    ### Criação da GAN ###
# Gerador
gerador = Sequential()
gerador.add(Dense(units=500, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5), input_dim=100))
 # L1L2: add função de penalidade na aprendizagem para evitar overfitting
gerador.add(Dense(units=500, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)))
gerador.add(Dense(units=784, activation='sigmoid', kernel_regularizer=L1L2(1e-5, 1e-5)))
gerador.add(Reshape(target_shape=(28,28)))

# Discriminador
discriminador = Sequential()
discriminador.add(InputLayer(input_shape=(28,28)))
discriminador.add(Flatten())
discriminador.add(Dense(units=500, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)))
discriminador.add(Dense(units=500, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)))
discriminador.add(Dense(units=1, activation='sigmoid', kernel_regularizer=L1L2(1e-5, 1e-5)))

# Junção do Gerador e Discriminador na GAN
gan = simple_gan(gerador, discriminador, normal_latent_sampling((100,)))
model = AdversarialModel(base_model=gan, 
                         player_params=[gerador.trainable_weights, discriminador.trainable_weights])
model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(),
                          player_optimizers=['adam', 'adam'],
                          loss='binary_crossentropy')

    ### Treino da GAN ###
model.fit(x=prev_train, y=gan_targets(60000), batch_size=256, epochs=100)


    ### Visualizar Imagens Geradas   ###
n_imgs = 10
amostras = np.random.normal(size=(n_imgs,100)) # gera dados aleatórios iniciais
previsao = gerador.predict(amostras)

for i in range(previsao.shape[0]):
    plt.imshow(previsao[i, :], cmap='gray')
    plt.show()
