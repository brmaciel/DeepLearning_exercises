import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization


# Import base de dados de números escritos a mão
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Exibição da imagem
def show_num_img(index):
    plt.imshow(x_train[index], cmap='gray')
    plt.title('Classe ' + str(y_train[index]))
show_num_img(5)


    ###   Pre processamento dos dados   ###
prev_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
prev_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # 1: numero de canais (somente escala de cinza)

prev_train = prev_train.astype('float32')
prev_test = prev_test.astype('float32')

# Normalização dos valores para ficarem em escala 0~1, e reduzir o esforço computacional
prev_train /= 255
prev_test /= 255

# Transforma as n classes em n dimensões
clas_train = np_utils.to_categorical(y_train, 10)
clas_test = np_utils.to_categorical(y_test, 10)


    ###   Criação do modelo de Classificação   ###
classificador = Sequential()
classificador.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
classificador.add(MaxPooling2D(pool_size=(2,2)))
classificador.add(Flatten())
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dense(units=10, activation='softmax')) # output layer
classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    ###   Treino e Teste do modelo   ###
classificador.fit(prev_train, clas_train, batch_size=128, epochs=5, 
                  validation_data=(prev_test, clas_test))

    ###   Avaliação dos Resultados   ###
resultado2 = classificador.evaluate(prev_test, clas_test)





        # =====   Melhorias no classificador   ===== #
classificador2 = Sequential()
classificador2.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
classificador2.add(BatchNormalization())
    # normalização para a camada de convolução, reduzindo tempo de treinamento
classificador2.add(MaxPooling2D(pool_size=(2,2)))

classificador2.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
classificador2.add(BatchNormalization())
classificador2.add(MaxPooling2D(pool_size=(2,2)))
classificador2.add(Flatten())

classificador2.add(Dense(units=128, activation='relu'))
classificador2.add(Dropout(rate=0.2))
classificador2.add(Dense(units=128, activation='relu'))
classificador2.add(Dropout(rate=0.2))
classificador2.add(Dense(units=10, activation='softmax')) # output layer
classificador2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    ###   Treino e Teste do modelo   ###
classificador2.fit(prev_train, clas_train, batch_size=128, epochs=5, 
                  validation_data=(prev_test, clas_test))

    ###   Avaliação dos Resultados   ###
resultado = classificador2.evaluate(prev_test, clas_test)





        # =====   Classificação usando Validação Cruzada   ===== #
seed = 5
np.random.seed(seed)

    ###   Pre processamento dos dados   ###
(x_train, y_train), (x_test, y_test) = mnist.load_data()
previsores = x_train.reshape(x_train.shape[0], 28, 28, 1)
previsores = previsores.astype('float32')
previsores /= 255
classe = np_utils.to_categorical(y_train, num_classes=10)

    ###   Criação, Treino e Teste do modelo   ###
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
resultados = []

for i_train, i_test in kfold.split(previsores, np.zeros(shape=(classe.shape[0], 1))):
    classificador = Sequential()
    classificador.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
    classificador.add(MaxPooling2D(pool_size=(2,2)))
    classificador.add(Flatten())
    classificador.add(Dense(units=128, activation='relu'))
    classificador.add(Dense(units=10, activation='softmax')) # output layer
    classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    classificador.fit(previsores[i_train], classe[i_train], batch_size=128, epochs=5)
    
    precisao = classificador.evaluate(previsores[i_test], classe[i_test])
    resultados.append(precisao[1])

    ###   Avaliação dos Resultados   ###
resultados = np.array(resultados)
resultados.mean()





        # =====   Tecnica de Augumentation   ===== #
# quando se possui poucas imagens de exemplos
# faz rotações em imagens, zoom, etc, gerando novas imagens
from keras.preprocessing.image import ImageDataGenerator

    ###   Pre processamento dos dados   ###
(x_train, y_train), (x_test, y_test) = mnist.load_data()
prev_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
prev_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
prev_train = prev_train.astype('float32')
prev_test = prev_test.astype('float32')
prev_train /= 255
prev_test /= 255
clas_train = np_utils.to_categorical(y_train, 10)
clas_test = np_utils.to_categorical(y_test, 10)

    ### Cria os processos de alterações nas imagens ###
gerador_train = ImageDataGenerator(rotation_range=7, horizontal_flip=True, shear_range=0.2,
                                   height_shift_range=0.07, zoom_range=0.2)
gerador_test = ImageDataGenerator()

    ### Aplicação das alterações ###
base_train = gerador_train.flow(prev_train, clas_train, batch_size=128)
base_test = gerador_test.flow(prev_test, clas_test, batch_size=128)

    ###   Criação do modelo de Classificação   ###
classificador = Sequential()
classificador.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
classificador.add(MaxPooling2D(pool_size=(2,2)))
classificador.add(Flatten())
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dense(units=10, activation='softmax')) # output layer
classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    ###   Treino e Teste do modelo   ###
classificador.fit_generator(base_train, steps_per_epoch=60000/128, epochs=5, 
                            validation_data=base_test, validation_steps=10000/128)
