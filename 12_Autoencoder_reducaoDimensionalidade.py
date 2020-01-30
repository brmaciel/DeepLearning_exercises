import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense


(prev_train, clas_train), (prev_test, clas_test) = mnist.load_data()

    ### Pre processamento dos Dados ###
prev_train = prev_train.astype('float32') / 255
prev_test = prev_test.astype('float32') / 255

prev_train = prev_train.reshape(prev_train.shape[0], np.prod(prev_train.shape[1:]))
prev_test = prev_test.reshape(prev_test.shape[0], np.prod(prev_test.shape[1:]))


    ### Criação do modelo de Redução de Dimensionalidade ###
model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=784))
model.add(Dense(units=784, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

    ### Treino e Teste do modelo de Redução de Dimensionalidade ###
model.fit(prev_train, prev_train, batch_size=256, epochs=50, 
          validation_data=(prev_test, prev_test))

    ### Construção do Modelo de Codificação ###
dim_original = Input(shape=(784,))
camada_encoder = model.layers[0]
encoder = Model(inputs=dim_original, outputs=camada_encoder(dim_original))
encoder.summary()


    ### Visualizar Imagens Codificadas e Decodificadas   ###
# Cria as imagens codificadas e decodificadas
imgs_codificadas = encoder.predict(prev_test)
imgs_decodificadas = model.predict(prev_test)

# Visualiza as imagens codificadas e decodificadas
n_imgs = 10
imgs_test = np.random.randint(prev_test.shape[0], size=n_imgs)

plt.figure(figsize=(10,8))
for i, indice_img in enumerate(imgs_test):
    # imagem original
    plt.subplot(10,10,i+1)
    plt.imshow(prev_test[indice_img].reshape(28,28))
    plt.xticks(())
    plt.yticks(())
    
    # imagem codificada
    plt.subplot(10,10,i+1 +10)
    plt.imshow(imgs_codificadas[indice_img].reshape(8,4))
    plt.xticks(())
    plt.yticks(())
    
    # imagem decodificada
    plt.subplot(10,10,i+1 +20)
    plt.imshow(imgs_decodificadas[indice_img].reshape(28,28))
    plt.xticks(())
    plt.yticks(())





        # =====   Classificação com/sem Redução de Dimensionalidade   ===== #
from keras.utils import np_utils

clas_train= np_utils.to_categorical(clas_train)
clas_test = np_utils.to_categorical(clas_test) 

# Codificação dos atributos previsores
prev_train_coded = encoder.predict(prev_train)
prev_test_coded = encoder.predict(prev_test)

# Sem Redução de Dimensionalidade
model_semRed = Sequential()
model_semRed.add(Dense(units=397, activation='relu', input_dim=784))
model_semRed.add(Dense(units=397, activation='relu'))
model_semRed.add(Dense(units=10, activation='softmax'))
model_semRed.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_semRed.fit(prev_train, clas_train, batch_size=256, epochs=100,
                 validation_data = (prev_test, clas_test))
# score: 0.9852

# Com Redução de Dimensionalidade
model_comRed = Sequential()
model_comRed.add(Dense(units=21, activation='relu', input_dim=32))
model_comRed.add(Dense(units=21, activation='relu'))
model_comRed.add(Dense(units=10, activation='softmax'))
model_comRed.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_comRed.fit(prev_train_coded, clas_train, batch_size=256, epochs=100,
                 validation_data = (prev_test_coded, clas_test))
# score: 0.9559





        # =====   Redução de Dimensionalidade com Deep Autoencoder   ===== #
    ### Criação do modelo de Redução de Dimensionalidade ###
autoencoder = Sequential()
# Encoder
autoencoder.add(Dense(units=128, activation='relu', input_dim=784))
autoencoder.add(Dense(units=64, activation='relu'))
autoencoder.add(Dense(units=32, activation='relu'))
# Decoder
autoencoder.add(Dense(units=64, activation='relu'))
autoencoder.add(Dense(units=128, activation='relu'))
autoencoder.add(Dense(units=784, activation='sigmoid'))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    ### Treino e Teste do modelo de Redução de Dimensionalidade ###
autoencoder.fit(prev_train, prev_train, batch_size=256, epochs=50,
          validation_data=(prev_test, prev_test))

    ### Construção do Modelo de Codificação ###
dim_original = Input(shape=(784,))
camada_encoder1 = model.layers[0]
camada_encoder2 = model.layers[1]
camada_encoder3 = model.layers[2]
encoder = Model(inputs=dim_original, 
                outputs=camada_encoder3(camada_encoder2(camada_encoder1(dim_original))))
encoder.summary()

    ### Visualizar Imagens Codificadas e Decodificadas   ###
# Cria as imagens codificadas e decodificadas
imgs_codificadas = encoder.predict(prev_test)
imgs_decodificadas = autoencoder.predict(prev_test)

# Visualiza as imagens codificadas e decodificadas
n_imgs = 10
imgs_test = np.random.randint(prev_test.shape[0], size=n_imgs)

plt.figure(figsize=(10,8))
for i, indice_img in enumerate(imgs_test):
    # imagem original
    plt.subplot(10,10,i+1)
    plt.imshow(prev_test[indice_img].reshape(28,28))
    plt.xticks(())
    plt.yticks(())
    
    # imagem codificada
    plt.subplot(10,10,i+1 +10)
    plt.imshow(imgs_codificadas[indice_img].reshape(8,4))
    plt.xticks(())
    plt.yticks(())
    
    # imagem decodificada
    plt.subplot(10,10,i+1 +20)
    plt.imshow(imgs_decodificadas[indice_img].reshape(28,28))
    plt.xticks(())
    plt.yticks(())





        # =====   Redução de Dimensionalidade com Convolutional Autoencoder   ===== #
from keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D

    ### Pre processamento dos Dados ###
prev_train = prev_train.reshape(prev_train.shape[0], prev_train.shape[1], prev_train.shape[2], 1)
prev_test = prev_test.reshape(prev_test.shape[0], prev_test.shape[1], prev_test.shape[2], 1)

prev_train = prev_train.astype('float32') / 255
prev_test = prev_test.astype('float32') / 255

    ### Criação do modelo de Redução de Dimensionalidade ###
autoencoder = Sequential()
#Encoder
autoencoder.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
autoencoder.add(MaxPooling2D(pool_size=(2,2)))
autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same'))
autoencoder.add(MaxPooling2D(pool_size=(2,2), padding='same'))
autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2)))
autoencoder.add(Flatten())
#Decoder
autoencoder.add(Reshape(target_shape=(4,4,8)))
autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D(size=(2,2)))
autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D(size=(2,2)))
autoencoder.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
autoencoder.add(UpSampling2D(size=(2,2)))
autoencoder.add(Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same'))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.summary()

    ### Treino e Teste do modelo de Redução de Dimensionalidade ###
autoencoder.fit(prev_train, prev_train, batch_size=256, epochs=10,
                validation_data=(prev_test, prev_test))

    ### Construção do Modelo de Codificação ###
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('flatten_7').output)
encoder.summary()


    ### Visualizar Imagens Codificadas e Decodificadas   ###
# Cria as imagens codificadas e decodificadas
imgs_codificadas = encoder.predict(prev_test)
imgs_decodificadas = autoencoder.predict(prev_test)

# Visualiza as imagens codificadas e decodificadas
n_imgs = 10
imgs_test = np.random.randint(prev_test.shape[0], size=n_imgs)

plt.figure(figsize=(10,8))
for i, indice_img in enumerate(imgs_test):
    # imagem original
    plt.subplot(10,10,i+1)
    plt.imshow(prev_test[indice_img].reshape(28,28))
    plt.xticks(())
    plt.yticks(())
    
    # imagem codificada
    plt.subplot(10,10,i+1 +10)
    plt.imshow(imgs_codificadas[indice_img].reshape(16,8)) # 16*8 = 128
    plt.xticks(())
    plt.yticks(())
    
    # imagem decodificada
    plt.subplot(10,10,i+1 +20)
    plt.imshow(imgs_decodificadas[indice_img].reshape(28,28))
    plt.xticks(())
    plt.yticks(())