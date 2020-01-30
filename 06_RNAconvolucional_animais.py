from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


    ###   Processo de Augumentation   ###
gerador_train = ImageDataGenerator(rescale=1./255, rotation_range=7, horizontal_flip=True,
                                   shear_range=0.2, height_shift_range=0.07, zoom_range=0.2)
gerador_test = ImageDataGenerator(rescale=1./255)

    ### Aplicação das alterações ###
base_train = gerador_train.flow_from_directory('dataset-animais/training_set',
                                               target_size=(64,64), batch_size=32, 
                                               class_mode='binary')
base_test = gerador_test.flow_from_directory('dataset-animais/test_set', 
                                             target_size=(64,64), batch_size=32, 
                                             class_mode='binary')

    ###   Criação do modelo de Classificação   ###
classificador = Sequential()
classificador.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(64,64,3), activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))
classificador.add(Flatten())

classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(rate=0.2))
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(rate=0.2))
classificador.add(Dense(units=1, activation='sigmoid')) # output layer

classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    ###   Treino e Teste do modelo   ###
classificador.fit_generator(base_train, steps_per_epoch=4000/16,
                            epochs=5, validation_data=base_test,
                            validation_steps=1000/16)



        # =====   Classificação de uma imagem   ===== #
import numpy as np
from keras.preprocessing import image

    ### Carregamento da Imagem ###
imagem = image.load_img('dataset-animais/test_set/gato/cat.3726.jpg', target_size=(64,64))
imagem = image.img_to_array(imagem)
imagem /= 255
imagem = np.expand_dims(imagem, axis=0) # transforma para o formato que trabalha o TensorFlow

    ### Classificação da Imagem ###
previsao = classificador.predict(imagem)
previsao = np.round(previsao)

for valor_previsto in previsao[0]:
    for value, key in enumerate(base_train.class_indices):
        if valor_previsto == value:
            print("Classe: {}".format(key))
            break
