from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU


def Time_Series_CNN(dropout_rate, num_classes, num_features):
    
    input_shape = (num_features, 30, 1)
    model = Sequential()
    model.add(Conv2D(64, (1,5), name='conv1', padding='valid', kernel_initializer='he_normal', input_shape=input_shape))
    model.add(LeakyReLU(0.01))
    model.add(Dropout(dropout_rate, seed=1))
    model.add(MaxPooling2D((1, 2), strides=(1, 2)))
    model.add(Conv2D(128, (1, 5), name='conv2', padding='valid', kernel_initializer='he_normal'))
    model.add(LeakyReLU(0.01))
    model.add(Dropout(dropout_rate, seed=1))
    model.add(MaxPooling2D((1, 2), strides=(1, 2)))

    model.add(AveragePooling2D((num_features,num_features)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU(0.01))
    model.add(Dropout(dropout_rate, seed=1))
    model.add(Dense(num_classes, activation='softmax'))
    

    return model

