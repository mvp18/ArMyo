import os
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


from model import *
from config_cnn import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

actions = ['bicep_down', 'bicep_medium', 'elbowRot', 'fist_close', 'reset_fist','shoulder_up','shoulder_down','wrist_up']

label = {}

num_classes = len(actions)

for i in range(len(actions)):

    label[actions[i]] = i

DATA_DIR = "../dataset/"

# Data processing

data = []
labels = []

for file in os.listdir(DATA_DIR):
    action = file[:-5]
    if action in actions:
        df = pd.read_csv(os.path.join(DATA_DIR, file))
        dnp = df.values[100:,1:5]
        dnp = dnp[:int(dnp.shape[0]/WINDOW_SIZE)*WINDOW_SIZE]
        scaler = MinMaxScaler()
        scaler.fit(dnp)
        dnp_tr = scaler.transform(dnp)
        dnp_tr = np.reshape(dnp_tr,(-1, dnp.shape[1], WINDOW_SIZE, 1))
        lab = np.zeros(dnp_tr.shape[0])
        lab.fill(label[action])
        data.append(dnp_tr)
        labels.append(lab)

fin_data = data[0]
fin_labels = labels[0]
for i in range(1,len(data)):
    fin_data = np.concatenate((fin_data,data[i]))
    fin_labels = np.concatenate((fin_labels,labels[i]))

print('Data Shape : {} with {} labels.'.format(fin_data.shape, fin_labels.shape[0]))

X_train, X_test, y_train, y_test = train_test_split(fin_data, fin_labels, test_size=0.2, random_state=42, stratify=fin_labels)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

thresholds = np.arange(1, 100) / 100.0
thresholds = thresholds[:, np.newaxis]


class Score_History(keras.callbacks.Callback):
    
    def on_epoch_end(self, epoch, logs={}):
        
        y_pred = self.model.predict(self.validation_data[0])
        y_true = self.validation_data[1]

        # print(y_pred.shape, y_true.shape)
        
        pred = y_pred[:, 0] >= thresholds
        pred = pred.astype(int)
        true = y_true[:, 0] >= thresholds
        true = true.astype(int)
        
        # print(pred.shape, true.shape)

        accs = (((y_true.shape[0] - np.sum(np.abs(pred - true), axis=1))) / float(y_true.shape[0])) * 100.0
        # print(accs.shape)
        best = np.argmax(accs)
        # print("best threshold, acc", thresholds[best], accs[best])

        accuracy = accs[best]
       
        print("validation accuracy with thresholds: {}".format(accuracy))
        
        return


def train():
    
    model = Time_Series_CNN(DROPOUT_RATE)
    
    model_save_path = './weights/time_series_cnn' + '.h5'

    # opt = SGD(lr=LEARNING_RATE, momentum=0.9, nesterov=True)
    
    opt = Adam(lr=LEARNING_RATE)
    
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    print(model.summary())

    checkpoint = ModelCheckpoint(filepath=model_save_path,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_weights_only=False,
                                 save_best_only=True,
                                 mode='auto')
    earlyStopping = EarlyStopping(monitor='val_acc',
                                  patience=EARLY_STOPPING,
                                  verbose=1,
                                  mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_acc',
                                  factor=0.2,
                                  patience=REDUCE_LR,
                                  verbose=1,
                                  min_lr=1e-8)


    # my call back function 
    histories = Score_History()

    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
              callbacks=[checkpoint, earlyStopping, reduce_lr, histories], shuffle=True, validation_data=(X_test, y_test))

    print("Finished!")


if __name__ == '__main__':
    train()
