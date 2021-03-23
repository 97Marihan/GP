import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json, Model
from keras.layers import Dense, Activation, Conv2D, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout, Flatten
from keras import optimizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, CSVLogger
from keras.initializers import glorot_normal

seed = 42
init = glorot_normal()


def classification_report_csv(report, directory):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        try:
            row = {}
            row_data_ = line.split('      ')
            row_data = [x for x in row_data_ if x]
            row['class'] = row_data[0]
            row['precision'] = float(row_data[1])
            row['recall'] = float(row_data[2])
            row['f1_score'] = float(row_data[3])
            row['support'] = float(row_data[4])
            report_data.append(row)
        except:
            pass
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(directory + '_classification_report.csv', index=False)

### calcualte accuracy
def decode(data):
    decoded_datum = []
    for i in range(data.shape[0]):
        decoded_data.append(np.argmax(data[i]))
    return np.array(decoded_data)


####   Loadind training and testing pickle files
data_fid = open('data_training_croped_4.pkl', 'rb')
[trainX, trainY] = pickle.load(data_fid)  #trainX images of training TrainY label of the trainig image
data_fid.close()
data_fid = open('data_testing_croped_4.pkl', 'rb')
[testX, testY] = pickle.load(data_fid) #TsetX images of testing TestY label of testing set
data_fid.close()

#    Initial training parameters
inputShape = (300, 300, 3)
epoch = 200
BS = 32
num_classes = 4
random.seed(seed)
INIT_LR = 0.01
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
def own_model():
    ###    Model Structure
    images = Input(shape=inputShape)
    x = Conv2D(32, (3, 3), strides=(1, 1), kernel_initializer=init, padding='same')(images)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer=init, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer=init, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer=init, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer=init, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer=init, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer=init, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(500, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=images, outputs=outputs)
    return model

training = True

if training:

    model = own_model()

    ### Optimizer and callbacks
    checkpoint = ModelCheckpoint('models/model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1, monitor='val_acc', save_best_only=True, mode='auto')
    csv_logger = CSVLogger('report/log_' + str(INIT_LR) + '.csv', append=False, separator=';')
##by2all l learningg rate 3shan ywsl l global minimum
    sgd = optimizers.SGD(lr=INIT_LR, decay=1e-2, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())

    ### Training

    H = model.fit(x=trainX, y=trainY, batch_size=BS, validation_data=(testX, testY), epochs=epoch, callbacks=[csv_logger, checkpoint])


    # plot the training loss and accuracy
    N = np.arange(0, epoch)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.plot(N, H.history["acc"], label="train_acc")
    plt.plot(N, H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy " + str(INIT_LR))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig('Plots/history_fig')

else:
    ###   Testing


    model = own_model()
    # load weights into new model
    model.load_weights('models/model-045-1.000000-0.881117.h5')
    # print("Loaded model from disk")

    #report that contains the results of the confusion metrix
    predictions = model.predict(testX)
    report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1))
    print(report)
    classification_report_csv(report, 'test/' + str(INIT_LR))


#3shan n7sb l accuracy ba call function decode
result = decode(model.predict(testX))
print(result)
ref_result = decode(testY)
print(ref_result)
acc = 100 * (1 - float(np.count_nonzero(result - ref_result)) / float(len(result)))
print('Acc = ' + str(acc))
