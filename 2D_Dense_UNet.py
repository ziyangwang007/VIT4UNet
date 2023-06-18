import os
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Conv2D,Input, concatenate, MaxPooling2D, Conv2DTranspose, BatchNormalization, Dropout
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import Model
from keras import backend as K

from Process_Data_for_2D_NoisyLabel import load_train_data, load_test_data, load_mask_train, preprocess_squeeze, load_mask_test

K.set_image_data_format('channels_last') # tensorflow - channels_last

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


project_name = '2DDenseUnet'

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2 * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


_dir = os.getcwd()
os.chdir(_dir)

# save model
weight_dir = os.path.join(_dir, 'weights/')
if not os.path.exists(weight_dir):
    os.mkdir(weight_dir)

# save results
pred_dir = os.path.join(_dir, '2DDenseUnetpreds/')
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)

# save log
log_dir = os.path.join(_dir, 'logs/')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)


image_rows = int(256)
image_cols = int(256)
image_depth = int(16)

case_depth = [559,507,560,625,601,562,509,548,572,552] # depth of each cases, 10 cases in total
case_depth = list(map(int,case_depth))



def get_unet():
    inputs = Input((256, 256,1))
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conc11 = concatenate([inputs, conv11], axis=3)
    conv12 = Conv2D(32, (3, 3), activation='relu', padding='same')(conc11)
    conc12 = concatenate([inputs, conv12], axis=3)
    conc12 = BatchNormalization()(conc12)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conc12)
    pool1 = Dropout(0.1)(pool1)

    conv21 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conc21 = concatenate([pool1, conv21], axis=3)
    conv22 = Conv2D(64, (3, 3), activation='relu', padding='same')(conc21)
    conc22 = concatenate([pool1, conv22], axis=3)
    conc22 = BatchNormalization()(conc22)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conc22)
    pool2 = Dropout(0.1)(pool2)

    conv31 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conc31 = concatenate([pool2, conv31], axis=3)
    conv32 = Conv2D(128, (3, 3), activation='relu', padding='same')(conc31)
    conc32 = concatenate([pool2, conv32], axis=3)
    conc32 = BatchNormalization()(conc32)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conc32)
    pool3 = Dropout(0.1)(pool3)

    conv41 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conc41 = concatenate([pool3, conv41], axis=3)
    conv42 = Conv2D(256, (3, 3), activation='relu', padding='same')(conc41)
    conc42 = concatenate([pool3, conv42], axis=3)
    conc42 = BatchNormalization()(conc42)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conc42)
    pool4 = Dropout(0.1)(pool4)

    conv51 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conc51 = concatenate([pool4, conv51], axis=3)
    conv52 = Conv2D(512, (3, 3), activation='relu', padding='same')(conc51)
    conc52 = concatenate([pool4, conv52], axis=3)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conc52), conc42], axis=3)
    up6 = Dropout(0.1)(up6)
    conv61 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conc61 = concatenate([up6, conv61], axis=3)
    conv62 = Conv2D(256, (3, 3), activation='relu', padding='same')(conc61)
    conc62 = concatenate([up6, conv62], axis=3)
    conc62 = BatchNormalization()(conc62)


    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conc62), conv32], axis=3)
    up7 = Dropout(0.1)(up7)
    conv71 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conc71 = concatenate([up7, conv71], axis=3)
    conv72 = Conv2D(128, (3, 3), activation='relu', padding='same')(conc71)
    conc72 = concatenate([up7, conv72], axis=3)
    conc72 = BatchNormalization()(conc72)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conc72), conv22], axis=3)
    up8 = Dropout(0.1)(up8)
    conv81 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conc81 = concatenate([up8, conv81], axis=3)
    conv82 = Conv2D(64, (3, 3), activation='relu', padding='same')(conc81)
    conc82 = concatenate([up8, conv82], axis=3)
    conc82 = BatchNormalization()(conc82)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conc82), conv12], axis=3)
    up9 = Dropout(0.1)(up9)
    conv91 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conc91 = concatenate([up9, conv91], axis=3)
    conv92 = Conv2D(32, (3, 3), activation='relu', padding='same')(conc91)
    conc92 = concatenate([up9, conv92], axis=3)
    conc92 = BatchNormalization()(conc92)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conc92)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.summary()
    #plot_model(model, to_file='model.png')

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def train():

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()

    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=os.path.join(weight_dir, '2dDenseunetcallback.h5'), monitor='val_loss', save_best_only=True)

    csv_logger = keras.callbacks.CSVLogger(os.path.join(log_dir,  '2dDensev2v3.txt'), separator=',', append=False)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)


            
    imgs_train = load_train_data()
    imgs_train = imgs_train.astype('float32')
    imgs_train /= 255.
    imgs_masks_train = load_mask_train()
    imgs_masks_train = imgs_masks_train.astype('float32')
    imgs_masks_train /= 255. 

    imgs_test = load_test_data()
    imgs_test = imgs_test.astype('float32')
    imgs_test = imgs_test/255.

    imgs_test_mask = load_mask_test()
    imgs_test_mask = imgs_test_mask.astype('float32')
    imgs_test_mask = imgs_test_mask/255.  

    # train_dataset = tf.data.Dataset.from_tensor_slices((imgs_train, imgs_masks_train))
    # train_dataset = train_dataset.batch(1)
    # model.fit(train_dataset, epochs=50, verbose=1, callbacks=[model_checkpoint, csv_logger])    # [model_checkpoint, csv_logger] 
    model.fit(imgs_train, imgs_masks_train, batch_size=16, epochs=200, verbose=1, shuffle=False, validation_data=(imgs_test,imgs_test_mask), callbacks=[model_checkpoint, csv_logger])    # [model_checkpoint, csv_logger] 
    # model.fit(imgs_train, imgs_masks_train, batch_size=16, epochs=50, verbose=1, shuffle=True, validation_split=0.10, callbacks=[model_checkpoint, csv_logger])    # [model_checkpoint, csv_logger] 




    print('-'*30)
    print('Training finished')
    print('-'*30)
    os.chdir(weight_dir)
    model.save_weights('2dDenseunet.h5')


def predict():


    model = get_unet()
    os.chdir(weight_dir)
    model.load_weights('2dDenseunetcallback.h5')
    # model = keras.models.load_model(os.path.join(weight_dir, project_name + '.h5'))
    
    imgs_test = load_test_data()
    imgs_test = imgs_test.astype('float32')
    imgs_test /= 255.

    imgs_test_results = model.predict(imgs_test, batch_size=16, verbose=1)
    imgs_test_results = preprocess_squeeze(imgs_test_results)
    imgs_test_results = np.around(imgs_test_results, decimals=0)
    imgs_test_results = (imgs_test_results*255.).astype(np.uint8)
    os.chdir(pred_dir)
    np.save('results_2dDenseunet.npy',imgs_test_results)

    print('-'*30)
    print('Prediction finished')
    print('-'*30)


if __name__ == '__main__':
    train()
    predict()
