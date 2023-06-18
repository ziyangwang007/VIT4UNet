import os
import numpy as np
# import tensorflow as tf
import keras
from keras.layers import Conv3D,Input, concatenate, MaxPooling3D, Conv3DTranspose, BatchNormalization, Dropout
from keras.optimizers import RMSprop, Adam, SGD
# from keras.models import Model
# from keras.utils import multi_gpu_model
# from keras import backend as K

from Process_Data import load_train_data, load_test_data, load_mask_train, preprocess_squeeze, load_mask_test

keras.backend.set_image_data_format('channels_last') # tensorflow - channels_last

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"
# batch_size_per_replica = 2
# strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.Strategy.experimental_run_v2()
# print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
# batch_size = batch_size_per_replica * strategy.num_replicas_in_sync




project_name = '3DdenseUnet'



def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2 * intersection + smooth) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return (1-dice_coef(y_true, y_pred))




_dir = os.getcwd()
os.chdir(_dir)

# save model
weight_dir = os.path.join(_dir, 'weights/')
if not os.path.exists(weight_dir):
    os.mkdir(weight_dir)

# save results
pred_dir = os.path.join(_dir, '3DdenseUnetpreds/')
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
    inputs = Input((16, 256, 256,1))
    conv11 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conc11 = concatenate([inputs, conv11], axis=4)
    conv12 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conc11)
    conc12 = concatenate([inputs, conv12], axis=4)
    conc12 = BatchNormalization()(conc12)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conc12)
    pool1 = Dropout(0.1)(pool1)

    conv21 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conc21 = concatenate([pool1, conv21], axis=4)
    conv22 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conc21)
    conc22 = concatenate([pool1, conv22], axis=4)
    conc22 = BatchNormalization()(conc22)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conc22)
    pool2 = Dropout(0.1)(pool2)

    conv31 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conc31 = concatenate([pool2, conv31], axis=4)
    conv32 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conc31)
    conc32 = concatenate([pool2, conv32], axis=4)
    conc32 = BatchNormalization()(conc32)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conc32)
    pool3 = Dropout(0.1)(pool3)

    conv41 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conc41 = concatenate([pool3, conv41], axis=4)
    conv42 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conc41)
    conc42 = concatenate([pool3, conv42], axis=4)
    conc42 = BatchNormalization()(conc42)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conc42)
    pool4 = Dropout(0.1)(pool4)

    conv51 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    conc51 = concatenate([pool4, conv51], axis=4)
    conv52 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conc51)
    conc52 = concatenate([pool4, conv52], axis=4)

    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc52), conc42], axis=4)
    up6 = Dropout(0.1)(up6)
    conv61 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
    conc61 = concatenate([up6, conv61], axis=4)
    conv62 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conc61)
    conc62 = concatenate([up6, conv62], axis=4)
    conc62 = BatchNormalization()(conc62)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc62), conv32], axis=4)
    up7 = Dropout(0.1)(up7)
    conv71 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
    conc71 = concatenate([up7, conv71], axis=4)
    conv72 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conc71)
    conc72 = concatenate([up7, conv72], axis=4)
    conc72 = BatchNormalization()(conc72)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc72), conv22], axis=4)
    up8 = Dropout(0.1)(up8)
    conv81 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
    conc81 = concatenate([up8, conv81], axis=4)
    conv82 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conc81)
    conc82 = concatenate([up8, conv82], axis=4)
    conc82 = BatchNormalization()(conc82)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc82), conv12], axis=4)
    up9 = Dropout(0.1)(up9)
    conv91 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
    conc91 = concatenate([up9, conv91], axis=4)
    conv92 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conc91)
    conc92 = concatenate([up9, conv92], axis=4)
    conc92 = BatchNormalization()(conc92)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conc92)

    model = keras.models.Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss=dice_coef_loss, metrics=[dice_coef])

    model.summary()
    #plot_model(model, to_file='model.png')


    return model


def train():

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)


    # strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])


    model = get_unet()
    # Pmodel = multi_gpu_model(model,gpus =2)
    # Pmodel.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss=dice_coef_loss, metrics=[dice_coef])
    
    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])


    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=os.path.join(weight_dir, '3ddenseunetcallback.h5'), monitor='val_loss', save_best_only=True)

    csv_logger = keras.callbacks.CSVLogger(os.path.join(log_dir,  '3ddenseunet.txt'), separator=',', append=False)

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
    imgs_test /= 255. 
    imgs_masks_test = load_mask_test()
    imgs_masks_test = imgs_masks_test.astype('float32')
    imgs_masks_test /= 255. 
    # model.load_weights(os.path.join(weight_dir, '3ddenseunetcallback.h5'))
    # model.fit(imgs_train, imgs_masks_train, batch_size=2, epochs=10, verbose=1, shuffle=True, validation_data=(imgs_test,imgs_masks_test), callbacks=[model_checkpoint, csv_logger])    # [model_checkpoint, csv_logger] 
    model.fit(imgs_train, imgs_masks_train, batch_size=1, epochs=150, verbose=1, shuffle=True, validation_data=(imgs_test,imgs_masks_test), callbacks=[model_checkpoint, csv_logger])    # [model_checkpoint, csv_logger] 

    # train_dataset = tf.data.Dataset.from_tensor_slices((imgs_train, imgs_masks_train))
    # train_dataset = train_dataset.batch(1)
    # model.fit(train_dataset, epochs=50, verbose=1, callbacks=[model_checkpoint, csv_logger])    # [model_checkpoint, csv_logger] 
    # model.fit(imgs_train, imgs_masks_train, batch_size=2, epochs=50, verbose=1, shuffle=True, validation_split=0.10, callbacks=[model_checkpoint, csv_logger])    # [model_checkpoint, csv_logger] 
    # model.fit(imgs_train, imgs_masks_train, batch_size=16, epochs=50, verbose=1, shuffle=True, validation_split=0.10, callbacks=[model_checkpoint, csv_logger])    # [model_checkpoint, csv_logger] 




    print('-'*30)
    print('Training finished')
    print('-'*30)
    os.chdir(weight_dir)
    # model.save_weights('3dunet.h5')
    model.save_weights('3ddenseunet.h5')


def predict():


    model = get_unet()
    os.chdir(weight_dir)
    # model.load_weights('3dunetcallback.h5')
    model.load_weights('3ddenseunetcallback.h5')
    # model = keras.models.load_model(os.path.join(weight_dir, project_name + '.h5'))
    
    imgs_test = load_test_data()
    imgs_test = imgs_test.astype('float32')
    imgs_test /= 255.

    # imgs_test_results = model.predict(imgs_test, batch_size=2, verbose=1
    imgs_test_results = model.predict(imgs_test, batch_size=2, verbose=1)
    imgs_test_results = preprocess_squeeze(imgs_test_results)
    imgs_test_results = np.around(imgs_test_results, decimals=0)
    imgs_test_results = (imgs_test_results*255.).astype(np.uint8)

    os.chdir(pred_dir)
    np.save('results_3ddenseunet.npy',imgs_test_results)

    print('-'*30)
    print('Prediction finished')
    print('-'*30)


if __name__ == '__main__':
    train()
    predict()
