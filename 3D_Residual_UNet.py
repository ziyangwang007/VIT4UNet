



import os
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Conv3D,Input, concatenate, MaxPooling3D, Conv3DTranspose, BatchNormalization, Dropout
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import Model
from keras.utils import multi_gpu_model
from keras import backend as K

from Process_Data import load_train_data, load_test_data, load_mask_train, preprocess_squeeze, load_mask_test

K.set_image_data_format('channels_last') # tensorflow - channels_last

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
# batch_size_per_replica = 2
# strategy = tf.distribute.MirroredStrategy()
# print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
# batch_size = batch_size_per_replica * strategy.num_replicas_in_sync




project_name = '3DresUnet'

def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return (1-dice_coef(y_true, y_pred))





_dir = os.getcwd()
os.chdir(_dir)

# save model
weight_dir = os.path.join(_dir, 'weights/')
if not os.path.exists(weight_dir):
    os.mkdir(weight_dir)

# save results
pred_dir = os.path.join(_dir, '3DresUnetpreds/')
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
    inputs = Input((16, 256, 256, 1))
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    conc1 = concatenate([inputs, conv1], axis=4)
    conc1 = BatchNormalization()(conc1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conc1)
    pool1 = Dropout(0.1)(pool1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    conc2 = concatenate([pool1, conv2], axis=4)
    conc2 = BatchNormalization()(conc2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conc2)
    pool2 = Dropout(0.1)(pool2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    conc3 = concatenate([pool2, conv3], axis=4)
    conc3 = BatchNormalization()(conc3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conc3)
    pool3 = Dropout(0.1)(pool3)

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
    conc4 = concatenate([pool3, conv4], axis=4)
    conc4 = BatchNormalization()(conc4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conc4)
    pool4 = Dropout(0.1)(pool4)

    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)
    conc5 = concatenate([pool4, conv5], axis=4)

    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc5), conv4], axis=4)
    up6 = Dropout(0.1)(up6)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)
    conc6 = concatenate([up6, conv6], axis=4)
    conc6 = BatchNormalization()(conc6)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc6), conv3], axis=4)
    up7 = Dropout(0.1)(up7)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)
    conc7 = concatenate([up7, conv7], axis=4)
    conc7 = BatchNormalization()(conc7)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc7), conv2], axis=4)
    up8 = Dropout(0.1)(up8)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)
    conc8 = concatenate([up8, conv8], axis=4)
    conc8 = BatchNormalization()(conc8)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc8), conv1], axis=4)
    up9 = Dropout(0.1)(up9)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)
    conc9 = concatenate([up9, conv9], axis=4)
    conc9 = BatchNormalization()(conc9)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conc9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.summary()
    #plot_model(model, to_file='model.png')

    model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def train():

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)


    # strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

    # with strategy.scope():


    model = get_unet()
    # model = multi_gpu_model(model,gpus =2)
    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])


    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=os.path.join(weight_dir, '3dresunetcallback.h5'), monitor='val_loss', save_best_only=True)

    csv_logger = keras.callbacks.CSVLogger(os.path.join(log_dir,  '3dresunet.txt'), separator=',', append=False)

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

    # model.load_weights(os.path.join(weight_dir, '3dunetcallback.h5'))
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
    model.save_weights('3dresunet.h5')


def predict():


    model = get_unet()
    os.chdir(weight_dir)
    # model.load_weights('3dunetcallback.h5')
    model.load_weights('3dresunetcallback.h5')
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
    np.save('results_3dresunet.npy',imgs_test_results)

    print('-'*30)
    print('Prediction finished')
    print('-'*30)


if __name__ == '__main__':
    train()
    predict()
