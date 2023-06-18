import os
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Conv2D,Input, concatenate, MaxPooling2D, Conv2DTranspose, BatchNormalization,Dropout
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import Model
from keras import backend as K
os.environ["SM_FRAMEWORK"]="tf.keras"
import segmentation_models as sm

from Process_Data_for_2D import load_train_data, load_mask_train, load_test_data, preprocess_squeeze, load_mask_test





K.set_image_data_format('channels_last') # tensorflow - channels_last

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"




project_name = '2DFPN'


_dir = os.getcwd()
os.chdir(_dir)

# save model
weight_dir = os.path.join(_dir, 'weights/')
if not os.path.exists(weight_dir):
    os.mkdir(weight_dir)

# save results
pred_dir = os.path.join(_dir, '2DFPN/')
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

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2 * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)




    



def train():

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    
    # model = get_unet()

    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=os.path.join(weight_dir, '2dFPNcallback.h5'), monitor='val_loss', save_best_only=True)

    csv_logger = keras.callbacks.CSVLogger(os.path.join(log_dir,  '2dFPN.txt'), separator=',', append=False)

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

    
    # preprocess_input = sm.get_preprocessing('inceptionv3')
    # imgs_train = preprocess_input(imgs_train)
    # imgs_test = preprocess_input(imgs_test)
    # model = sm.Unet('resnet34', encoder_weights='imagenet')
    # model = sm.Unet('inceptionv3', input_shape=(256,256,1), encoder_weights=None)
    base_model = sm.FPN('inceptionv3',input_shape=(256,256,3), encoder_weights='imagenet', classes=1, activation='sigmoid')
    inp = Input(shape=(256,256,1))
    l1 = Conv2D(3,(1,1))(inp)
    out = base_model(l1)
    model = Model(inp,out)
    
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    
    model.summary()

    model.fit(imgs_train, imgs_masks_train, batch_size=16, epochs=100, verbose=1, shuffle=False, validation_data=(imgs_test,imgs_test_mask), callbacks=[model_checkpoint, csv_logger])    # [model_checkpoint, csv_logger] 
    
    # loss = history.history['loss']



    # model.fit(imgs_train, imgs_masks_train, batch_size=16, epochs=20, verbose=1, shuffle=False, validation_data=(imgs_test,imgs_test_mask), callbacks=[model_checkpoint, csv_logger])    # [model_checkpoint, csv_logger] 
 


    # model.fit(imgs_train, imgs_masks_train, batch_size=16, epochs=50, verbose=1, shuffle=True, vali


    print('-'*30)
    print('Training finished')
    print('-'*30)
    os.chdir(weight_dir)
    model.save_weights('2dFPN.h5')


def predict():


    model = sm.FPN(input_shape=(256,256,1), encoder_weights=None)
    os.chdir(weight_dir)
    model.load_weights('2dFPNcallback.h5')
    # model = keras.models.load_model(os.path.join(weight_dir, project_name + '.h5'))
    
    imgs_test = load_test_data()
    imgs_test = imgs_test.astype('float32')
    imgs_test = imgs_test/255.

    imgs_test_results = model.predict(imgs_test, batch_size=8, verbose=1)
    imgs_test_results = preprocess_squeeze(imgs_test_results)
    imgs_test_results = np.around(imgs_test_results, decimals=0)
    imgs_test_results = (imgs_test_results*255.).astype(np.uint8)
    os.chdir(pred_dir)
    np.save('results_2dFPN.npy',imgs_test_results)

    print('-'*30)
    print('Prediction finished')
    print('-'*30)


if __name__ == '__main__':
    train()
    predict()
