import os
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Conv2D,Input, concatenate, MaxPooling2D, Conv2DTranspose, BatchNormalization,Dropout,add, Input,Activation
from keras.optimizers import RMSprop, Adam, SGD
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.models import Model, model_from_json
from keras import backend as K

from Process_Data_for_2D_NoisyLabel_MultiRes import load_train_data, load_test_data, load_mask_train_noisy,load_mask_train_original, preprocess_squeeze, load_mask_test





K.set_image_data_format('channels_last') # tensorflow - channels_last

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"




project_name = '2DMultiResUnet'


_dir = os.getcwd()
os.chdir(_dir)

# save model
weight_dir = os.path.join(_dir, 'weights/')
if not os.path.exists(weight_dir):
    os.mkdir(weight_dir)

# save results
pred_dir = os.path.join(_dir, '2DMultiResUnetpreds/')
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



def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    '''
    2D Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if(activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x


def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
    '''
    2D Transposed Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2)})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    
    return x


def MultiResBlock(U, inp, alpha = 1.67):
    '''
    MultiRes Block
    
    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''

    W = alpha * U

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                         int(W*0.5), 1, 1, activation=None, padding='same')

    conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,
                        activation='relu', padding='same')

    conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3,
                        activation='relu', padding='same')

    conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3,
                        activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    return out


def ResPath(filters, length, inp):
    '''
    ResPath
    
    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''


    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1,
                         activation=None, padding='same')

    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                             activation=None, padding='same')

        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out


def MultiResUnet(height, width, n_channels):
    '''
    MultiResUNet
    
    Arguments:
        height {int} -- height of image 
        width {int} -- width of image 
        n_channels {int} -- number of channels in image
    
    Returns:
        [keras model] -- MultiResUNet model
    '''


    inputs = Input((height, width, n_channels))

    mresblock1 = MultiResBlock(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = ResPath(32, 4, mresblock1)

    mresblock2 = MultiResBlock(32*2, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = ResPath(32*2, 3, mresblock2)

    mresblock3 = MultiResBlock(32*4, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = ResPath(32*4, 2, mresblock3)

    mresblock4 = MultiResBlock(32*8, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = ResPath(32*8, 1, mresblock4)

    mresblock5 = MultiResBlock(32*16, pool4)

    up6 = concatenate([Conv2DTranspose(
        32*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = MultiResBlock(32*8, up6)

    up7 = concatenate([Conv2DTranspose(
        32*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
    mresblock7 = MultiResBlock(32*4, up7)

    up8 = concatenate([Conv2DTranspose(
        32*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
    mresblock8 = MultiResBlock(32*2, up8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(
        2, 2), padding='same')(mresblock8), mresblock1], axis=3)
    mresblock9 = MultiResBlock(32, up9)

    conv10 = conv2d_bn(mresblock9, 1, 1, 1, activation='sigmoid')
    
    model = Model(inputs=[inputs], outputs=[conv10])

    print(model.summary())

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model
   


def train():

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    
    model = MultiResUnet(256, 256, 1)

    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=os.path.join(weight_dir, '2dMultiResunetcallback.h5'), monitor='val_loss', save_best_only=True)

    csv_logger = keras.callbacks.CSVLogger(os.path.join(log_dir,  '2dMultiResunet.txt'), separator=',', append=False)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)


            
    imgs_train = load_train_data()
    imgs_train = imgs_train.astype('float32')
    imgs_train /= 255.
    imgs_masks_train = load_mask_train_original()
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
    
    # loss = history.history['loss']



    # model.fit(imgs_train, imgs_masks_train, batch_size=16, epochs=20, verbose=1, shuffle=False, validation_data=(imgs_test,imgs_test_mask), callbacks=[model_checkpoint, csv_logger])    # [model_checkpoint, csv_logger] 
 


    # model.fit(imgs_train, imgs_masks_train, batch_size=16, epochs=50, verbose=1, shuffle=True, vali


    print('-'*30)
    print('Training finished')
    print('-'*30)
    os.chdir(weight_dir)
    model.save_weights('2dMultiResunet.h5')


def predict():


    model = MultiResUnet(256, 256, 1)
    os.chdir(weight_dir)
    model.load_weights('2dMultiResunetcallback.h5')
    # model = keras.models.load_model(os.path.join(weight_dir, project_name + '.h5'))
    
    imgs_test = load_test_data()
    imgs_test = imgs_test.astype('float32')
    imgs_test = imgs_test/255.

    imgs_test_results = model.predict(imgs_test, batch_size=8, verbose=1)
    imgs_test_results = preprocess_squeeze(imgs_test_results)
    imgs_test_results = np.around(imgs_test_results, decimals=0)
    imgs_test_results = (imgs_test_results*255.).astype(np.uint8)
    os.chdir(pred_dir)
    np.save('results_2dMultiResunet.npy',imgs_test_results)

    print('-'*30)
    print('Prediction finished')
    print('-'*30)


if __name__ == '__main__':
    train()
    predict()
