import numpy as np
from glob import glob
import os

import tensorflow as tf
from tensorflow import keras

print('TensorFlow {}; Keras {}'.format(tf.__version__, keras.__version__))

from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Dropout, Activation, UpSampling2D, GlobalMaxPooling2D, multiply
from tensorflow.keras.backend import max


from keras_unet_collection import models, base, utils, losses

from keras import backend as K

# def dice_coef(y_true, y_pred):
#     y_true = y_true[...]
#     y_pred = y_pred[...]
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2 * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))

# def dice_coef_loss(y_true, y_pred):
#     return (1. - dice_coef(y_true, y_pred))


_dir = os.getcwd()
os.chdir(_dir)


data_dir = os.path.join(_dir, 'data/')
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

weight_dir = os.path.join(_dir, 'weight/')
if not os.path.exists(weight_dir):
    os.mkdir(weight_dir)


first_time_running = False

if first_time_running:
    # downloading and executing data files
    import tarfile
    import urllib.request
    
    filename_image = data_dir+'images.tar.gz'
    filename_target = data_dir+'annotations.tar.gz'
    
    urllib.request.urlretrieve('http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz', filename_image);
    urllib.request.urlretrieve('https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz', filename_target);
    
    with tarfile.open(filename_image, "r:gz") as tar_io:
        tar_io.extractall(path=data_dir)
    with tarfile.open(filename_target, "r:gz") as tar_io:
        tar_io.extractall(path=data_dir)




#############################################################################################################################################



attunet = models.att_unet_2d((128, 128, 3), [64, 128, 256, 512], n_labels=3,
                           stack_num_down=2, stack_num_up=2,
                           activation='ReLU', atten_activation='ReLU', attention='add', output_activation='Softmax', 
                           batch_norm=True, pool=False, unpool='bilinear', name='attunet')


def hybrid_loss(y_true, y_pred):

    loss_focal = losses.focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3)
    loss_iou = losses.iou_seg(y_true, y_pred)
    
    # (x) 
    #loss_ssim = losses.ms_ssim(y_true, y_pred, max_val=1.0, filter_size=4)
    
    return loss_focal+loss_iou #+loss_ssim


def input_data_process(input_array):
    '''converting pixel vales to [0, 1]'''
    return input_array/255.

def target_data_process(target_array):
    '''Converting tri-mask of {1, 2, 3} to three categories.'''
    return keras.utils.to_categorical(target_array-1)

sample_names = np.array(sorted(glob(data_dir+'images/*.jpg')))
label_names = np.array(sorted(glob(data_dir+'annotations/trimaps/*.png')))

L = len(sample_names)
ind_all = utils.shuffle_ind(L)

L_train = int(0.8*L); L_valid = int(0.1*L); L_test = L - L_train - L_valid
ind_train = ind_all[:L_train]; ind_valid = ind_all[L_train:L_train+L_valid]; ind_test = ind_all[L_train+L_valid:]
print("Training:validation:testing = {}:{}:{}".format(L_train, L_valid, L_test))

valid_input = input_data_process(utils.image_to_array(sample_names[ind_valid], size=128, channel=3))
valid_target = target_data_process(utils.image_to_array(label_names[ind_valid], size=128, channel=1))
test_input = input_data_process(utils.image_to_array(sample_names[ind_test], size=128, channel=3))
test_target = target_data_process(utils.image_to_array(label_names[ind_test], size=128, channel=1))

train_input = input_data_process(utils.image_to_array(sample_names[ind_train], size=128, channel=3))
train_target = target_data_process(utils.image_to_array(label_names[ind_train], size=128, channel=1))

model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=os.path.join(weight_dir, '2dattunetcallback.h5'),monitor='val_loss', save_best_only=True)


attunet.compile(loss=[hybrid_loss, hybrid_loss, hybrid_loss, hybrid_loss, hybrid_loss],
                  loss_weights=[0.25, 0.25, 0.25, 0.25, 1.0],
                  optimizer=keras.optimizers.Adam(learning_rate=1e-4))

attunet.fit(
    train_input, train_target,
    batch_size=32, epochs=10,
    verbose=2, shuffle=True,
    validation_data=(valid_input, valid_target),
    callbacks=[model_checkpoint])



attunet.load_weights(os.path.join(weight_dir, '2dattunetcallback.h5'))
temp_out = attunet.predict([test_input,])
# y_pred = temp_out[-1]
y_pred = temp_out
print('Testing set cross-entropy loss = {}'.format(np.mean(keras.losses.categorical_crossentropy(test_target, y_pred))))
print('Testing set focal Tversky loss = {}'.format(np.mean(losses.focal_tversky(test_target, y_pred))))
print('Testing set IoU loss = {}'.format(np.mean(losses.iou_seg(test_target, y_pred))))

np.save('raw.npy', test_input)
np.save('mask.npy', test_target)
np.save('pred_attunet.npy', y_pred)






temp_out = attunet.predict([test_input,])
# y_pred = temp_out[-1]
y_pred = temp_out
d = y_pred
d = np.squeeze(d)
print(np.shape(d))
d = np.where(d > 0.5, 1., 0.)
test_target = target_data_process(utils.image_to_array(label_names[ind_test], size=128, channel=1))

a = test_target
a = np.squeeze(a)
print('*' * 50)
print(np.shape(a))
a = np.where(a > 0.5, 1., 0.)

dice = np.sum(d[a == 1]) * 2.0 / (np.sum(d) + np.sum(a))
# dice = dice_coef(y_true=a, y_pred=d)
print('Dice similarity score is {}'.format(dice))

IOU = dice / (2 - dice)
print('IOU score is {}'.format(IOU))

true_labels = a
pred_labels = d
# True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))

# True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))

# False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))

# False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

print('TP:{}, FP:{}, TN:{}, FN:{}'.format(TP, FP, TN, FN))
TP = float(TP)
FP = float(FP)
TN = float(TN)
FN = float(FN)

accuracy = (TP + TN) / (TP + FN + FP + TN)
print('accuracy:{}'.format(accuracy))

precision = TP / (TP + FP)
print('precision:{}'.format(precision))

recall = TP / (TP + FN)
print('recall:{}'.format(recall))

specificity = TN / (TN + FP)
print('specificity:{}'.format(specificity))