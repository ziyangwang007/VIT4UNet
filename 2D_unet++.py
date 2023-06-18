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

from tensorflow.keras import backend as K

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



unet2plus = models.unet_plus_2d((256, 256, 1), [32, 64, 128, 256, 512], n_labels=3,
                            stack_num_down=2, stack_num_up=2,
                            activation='ReLU', output_activation='Softmax', 
                            batch_norm=True, pool=True, unpool=True, deep_supervision=True, name='unet2plus')


unet2plus.summary()
#############################################################################################################################################
# unet2plus = models.unet_3plus_2d((128, 128, 3), n_labels=2, filter_num_down=[64, 128, 256, 512], 
#                              filter_num_skip='auto', filter_num_aggregate='auto', 
#                              stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Sigmoid',
#                              batch_norm=True, pool='max', unpool=False, deep_supervision=True, name='unet2plus')

# #############################################################################################################################################
# name = 'unet2plus'
# activation = 'ReLU'
# filter_num_down = [32, 64, 128, 256, 512]
# filter_num_skip = [32, 32, 32, 32]
# filter_num_aggregate = 160

# stack_num_down = 2
# stack_num_up = 1
# n_labels = 3

# # `unet_3plus_2d_base` accepts an input tensor 
# # and produces output tensors from different upsampling levels
# # ---------------------------------------- #
# input_tensor = keras.layers.Input((128, 128, 3))
# # base architecture
# X_decoder = base.unet_3plus_2d_base(
#     input_tensor, filter_num_down, filter_num_skip, filter_num_aggregate, 
#     stack_num_down=stack_num_down, stack_num_up=stack_num_up, activation=activation, 
#     batch_norm=True, pool=True, unpool=True, backbone=None, name=name)



# # allocating deep supervision tensors
# OUT_stack = []
# # reverse indexing `X_decoder`, so smaller tensors have larger list indices 
# X_decoder = X_decoder[::-1]

# # deep supervision outputs
# for i in range(1, len(X_decoder)):
#     # 3-by-3 conv2d --> upsampling --> sigmoid output activation
#     pool_size = 2**(i)
#     X = Conv2D(n_labels, 3, padding='same', name='{}_output_conv1_{}'.format(name, i-1))(X_decoder[i])
    
#     X = UpSampling2D((pool_size, pool_size), interpolation='bilinear', 
#                      name='{}_output_sup{}'.format(name, i-1))(X)
    
#     X = Activation('sigmoid', name='{}_output_sup{}_activation'.format(name, i-1))(X)
#     # collecting deep supervision tensors
#     OUT_stack.append(X)

# # the final output (without extra upsampling)
# # 3-by-3 conv2d --> sigmoid output activation
# X = Conv2D(n_labels, 3, padding='same', name='{}_output_final'.format(name))(X_decoder[0])
# X = Activation('sigmoid', name='{}_output_final_activation'.format(name))(X)
# # collecting final output tensors
# OUT_stack.append(X)



# # Classification-guided Module (CGM)
# # ---------------------------------------- #
# # dropout --> 1-by-1 conv2d --> global-maxpooling --> sigmoid
# X_CGM = X_decoder[-1]
# X_CGM = Dropout(rate=0.1)(X_CGM)
# X_CGM = Conv2D(filter_num_skip[-1], 1, padding='same')(X_CGM)
# X_CGM = GlobalMaxPooling2D()(X_CGM)
# X_CGM = Activation('sigmoid')(X_CGM)

# CGM_mask = max(X_CGM, axis=-1) # <----- This value could be trained with "none-organ image"

# for i in range(len(OUT_stack)):
#     if i < len(OUT_stack)-1:
#         # deep-supervision
#         OUT_stack[i] = multiply([OUT_stack[i], CGM_mask], name='{}_output_sup{}_CGM'.format(name, i))
#     else:
#         # final output
#         OUT_stack[i] = multiply([OUT_stack[i], CGM_mask], name='{}_output_final_CGM'.format(name))


# unet2plus = keras.models.Model([input_tensor,], OUT_stack)


def hybrid_loss(y_true, y_pred):

    loss_focal = losses.focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3)
    loss_iou = losses.iou_seg(y_true, y_pred)
    
    # (x) 
    #loss_ssim = losses.ms_ssim(y_true, y_pred, max_val=1.0, filter_size=4)
    
    return loss_focal+loss_iou #+loss_ssim
# #############################################################################################################################################3


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

model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=os.path.join(weight_dir, '2dunet2+callback.h5'),monitor='val_loss', save_best_only=True)


unet2plus.compile(loss=[hybrid_loss, hybrid_loss, hybrid_loss, hybrid_loss, hybrid_loss],
                  loss_weights=[0.25, 0.25, 0.25, 0.25, 1.0],
                  optimizer=keras.optimizers.Adam(learning_rate=1e-4))

# unet2plus.load_weights(os.path.join(weight_dir, '2dunet2+callback.h5'))

unet2plus.fit(
    train_input, train_target,
    batch_size=32, epochs=10,
    verbose=2, shuffle=True,
    validation_data=(valid_input, valid_target),
    callbacks=[model_checkpoint])



unet2plus.load_weights(os.path.join(weight_dir, '2dunet2+callback.h5'))

print('*' * 50)


temp_out = unet2plus.predict([test_input,])
print(np.shape(temp_out))

# y_pred = temp_out[-1]
y_pred = temp_out
print('Testing set cross-entropy loss = {}'.format(np.mean(keras.losses.categorical_crossentropy(test_target, y_pred))))
print('Testing set focal Tversky loss = {}'.format(np.mean(losses.focal_tversky(test_target, y_pred))))
print('Testing set IoU loss = {}'.format(np.mean(losses.iou_seg(test_target, y_pred))))

np.save('raw.npy', test_input)
np.save('mask.npy', test_target)
np.save('pred_unet2plus.npy', y_pred)



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