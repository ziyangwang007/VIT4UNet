
from __future__ import absolute_import

from keras_unet_collection.layer_utils import *
from keras_unet_collection.activations import GELU, Snake

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


def RSU(X, channel_in, channel_out, depth=5, activation='ReLU', batch_norm=True, pool=True, unpool=True, name='RSU'):
    '''
    The Residual U-blocks (RSU).
    
    RSU(X, channel_in, channel_out, depth=5, activation='ReLU', batch_norm=True, pool=True, unpool=True, name='RSU')
    
    ----------
    Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O.R. and Jagersand, M., 2020. 
    U2-Net: Going deeper with nested U-structure for salient object detection. 
    Pattern Recognition, 106, p.107404.
    
    Input
    ----------
        X: input tensor.
        channel_in: number of intermediate channels.
        channel_out: number of output channels.
        depth: number of down- and upsampling levels.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        batch_norm: True for batch normalization, False otherwise.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.   
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
        
    '''
    
    pool_size = 2
    
    X_skip = []
    
    X = CONV_stack(X, channel_out, kernel_size=3, stack_num=1, 
                   dilation_rate=1, activation=activation, batch_norm=batch_norm, 
                   name='{}_in'.format(name))
    X_skip.append(X)

    X = CONV_stack(X, channel_in, kernel_size=3, stack_num=1, dilation_rate=1, 
                   activation=activation, batch_norm=batch_norm, name='{}_down_0'.format(name))
    X_skip.append(X)
    
    for i in range(depth):
        
        X = encode_layer(X, channel_in, pool_size, pool, activation=activation, 
                         batch_norm=batch_norm, name='{}_encode_{}'.format(name, i))
        
        X = CONV_stack(X, channel_in, kernel_size=3, stack_num=1, dilation_rate=1, 
                       activation=activation, batch_norm=batch_norm, name='{}_down_{}'.format(name, i+1))
        X_skip.append(X)

    X = CONV_stack(X, channel_in, kernel_size=3, stack_num=1, 
               dilation_rate=2, activation=activation, batch_norm=batch_norm, 
               name='{}_up_0'.format(name))
    
    X_skip = X_skip[::-1]
    
    for i in range(depth):
        
        X = concatenate([X, X_skip[i]], axis=-1, name='{}_concat_{}'.format(name, i))
        
        X = CONV_stack(X, channel_in, kernel_size=3, stack_num=1, dilation_rate=1, 
                       activation=activation, batch_norm=batch_norm, name='{}_up_{}'.format(name, i+1))
        
        X = decode_layer(X, channel_in, pool_size, unpool, 
                         activation=activation, batch_norm=batch_norm, name='{}_decode_{}'.format(name, i))
        
    X = concatenate([X, X_skip[depth]], axis=-1, name='{}_concat_out'.format(name))

    X = CONV_stack(X, channel_out, kernel_size=3, stack_num=1, dilation_rate=1, 
                   activation=activation, batch_norm=batch_norm, name='{}_out'.format(name))
    X = add([X, X_skip[-1]], name='{}_out_add'.format(name))
    return X 

def RSU4F(X, channel_in, channel_out, dilation_num=[1, 2, 4, 8], activation='ReLU', batch_norm=True, name='RSU4F'):
    '''
    The Residual U-blocks with dilated convolutional kernels (RSU4F).
    
    RSU4F(X, channel_in, channel_out, dilation_num=[1, 2, 4, 8], activation='ReLU', batch_norm=True, name='RSU4F')
    
    ----------
    Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O.R. and Jagersand, M., 2020. 
    U2-Net: Going deeper with nested U-structure for salient object detection. 
    Pattern Recognition, 106, p.107404.
    
    Input
    ----------
        X: input tensor.
        channel_in: number of intermediate channels.
        channel_out: number of output channels.
        dilation_num: an iterable that defines dilation rates of convolutional layers.
                      Qin et al. (2020) suggested `[1, 2, 4, 8]`.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor
        
    '''
    
    X_skip = []    
    X = CONV_stack(X, channel_out, kernel_size=3, stack_num=1, dilation_rate=1, 
                   activation=activation, batch_norm=batch_norm, name='{}_in'.format(name))
    X_skip.append(X)
    
    for i, d in enumerate(dilation_num):

        X = CONV_stack(X, channel_in, kernel_size=3, stack_num=1, dilation_rate=d, 
                       activation=activation, batch_norm=batch_norm, name='{}_down_{}'.format(name, i))
        X_skip.append(X)

    X_skip = X_skip[:-1][::-1]
    dilation_num = dilation_num[:-1][::-1]
    
    for i, d in enumerate(dilation_num[:-1]):

        X = concatenate([X, X_skip[i]], axis=-1, name='{}_concat_{}'.format(name, i))
        X = CONV_stack(X, channel_in, kernel_size=3, stack_num=1, dilation_rate=d, 
                       activation=activation, batch_norm=batch_norm, name='{}_up_{}'.format(name, i))

    X = concatenate([X, X_skip[2]], axis=-1, name='{}_concat_out'.format(name))
    X = CONV_stack(X, channel_out, kernel_size=3, stack_num=1, dilation_rate=1, 
                   activation=activation, batch_norm=batch_norm, name='{}_out'.format(name))
    
    return add([X, X_skip[-1]], name='{}_out_add'.format(name))

def u2net_2d_base(input_tensor, 
                      filter_num_down, filter_num_up, 
                      filter_mid_num_down, filter_mid_num_up, 
                      filter_4f_num, filter_4f_mid_num, activation='ReLU',
                      batch_norm=False, pool=True, unpool=True, name='u2net'):
    
    '''
    The base of U^2-Net
    
    u2net_2d_base(input_tensor, 
                  filter_num_down, filter_num_up, 
                  filter_mid_num_down, filter_mid_num_up, 
                  filter_4f_num, filter_4f_mid_num, activation='ReLU',
                  batch_norm=False, pool=True, unpool=True, name='u2net')
    
    ----------
    Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O.R. and Jagersand, M., 2020. 
    U2-Net: Going deeper with nested U-structure for salient object detection. 
    Pattern Recognition, 106, p.107404.
    
    Input
    ----------
        input_tensor: the input tensor of the base, e.g., keras.layers.Inpyt((None, None, 3))
        filter_num_down: a list that defines the number of RSU output filters for each 
                         downsampling level. e.g., `[64, 128, 256, 512]`.
                         the network depth is expected as `len(filter_num_down) + len(filter_4f_num)`                         
        filter_mid_num_down: a list that defines the number of RSU intermediate filters for each 
                             downsampling level. e.g., `[16, 32, 64, 128]`.
                             * RSU intermediate and output filters must paired, i.e., list with the same length.
                             * RSU intermediate filters numbers are expected to be smaller than output filters numbers.
        filter_mid_num_up: a list that defines the number of RSU intermediate filters for each 
                           upsampling level. e.g., `[16, 32, 64, 128]`.
                           * RSU intermediate and output filters must paired, i.e., list with the same length.
                           * RSU intermediate filters numbers are expected to be smaller than output filters numbers.
        filter_4f_num: a list that defines the number of RSU-4F output filters for each 
                       downsampling and bottom level. e.g., `[512, 512]`.
                       the network depth is expected as `len(filter_num_down) + len(filter_4f_num)`.
        filter_4f_mid_num: a list that defines the number of RSU-4F intermediate filters for each 
                           downsampling and bottom level. e.g., `[256, 256]`.
                           * RSU-4F intermediate and output filters must paired, i.e., list with the same length.
                           * RSU-4F intermediate filters numbers are expected to be smaller than output filters numbers.    
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.
        deep_supervision: True for a model that supports deep supervision. Details see Qin et al. (2020).
        name: prefix of the created keras layers.
        
    Output
    ----------
        A list of tensors with the first/second/third tensor obtained from 
        the deepest/second deepest/third deepest upsampling block, etc.
        * The feature map sizes of these tensors are different, 
          with first tensor has the smallest size.
        
    * Dilation rates of RSU4F layers are fixed to `[1, 2, 4, 8]`.
    * Downsampling is achieved through maxpooling in Qin et al. (2020), 
      and can be replaced by strided convolutional layers here.
    * Upsampling is achieved through bilinear interpolation in Qin et al. (2020), 
      and can be replaced by transpose convolutional layers here.
    
    '''
    
    pool_size = 2
    
    X_skip = []; X_out = []; OUT_stack = []
    depth_backup = []
    depth_ = len(filter_num_down)
    
    X = input_tensor
    
    X = RSU(X, filter_mid_num_down[0], filter_num_down[0], depth=depth_+1, activation=activation, 
            batch_norm=batch_norm, pool=pool, unpool=unpool, name='{}_in'.format(name))
    X_skip.append(X)
    
    depth_backup.append(depth_+1)
    
    for i, f in enumerate(filter_num_down[1:]):
        
        X = encode_layer(X, f, pool_size, pool, activation=activation, 
                         batch_norm=batch_norm, name='{}_encode_{}'.format(name, i))
        
        X = RSU(X, filter_mid_num_down[i+1], f, depth=depth_-i, activation=activation, 
                batch_norm=batch_norm, pool=pool, unpool=unpool, name='{}_down_{}'.format(name, i))
        
        depth_backup.append(depth_-i)
        
        X_skip.append(X)

    for i, f in enumerate(filter_4f_num):
        
        X = encode_layer(X, f, pool_size, pool, activation=activation, 
                         batch_norm=batch_norm, name='{}_encode_4f_{}'.format(name, i))
        
        X = RSU4F(X, filter_4f_mid_num[i], f, activation=activation, 
                  batch_norm=batch_norm, name='{}_down_4f_{}'.format(name, i))
        X_skip.append(X)
        
    X_out.append(X)
    
    # ---------- #
    X_skip = X_skip[:-1][::-1]
    depth_backup = depth_backup[::-1]
    
    filter_num_up = filter_num_up[::-1]
    filter_mid_num_up = filter_mid_num_up[::-1]
    
    filter_4f_num = filter_4f_num[:-1][::-1]
    filter_4f_mid_num = filter_4f_mid_num[:-1][::-1]
    
    tensor_count = 0
    for i, f in enumerate(filter_4f_num):
        
        X = decode_layer(X, f, pool_size, unpool, 
                         activation=activation, batch_norm=batch_norm, name='{}_decode_4f_{}'.format(name, i))
        
        X = concatenate([X, X_skip[tensor_count]], axis=-1, name='{}_concat_4f_{}'.format(name, i))
        
        X = RSU4F(X, filter_4f_mid_num[i], f, activation=activation, 
                  batch_norm=batch_norm, name='{}_up_4f_{}'.format(name, i))
        X_out.append(X)
        
        tensor_count += 1
    
    for i, f in enumerate(filter_num_up):
        
        X = decode_layer(X, f, pool_size, unpool, 
                         activation=activation, batch_norm=batch_norm, name='{}_decode_{}'.format(name, i))
        
        X = concatenate([X, X_skip[tensor_count]], axis=-1, name='{}_concat_{}'.format(name, i))
        
        X = RSU(X, filter_mid_num_up[i], f, depth=depth_backup[i], 
                activation=activation, batch_norm=batch_norm, pool=pool, unpool=unpool, name='{}_up_{}'.format(name, i))
        X_out.append(X)
        
        tensor_count += 1

    return X_out


def u2net_2d(input_size, n_labels, filter_num_down, filter_num_up='auto', filter_mid_num_down='auto', filter_mid_num_up='auto', 
             filter_4f_num='auto', filter_4f_mid_num='auto', activation='ReLU', output_activation='Sigmoid', 
             batch_norm=False, pool=True, unpool=True, deep_supervision=False, name='u2net'):
    
    '''
    U^2-Net
    
    u2net_2d(input_size, n_labels, filter_num_down, filter_num_up='auto', filter_mid_num_down='auto', filter_mid_num_up='auto', 
             filter_4f_num='auto', filter_4f_mid_num='auto', activation='ReLU', output_activation='Sigmoid', 
             batch_norm=False, deep_supervision=False, name='u2net')
    
    ----------
    Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O.R. and Jagersand, M., 2020. 
    U2-Net: Going deeper with nested U-structure for salient object detection. 
    Pattern Recognition, 106, p.107404.
    
    Input
    ----------
        input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
        filter_num_down: a list that defines the number of RSU output filters for each 
                         downsampling level. e.g., `[64, 128, 256, 512]`.
                         the network depth is expected as `len(filter_num_down) + len(filter_4f_num)`                         
        filter_mid_num_down: a list that defines the number of RSU intermediate filters for each 
                             downsampling level. e.g., `[16, 32, 64, 128]`.
                             * RSU intermediate and output filters must paired, i.e., list with the same length.
                             * RSU intermediate filters numbers are expected to be smaller than output filters numbers.
        filter_mid_num_up: a list that defines the number of RSU intermediate filters for each 
                           upsampling level. e.g., `[16, 32, 64, 128]`.
                           * RSU intermediate and output filters must paired, i.e., list with the same length.
                           * RSU intermediate filters numbers are expected to be smaller than output filters numbers.
        filter_4f_num: a list that defines the number of RSU-4F output filters for each 
                       downsampling and bottom level. e.g., `[512, 512]`.
                       the network depth is expected as `len(filter_num_down) + len(filter_4f_num)`.
        filter_4f_mid_num: a list that defines the number of RSU-4F intermediate filters for each 
                           downsampling and bottom level. e.g., `[256, 256]`.
                           * RSU-4F intermediate and output filters must paired, i.e., list with the same length.
                           * RSU-4F intermediate filters numbers are expected to be smaller than output filters numbers.         
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        output_activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interface or 'Sigmoid'.
                           Default option is 'Softmax'.
                           if None is received, then linear activation is applied.           
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.  
        deep_supervision: True for a model that supports deep supervision. Details see Qin et al. (2020).
        name: prefix of the created keras layers.
        
    Output
    ----------
        model: a keras model.
    
    * Automated hyper-parameter estimation will produce a slightly larger network, different from that of Qin et al. (2020).
    * Dilation rates of RSU4F layers are fixed to `[1, 2, 4, 8]`.
    * The default output activation is sigmoid, the same as in Qin et al. (2020).
    * Downsampling is achieved through maxpooling and can be replaced by strided convolutional layers.
    * Upsampling is achieved through bilinear interpolation and can be replaced by transpose convolutional layers.
    
    '''
    
    verbose = False
    
    if filter_num_up == 'auto':
        verbose = True
        filter_num_up = filter_num_down
        
    if filter_mid_num_down == 'auto':
        verbose = True
        filter_mid_num_down = [num//4 for num in filter_num_down]
        
    if filter_mid_num_up == 'auto':
        verbose = True
        filter_mid_num_up = filter_mid_num_down
        
    if filter_4f_num == 'auto':
        verbose = True
        filter_4f_num = [filter_num_down[-1], filter_num_down[-1]]
        
    if filter_4f_mid_num == 'auto':
        verbose = True
        filter_4f_mid_num = [num//2 for num in filter_4f_num]
        
    if verbose:
        print('Automated hyper-parameter determination is applied with the following details:\n----------')
        print('\tNumber of RSU output channels within downsampling blocks: filter_num_down = {}'.format(filter_num_down))
        print('\tNumber of RSU intermediate channels within downsampling blocks: filter_mid_num_down = {}'.format(filter_mid_num_down))
        print('\tNumber of RSU output channels within upsampling blocks: filter_num_up = {}'.format(filter_num_up))
        print('\tNumber of RSU intermediate channels within upsampling blocks: filter_mid_num_up = {}'.format(filter_mid_num_up))        
        print('\tNumber of RSU-4F output channels within downsampling and bottom blocks: filter_4f_num = {}'.format(filter_4f_num))
        print('\tNumber of RSU-4F intermediate channels within downsampling and bottom blocks: filter_4f_num = {}'.format(filter_4f_mid_num))
        print('----------\nExplicitly specifying keywords listed above if their "auto" settings do not satisfy your needs')
        
    print("----------\nThe depth of u2net_2d = len(filter_num_down) + len(filter_4f_num) = {}".format(len(filter_num_down)+len(filter_4f_num)))
    
    X_skip = []; X_out = []; OUT_stack = []
    depth_backup = []
    depth_ = len(filter_num_down)
    
    IN = Input(shape=input_size) 
    
    # base (before conv + activation + upsample)
    X_out = u2net_2d_base(IN, 
                          filter_num_down, filter_num_up, 
                          filter_mid_num_down, filter_mid_num_up, 
                          filter_4f_num, filter_4f_mid_num, activation=activation, 
                          batch_norm=batch_norm, pool=pool, unpool=unpool, name=name)
    
    # output layers
    X_out = X_out[::-1]
    L_out = len(X_out)
    
    X = CONV_output(X_out[0], n_labels, kernel_size=3, activation=output_activation, 
                    name='{}_output_sup0'.format(name))
    OUT_stack.append(X)
    
    for i in range(1, L_out):
        
        pool_size = 2**(i)
        
        X = Conv2D(n_labels, 3, padding='same', name='{}_output_conv_{}'.format(name, i))(X_out[i])
        
        X = decode_layer(X, n_labels, pool_size, unpool, 
                         activation=None, batch_norm=False, name='{}_sup{}'.format(name, i))
        
        if output_activation:
            if output_activation == 'Sigmoid':
                X = Activation('sigmoid', name='{}_output_sup{}_activation'.format(name, i))(X)
            else:
                activation_func = eval(output_activation)
                X = activation_func(name='{}_output_sup{}_activation'.format(name, i))(X)
                
        OUT_stack.append(X)
        
    D = concatenate(OUT_stack, axis=-1, name='{}_output_concat'.format(name))
    
    D = CONV_output(D, n_labels, kernel_size=1, activation=output_activation, 
                    name='{}_output_final'.format(name))
    
    if deep_supervision:
        
        OUT_stack.append(D)
        print('----------\ndeep_supervision = True\nnames of output tensors are listed as follows ("sup0" is the shallowest supervision layer;\n"final" is the final output layer):\n')
        
        if output_activation == None:
            if unpool is False:
                for i in range(L_out):
                    print('\t{}_output_sup{}_trans_conv'.format(name, i))
            else:
                for i in range(L_out):
                    print('\t{}_output_sup{}_unpool'.format(name, i))
                
            print('\t{}_output_final'.format(name))
        
        else:        
            for i in range(L_out):
                print('\t{}_output_sup{}_activation'.format(name, i))
                
            print('\t{}_output_final_activation'.format(name))
            
        model = Model([IN,], OUT_stack)
        
    else:
        model = Model([IN,], [D,])
        
    return model
