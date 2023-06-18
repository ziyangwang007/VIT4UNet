import os
import numpy as np
import SimpleITK as sitk
import cv2
import skimage.morphology as sm
import random

_dir = os.getcwd()
os.chdir(_dir)

train_data_path = os.path.join(_dir, 'train/')
mask_data_path = os.path.join(_dir, 'masks/')
test_data_path = os.path.join(_dir, 'test/')

image_rows = int(256)
image_cols = int(256)
image_depth = int(16)

case_depth = [559,507,560,625,601,562,509,548,572,552] # depth of each cases, 10 cases in total
case_depth = list(map(int,case_depth))

case_total = 10 # 9 case for train, 1 case for test
case_train_total = 9
case_test_total = 1

case_mask_total = 10


def create_train_data():

    print('-'*30)
    print('train: Convert mhd file to numpy file...')
    print('-'*30)

    print('-'*30)
    print('Loading of train data...')
    print('-'*30)

    os.chdir(train_data_path)

    img = np.ndarray((559, 512, 512), dtype=np.uint8)

    cases = os.path.join(train_data_path, 'case1.mhd')
    img = sitk.ReadImage(cases)
    img = sitk.GetArrayFromImage(img)
    img = img.astype(np.uint8)


    for i in range(1,9):

        print('process {}/{} case...'.format(i+1,case_train_total))

        imgtemp = np.ndarray((case_depth[i], image_rows, image_cols), dtype=np.uint8)

        cases = os.path.join(train_data_path, 'case{}.mhd'.format(i+1))
        imgtemp = sitk.ReadImage(cases)
        imgtemp = sitk.GetArrayFromImage(imgtemp)
        imgtemp = imgtemp.astype(np.uint8)
        img = np.concatenate((img,imgtemp))



    img_small = np.ndarray((5043,256,256), dtype=np.uint8)
    for i in range(5043):
        imgtemp = img[i,:,:]
        img_small_temp = cv2.resize(imgtemp,(256,256),interpolation=cv2.INTER_CUBIC)
        img_small[i,:,:] = img_small_temp

    img = img_small

    print('resize 512 to 256')
    print(np.shape(img))


    print('-'*30)
    print('Saving to .npy files...')
    print('-'*30)
    # img = img.astype(bool)
    # img = np.where(img>0,255,0)
    np.save('train2D.npy', img)






    

def create_test_data():
    os.chdir(test_data_path)

    print('-'*30)
    print('test: Convert mhd file to numpy file...')
    print('-'*30)

    print('-'*30)
    print('Loading of test data...')
    print('-'*30)

    img = np.ndarray((552, image_rows, image_cols), dtype=np.uint8)
    dirr = os.path.join(test_data_path, 'case10.mhd')
    img = sitk.ReadImage(dirr)
    img = sitk.GetArrayFromImage(img)
    img = img.astype(np.uint8)


    img_small = np.ndarray((552,256,256), dtype=np.uint8)
    for i in range(552):
        imgtemp = img[i,:,:]
        img_small_temp = cv2.resize(imgtemp,(256,256),interpolation=cv2.INTER_CUBIC)
        img_small[i,:,:] = img_small_temp

    img = img_small

    print('resize 512 to 256')
    print(np.shape(img))


    print('-'*30)
    print('Saving to .npy files...')
    print('-'*30)
    # img = img.astype(bool)
    # img = np.where(img>0,255,0)
    np.save('test2D.npy', img)




def create_mask_data():
    print('-'*30)
    print('mask: Convert mhd file to numpy file...')
    print('-'*30)

    print('-'*30)
    print('Loading of mask data...')
    print('-'*30)

    os.chdir(mask_data_path)

    img = np.ndarray((559, image_rows, image_cols), dtype=np.uint8)

    cases = os.path.join(mask_data_path, 'case1_label.mhd')
    img = sitk.ReadImage(cases)
    img = sitk.GetArrayFromImage(img)
    img = img.astype(np.uint8)

    for i in range(1,9):
        print('process {}/{} case...'.format(i+1, case_mask_total))
        # print(np.shape(img))

        imgtemp = np.ndarray((case_depth[i], image_rows, image_cols), dtype=np.uint8)

        cases = os.path.join(mask_data_path, 'case{}_label.mhd'.format(i+1))
        imgtemp = sitk.ReadImage(cases)
        imgtemp = sitk.GetArrayFromImage(imgtemp)
        imgtemp = imgtemp.astype(np.uint8)
        img = np.concatenate((img,imgtemp))
   

    img_small = np.ndarray((5043,256,256), dtype=np.uint8)
    for i in range(5043):
        imgtemp = img[i,:,:]
        img_small_temp = cv2.resize(imgtemp,(256,256),interpolation=cv2.INTER_CUBIC)
        img_small[i,:,:] = img_small_temp

    img = img_small

    print('resize 512 to 256')
    print(np.shape(img))


    print('-'*30)
    print('Saving to .npy files...')
    print('-'*30)
    # img = img.astype(bool)
    img = np.where(img>0,255,0)
    np.save('mask_train2D.npy', img) 





# create  noisy label


    noisylabellisterosion = random.sample(range(0,2000),1200)
    noisylabellistdilation = random.sample(range(2000,4000),1200)
    noisylabellistelastictransformation  = random.sample(range(4000,5043),450)

    noisy=20

    for i in noisylabellisterosion:
        print(i,'erosion')
        imgtemp = img[i,:,:]
        imgtemp = sm.erosion(imgtemp,sm.square(noisy))
        img[i,:,:] = imgtemp


    for i in noisylabellistdilation:
        print(i,'dilation')
        imgtemp = img[i,:,:]
        imgtemp = sm.dilation(imgtemp,sm.square(noisy))
        img[i,:,:] = imgtemp
     
    for i in noisylabellistelastictransformation:
        print(i,'transformation ')
        imgtemp = img[i,:,:]
        A = imgtemp.shape[0] / 3.0
        w = 2.0 / imgtemp.shape[1]
        shift = lambda x: A * np.sin(1.5*np.pi*x * w)
        for k in range(imgtemp.shape[0]):
            imgtemp[:,k] = np.roll(imgtemp[:,k], int(shift(k)))
        img[i,:,:] = imgtemp



    np.save('mask_train2D_V3.npy', img)




    cases = os.path.join(mask_data_path, 'case10_label.mhd')
    img = sitk.ReadImage(cases)
    img = sitk.GetArrayFromImage(img)
    img = img.astype(np.uint8)


    img_small = np.ndarray((552,256,256), dtype=np.uint8)
    for i in range(552):
        imgtemp = img[i,:,:]
        img_small_temp = cv2.resize(imgtemp,(256,256),interpolation=cv2.INTER_CUBIC)
        img_small[i,:,:] = img_small_temp

    img = img_small

    print('resize 512 to 256')
    print(np.shape(img))
    

    print('-'*30)
    print('Saving to .npy files... ')
    print('-'*30)
    # img = img.astype(bool)
    img = np.where(img>0,255,0)
    np.save('mask_test2D.npy', img)




def load_train_data():
    os.chdir(train_data_path)
    imgs_train = np.load('train2D.npy')
    imgs_train = np.expand_dims(imgs_train,axis=-1)   #channel dimension
    # print(np.shape(imgs_train))
    # imgs_train = imgs_train[None,:,:,:]
    return imgs_train

def load_mask_train_original():
    os.chdir(mask_data_path)
    imgs_mask_train_original = np.load('mask_train2D.npy')
    imgs_mask_train_original = np.expand_dims(imgs_mask_train_original,axis=-1)
    # imgs_mask_train = imgs_mask_train[None,:,:,:]
    return imgs_mask_train_original

def load_mask_train_noisy():
# def load_mask_train():
    os.chdir(mask_data_path)
    imgs_mask_train_noisy = np.load('mask_train2D_V1.npy')
    imgs_mask_train_noisy = np.expand_dims(imgs_mask_train_noisy,axis=-1)
    # imgs_mask_train = imgs_mask_train[None,:,:,:]
    return imgs_mask_train_noisy

def load_test_data():
    os.chdir(test_data_path)
    imgs_test = np.load('test2D.npy')
    imgs_test = np.expand_dims(imgs_test,axis=-1)
    # imgs_test = imgs_test[None,:,:,:]
    return imgs_test

def load_mask_test():
    os.chdir(mask_data_path)
    imgs_mask_test = np.load('mask_test2D.npy')
    imgs_mask_test = np.expand_dims(imgs_mask_test,axis=-1)
    # print(np.shape(imgs_mask_test))
    # imgs_mask_train = imgs_mask_train[None,:,:,:]
    return imgs_mask_test

def preprocess_squeeze(imgs):
    imgs = np.squeeze(imgs, axis=3)
    print(' ---------------- preprocessed squeezed -----------------')
    return imgs


if __name__ == '__main__':
    create_train_data()
    create_test_data()
    create_mask_data()
