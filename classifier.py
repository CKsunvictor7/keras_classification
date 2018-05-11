"""
###Template of Keras Classifier###

classifier of Food / Not-Food,
input: img
output: 0 = Not-Food, 1 = Food

"""

import os
os.environ['KERAS_BACKEND']='tensorflow'

import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import model_from_json
from keras import backend as K

model_path = os.path.join(os.path.sep, 'mnt2', 'models', 'Food_NFood', 'Inceptionv3_Mix.hdf5')
network_path = os.path.join(os.path.sep, 'mnt2', 'models', 'Food_NFood', 'Inceptionv3.json')
network_path = os.path.abspath(network_path)
nb_classes = 2
input_size = (229, 229)
img_channels = 3
batch_size = 128


# Image Augmentation
auggen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening, require very long time to fit the samples
    rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
    shear_range=0.01,
    zoom_range=0.01,
    width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images


def load_preprocess_img(path):
    try:
        img = img_to_array(load_img(path, target_size=input_size))
        # do aug
        # img = auggen.random_transform(img)
        # reshape to ()
        if K.image_data_format() == 'channels_first':  # (channels, rows, cols)
            img = img.reshape(img_channels, input_size[0], input_size[1])
            # for single image
            # img = img.reshape(1, img_channels, input_size[0], input_size[1])
        else:
            img = img.reshape(input_size[0], input_size[1], img_channels)
            # for single image
            # img = img.reshape(1, input_size[0], input_size[1], img_channels)
        return img
    except:
        return None


def read_imglist(super_dir):
    img_file_list = []
    for dir in os.listdir(super_dir):
        dir_path = os.path.join(super_dir, dir)
        if os.path.isdir(dir_path):
            img_file_list += [os.path.join(dir_path, f)
                              for f in os.listdir(dir_path)
                              if f.endswith(('.jpg', 'jpeg', '.png', '.bmp', '.JPG', 'JPEG', '.PNG', '.BMP'))]
    return img_file_list

def tester():
    for f in os.listdir('/mnt/dc/web_food_imgs/chūkadon'):
        with open('/mnt2/projects/food_nonfood/annam/prediction.csv', 'a+') as w:
            path = os.path.join('/mnt/dc/web_food_imgs/chūkadon',f)
            l = path.split(os.path.sep)
            print(l)
            # UnicodeEncodeError: 'ascii' codec can't encode characters in position 2-3: ordinal not in range(128)
            w.write('{}\n'.format(os.path.join(l[-2],l[-1])))


def main():
    img_file_list = read_imglist('/mnt/dc/web_food_imgs')
    nb_img = len(img_file_list)
    print('nb of images is ', nb_img)

    model = model_from_json(open(network_path).read())
    model.load_weights(model_path)
    opt_Adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=opt_Adam, metrics=['accuracy'],
                  loss='binary_crossentropy')

    batch_idx = 0
    nb_batch = round(nb_img/batch_size)
    while batch_idx*batch_size < len(img_file_list):
        print('batch {}/{}'.format(batch_idx+1, nb_batch))
        # even index exceeds bound, it will take the last one, won't cause error
        batch_img_file_list = img_file_list[(batch_idx * batch_size):(batch_idx + 1) * batch_size]
        batch_img = []
        for f in batch_img_file_list:
            img = load_preprocess_img(f)
            # add to batch if img is not None
            if img is not None:
                batch_img.append(img)
        batch_img = np.asarray(batch_img)

        # probs ＝ np.array
        probs = model.predict(batch_img, batch_size=batch_size)
        probs_list = np.max(probs, axis=1)
        preds_list = np.argmax(probs, axis=1)
        with open('/mnt2/projects/food_nonfood/annam/prediction.csv', 'a+') as w:
            for idx_f, f in enumerate(batch_img_file_list):
                # only record the nonfood images
                if preds_list[idx_f] == 0:
                    l = f.split(os.path.sep)
                    try:
                        w.write('{}\n'.format(os.path.join(l[-2], l[-1])))
                        print('{},{:.4f},{}'.format(os.path.join(l[-2], l[-1]), probs_list[idx_f], preds_list[idx_f]))
                    except:
                        # TODO UnicodeEncodeError: 'ascii' codec can't encode characters in position 2-3: ordinal not in range(128)
                        w.write('sth wrong during output\n')
                        print('sth wrong during output')

        batch_idx = batch_idx + 1

    print('prediction done')



if __name__ == '__main__':
    main()