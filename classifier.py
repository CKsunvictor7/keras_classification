"""
###Template of Keras Classifier###

input: dir of images to be classified
output: image_path, probability, classification result in csv
"""
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.optimizers import Adam
from keras.models import model_from_json
from keras import backend as K

os.environ['KERAS_BACKEND'] = 'tensorflow'
MODEL_PATH = os.path.join(os.path.sep, '__.hdf5')
NETWORK_PATH = os.path.join(os.path.sep, '__.json')
IMG_DIR = os.path.join(os.path.sep, '')
RESULT_CSV_PATH = os.path.join(os.path.sep, '_.csv')
NB_CLASSES = 2
INPUT_SIZE = (229, 229)
CHANNELS_NUM = 3
BATCH_SIZE = 128


# Image Augmentation
AUG_GENERATOR = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening, require very long time to fit the samples
    # randomly rotate images in the range (degrees, 0 to 180)
    rotation_range=180,
    shear_range=0.01,
    zoom_range=0.01,
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.05,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.05,
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images


def read_imglist(super_dir):
    """
    :param super_dir: dir of images to be classified
    :return: list of abs image path
    """
    img_file_list = []
    for dir in os.listdir(super_dir):
        dir_path = os.path.join(super_dir, dir)
        if os.path.isdir(dir_path):
            img_file_list += [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                               if f.endswith(('.jpg', 'jpeg', '.png', '.bmp',
                                              '.JPG', 'JPEG', '.PNG', '.BMP'))]
    return img_file_list


def load_preprocess_img(path):
    """
    :param path: path of image
    :return:  reshaped image array
    """
    try:
        img = img_to_array(load_img(path, target_size=INPUT_SIZE))
        # do aug
        # img = auggen.random_transform(img)
        if K.image_data_format() == 'channels_first':  # (channels, rows, cols)
            img = img.reshape(CHANNELS_NUM, INPUT_SIZE[0], INPUT_SIZE[1])
            # for single image
            # img = img.reshape(1, img_channels, input_size[0], input_size[1])
        else:
            img = img.reshape(INPUT_SIZE[0], INPUT_SIZE[1], CHANNELS_NUM)
            # for single image
            # img = img.reshape(1, input_size[0], input_size[1], img_channels)
        return img
    except BaseException:
        return None


def main():
    img_file_list = read_imglist(IMG_DIR)
    nb_img = len(img_file_list)
    print('nb of images is ', nb_img)

    model = model_from_json(open(NETWORK_PATH).read())
    model.load_weights(MODEL_PATH)
    opt_adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=opt_adam, metrics=['accuracy'],
                  loss='binary_crossentropy')

    batch_idx = 0
    nb_batch = round(nb_img / BATCH_SIZE)
    while batch_idx * BATCH_SIZE < len(img_file_list):
        print('batch {}/{}'.format(batch_idx + 1, nb_batch))
        # even index exceeds bound, it will take the last one, won't cause error
        batch_img_file_list = img_file_list[(
            batch_idx * BATCH_SIZE):(batch_idx + 1) * BATCH_SIZE]
        batch_img = []
        for img_path in batch_img_file_list:
            img = load_preprocess_img(img_path)
            # add to batch if img is not None
            if img is not None:
                batch_img.append(img)
        batch_img = np.asarray(batch_img)

        # probs ＝ np.array
        probs = model.predict(batch_img, batch_size=BATCH_SIZE)
        probs_list = np.max(probs, axis=1)
        preds_list = np.argmax(probs, axis=1)

        with open(RESULT_CSV_PATH, 'a+') as writer:
            for idx_f, img_path in enumerate(batch_img_file_list):
                img_path_pieces = img_path.split(os.path.sep)
                try:
                    writer.write('{}\n'.format(os.path.join(img_path_pieces[-2],
                                                       img_path_pieces[-1])))
                    # image_path, probability, classification result
                    print('{},{:.4f},{}'.format(os.path.join(
                        img_path_pieces[-2], img_path_pieces[-1]),
                        probs_list[idx_f], preds_list[idx_f]))
                except BaseException:
                    writer.write('sth wrong during output\n')
                    print('sth wrong during output')
        batch_idx = batch_idx + 1

    print('prediction done')


if __name__ == '__main__':
    main()
