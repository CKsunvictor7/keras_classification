"""
###Template of Keras model Trainer###
Food and Non-Food classification on Large Datasets


"""
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
# from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3, conv2d_bn
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, History, BaseLogger, Callback, ReduceLROnPlateau, CSVLogger
from keras import regularizers
from keras.layers import Input
from keras import layers
from keras import backend as K
from keras.utils.np_utils import to_categorical
from tea_coffee_black_tea import category_list
from math import ceil

import argparse
# python Food_NFood_generator.py -o 2 -n Xcept_Mix_2.json -m Xcept_Mix_2.hdf5 -l Xcept_Mix_2.csv

"""
python Food_NFood_generator.py -o 1 -n Inceptionv3.json -m Inceptionv3_Mix.hdf5 -l Inceptionv3_Mix.csv
python Food_NFood_generator.py -o 2 -n xinception.json -m Xcept_Mix.hdf5 -l Xcept_Mix.csv
python Food_NFood_generator.py -o 3 -n ResNet50.json -m ResNet50_Mix.hdf5 -l ResNet50_Mix.csv
"""

parser = argparse.ArgumentParser(description='~~~')
parser.add_argument(
    '-o',
    '--option',
    help='1=Inceptionv3, 2=xinception 3=ResNet50',
    type=int)
parser.add_argument('-n', '--network_f', help='network_filename', type=str)
parser.add_argument('-m', '--model_f', help='model_filename', type=str)
parser.add_argument('-l', '--log_f', help='log_filename', type=str)
args = parser.parse_args()

option = args.option
model_path = args.model_f  # 'Xception_mixed_1.hdf5'
network_path = args.network_f  # 'Xception_mixed.json'
log_path = args.log_f  # 'log_Food_NonFood_mixed_x1.csv'
nb_classes = 0
input_size = (229, 229)
img_channels = 3
total_epochs = 15
batch_size = 32


def suffle_data(id_list, label_list):
    nb_data = len(id_list)
    shuffled_index = np.random.permutation(
        np.arange(nb_data))  # make a shuffle idx array
    shuffled_list = []
    shuffled_label_list = []
    for idx in shuffled_index:
        shuffled_list.append(id_list[idx])
        shuffled_label_list.append(label_list[idx])

    return shuffled_list, shuffled_label_list


def split_data(ids, split_percentage=20, stratified=True):
    num_all = len(ids)

    shuffled_index = np.random.permutation(
        np.arange(num_all))  # make a shuffle idx array

    # calcualate the train & validation index
    num_small = int(num_all // (100 / split_percentage))
    num_big = num_all - num_small

    ix_big = shuffled_index[:num_big]
    ix_small = shuffled_index[num_big:]

    # divide
    id_big = []
    for idx in ix_big:
        id_big.append(ids[idx])
    id_small = []
    for idx in ix_small:
        id_small.append(ids[idx])

    print('num of id_big = ', len(id_big))
    print('num of id_small = ', len(id_small))

    # y_valid must be np array as 'int64'
    return id_big, id_small

# Image Augmentation
auggen = ImageDataGenerator(
    featurewise_center=True,  # set input mean to 0 over the dataset
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
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True)  # randomly flip images


class DataGenerator():
    """
    general image data generator,

    how to use it?
    params = {'dim_x': 32,
          'dim_y': 32,
          'batch_size': 32,
          'do_shuffle': True}
    training_generator = DataGenerator(**params).generate(img_list, label_list)
    """
    def __init__(self, dim_x, dim_y, batch_size=32, do_shuffle=True):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.batch_size = batch_size
        self.do_shuffle = do_shuffle

    def generator(self, img_list, label_list):
        """
        generator of data

        :param img_list:  list of absolute img path
        :param label_list:  list of label, corresponding to img
        :return: a batch size data (img(np.array), label(?))
        """
        assert len(label_list) == len(img_list), 'the length of img_list and label_list should be the same'

        nb_of_batch_each_epoch = ceil(len(label_list) / float(self.batch_size))
        while True:
            # get new index order each epoch
            idxs = self.__get_idxs_of_this_epoch(len(label_list))
            for i in range(nb_of_batch_each_epoch):
                tmp_img_list = [img_list[k] for k in idxs[i * self.batch_size:(i + 1) * self.batch_size]]
                tmp_label_list = [
                    label_list[k] for k in idxs[i * self.batch_size:(i + 1) * self.batch_size]]
                # TODO: know the shape of label & one hot
                # keras.utils.np_utils.to_categorical

                yield self.__read_one_batch_data(tmp_img_list), to_categorical(tmp_label_list, num_classes=nb_classes)

    def __get_idxs_of_this_epoch(self, nb_of_data):
        idxs = np.arange(nb_of_data)
        if self.do_shuffle:
            np.random.shuffle(idxs)
        return idxs

    def __read_one_batch_data(self, img_list):
        x = []
        # imgs loading
        for f in img_list:
            img = img_to_array(load_img(f, target_size=input_size))
            # do aug
            img = auggen.random_transform(img)
            x.append(img)
        x = np.asarray(x)

        # reshape to match the format of backend engine
        if K.image_data_format() == 'channels_first':  # (channels, rows, cols)
            x = x.reshape(x.shape[0], img_channels, input_size[0],
                          input_size[1])
        else:
            x = x.reshape(x.shape[0], input_size[0], input_size[1],
                          img_channels)
        return x


def make_dataset(category_list):
    superdir_path = '/mnt/dc/web_food_imgs'
    global nb_classes
    nb_classes = len(category_list)
    category_label_mapping = {}
    train_food_img_list = []
    train_food_img_labels = []
    val_food_img_list = []
    val_food_img_labels = []
    # 1. read all image name list
    for idx, category in enumerate(category_list):
        category_label_mapping[category] = idx
        dir_path = os.path.join(superdir_path, category)
        img_list = [ os.path.join(dir_path, f) for f in os.listdir(dir_path)]
        train_list, val_list= split_data(img_list, split_percentage=10)

        train_food_img_list = train_food_img_list + train_list
        val_food_img_list = val_food_img_list + val_list

        train_food_img_labels = train_food_img_labels + [idx]*len(train_list)
        val_food_img_labels = val_food_img_labels + [idx]*len(val_list)

    print(category_label_mapping)


    # 2. shuffle
    return suffle_data(train_food_img_list, train_food_img_labels), val_food_img_list, val_food_img_labels


def main():
    print('model_path=', model_path)

    (train_data, train_labels), val_data, val_labels  = make_dataset(category_list)
    print('nb of training data ={}, val data = {}'.format(
        len(train_data), len(val_data)))

    """
    # debug
    dataer = DataGenerator(dim_x=input_size[0], dim_y=input_size[1],
                              batch_size=batch_size, do_shuffle=True)
    for x, y in dataer.generator(img_list=train_data, label_list=train_labels):
        print(y)
    exit()
    """

    # TODO: random sampling generator
    x_fit_samples = []
    for index, img_filename in enumerate(train_data):
        img = img_to_array(
            load_img(img_filename, target_size=input_size))
        x_fit_samples.append(img)
        if index > 1000:
            break
    auggen.fit(x_fit_samples)
    # = [[[ 81.7374649   81.7374649   81.59357452]]]
    print("mean = ", auggen.mean)

    food_data = DataGenerator(dim_x=input_size[0], dim_y=input_size[1], batch_size=batch_size, do_shuffle=True)

    # build Model
    # base_model = VGG16(include_top=False, weights='imagenet')
    # base_model = InceptionV3(include_top=False, weights='imagenet')
    base_model = Xception(include_top=False, weights='imagenet')
    if option == 0:
        base_model = InceptionV3(include_top=False, weights='imagenet')
    elif option == 1:
        base_model = Xception(include_top=False, weights='imagenet')
    elif option == 2:
        base_model = ResNet50(include_top=False, weights='imagenet')

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    # add a fully-connected layer
    # x = Dense(256, activation='relu',)(x)
    # and a logistic layer
    predictions = Dense(
        nb_classes,
        kernel_regularizer=regularizers.l2(0.1),
        activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # print network architecture
    model.summary()

    json_string = model.to_json()
    open(network_path, 'w').write(json_string)

    # training
    print("start compiling...")
    opt_Adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(
        optimizer=opt_Adam,
        metrics=['accuracy'],
        loss='categorical_crossentropy')
    EStopping = EarlyStopping(
        monitor='val_loss',  # val_loss
        patience=3,
        verbose=1,
        mode='auto')
    Mcheckpoint = ModelCheckpoint(
        filepath=model_path,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='auto')
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=2,
        min_lr=0.00001)
    csv_logger = CSVLogger(log_path)
    # t0=time.time()
    print('start training...')

    model.fit_generator(
        generator=food_data.generator(img_list=train_data, label_list=train_labels),
        steps_per_epoch=len(train_data) // batch_size,
        epochs=total_epochs,
        verbose=1,
        validation_data=food_data.generator(img_list=val_data, label_list=val_labels),
        validation_steps=len(val_data) // batch_size,
        callbacks=[
            EStopping,
            reduce_lr,
            Mcheckpoint,
            csv_logger],
        max_queue_size=10)

    # evaluation
    # get loss & acc
    eval = model.evaluate_generator(
        food_data.generator(img_list=val_data, label_list=val_labels), steps=len(val_data) / batch_size)
    print("loss = {}, acc = {}".format(eval[0], eval[1]))
    with open(log_path, 'a+') as w:
        w.write(
            "\n loss = {}, acc = {} \n".format(
                eval[0], eval[1]))


if __name__ == '__main__':
    main()
