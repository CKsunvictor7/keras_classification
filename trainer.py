"""
###Template of Keras model Trainer###
Binary classification on Large Datasets


"""
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model, model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping, History, BaseLogger, Callback, ReduceLROnPlateau, CSVLogger
from keras import regularizers
from keras.layers import Input
from keras import layers
from keras import backend as K
from keras.utils.np_utils import to_categorical
from math import ceil
from sklearn.metrics import confusion_matrix
import utils


import argparse
# python Food_NFood_generator.py -o 2 -n Xcept_Mix_2.json -m Xcept_Mix_2.hdf5 -l Xcept_Mix_2.csv

"""
python trainer.py  -o 1 -n Inceptionv3.json -m Inceptionv3_fc.hdf5 -l Inceptionv3_fc.csv
python Food_NFood_generator.py -o 1 -n Inceptionv3.json -m Inceptionv3_Mix.hdf5 -l Inceptionv3_Mix.csv
python Food_NFood_generator.py -o 2 -n xinception.json -m Xcept_Mix.hdf5 -l Xcept_Mix.csv
python Food_NFood_generator.py -o 3 -n ResNet50.json -m ResNet50_Mix.hdf5 -l ResNet50_Mix.csv
"""

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
"""


nb_classes = 0


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
    def __init__(self, input_size, img_channels, batch_size=32, do_shuffle=True):
        self.input_size = input_size
        self.img_channels = img_channels
        self.batch_size = batch_size
        self.do_shuffle = do_shuffle

    def generator(self, img_list, label_list):
        """
        generator of data & label

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
                yield self.__read_one_batch_data(tmp_img_list), to_categorical(tmp_label_list, num_classes=nb_classes)

    def data_generator(self, img_list):
        """
        generator of data

        :param img_list:  list of absolute img path
        :return: a batch size data (img(np.array), label(?))
        """
        nb_of_batch_each_epoch = ceil(len(img_list) / float(self.batch_size))
        while True:
            # get new index order each epoch
            idxs = self.__get_idxs_of_this_epoch(len(img_list))
            for i in range(nb_of_batch_each_epoch):
                tmp_img_list = [img_list[k] for k in idxs[i * self.batch_size:(i + 1) * self.batch_size]]
                yield self.__read_one_batch_data(tmp_img_list)

    def __get_idxs_of_this_epoch(self, nb_of_data):
        idxs = np.arange(nb_of_data)
        if self.do_shuffle:
            np.random.shuffle(idxs)
        return idxs

    def __read_one_batch_data(self, img_list):
        x = []
        # imgs loading
        for f in img_list:
            img = img_to_array(load_img(f, target_size=self.input_size))
            # do aug
            img = auggen.random_transform(img)
            x.append(img)
        x = np.asarray(x)

        # reshape to match the format of backend engine
        if K.image_data_format() == 'channels_first':  # (channels, rows, cols)
            x = x.reshape(x.shape[0], self.img_channels, self.input_size[0],
                          self.input_size[1])
        else:
            x = x.reshape(x.shape[0], self.input_size[0], self.input_size[1],
                          self.img_channels)
        return x


def make_dataset(category_list):
    superdir_path = '/mnt/dc/web_food_imgs_f/'
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
        img_list = [ os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(('.jpg', 'jpeg', '.png', '.bmp', '.JPG', 'JPEG', '.PNG', '.BMP'))]

        # split to train & validation
        train_list, val_list= utils.split_by_KFold(img_list, nb_splits=10)
        train_food_img_list = train_food_img_list + train_list
        val_food_img_list = val_food_img_list + val_list

        train_food_img_labels = train_food_img_labels + [idx]*len(train_list)
        val_food_img_labels = val_food_img_labels + [idx]*len(val_list)

    print(category_label_mapping)

    # 2. shuffle
    return utils.suffle_data(train_food_img_list, train_food_img_labels), val_food_img_list, val_food_img_labels


def files_augmentation(file_list, desired_nb):
    """
    enlarge the size of file_list to desired_nb
    :param file_list:
    :param desired_nb:
    :return:
    """
    while desired_nb/len(file_list) > 2:
        file_list = file_list + file_list
    file_list = file_list + file_list[: desired_nb-len(file_list)]

    assert len(file_list)==desired_nb, 'different {} , {}'.format(len(file_list), desired_nb)
    return file_list


def make_balanced_dataset(category_list):
    superdir_path = '/mnt/dc/web_food_imgs_f/'
    global nb_classes
    nb_classes = len(category_list)
    category_label_mapping = {}
    train_data_list = []
    val_data_list = []

    train_data = []
    train_labels = []
    val_data = []
    val_labels = []

    # 1. read all image name list
    for idx, category in enumerate(category_list):
        img_list = []
        if isinstance(category, list):
            # ['steak', 'sirloin_steak', 'dice_steak', 'fillet_steak'] the 1st element 'steak' is confusion_category name, others are categories
            # for example [['steak', 'sirloin_steak', 'dice_steak', 'fillet_steak'], 'tomato_curry', 'bulgogi', 'roast_beef', 'yakiniku__karubi']
            category_label_mapping[category[0]] = idx
            for sub_c in category[1::]:
                # due to the fold in web_food_imgs_f is divided into '1' and '2'
                dir_path = os.path.join(superdir_path, sub_c, '1')
                img_list = img_list + [os.path.join(dir_path, f) for f in
                            os.listdir(dir_path) if
                            f.endswith(
                                ('.jpg', 'jpeg', '.png', '.bmp', '.JPG', 'JPEG',
                                 '.PNG', '.BMP'))]
                dir_path = os.path.join(superdir_path, sub_c, '2')
                img_list = img_list + [os.path.join(dir_path, f) for f in
                             os.listdir(dir_path)
                             if f.endswith(
                        ('.jpg', 'jpeg', '.png', '.bmp', '.JPG', 'JPEG',
                         '.PNG', '.BMP'))]
        else:
            category_label_mapping[category] = idx
            # due to the fold in web_food_imgs_f is divided into '1' and '2'
            dir_path = os.path.join(superdir_path, category, '1')
            img_list = img_list + [os.path.join(dir_path, f) for f in os.listdir(dir_path) if
                        f.endswith(('.jpg', 'jpeg', '.png', '.bmp', '.JPG', 'JPEG',
                                    '.PNG', '.BMP'))]
            dir_path = os.path.join(superdir_path, category, '2')
            img_list = img_list + [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                        if f.endswith(('.jpg', 'jpeg', '.png', '.bmp', '.JPG', 'JPEG',
                             '.PNG', '.BMP'))]

        print('{} = {}'.format(category, len(img_list)))
        tmp_train_data, tmp_val_data = utils.split_by_KFold(img_list, nb_splits=10)
        train_data_list.append(tmp_train_data)
        val_data_list.append(tmp_val_data)

    print(category_label_mapping)

    # 2. do balance data augmentation
    # for train data
    desired_nb =  max([len(x) for x in train_data_list])
    print('desired_nb=', desired_nb)
    for idx, x in enumerate(train_data_list):
        train_data = train_data + files_augmentation(x, desired_nb)
        train_labels = train_labels + [idx]*desired_nb

    # for val data
    desired_nb = max([len(x) for x in val_data_list])
    print('desired_nb=', desired_nb)
    for idx, x in enumerate(val_data_list):
        val_data = val_data + files_augmentation(x, desired_nb)
        val_labels = val_labels + [idx] * desired_nb

    # reform the categories
    merged_categories = []
    for k, v in category_label_mapping.items():
        merged_categories.append(k)

    # 3. shuffle
    return utils.suffle_data(train_data, train_labels), val_data, val_labels, merged_categories


def make_balanced_dataset_(category_list):
    superdir_path = '/mnt/dc/web_food_imgs/'
    global nb_classes
    nb_classes = len(category_list)
    category_label_mapping = {}
    train_data_list = []
    val_data_list = []

    train_data = []
    train_labels = []
    val_data = []
    val_labels = []


    # 1. read all image name list
    for idx, category in enumerate(category_list):
        category_label_mapping[category] = idx
        dir_path = os.path.join(superdir_path, category)
        # img_list_list.append([ os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(('.jpg', 'jpeg', '.png', '.bmp', '.JPG', 'JPEG', '.PNG', '.BMP'))])
        img_list = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if
                    f.endswith(('.jpg', 'jpeg', '.png', '.bmp', '.JPG', 'JPEG',
                                '.PNG', '.BMP'))]
        tmp_train_data, tmp_val_data = utils.split_by_KFold(img_list, nb_splits=10)
        train_data_list.append(tmp_train_data)
        val_data_list.append(tmp_val_data)

    print(category_label_mapping)

    # 2. do balance data augmentation
    # for train data
    desired_nb =  max([len(x) for x in train_data_list])
    print('desired_nb=', desired_nb)
    for idx, x in enumerate(train_data_list):
        train_data = train_data + files_augmentation(x, desired_nb)
        train_labels = train_labels + [idx]*desired_nb

    # for val data
    desired_nb = max([len(x) for x in val_data_list])
    print('desired_nb=', desired_nb)
    for idx, x in enumerate(val_data_list):
        val_data = val_data + files_augmentation(x, desired_nb)
        val_labels = val_labels + [idx] * desired_nb

    # 3. shuffle
    return utils.suffle_data(train_data, train_labels), val_data, val_labels


def runner(category_list, input_size, img_channels, nb_epochs, batch_size, val_batch_size, base_model, title='~', network_path='net.json',
           model_path='model.hdf5', log_path='log.csv'):
    # data, label reading
    # (train_data, train_labels), val_data, val_labels  = make_dataset(category_list) # make_dataset(category_list), make_balanced_dataset(category_list)
    (train_data, train_labels), val_data, val_labels, merged_categories = make_balanced_dataset(
        category_list)
    train_labels = train_labels
    val_labels = val_labels

    assert len(train_data) == len(
        train_labels), 'length are different: {} - {}'.format(len(train_data),
                                                              len(train_labels))
    assert len(val_data) == len(
        val_labels), 'length are different: {} - {}'.format(
        len(val_data), len(val_labels))

    print('nb of training data ={}, val data = {}'.format(
        len(train_data), len(val_data)))

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    x_fit_samples = []
    for img_filename in train_data[:1000]:
        img = img_to_array(
            load_img(img_filename, target_size=input_size))
        x_fit_samples.append(img)
    auggen.fit(x_fit_samples)
    # = [[[ 81.7374649   81.7374649   81.59357452]]]
    print("mean = ", auggen.mean)

    # set data generator
    food_data = DataGenerator(input_size, img_channels, batch_size=batch_size, do_shuffle=True)

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
    # model.summary()

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
        patience=5,
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
        patience=3,
        min_lr=0.000001)
    csv_logger = CSVLogger(log_path)
    # t0=time.time()
    print('start training...')

    model.fit_generator(
        generator=food_data.generator(img_list=train_data,
                                      label_list=train_labels),
        steps_per_epoch=ceil(len(train_data)/batch_size),
        epochs=nb_epochs,
        verbose=1,
        validation_data=food_data.generator(img_list=val_data, label_list=val_labels),
        validation_steps=ceil(len(val_data)/batch_size),
        callbacks=[
            EStopping,
            reduce_lr,
            Mcheckpoint,
            csv_logger],
        max_queue_size=10)

    # using val_data & best_model to generate confusion matrix
    best_model = model_from_json(open('net.json').read())
    best_model.load_weights('model.hdf5')
    best_model.compile(
        optimizer=opt_Adam,
        metrics=['accuracy'],
        loss='categorical_crossentropy')

    # TODO: bug generator already executing -> due to multiple threads access generator
    # solution = https://stackoverflow.com/questions/41194726/python-generator-thread-safety-using-keras
    val_food_data = DataGenerator(input_size, img_channels,
                                  batch_size=val_batch_size, do_shuffle=False)
    predictions = best_model.predict_generator(
        generator=val_food_data.data_generator(img_list=val_data),
        steps=ceil(len(val_data)/val_batch_size),
        max_queue_size=10, workers=1
    )

    """
    compute and plot confusion matrix 
    """
    cm = confusion_matrix(y_true=val_labels,
                          y_pred=np.argmax(predictions, axis=1).tolist())
    np.save('cm_matrix_{}'.format(title), cm)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    utils.plot_confusion_matrix(cm, classes=merged_categories,
                                title='cm_{}'.format(title))
    utils.plot_confusion_matrix(cm, classes=merged_categories, normalize=True,
                                title='cm_n_{}'.format(title))

    """
    # evaluation
    # get loss & acc
    eval = model.evaluate_generator(
        food_data.generator(img_list=val_data, label_list=val_labels), steps=len(val_data) / batch_size)
    print("loss = {}, acc = {}".format(eval[0], eval[1]))
    with open(log_path, 'a+') as w:
        w.write(
            "\n loss = {}, acc = {} \n".format(
                eval[0], eval[1]))
    """




