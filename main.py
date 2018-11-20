"""
read config, choose network then launch model training in trainer.py

"""
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from config import training_params
from trainer import runner

NETWORK_LIST = {
    'InceptionV3':InceptionV3(include_top=False, weights='imagenet'),
    'Xception':Xception(include_top=False, weights='imagenet'),
    'ResNet50':ResNet50(include_top=False, weights='imagenet')
}

SUPER_CATEGORY = {
    'exp1': ['cls1', 'cls2']
}


def main():
    for super_name, category_list in SUPER_CATEGORY.items():
        # train the models using networks in network_list
        for _, base_model in NETWORK_LIST.items():
            kwargs = {
                'category_list': category_list,
                'input_size':training_params['input_size'],
                'nb_epochs':training_params['nb_epochs'],
                'batch_size':training_params['batch_size'],
                'val_batch_size':training_params['val_batch_size'],
                'img_channels':training_params['img_channels'],
                'title': '{}'.format(super_name),
                'base_model':base_model,
                'network_path':'net.json',
                'model_path':'model.hdf5',
                'log_path':'log.csv'
            }
            runner(**kwargs)

if __name__ == '__main__':
    main()
