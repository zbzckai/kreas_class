# -*- coding: utf-8 -*-
# @Time    : 2019\7\9 0009 11:15
# @Author  : 凯
# @File    : VGG16_difficult.py

from __future__ import print_function

import glob
import numpy as np
import warnings
import os
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing import image
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from vgg_net.imagenet_utils import decode_predictions, preprocess_input,preprocess_input_image
from keras.models import model_from_json


##生成训练图片
def image_preprocess(train_dir ,validation_dir,IM_WIDTH ,IM_HEIGHT,batch_size):
    #   图片生成器
    # 　训练集的图片生成器，通过参数的设置进行数据扩增
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_image,
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode = 'nearest'
    )
#    '''
#rotation_range是一个0~180的度数，用来指定随机选择图片的角度。
#
#width_shift和height_shift用来指定水平和竖直方向随机移动的程度，这是两个0~1之间的比例。
#
#rescale值将在执行其他处理前乘到整个图像上，我们的图像在RGB通道都是0~255的整数，这样的操作可能使图像的值过高或过低，所以我们将这个值定为0~1之间的数。
#
#shear_range是用来进行剪切变换的程度，参考剪切变换
#
#zoom_range用来进行随机的放大
#
#horizontal_flip随机的对图片进行水平翻转，这个参数适用于水平翻转不影响图片语义的时候
#
#fill_mode用来指定当需要进行像素填充，如旋转，水平和竖直位移时，如何填充新出现的像素
#    '''
    #   验证集的图片生成器，不进行数据扩增，只进行数据预处理
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_image,
    )
    # 训练数据与测试数据
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size, class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size, class_mode='categorical')
    return train_generator, validation_generator

TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5'
TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def VGG16(include_top=True, weights='imagenet',
          input_tensor=None):
    '''
    # Arguments
        include_top: 是否包含最后的三层全连接层
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
    '''
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, 224, 224)
    else:
        input_shape = (224, 224, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)

    # Create model
    model = Model(img_input, x)

    # load weights
    if weights == 'imagenet':
        print('K.image_dim_ordering:', K.image_dim_ordering())
        if K.image_dim_ordering() == 'th':
            if include_top:
                weights_path = get_file('vgg16_weights_th_dim_ordering_th_kernels.h5',
                                        TH_WEIGHTS_PATH,
                                        cache_subdir='models')
            else:
                weights_path = get_file('vgg16_weights_th_dim_ordering_th_kernels_notop.h5',
                                        TH_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models')
            model.load_weights(weights_path)
        else:
            if include_top:
                weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                        TF_WEIGHTS_PATH,
                                        cache_subdir='models')
            else:
                weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        TF_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models')
            model.load_weights(weights_path)

    return model



##获取文件
def get_nb_files(directory):
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


def setup_to_transfer_learn(model):  # 是否冻结
    for layer in model.layers:
        layer.trainable = False


def add_new_last_layer(base_model, nb_classes):##增加后面的三层重新训练
    x = base_model.output
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    predictions = Dense(nb_classes, activation='softmax', name='predictions')(x)
    model = Model(input=base_model.input, output=predictions)
    return model
##保存模型
def model_save(model_type,model,model_name):
    json_string = model.to_json()
    open('{}_{}.json'.format(model_name,model_type), 'w').write(json_string)
    model.save_weights('{}_{}.h5'.format(model_name,model_type))
##定义微调模型
def setup_to_finetune(model,optimizer_fine_tune,NB_IV3_LAYERS_TO_FREEZE):
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=optimizer_fine_tune, loss='categorical_crossentropy', metrics=['accuracy'])
#训练模型
def model_train(train_dir, validation_dir, IM_WIDTH= 224, IM_HEIGHT=224, model_name='VGG16', include_top=False,
                batch_size=32, nb_epoch_no_top=1, epoch_finne=1, weights='imagenet',
                optimizer=SGD(lr=0.0001, momentum=0.9)##训练的lr太大会导致不收敛
                , loss='categorical_crossentropy', fine_tune = False,metrics=['accuracy'],NB_IV3_LAYERS_TO_FREEZE = 0,optimizer_fine_tune = SGD(lr=0.0001, momentum=0.9)):
    '''
    # :param train_dir:训练数据的路径，存储形式按照train/标签name/标签数据图片
    :param validation_dir: 测试数据的路径，存储形式按照test/标签name/标签数据图片
    :param IM_WIDTH,IM_HEIGHT:输入图片模型的像素
    :param model_name:模型的名字
    :param include_top:是否包含顶层
    :param batch_size:批次
    :param nb_epoch_no_top:模型retrain的次数
    :param epoch_finne:模型微调次数
    :param weights:权重
    :param optimizer:梯度参数
    :param loss:损失
    :param metrics:评价
    :param optimizer_fine_tune:微调梯度参数参数
    :param NB_IV3_LAYERS_TO_FREEZE:微调开始层数
    :param fine_tune 是否进行微调
    :return:返回训练好的模型，并保存模型
    '''
    nb_train_samples = get_nb_files(train_dir)  # 训练样本个数
    nb_classes = len(glob.glob(train_dir + "/*"))  # 分类数
    nb_val_samples = get_nb_files(validation_dir)  # 验证集样本个数
    train_generator, validation_generator = image_preprocess(train_dir, validation_dir, IM_WIDTH, IM_HEIGHT, batch_size)
    base_model = VGG16(include_top=include_top,
                             weights=weights)  ##include_top 如果是true 的话就会包含顶层训练的权重，若果不包含则会重新训练 fune_tine表示是否进行微调

    setup_to_transfer_learn(base_model)  ##设置固定的层数
    model = add_new_last_layer(base_model, nb_classes)  # 增加顶层
    for layer in model.layers:
        print(layer.trainable)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit_generator(
        train_generator,
        epochs=nb_epoch_no_top,
        steps_per_epoch=nb_train_samples // batch_size,
        validation_data=validation_generator,
        validation_steps=nb_val_samples // batch_size,
        class_weight='auto')  ##进行训练
    print('{}retrain已经训练完成'.format(model_name))
    model_save(model_type = 'retrain',model = model,model_name = model_name)
    print('{}retrain模型已经保存')
    setup_to_finetune(model,optimizer_fine_tune,NB_IV3_LAYERS_TO_FREEZE)
    if fine_tune:
        print('{}模型开始微调'.format(model_name))
        model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epoch_finne,
            validation_data=validation_generator,
            validation_steps=nb_val_samples // batch_size,
            class_weight='auto')
        model_save(model_type='fine_tune', model=model, model_name=model_name)
        print('{}微调模型已经保存')
    return model
if __name__ == '__main__':

    path = '../data'
    train_dir = os.path.join(path, 'train')
    validation_dir = os.path.join(path, 'test')
    ##训练模型
    model = model_train(validation_dir= validation_dir,train_dir=train_dir)




