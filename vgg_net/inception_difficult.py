# -*- coding: utf-8 -*-
# @Time    : 2019\7\8 0008 15:18
# @Author  : 凯
# @File    : inception_difficult.py

from __future__ import print_function

import glob
import os

import numpy as np
from keras import backend as K
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D,GlobalAveragePooling2D
from keras.layers import Flatten, Dense, Input, BatchNormalization, concatenate
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.data_utils import get_file

from vgg_net.imagenet_utils import decode_predictions


##生成训练图片
def image_preprocess(train_dir ,validation_dir,IM_WIDTH ,IM_HEIGHT,batch_size):
    #   图片生成器
    # 　训练集的图片生成器，通过参数的设置进行数据扩增
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
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
        preprocessing_function=preprocess_input,
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




def conv2d_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1),
              name=None):##定义一个流程卷积 - bn
    '''Utility function to apply conv + BN.
    '''
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_dim_ordering() == 'th':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      activation='relu',
                      border_mode=border_mode,
                      name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name)(x)
    return x


def InceptionV3(include_top=True, weights='imagenet',
                input_tensor=None):
    '''
    include_top 是否包含顶层参数，如果为true 则包含顶层参数，可以直接这样就能对原始数据直接进行预测，false则主要自主去训练顶层
    weights 权重采用的是与训练好的inagenet
    input_tensor 判断输入数据是否为none

    '''
    TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
    TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (3, 299, 299)
        else:
            input_shape = (3, None, None)
    else:
        if include_top:
            input_shape = (299, 299, 3)
        else:
            input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor

    if K.image_dim_ordering() == 'th':
        channel_axis = 1
    else:
        channel_axis = 3

    x = conv2d_bn(img_input, 32, 3, 3, subsample=(2, 2), border_mode='valid')
    x = conv2d_bn(x, 32, 3, 3, border_mode='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, border_mode='valid')
    x = conv2d_bn(x, 192, 3, 3, border_mode='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    for i in range(3):
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), border_mode='same')(x)
        branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
        x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                        axis=channel_axis,
                        name='mixed' + str(i))

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, subsample=(2, 2), border_mode='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3,
                             subsample=(2, 2), border_mode='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = concatenate([branch3x3, branch3x3dbl, branch_pool],
                    axis=channel_axis,
                    name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                    axis=channel_axis,
                    name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), border_mode='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                        axis=channel_axis,
                        name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 160, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                    axis=channel_axis,
                    name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          subsample=(2, 2), border_mode='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3,
                            subsample=(2, 2), border_mode='valid')

    branch_pool = AveragePooling2D((3, 3), strides=(2, 2))(x)
    x = concatenate([branch3x3, branch7x7x3, branch_pool],
                    axis=channel_axis,
                    name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = concatenate([branch3x3_1, branch3x3_2],
                                axis=channel_axis,
                                name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = concatenate([branch3x3dbl_1, branch3x3dbl_2],
                                   axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), border_mode='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                        axis=channel_axis,
                        name='mixed' + str(9 + i))

    if include_top: ##是否包含全全连接层
        # Classification block
        x = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
        x = Flatten(name='flatten')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)

    # Create model
    model = Model(img_input, x)

    # load weights 下载权重
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
                                    TF_WEIGHTS_PATH,
                                    cache_subdir='models'
                                    )
        else:
            weights_path = get_file('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models'
                                 )
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

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def setup_to_transfer_learn(model):  # 是否冻结
    for layer in model.layers:
        layer.trainable = False


def add_new_last_layer(base_model, nb_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model

def model_save(model_type,model,model_name):
    json_string = model.to_json()
    open('{}_{}.json'.format(model_name,model_type), 'w').write(json_string)
    model.save_weights('{}_{}.h5'.format(model_name,model_type))

def setup_to_finetune(model,optimizer_fine_tune,NB_IV3_LAYERS_TO_FREEZE):
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=optimizer_fine_tune, loss='categorical_crossentropy', metrics=['accuracy'])


def model_train(train_dir, validation_dir, IM_WIDTH= 224, IM_HEIGHT=224, model_name='ResNet50', include_top=False,
                batch_size=32, nb_epoch_no_top=1, epoch_finne=1, weights='imagenet',
                optimizer=SGD(lr=0.01, momentum=0.9)
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
    base_model = InceptionV3(include_top=include_top,
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
    setup_to_finetune(model, optimizer_fine_tune, NB_IV3_LAYERS_TO_FREEZE)
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
    model = model_train(validation_dir= validation_dir,train_dir=train_dir)




