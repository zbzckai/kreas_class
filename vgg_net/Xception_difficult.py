# -*- coding: utf-8 -*-
# @Time    : 2019\7\9 0009 13:28
# @Author  : 凯
# @File    : Xception_difficult.py
from __future__ import print_function

import glob
import os
import warnings
import numpy as np
from keras import backend as K
from keras.layers import Activation,merge,Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D,add
from keras.layers import Flatten, Dense, Input, BatchNormalization, concatenate
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.data_utils import get_file

from vgg_net.imagenet_utils import decode_predictions

'''获取文件的个数'''
os.getcwd()
model_name = 'Xception'
path = '../data'
train_dir = os.path.join(path, 'train')
test_dir = os.path.join(path, 'test')
nb_epoch = 1
epoch_frezz = 1
batch_size = 8

IM_WIDTH, IM_HEIGHT = 299, 299  # InceptionV3指定的图片尺寸
NB_IV3_LAYERS_TO_FREEZE = 0  ##在进行微调的时候选择冻结的


##生成训练图片
def image_preprocess():
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
        fill_mode='nearest'
    )
    #    '''
    # rotation_range是一个0~180的度数，用来指定随机选择图片的角度。
    #
    # width_shift和height_shift用来指定水平和竖直方向随机移动的程度，这是两个0~1之间的比例。
    #
    # rescale值将在执行其他处理前乘到整个图像上，我们的图像在RGB通道都是0~255的整数，这样的操作可能使图像的值过高或过低，所以我们将这个值定为0~1之间的数。
    #
    # shear_range是用来进行剪切变换的程度，参考剪切变换
    #
    # zoom_range用来进行随机的放大
    #
    # horizontal_flip随机的对图片进行水平翻转，这个参数适用于水平翻转不影响图片语义的时候
    #
    # fill_mode用来指定当需要进行像素填充，如旋转，水平和竖直位移时，如何填充新出现的像素
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
        test_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size, class_mode='categorical')
    return train_generator, validation_generator


TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'


def Xception(include_top=True, weights='imagenet',
             input_tensor=None):
    '''Instantiate the Xception architecture,
    optionally loading weights pre-trained
    on ImageNet. This model is available for TensorFlow only,
    and can only be used with inputs following the TensorFlow
    dimension ordering `(width, height, channels)`.
    You should set `image_dim_ordering="tf"` in your Keras config
    located at ~/.keras/keras.json.

    Note that the default input image size for this model is 299x299.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    if K.backend() != 'tensorflow':
        raise Exception('The Xception model is only available with '
                        'the TensorFlow backend.')
    if K.image_dim_ordering() != 'tf':
        warnings.warn('The Xception model is only available for the '
                      'input dimension ordering "tf" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'dimension ordering "th" (channels, width, height). '
                      'You should set `image_dim_ordering="tf"` in your Keras '
                      'config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "tf" dimension ordering.')
        K.set_image_dim_ordering('tf')
        old_dim_ordering = 'th'
    else:
        old_dim_ordering = None

    # Determine proper input shape
    if include_top:
        input_shape = (299, 299, 3)
    else:
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Conv2D(32, 3, 3, subsample=(2, 2), bias=False, name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, 3, 3, bias=False, name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, 1, 1, subsample=(2, 2),
                      border_mode='same', bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, 3, 3, border_mode='same', bias=False, name='block2_sepconv1')(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, 3, 3, border_mode='same', bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block2_pool')(x)

    x = add([x, residual])

    residual = Conv2D(256, 1, 1, subsample=(2, 2),
                      border_mode='same', bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, 3, 3, border_mode='same', bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, 3, 3, border_mode='same', bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block3_pool')(x)

    x = add([x, residual])
    residual = Conv2D(728, 1, 1, subsample=(2, 2),
                      border_mode='same', bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block4_pool')(x)
    x = add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = add([x, residual])

    residual = Conv2D(1024, 1, 1, subsample=(2, 2),
                      border_mode='same', bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024, 3, 3, border_mode='same', bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block13_pool')(x)
    x = add([x, residual])

    x = SeparableConv2D(1536, 3, 3, border_mode='same', bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(2048, 3, 3, border_mode='same', bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)

    # Create model
    model = Model(img_input, x)

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels.h5',
                                    TF_WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)

    if old_dim_ordering:
        K.set_image_dim_ordering(old_dim_ordering)
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
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    predictions = Dense(nb_classes, activation='softmax', name='predictions')(x)
    model = Model(input=base_model.input, output=predictions)
    return model


if __name__ == '__main__':
    nb_train_samples = get_nb_files(train_dir)  # 训练样本个数
    nb_classes = len(glob.glob(train_dir + "/*"))  # 分类数
    nb_val_samples = get_nb_files(test_dir)  # 验证集样本个数
    train_generator, validation_generator = image_preprocess()
    base_model = Xception(include_top=False,
                             weights='imagenet')  ##include_top 如果是true 的话就会包含顶层训练的权重，若果不包含则会重新训练 fune_tine表示是否进行微调
    for layer in base_model.layers:
        print(layer.trainable)
    setup_to_transfer_learn(base_model)  ##设置固定的层数
    model = add_new_last_layer(base_model, nb_classes)  # 增加顶层
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    history_tl = model.fit_generator(
        train_generator,
        epochs=epoch_frezz,
        steps_per_epoch=nb_train_samples // batch_size,
        validation_data=validation_generator,
        validation_steps=nb_val_samples // batch_size,
        class_weight='auto')  ##进行训练
    json_string = model.to_json()
    open('{}.json'.format(model_name), 'w').write(json_string)
    model.save_weights('{}.h5'.format(model_name))
    ##进行测试
    img_path = 'ceshi.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
