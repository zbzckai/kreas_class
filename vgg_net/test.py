# -*- coding: utf-8 -*-
# @Time    : 2019\7\9 0009 16:58
# @Author  : 凯
# @File    : test.py
from __future__ import print_function
import glob
import numpy as np
import warnings
import os
from keras import layers, backend, models
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Activation, merge, \
    ZeroPadding2D, add, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import cv2

import pandas as pd
from sklearn.preprocessing import LabelEncoder


##测试单张图片
def preprocess_input(net_name, x, dim_ordering):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}
    if net_name in ['VGG16', 'ResNet50']:
        if dim_ordering == 'th':
            x[:, 0, :, :] -= 103.939
            x[:, 1, :, :] -= 116.779
            x[:, 2, :, :] -= 123.68
            # 'RGB'->'BGR'
            x = x[:, ::-1, :, :]
        else:
            x[:, :, :, 0] -= 103.939
            x[:, :, :, 1] -= 116.779
            x[:, :, :, 2] -= 123.68
            # 'RGB'->'BGR'
            x = x[:, :, :, ::-1]
    else:
        x /= 255.
        x -= 0.5
        x *= 2.
    return x

##将图片的尺寸改为模型尺寸
def image_size(net_name):
    if net_name in ['VGG16', 'ResNet50']:
        return (224, 224)
    else:
        return (299, 299)

##获取文件夹下面的所有图片并进行预测使用
def get_inputs(predict_dir):
    predict_dir = predict_dir
    pre_x = []
    src = os.listdir(predict_dir)
    for s in src:
        images = image.load_img(os.path.join(predict_dir, s))
        x = image.img_to_array(images)
        input = cv2.resize(x, image_size(net_name))
        pre_x.append(input)  # input一张图片
    pre_x = np.array(pre_x)
    pre_x = preprocess_input(net_name, pre_x, 'default')
    return pre_x
##获得图片的原始名称进行对比使用
def gen_jpg_true_name(predict_dir):
    predict_dir = predict_dir
    old_name = os.listdir(predict_dir)
    from xpinyin import Pinyin
    p = Pinyin()
    true_name = pd.DataFrame({"name": old_name})
    true_name['汉语name'] = pd.Series(old_name).apply(lambda x: x.split("_")[0])
    wj_name_pinyin = []
    for name_i in true_name['汉语name']:
        wj_name_pinyin.append(p.get_pinyin(name_i, splitter=' '))
    true_name['eng_name'] = wj_name_pinyin
    return true_name
##将预测结果对应到标签，并整合为df 文件
def get_pre_solution(pre_y,train_label_path):
    true_name = gen_jpg_true_name()
    df = pd.DataFrame()
    list_test = os.listdir(train_label_path)
    le = LabelEncoder()
    le.fit(np.unique(list_test))
    df['label_index'] = le.transform(list_test)
    df['label_name'] = list_test
    all_pre = []
    for i in pre_y:
        # 排序，取出前5个概率最大的值（top-5),本数据集一共就5个
        # argsort()返回的是数组值从小到大排列所对应的索引值
        tmp_list = []
        top_5 = i.argsort()[-5:][::-1]
        for label_index in top_5:
            # 获取分类名称
            label_name = df.loc[df['label_index'] == label_index, 'label_name'].values[0]
            # 获取该分类的置信度
            label_score = i[label_index]
            tmp_list.append(label_name)
            tmp_list.append(label_score)
            print('%s (score = %.5f)' % (label_name, label_score))
        print('--------------------------------------------------------------')
        all_pre.append(tmp_list)
    all_pre = pd.DataFrame(all_pre)
    all_pre = pd.concat((true_name,all_pre),axis=1)
    return all_pre


if __name__ == "__main__":

    '''
    训练文件夹下图片
    '''
    ##训练一个文件夹下面的图片
    net_name = 'VGG16'
    predict_dir = r'D:\soft\git\kai\kreas_inception\testimages'
    train_label_path = '../data/train'
    os.listdir(train_label_path)
    ##导入图片
    pre_x = get_inputs(predict_dir)
    ##导入模型
    model = model_from_json(open('D:\soft\git\kai\kreas_inception\mnist_kreas_model_architecture.json').read())
    model.load_weights('D:\soft\git\kai\kreas_inception\mnist_kreas_mode_weights1.h5')
    #进行预测
    pre_y = model.predict(pre_x)
    #预测结果展示
    df_pre = get_pre_solution(pre_y,train_label_path)
    '''训练单张图片'''
    img_path = 'ceshi.jpg'
    img = image.load_img(img_path, target_size=image_size(net_name))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)



