# -*- coding: utf-8 -*-#
# Author:      weiz
# Date:        2019/12/18 下午4:39
# Name:        test.py
# Description:
import os
import numpy as np
from PIL import Image
from datetime import datetime

import tensorflow as tf
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from importlib import reload
from keras.layers.core import Dense, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard

import densenet

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def readTXT(path):
    """
    读取txt文件
    :param path:
    :return:
    """
    # 按行读取
    with open(path, "r+", encoding='utf-8') as f:
        wordLib = f.readlines()

    # 去掉换行符
    for index in range(len(wordLib)):
        wordLib[index] = wordLib[index].strip('\n')

    return wordLib


class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """
    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0
    def get(self, batchsize):
        r_n = []
        if(self.index + batchsize > self.total):
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batchsize) - self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
        else:
            r_n = self.range[self.index : self.index + batchsize]
            self.index = self.index + batchsize

        return r_n


def get_session(gpu_fraction=1.0):

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def readfile(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(' ')
        dic[p[0]] = p[1:]
    return dic


def gen(data_file, image_path, batchsize=128, maxlabellength=10, imagesize=(32, 280)):
    image_label = readfile(data_file)
    _imagefile = [i for i, j in image_label.items()]
    x = np.zeros((batchsize, imagesize[0], imagesize[1], 1), dtype=np.float)
    labels = np.ones([batchsize, maxlabellength]) * 10000
    input_length = np.zeros([batchsize, 1])
    label_length = np.zeros([batchsize, 1])

    r_n = random_uniform_num(len(_imagefile))
    _imagefile = np.array(_imagefile)
    while 1:
        shufimagefile = _imagefile[r_n.get(batchsize)]
        for i, j in enumerate(shufimagefile):
            img1 = Image.open(os.path.join(image_path, j)).convert('L')
            img = np.array(img1, 'f') / 255.0 - 0.5

            x[i] = np.expand_dims(img, axis=2)
            # print('imag:shape', img.shape)
            str = image_label[j]
            label_length[i] = len(str)

            if(len(str) <= 0):
                print("len < 0", j)
            input_length[i] = imagesize[1] // 8
            labels[i, :len(str)] = [int(k) - 1 for k in str]

        inputs = {'the_input': x,
                'the_labels': labels,
                'input_length': input_length,
                'label_length': label_length,
                }
        outputs = {'ctc': np.zeros([batchsize])}
        yield (inputs, outputs)


def getOldModel(img_h, nclass=None):
    """
    获得预训练模型
    :param img_h:
    :param nclass:
    :return:
    """
    if nclass == None:
        nclass = 5990
    input_base = Input(shape=(img_h, None, 1), name='the_input')
    y_pred = densenet.dense_cnn(input_base, nclass)                # 预训练的模型是5990个字符
    basemodel = Model(inputs=input_base, outputs=y_pred)

    pretrainModelPath = './models/pretrainedModel/weights_densenet.h5'
    print("--------------------------------------------------------------------------------------")
    if os.path.exists(pretrainModelPath):
        print("Loading model weights...")
        basemodel.load_weights(pretrainModelPath)
        print('done!')
    else:
        print("No pre-trained model loaded!")
    print("--------------------------------------------------------------------------------------")

    return basemodel



def getNewModel(img_h, nclass, basemodel, fine_tune_model=None):
    """
    获得需要fine-tune的模型
    :param img_h:
    :param nclass:
    :param basemodel:
    :return:
    """
    if fine_tune_model == None:
        fine_tune_model = 2
    # 在basemodel基础添加新的网络层（其实这里只是修改了最后的输出层）
    with tf.name_scope("output"):
        x = basemodel.get_layer("flatten").output
        y_pred = Dense(nclass, name='out', activation='softmax')(x)

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[basemodel.input, labels, input_length, label_length], outputs=loss_out)
    model.summary()

    # 注意：预训练模型一定要加载成功。没有加载成功但是却冻结网络，最后模型的效果肯定会很差
    # 有两种fine-tune的方法：第一种，只训练新添加的层训练；第二种，全局微调
    if fine_tune_model == 1:
        ## 第一种：冻结原始层位（即冻结basemodel网络层），在编译后生效
        for layer in basemodel.layers:
            layer.trainable = False
    elif fine_tune_model == 2:
        ## 第二种：设定各层是否用于训练，编译后生效
        for layer in model.layers[:20]:
            layer.trainable = False
        for layer in model.layers[20:]:
            layer.trainable = True


    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])

    return model


def getTrainingParameters(ep):
    checkpoint = ModelCheckpoint(filepath='./models/weights_densenet-{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss',
                                 save_best_only=False, save_weights_only=True)
    lr_schedule = lambda epoch: 0.0005 * 0.4 ** epoch
    learning_rate = np.array([lr_schedule(i) for i in range(ep)])
    changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    earlystop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "./models/logs"
    logdir = "{}/run-{}/".format(root_logdir, now)
    tensorboard = TensorBoard(log_dir=logdir, write_graph=True)

    return checkpoint, changelr, earlystop, tensorboard

epochs = 10
img_h = 32
img_w = 280
batch_size = 64
maxlabellength = 10
worldLib = "./cft/char_std_5990.txt"
label_train = "E:/dataSet/Chinese_ocr/data_train.txt"
label_test = "E:/dataSet/Chinese_ocr/data_test.txt"
imagesFile_train = "C:/dataSet/images"
imagesFile_test = "C:/dataSet/images"

if __name__ == "__main__":
    char_set = open(worldLib, 'r', encoding='utf-8').readlines()
    char_set = ''.join([ch.strip('\n') for ch in char_set][1:] + ['卍'])
    nclass = len(char_set)
    train_set_num = len(readTXT(label_train))
    test_set_num = len(readTXT(label_test))


    K.set_session(get_session())
    reload(densenet)
    basemodel = getOldModel(img_h)
    model = getNewModel(img_h, nclass, basemodel)

    train_loader = gen(label_train, imagesFile_train, batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))
    test_loader = gen(label_test, imagesFile_test, batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))

    checkpoint, changelr, earlystop, tensorboard = getTrainingParameters(epochs)
    print('-----------Start training-----------')
    model.fit_generator(train_loader,
                        steps_per_epoch=train_set_num // batch_size,
                        epochs=epochs,
                        initial_epoch=0,
                        validation_data=test_loader,
                        validation_steps=test_set_num // batch_size,
                        callbacks=[checkpoint, earlystop, changelr, tensorboard])