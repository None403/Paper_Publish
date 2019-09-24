#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author:GodNoFan
@Created On 2019-06-21
@Coding Environment: Anaconda Python 3.7
"""

import random
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from keras.layers import LeakyReLU
from keras.layers import PReLU
import operator
from keras.models import load_model
import copy
from keras.utils import multi_gpu_model
from keras.layers.normalization import BatchNormalization

# 数据预处理
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
print('Load data.....')
parapath = './dataset/Parasitized/'
uninpath = './dataset/Uninfected/'
parastized = os.listdir(parapath)
uninfected = os.listdir(uninpath)
data = []
label = []
for para in parastized:
    try:
        img =  image.load_img(parapath+para,target_size=(128, 128))
        x = image.img_to_array(img)
        data.append(x)
        label.append(1)
    except:
        print("Can't add "+para+" in the dataset")
for unin in uninfected:
    try:
        img =  image.load_img(uninpath+unin,target_size=(128, 128))
        x = image.img_to_array(img)
        data.append(x)
        label.append(0)
    except:
        print("Can't add "+unin+" in the dataset")
data = np.array(data)
label = np.array(label)
data = data/255
x_train, x_test, y_train, y_test = train_test_split(data,label,test_size = 0.2,random_state=0)

# -------------以上为数据预处理部分-----------------------------


save_dir = os.path.join(os.getcwd(),'saved_model_cnn')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)


class GeneticAlgorithm:
    # ----------初始数据定义----------
    # 遗传算法下那个管参数定义
    # JCL = 0.9 # 遗传时的交叉率
    # BTL = 0.09 # 遗传时的变异率

    def __init__(self,rows,times,num_classes,kill_num):
        self.rows = rows    # 染色体个数
        self.times = times      # 迭代次数
        self.accuracy = 0    # 模型准确率
        self.layer_list = ['Conv2D', 'Dense']    # 算法使用的网络层
        self.cnn_activation_function = ['linear', 'leaky_relu', 'prelu', 'relu']   # CNN层用到的激励函数
        self.dense_activation_function = ['linear', 'sigmoid', 'softmax', 'relu']  # 中间全连接层用到的激励函数
        self.last_dense_activation_function = ['sigmoid']   # 最后一个全连接层用到的激励函数
        self.unit_num_list = [64, 128, 256]     # 神经元数目选择
        self.filter_num_list = [4, 8, 16]    # 卷积核数目选择
        self.pooling_size = range(2, 4)     # max_pooling的选择范围
        self.filter_size = range(2, 4)   # 卷积核大小的选择范围
        self.layer_num_list = range(2, 4)   # 网络层次的选择范围
        self.max_size = 10  # 层数最大值
        self.threshold = 3  # 层数临界值
        self.batch_size = 512
        self.num_classes = num_classes  # 二分类。若输入时，num_classes = 2
        self.kill_num = kill_num  # 每次杀掉的网络个数


    def add_activation_funtion(self,model,activation_name):
        """
        添加激活函数
        :param model:
        :param activation_name:
        :return:
        """
        if activation_name == 'leaky relu':
            model.add(LeakyReLU())
        elif activation_name == 'prelu':
            model.add(PReLU())
        else:
            model.add(Activation(activation_name))
        return model

    def create_network(self,chromosome_dict):
        """
        :param chromosome_dict:
        :return:
        """
        layer_list = chromosome_dict['layer_list']
        layer_num = chromosome_dict['layer_num'] + 2     # 包括输入输出层
        unit_num_sum = 0    # 统计Dense神经元的个数
        model = Sequential()
        for i in range(len(layer_list)-1):
            if i==0:     # 是否为第一层，第一层固定为Conv2D层
                model.add(Conv2D(
                    layer_list[i]['conv_kernel_num'],
                    layer_list[i]['conv_kernel_size'],
                    padding=layer_list[i]['padding'],
                    input_shape=layer_list[i]['input_shape'],
                    kernel_initializer='he_normal'
                    )
                )

                model = self.add_activation_funtion(model,layer_list[i]['layer_activation_function'])

            else:    # Conv2D or Dense layer
                if layer_list[i]['layer_name'] == 'Conv2D':
                    model.add(Conv2D(layer_list[i]['conv_kernel_num'],
                                     layer_list[i]['conv_kernel_size'],
                                     padding=layer_list[i]['padding'],
                                     )
                              )
                    model = self.add_activation_funtion(model,layer_list[i]['layer_activation_function'])

                    if layer_list[i]['pooling_choice']: # 是否创建Pooling层次
                        try:
                            model.add(MaxPooling2D(pool_size=layer_list['pool_size'], dim_ordering="tf"))
                        except Exception as e:
                            print('Maxpooling大于输入的矩阵, 用pool_size=(1, 1)代替',e)
                            model.add(MaxPooling2D(pool_size=(1, 1),strides=(2, 2)))
                        layer_num += 1
                        model.add(BatchNormalization())

                    # Dropout层
                    model.add(Dropout(layer_list[i]['dropout_rate']))
                else: # Dense层
                    unit_num_sum += layer_list[i]['unit_num']
                    model.add(Dense(layer_list[i]['unit_num']))
                    model = self.add_activation_funtion(model, layer_list[i]['layer_activation_function'])

                    # Dropout层
                    model.add(Dropout(layer_list[i]['dropout_rate']))

        # 最后一层
        model.add(Flatten())
        model.add(Dense(1))
        model = self.add_activation_funtion(model, layer_list[-1]['layer_activation_function'])
        unit_num_sum += 2
        chromosome_dict['model'] = model
        chromosome_dict['punish_factor'] = (1 / layer_num) + (1 / unit_num_sum) # 惩罚因子

        return chromosome_dict




    def create_layer(self,layer_name):
        """
        创建网络层次属性
        :param layer_name:
        :return:
        """
        layer_dict = dict()
        layer_dict['layer_name'] = layer_name
        if layer_name == 'Conv2D':
            # 随机选取激励函数
            layer_activation_function = random.choice(self.cnn_activation_function)
            # 卷积核数量和大小
            conv_kernel_num = random.choice(self.filter_num_list)
            random_size = random.choice(self.filter_size)
            conv_kernel_size = (random_size,random_size)
            # 是否加入Pooling层
            pooling_choice = [True, False]
            if pooling_choice[random.randint(0, 1)]:
                layer_dict['pooling_choice'] = True
                random_size = random.choice(self.pooling_size)
                pool_size = (random_size,random_size)
                layer_dict['pool_size'] = pool_size
            else:
                layer_dict['pooling_choice'] = False

            layer_dict['layer_activation_function'] = layer_activation_function
            layer_dict['conv_kernel_num'] = conv_kernel_num
            layer_dict['conv_kernel_size'] = conv_kernel_size
            layer_dict['padding'] = 'same'
        else:   # Dense层
            # 激励函数
            layer_activation_function = random.choice(self.dense_activation_function)
            # 神经元个数
            unit_num = random.choice(self.unit_num_list)
            layer_dict['layer_activation_function'] = layer_activation_function
            layer_dict['unit_num'] = unit_num
        layer_dict['dropout_rate'] = round(random.uniform(0,1),3)   # uniform()随机生成一个实数；round()小数点后四舍五入
        return layer_dict

    def random_learning_rate(self):
        return random.uniform(0.01, 0.02)    # 在此范围内随机生成学习率

    def create_chromosome(self):
        """
        创建染色体
        :return:
        """
        chromosome_dict = dict()  # 用字典装载染色体的所有属性
        chromosome_dict['learning_rate'] = self.random_learning_rate() # 学习率
        layer_num = random.choice(self.layer_num_list) # 创建的网络层次， 输入层和输出层不计算在内
        chromosome_dict['layer_num'] = layer_num

        layer_list = list() # 网络层次顺序表
        # 第一层必须是卷积层
        layer_list.append({
            'layer_name': 'Conv2D',
            'conv_kernel_num': 32,
            'conv_kernel_size': (3, 3),
            'padding': 'same',
            'input_shape': (128, 128, 3),
            'layer_activation_function': random.choice(self.cnn_activation_function)
        })

        # 确定每层名称及对应属性
        for i in range(layer_num):
            # 选择层次类型
            layer_name = 'Conv2D'
            if i==0: # 第一层dropout_rate必须为0，即不存在
                layer_dict = self.create_layer(layer_name)
            else:
                layer_dict = self.create_layer(layer_name)
            layer_list.append(layer_dict) # 添加至层次列表

        # 最后一层必须是Dense层
        layer_list.append({
            'layer_name': 'Dense',
            'unit_num': self.num_classes,
            'layer_activation_function': random.choice(self.last_dense_activation_function)
        })

        # 将网络层次顺序表添加至染色体
        chromosome_dict['layer_list'] = layer_list

        return chromosome_dict

    def cal_fitness(self,line,epochs):
        """
        :param line: 染色体（网络）
        :param epochs: 迭代次数
        :return:
        """
        if epochs == 0:
            return line
        line = self.train_process(line,epochs)
        # 适应度函数，公式：准确率 + 训练参数个数的倒数，适应度越大，说明模型越好。
        fitness = line['accuracy']
        line['fitness'] = fitness
        return line

    def train_process(self,line,epochs):
        """
        :param line: 染色体
        :param epochs: 迭代次数
        :return:
        """
        print('learning_rate', line['learning_rate'])
        print('layer_num:', len(line['layer_list']))

        if line['is_saved']:    # 若已保存，则直接读入训练即可
            print('读取原有模型训练.....')
            model_path = line['model_path']
            model = load_model(model_path)
            accuracy = model.evaluate(x = x_test, y=y_test)[1]
            print('former accuracy:', accuracy)
        else:
            print('重新训练....')
            model = line['model']
            learning_rate = line['learning_rate']
            # 初始化adam优化器
            opt = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
                                        epsilon=None, decay=1e-6, amsgrad=False)

            # 模型编译
            model.compile(loss='binary_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])

        history = model.fit(x_train, y_train,epochs=epochs, batch_size=self.batch_size)
        # Score trained model.
        accuracy = model.evaluate(x=x_test,y=y_test)[1]
        line['accuracy'] = accuracy
        line['history'] = history  # 训练历史
        print('accuracy:', accuracy)

        # 保存模型
        model_name = line['model_name']
        model_path = os.path.join(save_dir,model_name)
        line['model_path'] = model_path  # 每一个模型的路径
        line['is_saved'] = True  # 是否保存
        line['model'] = model
        model.save(model_path)

        return line

    def random_select(self,ran_fit):
        """
        轮盘赌选择： 根据概率随机选择的染色体
        :param ran_fit:
        :return:
        """

        ran = random.random()
        for i in range(self.rows):
            if ran < ran_fit[i]:
                return i

    def mutation(self,line,name):
        """
        基因变异
        :param line:
        :param name:
        :return:
        """
        offspring1 = copy.deepcopy(line) # 深拷贝！！！！子代1
        offspring1['model_name'] = name
        offspring1['is_saved'] = False
        mutation_choice = [True, False]
        # 子代1变异：随机变异，改变学习率或者网络结构，二选一
        if mutation_choice[random.randint(0, 1)]:    # 改变学习率
            print('Mutation Operation:Change learning rate....')
            offspring1['learning_rate'] = self.random_learning_rate()   # 学习率
        else:   # 改变网络结构
            offspring1 = self.layer_mutation_operation(offspring1)
        offspring1 = self.create_network(offspring1)

        return offspring1

    def layer_mutation_operation(self, offspring):
        """
        :param offspring: 子代染色体
        :return:
        """
        mutation_layer_choice = [0, 1, 2]     # 0,1,2对应添加，替换，删除
        mutation_layer_choice_name = ['Add', 'Replace', 'Delete']
        layer_name = self.layer_list[random.randint(0, 1)]
        layer_dict = self.create_layer(layer_name)  # 选定layer_name之后，为对应的layer_name层去添加层相关属性
        choice_index = -1
        if self.threshold < offspring['layer_num'] < self.max_size:     # 最少层数临界值 < 层数 < 层数最大值
            choice_index = random.randint(0, 2)
            if mutation_layer_choice[choice_index] == 0:    # 添加操作
                insert_index = random.randint(1,len(offspring['layer_list']) - 1)
                offspring['layer_list'].insert(insert_index, layer_dict)
                offspring['layer_num'] += 1
            elif mutation_layer_choice[choice_index] == 1:  # 替换操作
                replace_index = random.randint(1, len(offspring['layer_list']) - 1)
                offspring['layer_list'][replace_index] = layer_dict
            else:    # 删除操作
                delete_index = random.randint(1, len(offspring['layer_list']) - 1)
                del offspring['layer_list'][delete_index]
                offspring['layer_num'] -= 1
        elif offspring['layer_num'] <= self.threshold:      # 网络层数<=临界值，智能添加或者替换
            choice_index = random.randint(0, 1)
            if mutation_layer_choice[choice_index] == 0:    # 添加操作
                insert_index = random.randint(1, len(offspring['layer_list']) - 1)      # 插入位置
                offspring['layer_list'].insert(insert_index, layer_dict)
                offspring['layer_num'] += 1
            else:
                replace_index = random.randint(1,len(offspring['layer_list']) - 1)
                offspring['layer_list'][replace_index] = layer_dict

        else:   # 层数达到最大值，只能进行深处操作
            delete_index = random.randint(1, len(offspring['layer_list']) - 1)      # 删除位置
            del offspring['layer_list'][delete_index]
            offspring['layer_num'] -= 1
        print('Mutation Operation:', mutation_layer_choice_name[choice_index])

        return offspring

    def get_best_chromosome(self,father,offspring1,offspring2,epochs):
        """
        比较父代、子代1、子代2的适应度，返回适应度最高的染色体
        :param father:
        :param offspring1:
        :param offspring2:
        :param epochs:
        :return:    返回适应度最高的染色体
        """
        print('子代1训练：',epochs)
        offspring1 = self.cal_fitness(offspring1,epochs)

        print('子代2训练：', epochs)
        offspring2 = self.cal_fitness(offspring2, epochs)

        tmp_lines = [father,offspring1,offspring2]
        sorted_lines = sorted(tmp_lines, key=operator.itemgetter('fitness'))

        return sorted_lines[-1]
# ----------遗传函数开始执行----------
    def run(self):

        print("开始迭代")

        # 初始化种群
        lines = [self.create_network(self.create_chromosome()) for i in range(self.rows)]

        # 初始化种群适应度
        fit = [0 for i in range(self.rows)]

        epochs = 1
        # 计算每个染色体()
        for i in range(0,self.rows):
            lines[i]['is_saved'] = False
            lines[i]['model_name'] = 'model_%s' % str(i)
            lines[i] = self.cal_fitness(lines[i], epochs)   # cal_fitness()调用train_process()函数，这里得到父代染色体的适应度
            fit[i] = lines[i]['fitness']

        # 开始迭代
        t = 0
        while t < self.times:
            print('迭代次数', t)
            random_fit = [0 for i in range(self.rows)]
            total_fit = 0
            tmp_fit = 0

            # 开始遗传
            # 根据轮盘赌选择父代
            # 计算原有种群的总适应度
            for i in range(self.rows):
                total_fit += fit[i]
            # 以下是轮盘赌的核心操作：通过 适应度/总适应度 的比例生成随机适应度
            for i in range(self.rows):
                random_fit[i] = tmp_fit + fit[i] / total_fit
                tmp_fit += random_fit[i]
            r = int(self.random_select(random_fit)) # 轮盘赌选择出的染色体，进行变异后的最优模型
            line = lines[r]

            # 不需要交叉，直接变异，遗传到下一代
            # 基因变异，连续迭代生两个子代，即子代和孙子代
            print("*****变异*****")
            offspring1 = self.mutation(line,'offspring1')
            offspring2 = self.mutation(offspring1,'offspring2')
            best_chromosome = self.get_best_chromosome(line, offspring1, offspring2, epochs)

            # 替换父代
            father_model_name = lines[r]['model_name']
            lines[r] = best_chromosome
            print('保存最佳变异个体。。。。')
            # 保存模型
            model_path = os.path.join(save_dir, father_model_name)
            lines[r]['model_path'] = model_path  # 每一个模型的路径
            lines[r]['is_saved'] = True  # 是否保存
            best_chromosome_model = lines[r]['model']
            best_chromosome_model.save(model_path)

            epochs += 1
            # 杀掉最差的self.kill_num个网络
            kill_index = 1
            sorted_lines = sorted(lines, key=operator.itemgetter('fitness')) # 按适应度从小到大排序
            if len(sorted_lines) > self.kill_num:
                # 第一次迭代杀死适应度小于0.55的网络
                for i in range(len(sorted_lines)):
                    if sorted_lines[i]['fitness'] < 0.55:
                        kill_index = i
                    else:
                        break
                if t == 0:
                    new_lines = sorted_lines[kill_index:]
                    self.rows -= kill_index
                else:
                    new_lines = sorted_lines[self.kill_num:]
                    self.rows -= self.kill_num
                lines = new_lines # 更新种群
                next_fit = [line['fitness'] for line in lines]  # 更新种群
                fit = next_fit
                print('..........Population size:%d .........' % self.rows)

            # 进行新的一次epochs，计算种群的适应度
            for i in range(0,self.rows):
                lines[i] = self.cal_fitness(lines[i],1)
                fit[i] = lines[i]['fitness']
            print('***************************************************')
            print()
            t += 1  # 代数+1


        # 提取适应度最高的
        m = fit[0]
        ml = 0
        for i in range(self.rows):
            if m < fit[i]:
                m = fit[i]
                ml = i
        print("迭代完成")
        # 输出结果
        excellent_chromosome = self.cal_fitness(lines[ml], 0)
        print('The best network:')
        print(excellent_chromosome['model'].summary())
        print('Fitness: ', excellent_chromosome['fitness'])
        print('Accuracy', excellent_chromosome['accuracy'])

        best_model_save_dir = os.path.join(os.getcwd(),'best_model')
        if not os.path.isdir(best_model_save_dir):
            os.makedirs(best_model_save_dir)
        best_model_path = os.path.join(best_model_save_dir, 'excellent_model')
        excellent_chromosome['model'].save(best_model_path)
        print(excellent_chromosome['layer_list'])
        with open('best_network_layer_list.txt', 'w') as fw:
            for layer in excellent_chromosome['layer_list']:
                fw.write(str(layer) + '\n')

    # ----------遗传函数执行完毕----------
"""
一个染色体代表一种网络结构，即随机生成的不同的层（卷积层(附带pooling层）或者是dense层）
，以及不同的激活函数等
"""

def main():
    pass


if __name__ == '__main__':
    main()
