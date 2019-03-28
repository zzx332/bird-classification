import numpy as np
import os


def get_files(filename):
    class_train = []
    label_train = []
    for train_class in os.listdir(filename):
        for pic in os.listdir(filename+train_class):
            name = train_class.split(sep='.')
            class_train.append(filename+train_class+'/'+pic)
            label_train.append(name[0])
    temp = np.array([class_train,label_train])
    temp = temp.transpose()
    # shuffle the samples
    np.random.shuffle(temp)
    # after transpose, images is in dimension 0 and label in dimension 1
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    # print(label_list)
    return image_list, label_list


data_dir = 'D:/data/birddata/CUB_200_2011/images/'
image, label = get_files(data_dir)
