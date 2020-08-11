from keras.layers import Lambda
import tensorflow as tf
from skimage import data, io, filters
import numpy as np
from numpy import array
from numpy.random import randint
from scipy.misc import imresize
import os
import sys
import cv2
import shutil



def load_path(path):
    directories = []
    if os.path.isdir(path):
        directories.append(path)

    return directories


def load_data_from_dirs(dirs, ext):
    files = []
    file_names = []

    for d in dirs:
        for f in os.listdir(d):
            if f.endswith(ext):
                files.append(f)
                file_names.append(os.path.join(d, f))

    return files,file_names

def load_data(directory, ext):

    files,filenames= load_data_from_dirs(load_path(directory), ext)
    files.sort(reverse=True)
    filenames.sort(reverse=True)
    return filenames,files


def plot_generated_images(output_dir, epoch, generator, x_test_hr, x_test_lr, dim=(1, 3), figsize=(15, 5)):

    examples = x_test_hr.shape[0]
    value = randint(0, examples)


    image_batch_hr = denormalize(x_test_hr)

    image_batch_lr = x_test_lr

    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)

    for i in range(len(generated_image)):

        path = os.path.join('./image_test/', "IMG-%s.png" % i)
        cv2.imwrite(path, cv2.cvtColor(generated_image[i], cv2.COLOR_RGB2BGR))


def normalize(input_data):
    return (input_data.astype(np.float32) - 127.5) / 127.5


def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)



def load_dcgan_train_model(dirs,ext):
    gen=[]
    dis=[]
    gen_name =[]
    dis_name =[]
    nfile=[]
    gen_num_list =[]
    dis_num_list =[]
    files = os.listdir(dirs)
    for file in files:
        if file.endswith(ext):
            nfile.append(file)
    if len(nfile)==0:
        print("Do not have model")
        gen=""
        dis=""
        epoch = 1
        return gen,dis,epoch
    elif len(nfile)!=0:
        for r, d, f in os.walk(dirs):
            for fi in f:
                if fi.startswith('gen'):
                    gen.append(os.path.join(dirs,fi))
                    gen_name.append(fi)

                elif fi.startswith('dis'):
                    dis.append(os.path.join(dirs,fi))
                    dis_name.append(fi)
        for i in gen_name:
            gen_num_list.append(int(i[9:-3]))
        for j in dis_name:
            dis_num_list.append(int(j[9:-3]))


        max_gen = max(gen_num_list)
        max_dis = max(dis_num_list)
        if max_gen > max_dis:
            max_gen = max_dis

        gen_max_index = gen_num_list.index(max_gen)
        dis_max_index = dis_num_list.index(max_dis)
        gen = gen[gen_max_index]
        dis = dis[dis_max_index]
        epoch = int(gen_name[gen_max_index][9:-3])
        return gen,dis,epoch

def remove_logs(logdir):
    logs = os.listdir(logdir)
    if len(logs) !=0:
        shutil.rmtree(logdir)
    os.mkdir(logdir)
















