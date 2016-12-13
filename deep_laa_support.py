import numpy as np
import numpy.matlib
import tensorflow as tf
import scipy.io
import os

from fpformat import sci

def convert_mat_to_one_hot_npz(read_file, write_file=""):
    data = scipy.io.loadmat(read_file)
    data_num, source_num = np.shape(data['L'])
    tmp_true_labels = data['true_labels']
    tmp_user_labels = data['L']
    
    # 'true_labels'
    tmp_min_category = np.min(tmp_true_labels)
    tmp_true_labels -= tmp_min_category
    category_num = np.max(tmp_true_labels) + 1
    true_labels = np.reshape(tmp_true_labels, (data_num, 1))
    
    # 'user_labels' & 'label_mask'
    user_labels = np.zeros((data_num, source_num*category_num))
    label_mask = np.zeros((data_num, source_num*category_num))
    for i in range(data_num):
        for j in range(source_num):
            tmp_label = tmp_user_labels[i, j]
            if tmp_label > 0:
                target_label = tmp_label - tmp_min_category
                user_labels[i, j*category_num+target_label] = 1
                label_mask[i, j*category_num:(j+1)*category_num] = 1
    
    if write_file == "":
        write_file = os.path.basename(read_file)
    np.savez(write_file, 
             true_labels=true_labels, 
             user_labels=user_labels, 
             label_mask=label_mask, 
             category_num=category_num, 
             source_num=source_num)

def get_constant_y(batch_size, category_size):
    constant_y = {}
    for i in range(category_size):
        constant_tmp = np.zeros((batch_size, category_size), dtype=np.float32)
        constant_tmp[:, i] = 1.0;
        constant_y[i] = tf.constant(constant_tmp)
    return constant_y

def get_majority_y(user_labels, category_num):
    n_samples, source_mul_category = np.shape(user_labels)
    source_num = source_mul_category / category_num
    tmp = np.eye(category_num)
    template = np.matlib.repmat(tmp, source_num, 1)
    majority_y = np.matmul(user_labels, template)
    majority_y = np.divide(majority_y, np.matlib.repmat(np.sum(majority_y, 1, keepdims=True), 1, category_num))
    return majority_y
    

def gen_data(filename, data_num, source_num, category_num):
    data_label_vectors = np.zeros((data_num, source_num*category_num))
    _tmp = np.random.multinomial(1, [1./category_num]*category_num, size=data_num*source_num)
    for i in range(data_num):
        for j in range(source_num):
            data_label_vectors[i, category_num*j:category_num*(j+1)] = _tmp[i*source_num+j, :]
    data_y_labels = np.argmax(np.random.multinomial(1, [1./category_num]*category_num, size=data_num), axis=1)
    np.savez(filename, data=data_label_vectors, labels=np.reshape(data_y_labels, (data_num, 1)))