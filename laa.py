'''
Created on Dec 4, 2016

@author: jianhua
'''
import random
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cholesky
#definition=================================================================================================
source_num = 5
source_label_per_object = 2
object_num = 4
simulate_lambda = 0.2
cate_num = 2;
#true_label==================================================================================================
true_label = [0 for x in range(object_num)]
for i in range(object_num):
    if i < object_num / 2:
        true_label[i] = 1
    else:
        true_label[i] = 2
#print(true_label)
#simulate data==============================================================================================
def initial(source_ability_mu,source_ability_sigma):
    label = [([0] * source_num * 2) for i in range(object_num)]
    simulate_source_ability = np.random.normal(source_ability_mu, source_ability_sigma, source_num)
    simulate_source_rate = 1 / (1 + np.exp(-simulate_lambda * simulate_source_ability));
    
    #instance_difficulty = np.random.normal(instance_difficulty_mu,instance_difficulty_sigma, object_num)
    '''ddad'''
    tmp_sort = []
    for i in range(0, source_num):
        tmp_sort.append(i)   
    for i in range(object_num):
        random.shuffle(tmp_sort)
        tmp_source_pool = tmp_sort[0:source_label_per_object]
        tmp_rate = simulate_source_rate[tmp_source_pool]
        tmp_rand = np.random.rand(source_label_per_object);
        #print(tmp_rate)
        #print(tmp_rand)
        for j in range(source_label_per_object):
            if tmp_rate[j] < tmp_rand[j]:
                label[i][2 * tmp_source_pool[j] + 2 - true_label[i]] = 1;
            else:
                label[i][2 * tmp_source_pool[j] + true_label[i] - 1] = 1;
    #print(simulate_source_rate)
    #print (label)
    return label
#=============================================================================================================
label = initial(1.5,16)
#print label[:][1]

#=============================================================================================================
tmp_encoder_h2 = [([0.0] * cate_num) for i in range(source_num * 2)]
for i in range (cate_num):
    for j in range(source_num * 2):
        if (i + j) % 2 == 0:
            tmp_encoder_h2[j][i] = 1.0
#print (tmp_encoder_h2)

tmp_decoder = [([0.0] * source_num * 2) for i in range(source_num * 2)]
for i in range (source_num * 2):
    tmp_decoder[i][i] = 1.0
    if i % 2 == 0:
        tmp_decoder[i][i + 1] = 1.0
        tmp_decoder[i + 1][i] = 1.0
print (tmp_decoder)
#=============================================================================================================
display_step = 1
learning_rate = 0.01
training_epochs = 20
n_input = source_num * 2 
n_hidden_left = n_input 
n_hidden_right = 2 
#print(n_input)
X = tf.placeholder("float", [None, n_input])
theta = tf.constant(0.5)
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_left],mean = 0.0, stddev = 0.01)),
    'encoder_h2': tf.Variable(tmp_encoder_h2),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_left, n_input],mean = 0.0, stddev = 0.01)),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_right, n_input],mean = 0.0, stddev = 0.01)),
}
biases = {
    'encoder_b1': tf.constant(0.0 ,shape=[n_hidden_left]),
    'encoder_b2': tf.constant(0.0, shape=[n_hidden_right]),
    'decoder_b1': tf.constant(0.0, shape=[n_input]),
    'decoder_b2': tf.constant(0.0, shape=[n_input]),
}
tmp_div = tf.Variable(tmp_decoder)
print (weights['encoder_h1'])
def encoder(x):
    print (x)
    # Encoder Hidden layer with sigmoid activation #1
    layer_left = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_right = tf.nn.softmax(tf.add(tf.matmul(x, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_left,layer_right

def decoder(x,y):
    tmp = tf.exp(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']) + tf.add(tf.matmul(y, weights['decoder_h2']),
                                   biases['decoder_b2']))

    layer_final = tf.div(tmp,tmp * tmp_div)
    return layer_final

encoder_left, encoder_right = encoder(X)
decoder_op = decoder(encoder_left, encoder_right)
y_pred = decoder_op
y_true = X
#cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
cost = tf.nn.sigmoid_cross_entropy_with_logits(y_pred, y_true) + tf.mul(theta, tf.nn.l2_loss(weights['encoder_h1'] - tmp_div))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        for i in range(object_num):
            _, c = sess.run([optimizer, cost], feed_dict={X: [label[i][:]]})
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{0}".format(c))



    
