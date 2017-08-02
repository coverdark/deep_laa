import tensorflow as tf
import numpy as np
import deep_laa_support as dls
import sys
import scipy.sparse
from numpy import save
import matplotlib.pyplot as plt

# read data
# filename = "web_processed_data_feature_2"
# filename = "bluebird_data"
filename = "flower_data"
if not filename == "millionaire_non_empty_sparse":
    data_all = np.load(filename+'.npz')
    user_labels = data_all['user_labels']
    label_mask = data_all['label_mask']
    true_labels = data_all['true_labels']
    category_size = data_all['category_num']
    source_num = data_all['source_num']
    n_samples, _ = np.shape(true_labels)
else:
    data_all = scipy.io.loadmat(filename+'.mat')
    user_labels = data_all['user_labels']
    label_mask = data_all['label_mask']
    true_labels = data_all['true_labels']
    category_size = data_all['category_num'][0,0]
    source_num = data_all['source_num'][0,0]
    n_samples, _ = np.shape(true_labels)

mv_y = dls.get_majority_y(user_labels, source_num, category_size)

input_size = source_num * category_size
batch_size = n_samples

n_z = 2 # number of latent aspects
flag_deep_z = False

# define x
x = tf.placeholder(dtype=tf.float32, shape=(batch_size, input_size))
mask = tf.placeholder(dtype=tf.float32, shape=(batch_size, input_size))

# define source-wise template
source_wise_template = np.zeros((input_size, input_size), dtype=np.float32)
for i in range(input_size):
    source_wise_template[i*category_size:(i+1)*category_size, i*category_size:(i+1)*category_size] = 1

# define constant_y
constant_y = dls.get_constant_y(batch_size, category_size)

# x -> z
with tf.variable_scope('encoder_x_z'):
    if not flag_deep_z:
        weights = tf.Variable(
            tf.truncated_normal(shape=(input_size, n_z), mean=0.0, stddev=0.01), name='w_encoder')
        biases = tf.Variable(
            tf.zeros(shape=([n_z]), dtype=tf.float32), name='b_encoder')
        z = tf.nn.softplus(tf.add(tf.matmul(x, weights), biases))
#         z = tf.nn.sigmoid(tf.add(tf.matmul(x, weights), biases))
    else:
        n_hz = 10
        weights_1 = tf.Variable(
            tf.truncated_normal(shape=(input_size, n_hz), mean=0.0, stddev=0.01), name='w_encoder')
        biases_1 = tf.Variable(
            tf.zeros(shape=([n_hz]), dtype=tf.float32), name='b_encoder')
        hz = tf.nn.softplus(tf.add(tf.matmul(x, weights_1), biases_1))
        weights_2 = tf.Variable(
            tf.truncated_normal(shape=(n_hz, n_z), mean=0.0, stddev=0.01), name='w_encoder')
        biases_2 = tf.Variable(
            tf.zeros(shape=([n_z]), dtype=tf.float32), name='b_encoder')
        z = tf.nn.softplus(tf.add(tf.matmul(hz, weights_2), biases_2))
        print "deep z"
    
    u = tf.Variable(
        tf.random_uniform(shape=(input_size, n_z), minval=1.0, maxval=2.0))
    r = tf.matmul(z, u, transpose_b=True)
    loss_mf = tf.nn.l2_loss(tf.mul(mask, (x - r)))
    loss_z = tf.nn.l2_loss(z)
    loss_u = tf.nn.l2_loss(u)
    
    print "x -> z, OK"
    
    # loss
    loss_z_entropy = - tf.reduce_mean(tf.reduce_sum(tf.mul(z, tf.log(z+1e-10)), reduction_indices=1))
    loss_z_mean = tf.reduce_sum(tf.square((tf.reduce_mean(z, reduction_indices=0) - 1.0/n_z*np.ones(shape=[1, n_z]))))
    if not flag_deep_z:
        loss_z_weights_l2 = tf.nn.l2_loss(weights)
        loss_z_biases_l2 = tf.nn.l2_loss(biases)
        loss_z_weights_l1 = tf.reduce_sum(tf.abs(weights))
        loss_z_biases_l1 = tf.reduce_sum(tf.abs(biases))
    else:
        loss_z_weights_l2 = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2)
        loss_z_biases_l2 = tf.nn.l2_loss(biases_1) + tf.nn.l2_loss(biases_2)
        loss_z_weights_l1 = tf.reduce_sum(tf.abs(weights_1)) + tf.reduce_sum(tf.abs(weights_2))
        loss_z_biases_l1 = tf.reduce_sum(tf.abs(biases_1)) + tf.reduce_sum(tf.abs(biases_2))
    loss_z_l1 = tf.reduce_sum(tf.abs(z))
    
# x, z -> y    
with tf.variable_scope('encoder_x_z_y'):
    c_weights = [tf.Variable(
        tf.truncated_normal(shape=(input_size, category_size), mean=0.0, stddev=0.01), name='w_encoder')
        for i in range(n_z)]
    biases = [tf.Variable(
        tf.zeros(shape=([category_size]), dtype=tf.float32), name='b_encoder')
        for i in range(n_z)]
    tmp_y = [tf.add(tf.matmul(x, c_weights[i]), biases[i]) for i in range(n_z)]
    for i in range(n_z):
        tmp_z = tf.slice(z, [0, i], [-1, 1])
        if i == 0:
            tmp_y_2 = tf.mul(tmp_y[i], tf.tile(tmp_z, [1, category_size]))
        else:
            tmp_y_2 += tf.mul(tmp_y[i], tf.tile(tmp_z, [1, category_size]))
    y = tf.nn.softmax(tmp_y_2)
    print "x, z -> y, OK"

    # loss
    constraint_template_classifier = np.matlib.repmat(np.eye(category_size), source_num, 1)
    for i in range(n_z):
        if i == 0:
            loss_w_classifier_l2 = tf.nn.l2_loss(c_weights[i] - constraint_template_classifier)
            # loss_w_classifier_l1 = tf.reduce_sum(tf.abs(weights[i] - constraint_template_classifier))
            loss_w_classifier_l1 = tf.reduce_sum(tf.abs(c_weights[i]))
            loss_b_classifier_l2 = tf.nn.l2_loss(biases[i])
            loss_b_classifier_l1 = tf.reduce_sum(tf.abs(biases[i]))
        else:
            loss_w_classifier_l2 += tf.nn.l2_loss(c_weights[i] - constraint_template_classifier)
            # loss_w_classifier_l1 += tf.reduce_sum(tf.abs(weights[i] - constraint_template_classifier))
            loss_w_classifier_l1 += tf.reduce_sum(tf.abs(c_weights[i]))
            loss_b_classifier_l2 += tf.nn.l2_loss(biases[i])
            loss_b_classifier_l1 += tf.reduce_sum(tf.abs(biases[i]))
    if n_z > 1:
        loss_w_diff = -tf.nn.l2_loss(c_weights[0] - c_weights[1])

# y, z -> x
with tf.variable_scope('decoder_yz_x'):
    weights = [tf.Variable(
        tf.truncated_normal(shape=(category_size, input_size), mean=0.0, stddev=0.01), name='w_decoder')
        for i in range(n_z)]
    biases = [tf.Variable(
        tf.zeros(shape=([input_size]), dtype=tf.float32), name='b_decoder')
        for i in range(n_z)]
    
    def reconstruct_yz_x(y, z):
        tmp_x = [tf.add(tf.matmul(y, weights[i]), biases[i]) for i in range(n_z)]
        tmp_x_2 = []
        for i in range(n_z):
            tmp_z = tf.slice(z, [0, i], [-1, 1])
            if i == 0:
                tmp_x_2 = tf.mul(tmp_x[i], tf.tile(tmp_z, [1, input_size]))
            else:
                tmp_x_2 += tf.mul(tmp_x[i], tf.tile(tmp_z, [1, input_size]))
        x_reconstr = tf.div(tf.exp(tmp_x_2), tf.matmul(tf.exp(tmp_x_2), source_wise_template))
        return x_reconstr
    
    tmp_reconstr = []
    for i in range(category_size):
        _tmp_reconstr_x = reconstruct_yz_x(constant_y[i], z)
        _tmp_cross_entropy = - tf.mul(x, tf.log(1e-10 + _tmp_reconstr_x))
        tmp_reconstr.append(tf.reduce_mean(tf.mul(mask, _tmp_cross_entropy), reduction_indices=1, keep_dims=True))
    reconstr_x = tf.concat(1, tmp_reconstr)
    print "y, z -> x, OK"

    # loss
    constraint_template_decoder = np.matlib.repmat(np.eye(category_size), 1, source_num)
    for i in range(n_z):
        if i == 0:
            loss_w_decoder_l2 = tf.nn.l2_loss(weights[i] - constraint_template_decoder)
            # loss_w_decoder_l1 = tf.reduce_sum(tf.abs(weights[i] - constraint_template_decoder))
            loss_w_decoder_l1 = tf.reduce_sum(tf.abs(weights[i]))
            loss_b_decoder_l2 = tf.nn.l2_loss(biases[i] - np.zeros([input_size]))
            loss_b_decoder_l1 = tf.reduce_sum(tf.abs(biases[i]))
        else:
            loss_w_decoder_l2 += tf.nn.l2_loss(weights[i] - constraint_template_decoder)
            # loss_w_decoder_l1 += tf.reduce_sum(tf.abs(weights[i] - constraint_template_decoder))
            loss_w_decoder_l1 += tf.reduce_sum(tf.abs(weights[i]))
            loss_b_decoder_l2 += tf.nn.l2_loss(biases[i] - np.zeros([input_size]))
            loss_b_decoder_l1 += tf.reduce_sum(tf.abs(biases[i]))

# loss classifier
y_target = tf.placeholder(dtype=tf.float32, shape=(batch_size, category_size))
_tmp_classifier_cross_entropy = - tf.mul(y_target, tf.log(1e-10 + y))
loss_classifier_x_y = tf.reduce_mean(tf.reduce_sum(_tmp_classifier_cross_entropy, reduction_indices=1))

_tmp_loss_backprop = tf.mul(y, reconstr_x)
loss_classifier_y_x = tf.reduce_mean(tf.reduce_sum(_tmp_loss_backprop, reduction_indices=1))
y_prior = tf.placeholder(dtype=tf.float32, shape=(batch_size, category_size))
loss_y_kl = tf.reduce_mean(tf.reduce_sum(tf.mul(y, tf.log(1e-10 + y)) - tf.mul(y, tf.log(1e-10 + y_prior)), reduction_indices=1))
# use proper parameters
loss_classifier = loss_classifier_y_x \
    + 0.0001*loss_y_kl \
    + 0.05/source_num/category_size/category_size * (loss_w_classifier_l1 + loss_b_classifier_l1 + loss_w_decoder_l1 + loss_b_decoder_l1) \
    + 0.1/source_num/n_z/n_z * (loss_z_weights_l2 + loss_z_biases_l2)

# optimizer
learning_rate = 0.001
optimizer_classifier_x_y = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_classifier_x_y)
# optimizer_VAE = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_VAE)
optimizer_classifier = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_classifier)

# evaluate with true labels
y_label = tf.placeholder(dtype=tf.int64, shape=(batch_size, 1))
inferred_category = tf.reshape(tf.argmax(y, 1), (batch_size, 1))
hit_num = tf.reduce_sum(tf.to_int32(tf.equal(inferred_category, y_label)))

# session
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    # initialize x -> y
    print "Initialize x -> y ..."
    epochs = 50
    total_batches = int(n_samples / batch_size)
    for epoch in xrange(epochs):
        total_hit = 0
        for batch in xrange(total_batches):
            batch_x, batch_mask, batch_y_label, batch_majority_y = user_labels, label_mask, true_labels, mv_y
            # x -> y, update classifier
            _, batch_y_classifier, batch_hit_num = sess.run(
                [optimizer_classifier_x_y, y, hit_num], 
                feed_dict={x:batch_x, y_label:batch_y_label, y_target:batch_majority_y})
            total_hit += batch_hit_num
                
        print "epoch: {0} accuracy: {1}".format(epoch, float(total_hit) / n_samples)
    
    print "Train the whole network ..."
    epochs = 500
    total_batches = int(n_samples / batch_size)
    for epoch in xrange(epochs):
        total_hit = 0
        for batch in xrange(total_batches):
            batch_x, batch_mask, batch_y_label, batch_majority_y = user_labels, label_mask, true_labels, mv_y
            # get y_prob from classifier x -> y
            _y_prob_classifier = sess.run([y], feed_dict={x:batch_x})
            # x -> y, update classifier
            _, batch_y_classifier, batch_hit_num = sess.run(
                [optimizer_classifier, y, hit_num], 
                feed_dict={x:batch_x, mask:batch_mask, y_label:batch_y_label, y_prior:batch_majority_y})
            total_hit += batch_hit_num
              
        print "epoch: {0} accuracy: {1}".format(epoch, float(total_hit)/n_samples)            
print "Done!" 