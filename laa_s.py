import tensorflow as tf
import numpy as np
import deep_laa_support as dls
import sys

# read data
# filename = "bluebird_data"
# filename = "flower_data"
# filename = "web_processed_data_feature_2"
filename = "age_data_3_category"
data_all = np.load(filename+'.npz')
user_labels = data_all['user_labels']
label_mask = data_all['label_mask']
true_labels = data_all['true_labels']
category_size = data_all['category_num']
source_num = data_all['source_num']
n_samples, _ = np.shape(true_labels)

majority_y = dls.get_majority_y(user_labels, category_size)

input_size = source_num * category_size
batch_size = n_samples

# define x
x = tf.placeholder(dtype=tf.float32, shape=(batch_size, input_size))
mask = tf.placeholder(dtype=tf.float32, shape=(batch_size, input_size))

# define source0-wise template
source_wise_template = np.zeros((input_size, input_size), dtype=np.float32)
for i in range(input_size):
    source_wise_template[i*category_size:(i+1)*category_size, i*category_size:(i+1)*category_size] = 1

# define constant_y
constant_y = dls.get_constant_y(batch_size, category_size)

# y -> x
with tf.name_scope('decoder_y_x'):
    # reconstruct
    weights_reconstr = tf.Variable(
        tf.truncated_normal(shape=(category_size, input_size), mean=0.0, stddev=.01), name='w_recons')
    biases_reconstr = tf.Variable(
        tf.zeros(shape=([input_size]), dtype=tf.float32), name='b_recons')

    def reconstruct_x_y(y):
        x_reconstr_tmp = tf.add(tf.matmul(y, weights_reconstr), biases_reconstr)
        x_reconstr = tf.div(tf.exp(x_reconstr_tmp), tf.matmul(tf.exp(x_reconstr_tmp), source_wise_template))
        return x_reconstr

    tmp_reconstr = []
    for i in range(category_size):
        _tmp_reconstr_x = reconstruct_x_y(constant_y[i])
        _tmp_cross_entropy = - tf.mul(x, tf.log(1e-10 + _tmp_reconstr_x))
        tmp_reconstr.append(tf.reduce_mean(tf.mul(mask, _tmp_cross_entropy), reduction_indices=1, keep_dims=True))
    reconstr_x = tf.concat(1, tmp_reconstr)

    print "y -> x, OK"

# x -> y
with tf.name_scope('classifier'):
    # classifier y
    weights_y_classifier = tf.Variable(
        tf.truncated_normal(shape=(input_size, category_size), mean=0.0, stddev=.01), name='w_h_2_encoder')
    biases_y_classifier = tf.Variable(
        tf.zeros(shape=([category_size]), dtype=tf.float32), name='b_h_2_encoder')
    y_classifier = tf.nn.softmax(
        tf.add(tf.matmul(x, weights_y_classifier), biases_y_classifier))
    
    print "x -> y, OK"

# constraints
# classifier
constraint_template_classifier = np.matlib.repmat(np.eye(category_size), source_num, 1)
loss_w_classifier = tf.nn.l2_loss(weights_y_classifier - constraint_template_classifier)
# decoder
constraint_template_decoder = np.matlib.repmat(np.eye(category_size), 1, source_num)
loss_w_decoder = tf.nn.l2_loss(weights_reconstr - constraint_template_decoder)

# loss VAE by given y_prob
y_prob = tf.placeholder(dtype=tf.float32, shape=(batch_size, category_size))
loss_reconstr = tf.reduce_mean(
    tf.reduce_sum(tf.mul(reconstr_x, y_prob), reduction_indices=1))
loss_VAE = loss_reconstr + 0.0001/source_num * loss_w_decoder

# loss classifier
y_target = tf.placeholder(dtype=tf.float32, shape=(batch_size, category_size))
_tmp_classifier_cross_entropy = - tf.mul(y_target, tf.log(1e-10 + y_classifier))
loss_classifier_x_y = tf.reduce_mean(tf.reduce_sum(_tmp_classifier_cross_entropy, reduction_indices=1))

# tf.constant(reconstr_all): constant?
_tmp_loss_backprop = tf.mul(y_classifier, reconstr_x)
loss_classifier_y_x = tf.reduce_mean(tf.reduce_sum(_tmp_loss_backprop, reduction_indices=1))
y_prior = tf.placeholder(dtype=tf.float32, shape=(batch_size, category_size))
loss_y_kl = tf.reduce_mean(tf.reduce_sum(tf.mul(y_classifier, tf.log(1e-10 + y_classifier)) - tf.mul(y_classifier, tf.log(1e-10 + y_prior)), reduction_indices=1))
y_kl_strength = tf.placeholder(dtype=tf.float32)
loss_classifier = loss_classifier_y_x + y_kl_strength * loss_y_kl + 0.0001/source_num * (loss_w_classifier + loss_w_decoder)

# optimizer
learning_rate = 0.02
optimizer_classifier_x_y = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_classifier_x_y)
optimizer_VAE = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_VAE)
optimizer_classifier = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_classifier)

# evaluate with true labels
y_label = tf.placeholder(dtype=tf.int64, shape=(batch_size, 1))
inferred_category = tf.reshape(tf.argmax(y_classifier, 1), (batch_size, 1))
hit_num = tf.reduce_sum(tf.to_int32(tf.equal(inferred_category, y_label)))

# session
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    # initialize x -> y
    epochs = 100
    total_batches = int(n_samples / batch_size)
    for epoch in xrange(epochs):
        total_cost = 0.0
        total_hit = 0
        for batch in xrange(total_batches):
            batch_x, batch_mask, batch_y_label, batch_majority_y = user_labels, label_mask, true_labels, majority_y
            # x -> y, update classifier
            _, batch_y_classifier, batch_hit_num = sess.run(
                [optimizer_classifier_x_y, y_classifier, hit_num], 
                feed_dict={x:batch_x, mask:batch_mask, y_label:batch_y_label, y_target:batch_majority_y})
            total_hit += batch_hit_num        
            
#             if epoch == epochs-1:
#                 debug_y_classifier = sess.run(
#                     [y_classifier],
#                     feed_dict={x:batch_x, mask:batch_mask})
#                 print debug_y_classifier
                
        print "epoch: {0} accuracy: {1}".format(epoch, float(total_hit) / n_samples)
    
    epochs = 500
    total_batches = int(n_samples / batch_size)
    for epoch in xrange(epochs):
        total_cost = 0.0
        total_hit = 0
        for batch in xrange(total_batches):
            batch_x, batch_mask, batch_y_label, batch_majority_y = user_labels, label_mask, true_labels, majority_y
            # get y_prob from classifier x -> y
            _y_prob_classifier = sess.run([y_classifier], feed_dict={x:batch_x})
            # x, y -> x, update VAE
#             y_prob_classifier = np.reshape(_y_prob_classifier, (batch_size, category_size))
#             _, batch_loss= sess.run(
#                 [optimizer_VAE, loss_VAE],
#                 feed_dict={x:batch_x, mask:batch_mask, y_prob:y_prob_classifier})
            # x -> y, update classifier
            _, batch_y_classifier, batch_hit_num = sess.run(
                [optimizer_classifier, y_classifier, hit_num], 
                feed_dict={x:batch_x, mask:batch_mask, y_label:batch_y_label, y_prior:batch_majority_y, y_kl_strength:0.0001})
            
            total_hit += batch_hit_num
            
#             if epoch == epochs-1:
#                 print 'z'
#                 print batch_z
#                 print 'z_mean'
#                 print batch_z_mean
#                 print 'z_log_sigma_sq'
#                 print batch_z_log_sigma_sq
             
            # debug output
#             if epoch == epochs-1:
#                 debug_y_classifier = sess.run(
#                     [y_classifier],
#                     feed_dict={x:batch_x, mask:batch_mask, y_prob:y_prob_classifier})
#                 print debug_y_classifier
             
        print "epoch: {0} accuracy: {1}".format(epoch, float(total_hit)/n_samples)
            
print "Done!"            
                