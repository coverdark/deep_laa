import tensorflow as tf
import numpy as np
import deep_laa_support as dls
import sys

# read data
# filename = "web_processed_data_feature_2"
# filename = "age_data_3_category"
filename = "bluebird_data"
# filename = "flower_data"
data_all = np.load(filename+'.npz')
user_labels = data_all['user_labels']
label_mask = data_all['label_mask']
true_labels = data_all['true_labels']
category_size = data_all['category_num']
source_num = data_all['source_num']
n_samples, _ = np.shape(true_labels)

input_size = source_num * category_size
batch_size = n_samples

# define x
x = tf.placeholder(dtype=tf.float32, shape=(batch_size, input_size))
mask = tf.placeholder(dtype=tf.float32, shape=(batch_size, input_size))

# define source-wise template
source_wise_template = np.zeros((input_size, input_size), dtype=np.float32)
for i in range(input_size):
    source_wise_template[i*category_size:(i+1)*category_size, i*category_size:(i+1)*category_size] = 1
# define constraint template
constraint_template_encoder = np.matlib.repmat(np.eye(category_size), source_num, 1)
constraint_template_decoder = np.matlib.repmat(np.eye(category_size), 1, source_num)

# mv_y
mv_y = dls.get_majority_y(user_labels, category_size)

# define constant_y
constant_y = dls.get_constant_y(batch_size, category_size)

# x -> y1 (confusion matrix)
with tf.variable_scope('encoder_x_y1'):
    # network
    weights = tf.Variable(
        tf.truncated_normal(shape=(input_size, category_size), mean=0.0, stddev=0.01), name='w_encoder')
    biases = tf.Variable(
        tf.zeros(shape=([category_size]), dtype=tf.float32), name='b_encoder')
    y1 = tf.nn.softmax(
        tf.add(tf.matmul(x, weights), biases))
    # loss
    loss_w_encoder_l2 = tf.nn.l2_loss(weights - constraint_template_encoder)
    loss_b_encoder_l2 = tf.nn.l2_loss(biases)
    y1_encoder_constraint_strength = tf.placeholder(dtype=tf.float32)
    loss_y1_encoder = y1_encoder_constraint_strength/source_num/category_size/category_size * (loss_w_encoder_l2 + loss_b_encoder_l2)
    
# x -> y2 (weighted majority voting)
with tf.variable_scope('encoder_x_y2'):
    # network
    weights = tf.Variable(
        tf.truncated_normal(shape=(input_size, category_size), mean=0.0, stddev=0.01), name='w_encoder')
    biases = tf.Variable(
        tf.zeros(shape=([category_size]), dtype=tf.float32), name='b_encoder')
    y2 = tf.nn.softmax(
        tf.add(tf.matmul(x, weights), biases))
    # loss
    loss_w_encoder_l1 = tf.reduce_sum(tf.abs(weights))
    loss_b_encoder_l1 = tf.reduce_sum(tf.abs(biases))
    y2_encoder_constraint_strength = tf.placeholder(dtype=tf.float32)
    loss_y2_encoder = y2_encoder_constraint_strength/source_num/category_size/category_size * (loss_w_encoder_l1 + loss_b_encoder_l1)
            
# y1, y2 -> y
with tf.variable_scope('encoder_y1_y2_y'):
    # network
    weights_y1 = tf.Variable(
        tf.truncated_normal(shape=(category_size, category_size), mean=0.0, stddev=0.01), name='w_encoder')
    weights_y2 = tf.Variable(
        tf.truncated_normal(shape=(category_size, category_size), mean=0.0, stddev=0.01), name='w_encoder')
    biases = tf.Variable(
        tf.zeros(shape=([category_size]), dtype=tf.float32), name='b_encoder')
    y = tf.nn.softmax(
        tf.add(tf.matmul(y1, weights_y1) + tf.matmul(y2, weights_y2), biases))    
    # loss
    y_prior = tf.placeholder(dtype=tf.float32, shape=(batch_size, category_size))
    loss_y_kl = tf.reduce_mean(tf.reduce_sum(tf.mul(y, tf.log(1e-10 + y)) - tf.mul(y, tf.log(1e-10 + y_prior)), reduction_indices=1))
    y_kl_strength = tf.placeholder(dtype=tf.float32)
    loss_w1_encoder_l2 = tf.nn.l2_loss(weights_y1 - np.eye(category_size))
    loss_w2_encoder_l2 = tf.nn.l2_loss(weights_y2 - np.eye(category_size))
    loss_b_encoder_l2 = tf.nn.l2_loss(biases)
    y_encoder_constraint_strength = tf.placeholder(dtype=tf.float32)
    loss_y = y_kl_strength * loss_y_kl + \
        y_encoder_constraint_strength/category_size/category_size/2 * (loss_w1_encoder_l2 + loss_w2_encoder_l2 + loss_b_encoder_l2)

# y -> y1
with tf.variable_scope('decoder_y_y1'):
    # network
    weights = tf.Variable(
        tf.truncated_normal(shape=(category_size, category_size), mean=0.0, stddev=0.01), name='w_decoder')
    biases = tf.Variable(
        tf.zeros(shape=([category_size]), dtype=tf.float32), name='b_decoder')
    y1 = tf.nn.softmax(
        tf.add(tf.matmul(y, weights), biases))
    # loss 
    loss_w_decoder_l2 = tf.nn.l2_loss(weights - np.eye(category_size))
    loss_b_encoder_l2 = tf.nn.l2_loss(biases)
    y1_decoder_constraint_strength = tf.placeholder(dtype=tf.float32)
    loss_y1_decoder = y1_decoder_constraint_strength/category_size/category_size * (loss_w_decoder_l2 + loss_b_encoder_l2)

# y -> y2
with tf.variable_scope('decoder_y_y2'):
    # network
    weights = tf.Variable(
        tf.truncated_normal(shape=(category_size, category_size), mean=0.0, stddev=0.01), name='w_decoder')
    biases = tf.Variable(
        tf.zeros(shape=([category_size]), dtype=tf.float32), name='b_decoder')
    y2 = tf.nn.softmax(
        tf.add(tf.matmul(y, weights), biases))
    # loss 
    loss_w_decoder_l2 = tf.nn.l2_loss(weights - np.eye(category_size))
    loss_b_encoder_l2 = tf.nn.l2_loss(biases)
    y2_decoder_constraint_strength = tf.placeholder(dtype=tf.float32)
    loss_y2_decoder = y2_decoder_constraint_strength/category_size/category_size * (loss_w_decoder_l2 + loss_b_encoder_l2)


# y1 -> x1
with tf.variable_scope('decoder_y1_x1'):
    # network
    weights = tf.Variable(
        tf.truncated_normal(shape=(category_size, input_size), mean=0.0, stddev=.01), name='w_recons')
    biases = tf.Variable(
        tf.zeros(shape=([input_size]), dtype=tf.float32), name='b_recons')
    x_reconstr_tmp = tf.add(tf.matmul(y1, weights), biases)
    x_reconstr_1 = tf.div(tf.exp(x_reconstr_tmp), tf.matmul(tf.exp(x_reconstr_tmp), source_wise_template))
    #loss
    _tmp_cross_entropy = - tf.mul(x, tf.log(1e-10 + x_reconstr_1))
    # divide label numbers !!
    loss_x1_cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.mul(mask, _tmp_cross_entropy), reduction_indices=1))
    
    loss_w_decoder_l2 = tf.nn.l2_loss(weights - constraint_template_decoder)
    loss_b_decoder_l2 = tf.nn.l2_loss(biases)
    x1_decoder_constraint_strength = tf.placeholder(dtype=tf.float32)
    loss_x1_decoder = x1_decoder_constraint_strength/source_num/category_size/category_size * (loss_w_decoder_l2 + loss_b_decoder_l2)

# y2 -> x2
with tf.variable_scope('decoder_y2_x2'):
    # network
    weights = tf.Variable(
        tf.truncated_normal(shape=(category_size, input_size), mean=0.0, stddev=.01), name='w_recons')
    biases = tf.Variable(
        tf.zeros(shape=([input_size]), dtype=tf.float32), name='b_recons')
    x_reconstr_tmp = tf.add(tf.matmul(y2, weights), biases)
    x_reconstr_2 = tf.div(tf.exp(x_reconstr_tmp), tf.matmul(tf.exp(x_reconstr_tmp), source_wise_template))
    # loss
    _tmp_cross_entropy = - tf.mul(x, tf.log(1e-10 + x_reconstr_2))
    # divide label numbers !!
    loss_x2_cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.mul(mask, _tmp_cross_entropy), reduction_indices=1))

    loss_w_decoder_l1 = tf.reduce_sum(tf.abs(weights))
    loss_b_decoder_l1 = tf.reduce_sum(tf.abs(biases))
    x2_decoder_constraint_strength = tf.placeholder(dtype=tf.float32)
    loss_x2_decoder = x2_decoder_constraint_strength/source_num/category_size/category_size * (loss_w_decoder_l1 + loss_b_decoder_l1)

# loss
loss_x_cross_entropy = loss_x1_cross_entropy + loss_x2_cross_entropy
loss_overall = loss_x_cross_entropy \
     + loss_y1_encoder + loss_y2_encoder \
     + loss_y \
     + loss_y1_decoder + loss_y2_decoder \
     + loss_x1_decoder + loss_x2_decoder

y_target = tf.placeholder(dtype=tf.float32, shape=(batch_size, category_size))
_tmp_y_cross_entropy = - tf.mul(y_target, tf.log(1e-10 + y))
loss_pre_train = tf.reduce_mean(tf.reduce_sum(_tmp_y_cross_entropy, reduction_indices=1))

# optimizer
learning_rate = 0.02
optimizer_autoencoder = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_overall)
optimizer_pre_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_pre_train)

# evaluate with true labels
y_label = tf.placeholder(dtype=tf.int64, shape=(batch_size, 1))
inferred_category = tf.reshape(tf.argmax(y, 1), (batch_size, 1))
hit_num = tf.reduce_sum(tf.to_int32(tf.equal(inferred_category, y_label)))

# session
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    # pre-train
    epochs = 50
    total_batches = int(n_samples / batch_size)
    for epoch in xrange(epochs):
        total_hit = 0
        for batch in xrange(total_batches):
            batch_x, batch_mask, batch_y_label, batch_majority_y = user_labels, label_mask, true_labels, mv_y
            # x -> y, update classifier
            _, batch_y, batch_hit_num = sess.run(
                [optimizer_pre_train, y, hit_num], 
                feed_dict={x:batch_x, mask:batch_mask, y_label:batch_y_label, y_prior:mv_y, y_target:mv_y,
                           y1_encoder_constraint_strength:0.001,
                           y2_encoder_constraint_strength:0.001,
                           y_kl_strength:0.0001,
                           y_encoder_constraint_strength:0.001})
            total_hit += batch_hit_num             
        print "epoch: {0} accuracy: {1}".format(epoch, float(total_hit)/n_samples)
    
    
    epochs = 300
    total_batches = int(n_samples / batch_size)
    for epoch in xrange(epochs):
        total_cost = 0.0
        total_hit = 0
        for batch in xrange(total_batches):
            batch_x, batch_mask, batch_y_label, batch_majority_y = user_labels, label_mask, true_labels, mv_y
            # x -> y, update classifier
            _, batch_y, batch_hit_num = sess.run(
                [optimizer_autoencoder, y, hit_num], 
                feed_dict={x:batch_x, mask:batch_mask, y_label:batch_y_label, y_prior:mv_y,
                           y1_encoder_constraint_strength:0.1,
                           y2_encoder_constraint_strength:0.1,
                           y_kl_strength:0.001,
                           y_encoder_constraint_strength:0.1,
                           y1_decoder_constraint_strength:0.1,
                           y2_decoder_constraint_strength:0.1,
                           x1_decoder_constraint_strength:0.1,
                           x2_decoder_constraint_strength:0.1})
            
            total_hit += batch_hit_num
             
        print "epoch: {0} accuracy: {1}".format(epoch, float(total_hit)/n_samples)
        
        if epoch == epochs-1:
            debug_y1, debug_y2, _ = sess.run(
                [y1, y2, y],
                feed_dict={x:batch_x, mask:batch_mask, y_label:batch_y_label, y_prior:mv_y,
                           y1_encoder_constraint_strength:0.1,
                           y2_encoder_constraint_strength:0.1,
                           y_kl_strength:0.001,
                           y_encoder_constraint_strength:0.1,
                           y1_decoder_constraint_strength:0.1,
                           y2_decoder_constraint_strength:0.1,
                           x1_decoder_constraint_strength:0.1,
                           x2_decoder_constraint_strength:0.1})
            # print debug_z
            print "y1:", debug_y1
            print "y2:", debug_y2
        
print "Done!" 