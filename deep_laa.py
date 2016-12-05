import tensorflow as tf
import numpy as np

input_size = 784
batch_size = 100
n_z = 10
category_size = 2

x = tf.placeholder(dtype=tf.float32, shape=(batch_size, input_size))

n_hidden_encoder = [500, 501]
n_hidden_decoder = [500, 501]
n_hidden_classifier = [500, 501]

# !!!!
pass
source_wise_template = tf.constant(0.0, dtype=tf.float32, shape=(input_size, input_size))

# encoder h_1
weights_h_1_encoder = tf.Variable(tf.truncated_normal(shape=(input_size, n_hidden_encoder[0]), mean=0.0, stddev=.01), name='w_h_1_encoder')
biases_h_1_encoder = tf.Variable(tf.zeros(shape=([n_hidden_encoder[0]]), dtype=tf.float32), name='b_h_1_encoder')
h_1_encoder = tf.nn.softplus(tf.add(tf.matmul(x, weights_h_1_encoder), biases_h_1_encoder))

# encoder h_2
weights_h_2_encoder = tf.Variable(tf.truncated_normal(shape=(n_hidden_encoder[0], n_hidden_encoder[1]), mean=0.0, stddev=.01), name='w_h_2_encoder')
biases_h_2_encoder = tf.Variable(tf.zeros(shape=([n_hidden_encoder[1]]), dtype=tf.float32), name='b_h_2_encoder')
h_2_encoder = tf.nn.softplus(tf.add(tf.matmul(h_1_encoder, weights_h_2_encoder), biases_h_2_encoder))

# z
weights_z_mean = tf.Variable(tf.truncated_normal(shape=(n_hidden_encoder[1], n_z), mean=0.0, stddev=.01), name='w_z_mean')
biases_z_mean = tf.Variable(tf.zeros(shape=([n_z]), dtype=tf.float32), name='b_z_mean')
z_mean = tf.add(tf.matmul(h_2_encoder, weights_z_mean), biases_z_mean)
weights_z_log_sigma_sq = tf.Variable(tf.truncated_normal(shape=(n_hidden_encoder[1], n_z), mean=0.0, stddev=.01), name='w_z_sigma')
biases_z_log_sigma_sq = tf.Variable(tf.zeros(shape=([n_z]), dtype=tf.float32), name='b_z_sigma')
z_log_sigma_sq = tf.add(tf.matmul(h_2_encoder, weights_z_log_sigma_sq), biases_z_log_sigma_sq)

eps = tf.random_normal(shape=(batch_size, n_z), mean=0.0, stddev=1.0, dtype=tf.float32)
z = tf.add(z_mean, tf.mul(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))

# y
y = tf.placeholder(dtype=tf.float32, shape=(batch_size, category_size))

# decoder h_1
weights_h_z_decoder = tf.Variable(tf.truncated_normal(shape=(n_z, n_hidden_decoder[0]), mean=0.0, stddev=.01), name='w_h_1_decoder')
weights_h_y_decoder = tf.Variable(tf.truncated_normal(shape=(category_size, n_hidden_decoder[0]), mean=0.0, stddev=.01), name='w_h_1_decoder')
biases_h_1_decoder = tf.Variable(tf.zeros(shape=([n_hidden_decoder[0]]), dtype=tf.float32), name='b_h_1_decoder')
h_1_decoder = tf.nn.softplus(tf.add(tf.matmul(z, weights_h_z_decoder) + tf.matmul(y, weights_h_y_decoder), biases_h_1_decoder))

# decoder h_2
weights_h_2_decoder = tf.Variable(tf.truncated_normal(shape=(n_hidden_decoder[0], n_hidden_decoder[1]), mean=0.0, stddev=.01), name='w_h_2_decoder')
biases_h_2_decoder = tf.Variable(tf.zeros(shape=([n_hidden_decoder[1]]), dtype=tf.float32), name='b_h_2_decoder')
h_2_decoder = tf.nn.softplus(tf.add(tf.matmul(h_1_decoder, weights_h_2_decoder), biases_h_2_decoder))

# reconstruct
weights_reconstr = tf.Variable(tf.truncated_normal(shape=(n_hidden_decoder[1], input_size), mean=0.0, stddev=.01), name='w_recons')
biases_reconstr = tf.Variable(tf.zeros(shape=([input_size]), dtype=tf.float32), name='b_recons')
x_reconstr_tmp = tf.add(tf.matmul(h_2_decoder, weights_reconstr), biases_reconstr)
x_reconstr = tf.div(tf.exp(x_reconstr_tmp), tf.matmul(tf.exp(x_reconstr_tmp), source_wise_template))

# x -> y
# classifier h_1
weights_h_1_classifier = tf.Variable(tf.truncated_normal(shape=(input_size, n_hidden_classifier[0]), mean=0.0, stddev=.01), name='w_h_1_encoder')
biases_h_1_classifier = tf.Variable(tf.zeros(shape=([n_hidden_encoder[0]]), dtype=tf.float32), name='b_h_1_encoder')
h_1_classifier = tf.nn.softplus(tf.add(tf.matmul(x, weights_h_1_classifier), biases_h_1_classifier))

# classifier h_2
weights_h_2_classifier = tf.Variable(tf.truncated_normal(shape=(n_hidden_encoder[0], n_hidden_encoder[1]), mean=0.0, stddev=.01), name='w_h_2_encoder')
biases_h_2_classifier = tf.Variable(tf.zeros(shape=([n_hidden_encoder[1]]), dtype=tf.float32), name='b_h_2_encoder')
h_2_classifier = tf.nn.softplus(tf.add(tf.matmul(h_1_classifier, weights_h_2_classifier), biases_h_2_classifier))

# classifier y
weights_y_classifier = tf.Variable(tf.truncated_normal(shape=(n_hidden_encoder[1], category_size), mean=0.0, stddev=.01), name='w_h_2_encoder')
biases_y_classifier = tf.Variable(tf.zeros(shape=([category_size]), dtype=tf.float32), name='b_h_2_encoder')
y_classifier = tf.nn.softmax(tf.add(tf.matmul(h_2_classifier, weights_y_classifier), biases_y_classifier))

# loss_VAE
# !!!! y_prob
pass
loss_reconstr = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(x_reconstr, x))
loss_z = -0.5 * tf.reduce_mean(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq))
loss_VAE = loss_reconstr + loss_z

# loss classifier
pass
loss_classifier = sigmoid_cross_entropy_with_logits(y_classifier, loss_VAE) # !!!!

# optimizer
learning_rate = 0.001
optimizer_VAE = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_VAE)
optimizer_classifier = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_classifier)

def read_data():
    pass
    return x, y

# session
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    epochs = 100
    total_batches = int(n_samples / batch_size)
    for epoch in xrange(epochs):
        total_cost = 0.0
        for batch in xrange(total_batches):
            batch_x, batch_y = read_data(batch_size)
            feed_dict = {x:batch_x}
            y_prob = sess.run([y_classifier], feed_dict=feed_dict)
            feed_dict = {x:batch_x, y:y_prob}
            _, batch_loss_VAE, _, batch_loss_classifier = sess.run([optimizer_VAE, loss_VAE, optimizer_classifier, loss_classifier], feed_dict=feed_dict)
            