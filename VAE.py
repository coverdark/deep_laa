import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# load data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples
print("Number of samples {} Shape of y[{}] Shape of X[{}]".format(n_samples, mnist.train.labels.shape, mnist.train.images.shape))
# plt.imshow(np.reshape(-mnist.train.images[4242], (28, 28)), interpolation='none', cmap=plt.get_cmap('gray'))
# plt.show()

n_z = 10
input_size = 784
batch_size = 100

x = tf.placeholder(dtype=tf.float32, shape=(batch_size, input_size))

n_hidden_encoder = [500, 501]

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

n_hidden_decoder = [500, 501]

# decoder h_1
weights_h_1_decoder = tf.Variable(tf.truncated_normal(shape=(n_z, n_hidden_decoder[0]), mean=0.0, stddev=.01), name='w_h_1_decoder')
biases_h_1_decoder = tf.Variable(tf.zeros(shape=([n_hidden_decoder[0]]), dtype=tf.float32), name='b_h_1_decoder')
h_1_decoder = tf.nn.softplus(tf.add(tf.matmul(z, weights_h_1_decoder), biases_h_1_decoder))

# decoder h_2
weights_h_2_decoder = tf.Variable(tf.truncated_normal(shape=(n_hidden_decoder[0], n_hidden_decoder[1]), mean=0.0, stddev=.01), name='w_h_2_decoder')
biases_h_2_decoder = tf.Variable(tf.zeros(shape=([n_hidden_decoder[1]]), dtype=tf.float32), name='b_h_2_decoder')
h_2_decoder = tf.nn.softplus(tf.add(tf.matmul(h_1_decoder, weights_h_2_decoder), biases_h_2_decoder))

# reconstruct
weights_reconstr = tf.Variable(tf.truncated_normal(shape=(n_hidden_decoder[1], input_size), mean=0.0, stddev=.01), name='w_recons')
biases_reconstr = tf.Variable(tf.zeros(shape=([input_size]), dtype=tf.float32), name='b_recons')
x_reconstr_mean = tf.add(tf.matmul(h_2_decoder, weights_reconstr), biases_reconstr)

# loss
loss_reconstr = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(x_reconstr_mean, x))
# loss_reconstr = -tf.reduce_sum(x * tf.log(1e-10 + x_reconstr_mean) + (1-x) * tf.log(1e-10 + 1 - x_reconstr_mean), 1)
loss_z = -0.5 * tf.reduce_mean(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq))
loss_all = loss_reconstr + loss_z

# optimizer
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_all)

# train session
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    epochs = 100
    total_batches = int(n_samples / batch_size)
    for epoch in xrange(epochs):
        total_cost = 0.0
        for batch in xrange(total_batches):
            batch_data, _ = mnist.train.next_batch(batch_size)
            _, batch_cost, batch_recons, batch_z, x_recons, print_z = sess.run([optimizer, loss_all, loss_reconstr, loss_z, x_reconstr_mean, z], feed_dict={x:batch_data})
            if batch == 0:
                print "batch {0}, recons: {1}, z: {2}, x_recons: {3}, z: {4}".format(batch, batch_recons, batch_z, x_recons, print_z)
        total_cost += batch_cost * batch_size / n_samples
        print "epoch {0}, cost: {1}".format(epoch, total_cost)
    