import tensorflow as tf
import numpy as np

n_input_size = 2
n_batch_size = 10
n_output_size = 1
learning_rate = 0.01

data = np.random.normal(0.0, 1.0, [n_batch_size, n_input_size])
label = np.random.binomial(1, 0.5, [n_batch_size, 1])

input_units = tf.placeholder(dtype=tf.float32, shape=(n_batch_size, n_input_size))
labels = tf.placeholder(dtype=tf.int32, shape=(n_batch_size, n_output_size))

weights = tf.Variable(tf.truncated_normal(shape=(n_input_size, n_output_size)))
biases = tf.Variable(tf.zeros(shape=([n_output_size]), dtype=tf.float32))
output_units = tf.add(tf.matmul(input_units, weights), biases)

loss = tf.reduce_mean(tf.square(tf.add(output_units, -tf.to_float(labels))))
# train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

feed_dict = {input_units: data,
        labels: label}

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    print sess.run(loss, feed_dict=feed_dict)