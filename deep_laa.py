import tensorflow as tf
import numpy as np
import sys

def gen_data(filename, data_num, source_num, category_num):
    data_label_vectors = np.zeros((data_num, source_num*category_num))
    _tmp = np.random.multinomial(1, [1./category_num]*category_num, size=data_num*source_num)
    for i in range(data_num):
        for j in range(source_num):
            data_label_vectors[i, category_num*j:category_num*(j+1)] = _tmp[i*source_num+j, :]
    data_y_labels = np.argmax(np.random.multinomial(1, [1./category_num]*category_num, size=data_num), axis=1)
    np.savez(filename, data=data_label_vectors, labels=np.reshape(data_y_labels, (data_num, 1)))


source_num = 5
category_size = 2
n_samples = 10
input_size = source_num * category_size
batch_size = n_samples
n_z = 2

filename = 'test_data'
gen_data(filename, n_samples, source_num, category_size)

data_all = np.load(filename+'.npz')
data = data_all['data']
y_labels = data_all['labels']


x = tf.placeholder(dtype=tf.float32, shape=(batch_size, input_size))

n_hidden_encoder = [10, 5]
n_hidden_decoder = [5, 10]
n_hidden_classifier = [10, 5]

def get_constant_y():
    constant_y = {}
    for i in range(category_size):
        constant_tmp = np.zeros((batch_size, category_size), dtype=np.float32)
        constant_tmp[:, i] = 1.0;
        constant_y[i] = tf.constant(constant_tmp)
    return constant_y

source_wise_template = np.zeros((input_size, input_size), dtype=np.float32)
for i in range(input_size):
    source_wise_template[i*category_size:(i+1)*category_size, i*category_size:(i+1)*category_size] = 1

constant_y = get_constant_y()

# x -> z

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

# z, y -> x
# decoder h_1
weights_h_z_decoder = tf.Variable(tf.truncated_normal(shape=(n_z, n_hidden_decoder[0]), mean=0.0, stddev=.01), name='w_h_1_decoder')
weights_h_y_decoder = tf.Variable(tf.truncated_normal(shape=(category_size, n_hidden_decoder[0]), mean=0.0, stddev=.01), name='w_h_1_decoder')
biases_h_1_decoder = tf.Variable(tf.zeros(shape=([n_hidden_decoder[0]]), dtype=tf.float32), name='b_h_1_decoder')
# decoder h_2
weights_h_2_decoder = tf.Variable(tf.truncated_normal(shape=(n_hidden_decoder[0], n_hidden_decoder[1]), mean=0.0, stddev=.01), name='w_h_2_decoder')
biases_h_2_decoder = tf.Variable(tf.zeros(shape=([n_hidden_decoder[1]]), dtype=tf.float32), name='b_h_2_decoder')
# reconstruct
weights_reconstr = tf.Variable(tf.truncated_normal(shape=(n_hidden_decoder[1], input_size), mean=0.0, stddev=.01), name='w_recons')
biases_reconstr = tf.Variable(tf.zeros(shape=([input_size]), dtype=tf.float32), name='b_recons')


def reconstruct_x_z_y(z, y):
    h_1_decoder = tf.nn.softplus(tf.add(tf.matmul(z, weights_h_z_decoder) + tf.matmul(y, weights_h_y_decoder), biases_h_1_decoder))
    h_2_decoder = tf.nn.softplus(tf.add(tf.matmul(h_1_decoder, weights_h_2_decoder), biases_h_2_decoder))
    x_reconstr_tmp = tf.add(tf.matmul(h_2_decoder, weights_reconstr), biases_reconstr)
    x_reconstr = tf.div(tf.exp(x_reconstr_tmp), tf.matmul(tf.exp(x_reconstr_tmp), source_wise_template))
    return x_reconstr

tmp_reconstr = []
for i in range(category_size):
    _tmp_reconstr_x = reconstruct_x_z_y(z, constant_y[i])
    _tmp_cross_enrtopy = -(tf.mul(x, tf.log(1e-10 + _tmp_reconstr_x)) +  tf.mul((1-x), tf.log(1 - 1e-10 + _tmp_reconstr_x)))
    tmp_reconstr.append(tf.reduce_mean(_tmp_cross_enrtopy, reduction_indices=1, keep_dims=True))
reconstr_all = tf.concat(1, tmp_reconstr)

y_prob = tf.placeholder(dtype=tf.float32, shape=(batch_size, category_size))

# loss VAE
loss_reconstr = tf.reduce_mean(tf.reduce_sum(tf.mul(reconstr_all, y_prob), reduction_indices=1))
loss_z = -0.5 * tf.reduce_mean(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq))
loss_VAE = loss_reconstr + loss_z


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


y_label = tf.placeholder(dtype=tf.int64, shape=(batch_size, 1))

inferred_category = tf.argmax(y_classifier, 1)
hit_num = tf.reduce_sum(tf.to_int32(tf.equal(inferred_category, y_label)))

# loss classifier
# tf.constant(reconstr_all): constant?
_tmp_loss = tf.mul(y_classifier, reconstr_all)
loss_classifier = tf.reduce_mean(tf.reduce_sum(_tmp_loss, reduction_indices=1))
# loss_y = 0.0 !!!!

# optimizer
learning_rate = 0.05
optimizer_VAE = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_VAE)
optimizer_classifier = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_classifier)

# session
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    epochs = 100
    total_batches = int(n_samples / batch_size)
    for epoch in xrange(epochs):
        total_cost = 0.0
        total_hit = 0
        for batch in xrange(total_batches):
            batch_x, batch_y_label = data, y_labels
            # x -> y
            _y_prob_classifier = sess.run([y_classifier], feed_dict={x:batch_x})
            
            # x, y -> x, update AVE
            y_prob_classifier = np.reshape(_y_prob_classifier, (batch_size, category_size))
            _, batch_reconstr_all = sess.run([optimizer_VAE, reconstr_all], feed_dict={x:batch_x, y_prob:y_prob_classifier})
            
            # x -> y, update classifier
            _, batch_y_classifier, batch_hit_num = sess.run([optimizer_classifier, y_classifier, hit_num], feed_dict={x:batch_x, y_label:batch_y_label})     
            total_hit += batch_hit_num
            print batch_y_classifier
        print "epoch: {0} accuracy: {1}".format(epoch, float(total_hit) / n_samples)
            
print "Done!"            
                