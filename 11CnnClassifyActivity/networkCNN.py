import tensorflow as tf
import nn
import math
init_kernel = tf.random_normal_initializer(mean=0, stddev=0.05)


def leakyReLu(x, alpha=0.2, name=None):
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)


def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


def classifier(inp, is_training, init=False, reuse=False, getter =None,category=125):
    with tf.variable_scope('discriminator_model', reuse=reuse,custom_getter=getter):
        counter = {}
        #x = tf.reshape(inp, [-1, 32, 32, 3])
        x = tf.reshape(inp, [-1, 200, 30, 3])
        x = tf.layers.dropout(x, rate=0.2, training=is_training, name='dropout_0')

        x = nn.conv2d(x, 96, nonlinearity=leakyReLu, init=init, counters=counter)                #  64*200*30*96
        x = nn.conv2d(x, 96, nonlinearity=leakyReLu, init=init, counters=counter)                #  64*200*30*96
        #x = nn.conv2d(x, 96, stride=[2, 2], nonlinearity=leakyReLu, init=init, counters=counter) 
        x = nn.conv2d(x, 96, stride=[5, 2], nonlinearity=leakyReLu, init=init, counters=counter) #  64*40*15*96
        
        x = tf.layers.dropout(x, rate=0.5, training=is_training, name='dropout_1')               #  64*40*15*96

        x = nn.conv2d(x, 192, nonlinearity=leakyReLu, init=init, counters=counter)               #  64*40*15*192
        x = nn.conv2d(x, 192, nonlinearity=leakyReLu, init=init, counters=counter)               #  64*40*15*192
        #x = nn.conv2d(x, 192, stride=[2, 2], nonlinearity=leakyReLu, init=init, counters=counter)
        x = nn.conv2d(x, 192, stride=[5, 2], nonlinearity=leakyReLu, init=init, counters=counter)#  64*8*8*192

        x = tf.layers.dropout(x, rate=0.5, training=is_training, name='dropout_2')               #  64*8*8*192

        x = nn.conv2d(x, 192, pad='VALID', nonlinearity=leakyReLu, init=init, counters=counter)  #  64*6*6*192
        x = nn.nin(x, 192, counters=counter, nonlinearity=leakyReLu, init=init)                  #  64*6*6*192
        x = nn.nin(x, 192, counters=counter, nonlinearity=leakyReLu, init=init)                  #  64*6*6*192
        x = tf.layers.max_pooling2d(x, pool_size=6, strides=1, name='avg_pool_0')                #  64*1*1*192
        x = tf.squeeze(x, [1, 2])                                                                #  64*192

        intermediate_layer = x

        #logits = nn.dense(x, 10, nonlinearity=None, init=init, counters=counter, init_scale=0.1)
        logits = nn.dense(x, category, nonlinearity=None, init=init, counters=counter, init_scale=0.1) # 64*125
        print('logits:',logits)

        return logits, intermediate_layer

