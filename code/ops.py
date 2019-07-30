import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def batch_normalization(input, is_training, name):
    with tf.variable_scope(name):
        return tf.contrib.layers.batch_norm(input, decay=0.9, epsilon=1e-5, scale=True, is_training=is_training)

def conv2d(input, output_chn, kernel_size, stride, dilation, use_bias=False, name='conv'):
    return tf.layers.conv2d(inputs=input, filters=output_chn, kernel_size=kernel_size, strides=stride,
                            dilation_rate=dilation,padding='same', data_format='channels_last',
                            kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            kernel_regularizer=slim.l2_regularizer(0.0005), use_bias=use_bias, name=name)

def conv_bn_relu(input, output_chn, kernel_size, stride, dilation, use_bias, is_training, name):
    with tf.variable_scope(name):
        conv = conv2d(input, output_chn, kernel_size, stride, dilation, use_bias, name='conv')
        bn = batch_normalization(conv, is_training=is_training, name="batch_norm")
        relu = tf.nn.relu(bn, name='relu')

    return relu

def Deconv2d(input, output_chn, kernel_size, stride, name):
    static_input_shape = input.get_shape().as_list()
    dyn_input_shape = tf.shape(input)
    filter = tf.get_variable(name+"/filter", shape=[kernel_size, kernel_size, output_chn, static_input_shape[3]], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(0, 0.01), regularizer=slim.l2_regularizer(0.0005))

    conv = tf.nn.conv2d_transpose(value=input, filter=filter, output_shape=[dyn_input_shape[0], dyn_input_shape[1] * stride, dyn_input_shape[2] * stride, output_chn],
                                  strides=[1, stride, stride, 1], padding="SAME", name=name)
    return conv

def deconv_bn_relu(input, output_chn, kernel_size, stride, is_training, name):
    with tf.variable_scope(name):
        conv = Deconv2d(input, output_chn, kernel_size, stride, name='deconv')
        bn = batch_normalization(conv, is_training=is_training, name="batch_norm")
        relu = tf.nn.relu(bn, name='relu')
    return relu

def conv_bn_relu_x2(input, output_chn, kernel_size, stride, dilation, use_bias, is_training, name):
    with tf.variable_scope(name):
        z=conv_bn_relu(input, output_chn, kernel_size, stride, dilation, use_bias, is_training, "dense1")
        z_out = conv_bn_relu(z, output_chn, kernel_size, stride, dilation, use_bias, is_training, "dense2")
        return z_out

def bottleneck_block(input, input_chn, output_chn, kernel_size, stride, dilation, use_bias, is_training, name):
    
    with tf.variable_scope(name):
        layer_conv1 = conv_bn_relu(input=input, output_chn=output_chn, kernel_size=kernel_size, stride=stride, dilation=dilation, use_bias=use_bias, is_training=is_training, name=name+'_conv1')
        layer_conv2 = conv2d(input=layer_conv1, output_chn=output_chn, kernel_size=kernel_size, stride=1, dilation=dilation, use_bias=use_bias, name=name+'_conv2')
        bn = batch_normalization(layer_conv2, is_training=is_training, name="batch_norm")
	if input_chn == output_chn and stride == 1:
		res = bn + input
	else:
		layer_conv3 = conv2d(input=input, output_chn=output_chn, kernel_size=1, stride=stride, dilation=(1,1), use_bias=use_bias, name=name+'_conv3')
		bn1 = batch_normalization(layer_conv3, is_training=is_training, name="batch_norm1")
		res = bn + bn1

	return tf.nn.relu(res, name='relu')

def bilinear_pooling(input_num):
    
    phi_I = tf.einsum('ijkm,ijkn->imn',input_num,input_num)
    y_ssqrt = tf.multiply(tf.sign(phi_I),tf.sqrt(tf.abs(phi_I)+1e-12))
    z_l2 = tf.nn.l2_normalize(y_ssqrt, dim=2)
    z_l2 = tf.reduce_mean(z_l2, 2)
    return z_l2

def couple_map(input_score, input_map, input_num):
    input_score_1 = input_score[:,:,:,1]
    dyn_input_shape = tf.shape(input_map)
    dyn_input_num_shape = tf.shape(input_num)
    input_score_1=input_score_1[:,:,:,tf.newaxis]       
    input_score_1 = tf.tile(input_score_1,[1,1,1,dyn_input_shape[3]])

    input_num_1 = tf.reshape(input_num,[dyn_input_num_shape[0],1,1,dyn_input_num_shape[1]]);
    input_num_1 = tf.tile(input_num_1,[1,dyn_input_shape[1],dyn_input_shape[2],1])   
    input_attention_score=tf.multiply(input_score_1, input_map)
    return tf.multiply(input_attention_score,input_num_1)
