import tensorflow as tf
from keras.models import Model
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers import add
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras import backend as K
from keras.models import load_model


def conv2d_bn(x, filters, kernel_size, strides=1, padding='same', activation='relu', use_bias=False, name=None):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, name=name)(x)
    if not use_bias:
        bn_axis = 3
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x


def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    if block_type == 'block35':
        branch_0 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(branch_1, 32, 3)
        branch_2 = conv2d_bn(x, 32, 1)
        branch_2 = conv2d_bn(branch_2, 48, 3)
        branch_2 = conv2d_bn(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 128, 1)
        branch_1 = conv2d_bn(branch_1, 160, [1, 7])
        branch_1 = conv2d_bn(branch_1, 192, [7, 1])
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn(branch_1, 256, [3, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    mixed = Concatenate(
        axis=channel_axis, name=block_name + '_mixed')(branches)
    up = conv2d_bn(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=block_name + '_conv')

    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=K.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=block_name)([x, up])
    if activation is not None:
        x = Activation(activation, name=block_name + '_ac')(x)
    return x


# summary: tạo model inceptionresnet v2
# param:
# 	return
# 		inceptionresnet v2 model (output 512 params)
def InceptionResNetV2():

	inputs = Input(shape=(160, 160, 3))

	# Stem block: 35 x 35 x 192
	x = conv2d_bn(inputs, 32, 3, strides=2, padding='valid')
	x = conv2d_bn(x, 32, 3, padding='valid')
	x = conv2d_bn(x, 64, 3)
	x = MaxPooling2D(3, strides=2)(x)
	x = conv2d_bn(x, 80, 1, padding='valid')
	x = conv2d_bn(x, 192, 3, padding='valid')
	x = MaxPooling2D(3, strides=2)(x)

	# Mixed 5b (Inception-A block): 35 x 35 x 320
	branch_0 = conv2d_bn(x, 96, 1)
	branch_1 = conv2d_bn(x, 48, 1)
	branch_1 = conv2d_bn(branch_1, 64, 5)
	branch_2 = conv2d_bn(x, 64, 1)
	branch_2 = conv2d_bn(branch_2, 96, 3)
	branch_2 = conv2d_bn(branch_2, 96, 3)
	branch_pool = AveragePooling2D(3, strides=1, padding='same')(x)
	branch_pool = conv2d_bn(branch_pool, 64, 1)
	branches = [branch_0, branch_1, branch_2, branch_pool]
	channel_axis = 3
	x = Concatenate(axis=channel_axis, name='mixed_5b')(branches)

	# 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
	for block_idx in range(1, 11):
		x = inception_resnet_block(x, scale=0.17, block_type='block35', block_idx=block_idx)

	# Mixed 6a (Reduction-A block): 17 x 17 x 1088
	branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='valid')
	branch_1 = conv2d_bn(x, 256, 1)
	branch_1 = conv2d_bn(branch_1, 256, 3)
	branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='valid')
	branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
	branches = [branch_0, branch_1, branch_pool]
	x = Concatenate(axis=channel_axis, name='mixed_6a')(branches)

	# 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
	for block_idx in range(1, 21):
		x = inception_resnet_block(x, scale=0.1, block_type='block17', block_idx=block_idx)

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
	branch_0 = conv2d_bn(x, 256, 1)
	branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='valid')
	branch_1 = conv2d_bn(x, 256, 1)
	branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='valid')
	branch_2 = conv2d_bn(x, 256, 1)
	branch_2 = conv2d_bn(branch_2, 288, 3)
	branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='valid')
	branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
	branches = [branch_0, branch_1, branch_2, branch_pool]
	x = Concatenate(axis=channel_axis, name='mixed_7a')(branches)

	# 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
	for block_idx in range(1, 10):
		x = inception_resnet_block(x, scale=0.2, block_type='block8', block_idx=block_idx)
	x = inception_resnet_block(x, scale=1., activation=None, block_type='block8', block_idx=10)

	# Final convolution block: 8 x 8 x 1536
	x = conv2d_bn(x, 1536, 1, name='conv_7b')
	
	model = Model(inputs, x, name='inception_resnet_v2')

	return model
	

def feature_extractor(inputs):
	feature_extractor_layers = InceptionResNetV2()
	feature_extractor_layers.trainable = False
	feature_extractor_layers = feature_extractor_layers(inputs)
	return feature_extractor_layers

from keras.applications.inception_resnet_v2 import InceptionResNetV2 as InceptionResNetV2_keras

def feature_extractor_pretrained(inputs):
  feature_extractor_layers = InceptionResNetV2_keras(include_top=False, weights="imagenet", input_shape=(160, 160, 3))
  feature_extractor_layers.trainable = False
  feature_extractor_layers = feature_extractor_layers(inputs)
  return feature_extractor_layers

def classifier_layer_embedding(inputs):
	x = GlobalAveragePooling2D(name='AvgPool')(x)
	x = Dropout(1.0 - 0.8, name='Dropout')(x)
	# Bottleneck
	x = Dense(512, use_bias=False, name='Bottleneck')(x)
	x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False, name='Bottleneck_BatchNorm')(x)	
	return x
	
def classifier_layer_train(inputs, num_class):
	x = classifier_layer_embedding(inputs)
	x = Dense(num_class, activation='softmax', name='predictions')(x)
	return x

def get_train_model(num_class):
	inputs = Input(shape=(160, 160, 3))
	res_feature_extractor = feature_extractor_pretrained(inputs)
	classification_output = classifier_layer_train(res_feature_extractor, num_class)
	model = Model(inputs, classification_output)
	return model

def get_train_model_fine_tune(num_class):
	inputs = Input(shape=(160, 160, 3))
	res_feature_extractor = feature_extractor(inputs)
	classification_output = classifier_layer_train(res_feature_extractor, num_class)
	model = Model(inputs, classification_output)
	return model

def get_model(weights_path):
	inputs = Input(shape=(160, 160, 3))
	res_feature_extractor = feature_extractor(inputs)
	classification_output = classifier_layer_embedding(res_feature_extractor)
	model = Model(inputs, classification_output)
	model.load_weights(weights_path)
	return model