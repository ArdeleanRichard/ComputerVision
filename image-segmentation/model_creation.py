import tensorflow as tf

from constants import CLASSES, IMAGES_SIZE2, IMAGES_SIZE1, DROPOUT


class InstanceNormalization(tf.keras.layers.Layer):
  """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)

  def call(self, x):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + self.epsilon)
    normalized = (x - mean) * inv
    return self.scale * normalized + self.offset

def upsample(filters, kernel_size, norm_type='batchnorm', strides=2, apply_dropout=None):
  """
  Upsamples an input.

  Conv2DTranspose => Batchnorm => Dropout => Relu

  Args:
    filters: number of filters
    kernel_size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer

  Returns:
    Upsample Sequential Model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(tf.keras.layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(InstanceNormalization())

  if apply_dropout is not None:
    result.add(tf.keras.layers.Dropout(apply_dropout))

  result.add(tf.keras.layers.ReLU())

  return result

def eff_b2(input_shape):
    effb2_base_model = tf.keras.applications.EfficientNetB2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    # print(effb2_base_model.summary())

    effb2_layer_names = [
        'block1a_activation',   # 104x104
        'block1b_activation',   # 104x104
        'block2a_expand_activation',   # 104x104
        'block2a_activation',   # 104x104
        'block2b_expand_activation',   # 104x104
        'block2b_activation',   # 104x104
        'block2c_expand_activation',   # 104x104
        'block2c_activation',   # 104x104
        'block2c_activation',   # 104x104
        'block3a_expand_activation',   # 104x104
        'block3a_activation',   # 104x104
        'block3b_expand_activation',   # 104x104
        'block3b_activation',   # 104x104
        'block3c_expand_activation',   # 104x104
        'block3c_activation',   # 104x104
        'block4a_expand_activation',   # 104x104
        'block4a_activation',   # 104x104
        'block4b_expand_activation',   # 104x104
        'block4b_activation',   # 104x104
        'block4c_expand_activation',   # 104x104
        'block4c_activation',   # 104x104
        'block4d_expand_activation',   # 104x104
        'block4d_activation',   # 13x13
        'block5a_expand_activation',   # 13x13
        'block5a_activation',   # 13x13
        'block5b_expand_activation',   # 13x13
        'block5b_activation',   # 13x13
        'block5c_expand_activation',   # 13x13
        'block5c_activation',   # 13x13
        'block5d_expand_activation',   # 13x13
        'block5d_activation',   # 13x13
        'block6a_expand_activation',   # 13x13


        # 'block_13_expand_relu',  # 8x8
        # 'block_16_project',      # 4x4
    ]

    effb2_layers = [effb2_base_model.get_layer(name).output for name in effb2_layer_names]
    # print(effb2_layers)

    # Create the feature extraction model
    effb2_down_stack = tf.keras.Model(inputs=effb2_base_model.input, outputs=effb2_layers)
    # print(effb2_down_stack.summary())
    effb2_down_stack.trainable = False

    effb2_up_stack = [
        upsample(1024, 3, strides=1, apply_dropout=DROPOUT),  # 16x16 -> 32x32
        upsample(1024, 3, strides=1, apply_dropout=DROPOUT),  # 16x16 -> 32x32
        upsample(1024, 3, strides=1, apply_dropout=DROPOUT),  # 16x16 -> 32x32
        upsample(1024, 3, strides=1, apply_dropout=DROPOUT),  # 16x16 -> 32x32
        upsample(1024, 3, strides=1, apply_dropout=DROPOUT),  # 16x16 -> 32x32
        upsample(1024, 3, strides=1, apply_dropout=DROPOUT),  # 16x16 -> 32x32
        upsample(1024, 3, strides=1, apply_dropout=DROPOUT),  # 16x16 -> 32x32
        upsample(1024, 3, strides=1, apply_dropout=DROPOUT),  # 16x16 -> 32x32
        upsample(1024, 3, strides=1, apply_dropout=DROPOUT),  # 16x16 -> 32x32
        upsample(1024, 3, strides=1, apply_dropout=DROPOUT),  # 16x16 -> 32x32
        upsample(1024, 3, strides=1, apply_dropout=DROPOUT),  # 16x16 -> 32x32
        upsample(1024, 3, strides=1, apply_dropout=DROPOUT),  # 16x16 -> 32x32
        upsample(1024, 3, strides=1, apply_dropout=DROPOUT),  # 16x16 -> 32x32
        upsample(1024, 3, strides=1, apply_dropout=DROPOUT),  # 16x16 -> 32x32
        upsample(1024, 3, strides=1, apply_dropout=DROPOUT),  # 16x16 -> 32x32
        upsample(512, 3, apply_dropout=DROPOUT),  # 4x4 -> 8x8
        upsample(512, 3, strides=1, apply_dropout=DROPOUT),  # 16x16 -> 32x32
        upsample(512, 3, strides=1, apply_dropout=DROPOUT),  # 16x16 -> 32x32
        upsample(512, 3, strides=1, apply_dropout=DROPOUT),  # 16x16 -> 32x32
        upsample(512, 3, strides=1, apply_dropout=DROPOUT),  # 16x16 -> 32x32
        upsample(512, 3, strides=1, apply_dropout=DROPOUT),  # 16x16 -> 32x32
        upsample(256, 3, apply_dropout=DROPOUT),  # 8x8 -> 16x16
        upsample(256, 3, strides=1, apply_dropout=DROPOUT),  # 8x8 -> 16x16
        upsample(256, 3, strides=1, apply_dropout=DROPOUT),  # 8x8 -> 16x16
        upsample(256, 3, strides=1, apply_dropout=DROPOUT),  # 8x8 -> 16x16
        upsample(256, 3, strides=1, apply_dropout=DROPOUT),  # 8x8 -> 16x16
        upsample(256, 3, strides=1, apply_dropout=DROPOUT),  # 8x8 -> 16x16
        upsample(256, 3, strides=1, apply_dropout=DROPOUT),  # 8x8 -> 16x16
        upsample(128, 3, apply_dropout=DROPOUT),  # 16x16 -> 32x32
        upsample(128, 3, strides=1, apply_dropout=DROPOUT),  # 8x8 -> 16x16
        upsample(128, 3, strides=1, apply_dropout=DROPOUT),  # 8x8 -> 16x16
        upsample(64, 3, apply_dropout=DROPOUT),  # 32x32 -> 64x64
        # upsample( 32, 3, apply_dropout=False),  # 32x32 -> 64x64
        # upsample( 16, 3, apply_dropout=True),  # 32x32 -> 64x64
        # upsample( 16, 3, apply_dropout=True),  # 32x32 -> 64x64
    ]
    model = unet_model(effb2_down_stack, effb2_up_stack, input_shape, CLASSES)

    return model

def mobile_large(input_shape):
    mobilev2_base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)

    # LARGE
    # Use the activations of these layers - input 208x208
    mobilev2_layer_names = [
        'block_1_expand_relu',  # 104x104
        'block_2_expand_relu',   # 104x104
        'block_3_expand_relu',  # 52x52
        'block_4_expand_relu',   # 52x52
        'block_5_expand_relu',   # 52x52
        'block_6_expand_relu',   # 26x26
        'block_7_expand_relu',   # 26x26
        'block_8_expand_relu',   # 26x26
        'block_9_expand_relu',   # 26x26
        'block_10_expand_relu',   # 13x13
        'block_11_expand_relu',   # 13x13
        'block_12_expand_relu',   # 13x13
        # 'block_13_expand_relu',  # 8x8
        # 'block_16_project',      # 4x4
    ]

    mobilev2_layers = [mobilev2_base_model.get_layer(name).output for name in mobilev2_layer_names]
    # print(mobilev2_layers)

    # Create the feature extraction model
    mobilev2_down_stack = tf.keras.Model(inputs=mobilev2_base_model.input, outputs=mobilev2_layers)
    # print(mobilev2_down_stack.summary())
    mobilev2_down_stack.trainable = False

    # LARGE
    mobilev2_up_stack = [
        upsample(1024, 3, strides=1, apply_dropout=DROPOUT),    # 13x13
        upsample(1024, 3, strides=1, apply_dropout=DROPOUT),    # 13x13
        upsample(1024, 3, strides=1, apply_dropout=DROPOUT),    # 13x13
        upsample(1024, 3, strides=1, apply_dropout=DROPOUT),    # 13x13
        upsample(1024, 3, strides=1, apply_dropout=DROPOUT),    # 13x13
        upsample(512, 3, apply_dropout=DROPOUT),                # 26x26
        upsample(512, 3, strides=1, apply_dropout=DROPOUT),     # 26x26
        upsample(512, 3, strides=1, apply_dropout=DROPOUT),     # 26x26
        upsample(256, 3, apply_dropout=DROPOUT),                # 52x52
        upsample(256, 3, strides=1, apply_dropout=DROPOUT),     # 52x52
        upsample(128, 3, apply_dropout=DROPOUT),                # 104x104
        upsample(64, 3, apply_dropout=DROPOUT),                # 208x208
        # upsample( 32, 3, apply_dropout=False),  # 32x32 -> 64x64
        # upsample( 16, 3, apply_dropout=True),  # 32x32 -> 64x64
        # upsample( 16, 3, apply_dropout=True),  # 32x32 -> 64x64
    ]

    model = unet_model(mobilev2_down_stack, mobilev2_up_stack, input_shape, CLASSES)

    return model

def mobile_small(input_shape):
    mobilev2_base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)

    # # Use the activations of these layers - input 208x208
    mobilev2_layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]

    mobilev2_layers = [mobilev2_base_model.get_layer(name).output for name in mobilev2_layer_names]

    # Create the feature extraction model
    mobilev2_down_stack = tf.keras.Model(inputs=mobilev2_base_model.input, outputs=mobilev2_layers)
    mobilev2_down_stack.trainable = False

    mobilev2_up_stack = [
        upsample(512, 3, apply_dropout=DROPOUT),  # 4x4 -> 8x8
        upsample(256, 3, apply_dropout=DROPOUT),  # 8x8 -> 16x16
        upsample(128, 3, apply_dropout=DROPOUT),  # 16x16 -> 32x32
        upsample(64, 3, apply_dropout=DROPOUT),   # 32x32 -> 64x64
    ]

    model = unet_model(mobilev2_down_stack, mobilev2_up_stack, input_shape, CLASSES)

    return model


def unet_model(downstack, upstack, input_shape, output_channels):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    # Downsampling through the model
    skips = downstack(x)

    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(upstack, skips):
        concat = tf.keras.layers.Concatenate()

        x = up(x)
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        output_channels,
        3,
        strides=2,
        padding='same',
        activation='softmax',
    )  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def model_factory(model):
    input_shape = [IMAGES_SIZE1, IMAGES_SIZE2, 3]

    if model == 'mobile_small':
        return mobile_small(input_shape=input_shape)
    elif model == 'mobile_large':
        return mobile_large(input_shape=input_shape)
    elif model == 'eff_b2':
        return eff_b2(input_shape=input_shape)
