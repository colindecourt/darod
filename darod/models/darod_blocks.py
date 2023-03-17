import tensorflow_addons as tfa
from tensorflow.keras import layers


class DARODBlock2D(layers.Layer):
    """
    Build a 2D convolutional block with num_conv convolution operations.
    """
    def __init__(self, filters, padding="same", kernel_size=(3, 3), num_conv=2, dilation_rate=(1, 1),
                 strides=(1, 1), activation="leaky_relu", block_norm=None,
                 pooling_size=(2, 2), pooling_strides=(2, 2), name=None):
        """

        :param filters: number of filters inside the block
        :param padding:  "same" or "valid". Same padding is applied to all convolutions
        :param kernel_size: the size of the kernel
        :param num_conv: number of convolution operations (2 or 3)
        :param dilation_rate: dilation rate to use inside the block
        :param strides: strides for the convolution operations
        :param activation: activation function to apply. Possible operations: ReLu, Leaky ReLu
        :param block_norm: normalization operation during training. Possible operations: batch norm, group norm
        :param pooling_size: size of the last pooling operation. If None, no pooling is applied.
        :param pooling_strides: strides of the last pooling opeation.
        :param name: name of the block
        """
        super(DARODBlock2D, self).__init__(name=name)
        #
        self.filters = filters
        self.padding = padding
        self.kernel_size = kernel_size
        self.num_conv = num_conv
        self.dilation_rate = dilation_rate
        self.strides = strides
        self.pooling_size = pooling_size
        self.pooling_strides = pooling_strides
        self.activation = activation
        self.block_norm = block_norm
        #

        assert self.num_conv == 2 or self.num_conv == 3, print(
            ValueError("Inappropriate number of convolutions used for this block."))

        self.conv1 = layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
                                   padding=self.padding, dilation_rate=self.dilation_rate, name=self.name + "_conv_1")
        #
        if block_norm == "batch_norm":
            self.norm1 = layers.BatchNormalization(name=name + "_normalization_1")
        elif block_norm == "group_norm":
            self.norm1 = tfa.layers.GroupNormalization(name=name + "_normalization_1")
        elif block_norm == "layer_norm":
            self.norm1 = tfa.layers.GroupNormalization(groups=1, name=name + "_normalization_1")
        elif block_norm == "instance_norm":
            assert filters is not None, print(
                "Error. Please provide input number of filters for instance normalization.")
            self.norm1 = tfa.layers.GroupNormalization(groups=filters, name=name + "_normalization_1")
        elif block_norm is None:
            pass
        else:
            raise NotImplementedError("Error. Please ue either batch norm or group norm or layer norm or instance norm or None \
                for normalization.")
        #
        if activation == "relu":
            self.activation1 = layers.ReLU(name=name + "_activation_1")
        elif activation == "leaky_relu":
            self.activation1 = layers.LeakyReLU(alpha=0.001, name=name + "_activation_1")
        else:
            raise NotImplementedError("Please use ReLu or Leaky ReLu activation function.")

        self.conv2 = layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
                                   padding=self.padding, dilation_rate=self.dilation_rate, name=self.name + "_conv_2")
        #
        if block_norm == "batch_norm":
            self.norm2 = layers.BatchNormalization(name=name + "_normalization_2")
        elif block_norm == "group_norm":
            self.norm2 = tfa.layers.GroupNormalization(name=name + "_normalization_2")
        elif block_norm == "layer_norm":
            self.norm2 = tfa.layers.GroupNormalization(groups=1, name=name + "_normalization_2")
        elif block_norm == "instance_norm":
            assert filters is not None, print(
                "Error. Please provide input number of filters for instance normalization.")
            self.norm2 = tfa.layers.GroupNormalization(groups=filters, name=name + "_normalization_2")
        elif block_norm is None:
            pass
        else:
            raise NotImplementedError("Error. Please ue either batch norm or group norm or layer norm or instance norm or None \
                for normalization.")
        #
        if activation == "relu":
            self.activation2 = layers.ReLU(name=name + "_activation_2")
        elif activation == "leaky_relu":
            self.activation2 = layers.LeakyReLU(alpha=0.001, name=name + "_activation_2")
        else:
            raise NotImplementedError("Please use ReLu or Leaky ReLu activation function.")

        if num_conv == 3:
            self.conv3 = layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
                                       padding=self.padding, dilation_rate=self.dilation_rate,
                                       name=self.name + "_conv_3")
            #
            if block_norm == "batch_norm":
                self.norm3 = layers.BatchNormalization(name=name + "_normalization_3")
            elif block_norm == "group_norm":
                self.norm3 = tfa.layers.GroupNormalization(name=name + "_normalization_3")
            elif block_norm == "layer_norm":
                self.norm3 = tfa.layers.GroupNormalization(groups=1, name=name + "_normalization_3")
            elif block_norm == "instance_norm":
                assert filters is not None, print(
                    "Error. Please provide input number of filters for instance normalization.")
                self.norm3 = tfa.layers.GroupNormalization(groups=filters, name=name + "_normalization_3")
            elif block_norm is None:
                pass
            else:
                raise NotImplementedError("Error. Please ue either batch norm or group norm or layer norm or instance norm or None \
                    for normalization.")
            #
            if activation == "relu":
                self.activation3 = layers.ReLU(name=name + "_activation_3")
            elif activation == "leaky_relu":
                self.activation3 = layers.LeakyReLU(alpha=0.001, name=name + "_activation_3")
            else:
                raise NotImplementedError("Please use ReLu or Leaky ReLu activation function.")

        self.maxpooling = layers.MaxPooling2D(pool_size=pooling_size, strides=pooling_strides, name=name + "_maxpool")

    def call(self, inputs, training=False):
        """
        DAROD forward pass
        :param inputs: RD spectrum
        :param training: training or inference
        :return: features
        """

        x = self.conv1(inputs)
        if self.block_norm is not None:
            x = self.norm1(x, training=training)
        x = self.activation1(x)

        x = self.conv2(x)
        if self.block_norm is not None:
            x = self.norm2(x, training=training)
        x = self.activation2(x)

        if self.num_conv == 3:
            x = self.conv3(x)
            if self.block_norm is not None:
                x = self.norm3(x, training=training)
            x = self.activation3(x)

        x = self.maxpooling(x)

        return x
