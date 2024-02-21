#%tensorflow_version 2.x
import tensorflow as tf
import numpy as np


################### LINEAR CLASS ###################
class Linear(layers.Layer):
    """
    Custom linear layer for neural networks.

    This layer performs a linear transformation on the input data using weights (w) and biases (b).
    The transformation is defined as: output = inputs * w + b.

    :param units: Number of output units/neurons in the layer.
    :type units: int, optional, default: 32
    :param input_dim: Dimensionality of the input data.
    :type input_dim: int, optional, default: 32
    :param is_transformer_regressor: If True, initializes weights with zeros and uses a trainable bias based on a
                                     transformer regressor approach. If False, initializes weights with random values
                                     and uses a trainable bias initialized with zeros.
    :type is_transformer_regressor: bool, optional, default: False

    :Attributes:
        - w (tf.Variable): Weight matrix for the linear transformation.
        - b (tf.Variable): Bias vector for the linear transformation.

    :Methods:
        - call(inputs): Performs the linear transformation on the input data.

    :Note: This class is designed to be used as a layer within a TensorFlow/Keras model.
    """
    def __init__(self, units=32, input_dim=32, is_transformer_regressor=False):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                                  dtype="float32"), trainable=True)
        if is_transformer_regressor:
            theta = tf.constant([ [1,0,0], [0,1,0] ])
            theta = tf.cast(theta, "float32")
            theta = tf.reshape(theta, shape=(units,))
            self.b = tf.Variable(initial_value=theta, trainable=True)
        else:
            b_init = tf.zeros_initializer()
            self.b = tf.Variable(initial_value=b_init(
                shape=(units,),
                dtype="float32"
                ), trainable=True)

    def call(self, inputs):
        """
        Apply the linear transformation on the input data.

        :param inputs: Input data tensor.
        :type inputs: tf.Tensor

        :return: Output tensor after the linear transformation.
        :rtype: tf.Tensor
        """
        return tf.matmul(inputs, self.w) + self.b


################### LOCNET CLASS ###################
class LocNet(tf.keras.layers.Layer):
    """
    Localization Network (LocNet) layer for spatial feature extraction.

    This layer consists of two convolutional blocks followed by max-pooling operations.

    :Attributes:
        - conv1 (Conv2D): First convolutional layer with 8 filters and a kernel size of 7x7, using ReLU activation.
        - maxpool1 (MaxPooling2D): First max-pooling layer with a pool size of 2x2 and strides of 2.
        - conv2 (Conv2D): Second convolutional layer with 10 filters and a kernel size of 5x5, using ReLU activation.
        - maxpool2 (MaxPooling2D): Second max-pooling layer with a pool size of 2x2 and strides of 2.

    :Methods:
        - call(inputs): Applies the LocNet layer on the input data to extract spatial features.`

    :Note: This class is intended to be used as a part of a neural network, particularly for spatial feature extraction
    in tasks such as image localization or object detection.
    """
    def __init__(self):
        super(LocNet, self).__init__()

        self.conv1 = Conv2D(filters=8, kernel_size=7, activation="relu")
        self.maxpool1 = MaxPooling2D(pool_size=(2,2), strides=2)
        self.conv2 = Conv2D(filters=10, kernel_size=5, activation="relu")
        self.maxpool2 = MaxPooling2D(pool_size=(2,2), strides=2)

    def call(self, inputs):
        """
        Apply the LocNet layer on the input data to extract spatial features.

        :param inputs: Input data tensor with shape (batch_size, height, width, channels).
        :type inputs: tf.Tensor

        :return: Output tensor representing the spatial features extracted by the LocNet layer.
        :rtype: tf.Tensor
        """
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        return x


################### REGRESSOR CLASS ###################
class TransformationRegressor(tf.keras.layers.Layer):
    """
    Transformation Regressor layer for predicting transformation parameters.

    This layer is designed to predict transformation parameters, such as rotation or translation, from input data.
    It consists of two linear layers with optional activation functions.

    :Attributes:
        - linear1 (Linear): First linear layer with units=32, responsible for extracting features from the input.
        - linear2 (Linear): Second linear layer with units=6, specifically designed for predicting transformation
                            parameters. If is_transformer_regressor is True, it uses a custom initialization for
                            transformer regression.

    :Methods:
        - call(inputs): Applies the Transformation Regressor layer on the input data to predict transformation
          parameters.

    :Note: This class is intended to be used as a part of a neural network, especially for tasks involving predicting
    transformation parameters from input data.
    """
    def __init__(self):
        super(TransformationRegressor, self).__init__()

        self.linear1 = Linear(units=32, input_dim=10*6*6)
        self.linear2 = Linear(units=6, input_dim=32, is_transformer_regressor=True)

    def call(self, inputs):
        """
        Apply the Transformation Regressor layer on the input data to predict transformation parameters.

        :param inputs: Input data tensor with shape (batch_size, input_dim).
        :type inputs: tf.Tensor

        :return: Output tensor representing the predicted transformation parameters.
        :rtype: tf.Tensor
        """
        x = self.linear1(inputs)
        x = tf.nn.tanh(x)
        x = self.linear2(x)
        x = tf.nn.tanh(x)
        return x


################### STN-CNN CLASS ###################
class SpatialTransformerCNN(tf.keras.models.Model):
    """
    Spatial Transformer CNN model for image transformation.

    This model combines a localization network (LocNet), a transformation regressor, and a convolutional neural network
    (CNN) for tasks involving spatial transformations on images.

    :Parameters:
        - drate (float): Dropout rate for regularization.

    :Attributes:
        - loc_net (LocNet): Localization network for spatial feature extraction.
        - flatten (Flatten): Flatten layer for transforming spatial features into a 1D vector.
        - transformer_regressor (TransformationRegressor): Regressor for predicting transformation parameters.
        - CNN (Sequential): Convolutional neural network with max-pooling layers for image classification.

    :Methods:
        - call(inputs): Applies the Spatial Transformer CNN model on the input data to perform spatial transformations
                        and classification.
        - transform(inputs): Applies the Spatial Transformer CNN model to perform spatial transformations without
                             dropout.

    :Note: This class is designed to be used as a Keras model for tasks involving spatial transformations on images with
    subsequent classification.
    """
    def __init__(self, drate=0.2):
        super(SpatialTransformerCNN, self).__init__()

        self.loc_net = LocNet()
        self.flatten = Flatten()
        self.transformer_regressor = TransformationRegressor()
        self.CNN = Sequential([Conv2D(filters=10, kernel_size=(5,5), strides=1, activation="relu"),
                               MaxPooling2D(pool_size=(2,2)),
                               Conv2D(filters=20, kernel_size=(5,5), strides=1, activation="relu"),
                               MaxPooling2D(pool_size=(2,2)),
                               Flatten(),
                               Dense(units=1024, activation="relu"),
                               Dropout(rate=drate),
                               Dense(units=10, activation="softmax")])

    def call(self, inputs):
        """
        Apply the Spatial Transformer CNN model on the input data to perform spatial transformations and classification.

        :param inputs: Input data tensor with shape (batch_size, H, W, channels).
        :type inputs: tf.Tensor

        :return: Logits representing the classification probabilities.
        :rtype: tf.Tensor
        """
        xs = self.loc_net(inputs)
        xs = self.flatten(xs)
        theta = self.transformer_regressor(xs)
        xt = transformer(inputs, theta, out_dims=[40,40])
        logits = self.CNN(xt)
        return(logits)

    def transform(self, inputs):
        """
        Apply the Spatial Transformer CNN model to perform spatial transformations without dropout.

        :param inputs: Input data tensor with shape (batch_size, H, W, channels).
        :type inputs: tf.Tensor

        :return: Transformed output tensor.
        :rtype: tf.Tensor
        """
        inputs = inputs.astype("float32")
        xs = self.loc_net(inputs)
        xs = self.flatten(xs)
        theta = self.transformer_regressor(xs)
        xt = transformer(inputs, theta, out_dims=[40,40])
        return xt
