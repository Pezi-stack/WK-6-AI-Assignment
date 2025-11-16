# model_architectures.py

import tensorflow as tf

class LightweightRecyclingModel:

    def __init__(self, input_shape=(128, 128, 3), num_classes=6):

        self.input_shape = input_shape

        self.num_classes = num_classes

        

    def create_mobile_net_variant(self):

        """Create lightweight CNN inspired by MobileNet"""

        model = tf.keras.Sequential([

            # Initial Convolution

            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 

                                 input_shape=self.input_shape,

                                 padding='same'),

            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.MaxPooling2D(2, 2),

            

            # Depthwise separable convolution block 1

            tf.keras.layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same'),

            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.MaxPooling2D(2, 2),

            

            # Depthwise separable convolution block 2

            tf.keras.layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same'),

            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.MaxPooling2D(2, 2),

            

            # Additional lightweight blocks

            tf.keras.layers.SeparableConv2D(256, (3, 3), activation='relu', padding='same'),

            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.GlobalAveragePooling2D(),

            

            # Classifier

            tf.keras.layers.Dense(128, activation='relu'),

            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(self.num_classes, activation='softmax')

        ])

        

        return model

    

    def create_tiny_cnn(self):

        """Create ultra-lightweight CNN for edge devices"""

        model = tf.keras.Sequential([

            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', 

                                 input_shape=self.input_shape, padding='same'),

            tf.keras.layers.MaxPooling2D(2, 2),

            

            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),

            tf.keras.layers.MaxPooling2D(2, 2),

            

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),

            tf.keras.layers.MaxPooling2D(2, 2),

            

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(64, activation='relu'),

            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Dense(self.num_classes, activation='softmax')

        ])

        

        return model

