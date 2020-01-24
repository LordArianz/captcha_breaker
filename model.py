from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from config import CONFIGURATION
from helper import class_to_letter


def build_model():
    class_count = 0
    if CONFIGURATION['data']['is_lower']:
        class_count += 26
    if CONFIGURATION['data']['is_upper']:
        class_count += 26
    if CONFIGURATION['data']['is_num']:
        class_count += 10

    learning_rate = CONFIGURATION['model']['learning_rate']
    width = CONFIGURATION['data']['width']
    height = CONFIGURATION['data']['height']
    dropout = CONFIGURATION['model']['dropout']

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(height, width, 1), padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=1024, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(units=class_count, activation='softmax'),
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model

