from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from config import CONFIGURATION
import argparse
import numpy as np
from model import build_model
from helper import class_to_letter, build_data_map, letter_to_class, prediction_to_captcha
import os
from image import split_letters
from cv2 import cv2
from data import build_captcha_data


def main(argv):
    data_x, data_y = build_captcha_data(CONFIGURATION['path']['test'])

    checkpoint_dir = os.path.dirname(CONFIGURATION['path']['checkpoint'])
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    model = build_model()
    model.load_weights(latest).expect_partial()

    counter = 0

    for i, data in enumerate(data_x):
        prediction = []
        for d in data:
            d = d.reshape([-1, 20, 20, 1])
            prediction.append(tf.argmax(model.predict(d), 1).numpy()[0])
        answer = prediction_to_captcha(prediction)
        correct = data_y[i]
        if answer == correct:
            counter += 1
    print(counter, len(data_y))

if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run(main)