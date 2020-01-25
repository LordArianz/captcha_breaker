from image import get_letters_b64, get_letters
from config import CONFIGURATION
from data import build_captcha_data
import os
import numpy as np
import tensorflow as tf
from model import build_model
from helper import prediction_to_captcha
# import argparse
from datetime import datetime
import sys

# parser = argparse.ArgumentParser()
# parser.add_argument('--fname', default='data/captcha/2ahckd.jpg',  help='captcha image path')


def main(argv):
    # args = parser.parse_args(argv[1:])
    # if args.fname == None:
    #     print('Capture Image is missing!')
    #     return None
    # fname = args.fname

    checkpoint_dir = os.path.dirname(CONFIGURATION['path']['checkpoint'])
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    model = build_model()
    model.load_weights(latest).expect_partial()

    while True:
        fname = input()
        data_x = []

        letters = get_letters(fname)
        if letters != None:
            X = []
            for _, letter in enumerate(letters):
                heigth, width = np.shape(letter)
                if heigth != CONFIGURATION['data']['height'] or width != CONFIGURATION['data']['width']:
                    continue
                X.append(letter)
            if len(X) == len(letters):
                data_x.append(X)

        for i, data in enumerate(data_x):
            prediction = []
            for d in data:
                d = d.reshape([-1, 20, 20, 1])
                prediction.append(tf.argmax(model.predict(d), 1).numpy()[0])
            answer = prediction_to_captcha(prediction)
            print(answer)


if __name__ == '__main__':
    tf.compat.v1.app.run(main)