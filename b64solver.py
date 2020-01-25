from image import get_letters_b64
from config import CONFIGURATION
from data import build_captcha_data
import os
import numpy as np
import tensorflow as tf
from model import build_model
from helper import prediction_to_captcha


b64 = input()

data_x = []

letters = get_letters_b64(b64)
if letters != None:
    X = []
    for _, letter in enumerate(letters):
        heigth, width = np.shape(letter)
        if heigth != CONFIGURATION['data']['height'] or width != CONFIGURATION['data']['width']:
            continue
        X.append(letter)
    if len(X) == len(letters):
        data_x.append(X)

checkpoint_dir = os.path.dirname(CONFIGURATION['path']['checkpoint'])
latest = tf.train.latest_checkpoint(checkpoint_dir)

model = build_model()
model.load_weights(latest).expect_partial()

for i, data in enumerate(data_x):
    prediction = []
    for d in data:
        d = d.reshape([-1, 20, 20, 1])
        prediction.append(tf.argmax(model.predict(d), 1).numpy()[0])
    answer = prediction_to_captcha(prediction)
    print(answer)
