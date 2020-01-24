from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from config import CONFIGURATION
import argparse
import numpy as np
from model import build_model
import matplotlib.pyplot as plt
from helper import class_to_letter


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=CONFIGURATION['model']['batch_size'], type=int, help='batch size')
parser.add_argument('--epochs', default=CONFIGURATION['model']['epochs'], type=int,
                      help='number of training epochs')

def main(argv):
    args = parser.parse_args(argv[1:])
    print(args.batch_size, args.epochs)

    with np.load(CONFIGURATION['path']['dataset'], allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)

    x_train = x_train.reshape([-1, 20, 20, 1])
    x_test = x_test.reshape([-1, 20, 20, 1])

    model = build_model()

    checkpoint_path = CONFIGURATION['path']['checkpoint']

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1, period=10)

    history = model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, 
                        validation_data=(x_test, y_test), callbacks=[cp_callback])

    model.save('data/my_model.h5') 

    # plt.plot(history.history['accuracy'], label='accuracy')
    # plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.ylim([0.5, 1])
    # plt.legend(loc='lower right')
    # plt.show()

    # test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    # print(test_acc, test_loss)


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run(main)