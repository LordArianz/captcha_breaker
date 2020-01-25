import os
from config import CONFIGURATION
from image import get_letters
import numpy as np
from cv2 import cv2
from helper import letter_to_class, build_data_map 


def main():
    data_x = []
    data_y = []

    dir_path = CONFIGURATION['path']['captcha']
    image_contents = build_data_map(dir_path)

    counter = 0

    for fname, contents in image_contents.items():
        counter += 1
        print(counter, fname, contents)

        letters = get_letters(os.path.join(dir_path, fname), len(contents))
        if letters != None:
            for i, letter in enumerate(letters):
                heigth, width = np.shape(letter)
                if heigth != CONFIGURATION['data']['height'] or width != CONFIGURATION['data']['width']:
                    print(i, 'Letter is not valid')
                    continue

                content = contents[i]

                data_x.append(letter)
                data_y.append(np.uint8(letter_to_class(content)))

                fpath = os.path.join(CONFIGURATION['path']['train'], content)
                if not os.path.exists(fpath):
                    os.makedirs(fpath)
                letter_fname = os.path.join(fpath, str(i + 1) + '-' + content + '.png')

                try:
                    cv2.imwrite(letter_fname, letter)
                except:
                    print('Saving letter failed')
        else:
            print('Letters is not valid')

    train_num = int(len(data_y) * CONFIGURATION['model']['split_ratio'])

    print('saving dataset')
    np.savez_compressed(dir_path,
        x_train=data_x[:train_num], y_train=data_y[:train_num],
        x_test=data_x[train_num:], y_test=data_y[train_num:])


def build_captcha_data(path):
    data_x = []
    data_y = []

    image_contents = build_data_map(path)

    for fname, contents in image_contents.items():        
        letters = get_letters(os.path.join(path, fname), len(contents))
        if letters != None:
            X = []
            for _, letter in enumerate(letters):
                heigth, width = np.shape(letter)
                if heigth != CONFIGURATION['data']['height'] or width != CONFIGURATION['data']['width']:
                    continue
                X.append(letter)
                
            if len(X) == len(letters):
                data_x.append(X)
                data_y.append(contents)
    return data_x, data_y


if __name__ == '__main__':
    main()