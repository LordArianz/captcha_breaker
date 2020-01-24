from config import CONFIGURATION
import os

acceptable_extensions = ['jpg', 'png']

def strip_extension(filename):
    return filename[:filename.rindex('.')]


def build_data_map(data_path):
    files = os.listdir(data_path)
    return {x: strip_extension(x) for x in files if x.split('.')[-1] in acceptable_extensions}


def letter_to_class(letter):
    prev = 0
    if CONFIGURATION['data']['is_lower']:
        if letter.islower():
            return ord(letter) - 97
        prev += 26
    if CONFIGURATION['data']['is_upper']:
        if letter.isupper():
            return ord(letter) - 65 + prev
        prev += 26
    if CONFIGURATION['data']['is_num']:
        if letter.isnumeric():
            return ord(letter) - 48 + prev
    

def class_to_letter(predicted_class):
    prev = 0
    if CONFIGURATION['data']['is_lower']:
        if predicted_class < 26:
            return chr(97 + predicted_class)
        prev += 26
    if CONFIGURATION['data']['is_upper']:
        if prev <= predicted_class < prev + 26:
            return chr(65 + predicted_class - prev)
        prev += 26
    if CONFIGURATION['data']['is_num']:
        if prev <= predicted_class < prev + 10:
            return chr(48 + predicted_class - prev)


def prediction_to_captcha(prediction):
    return ''.join([class_to_letter(p) for p in prediction])