CONFIGURATION = {
    'path': {
        'data': 'data',
        'captcha': 'data/captcha',
        'train': 'data/train',
        'test': 'data/test',
        'checkpoint': 'checkpoint/cp-{epoch:04d}.ckpt',
        'dataset': 'data/captcha.npz'
    }, 
    'data': {
        'width': 20,
        'height': 20,
        'is_upper': False,
        'is_lower': True,
        'is_num': True
    },
    'model': {
        'split_ratio': 0.9,
        'batch_size': 100,
        'epochs': 100,
        'learning_rate': 0.001,
        'dropout': 0.4
    },
    'cv2': {

    }
}