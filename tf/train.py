import os
import sys
import argparse

import keras
import tensorflow as tf

from model import Mymodel
from generator import Generator

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

def check_args(args):
    assert args.dataset, 'Dataset will use must be selected.'

    return args

def get_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    type=str,   default=None)
    parser.add_argument("--seblock",    action='store_true')
    parser.add_argument("--cbamblock",  action='store_true')
    parser.add_argument("--anrblock",   action='store_true')

    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--steps",      type=int,   default=0)
    parser.add_argument("--lr",         type=float, default=.0001)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--callbacks",  action='store_true')
    parser.add_argument("--summary",    action='store_true')

    return check_args(parser.parse_args(args))

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    
    args = get_arguments(args)
    get_session()

    if args.dataset == 'cifar10':
        from keras.datasets import cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        label_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        classes = 10

    model = Mymodel(classes=classes, 
                    _isseblock=args.seblock,
                    _iscbamblock=args.cbamblock,
                    _isanr=args.anrblock)

    if args.summary:
        model.summary()
        return
    
    model.compile(optimizer=keras.optimizers.Adam(lr=args.lr, clipnorm=.001),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['acc'])

    train_generator = Generator(x=x_train, y=y_train, mode='train', batch_size=args.batch_size)
    val_generator = Generator(x=x_test, y=y_test, mode='validation', batch_size=args.batch_size, shuffle=False)

    callback_name = ''
    if args.seblock:
        callback_name += 'seblock_'
    elif args.cbamblock:
        callback_name += 'cbamblock_'
    else:
        callback_name += 'no_'

    if not os.path.isdir('./checkpoint/{}'.format(callback_name)):
        os.makedirs('./checkpoint/{}'.format(callback_name))

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=args.steps if args.steps else int(len(y_train)//args.batch_size),
        epochs=args.epochs,
        validation_data=val_generator,
        validation_steps=int(len(y_test)//args.batch_size),
        callbacks=[
            keras.callbacks.ModelCheckpoint(filepath='./checkpoint/{}'.format(callback_name)+'/{epoch:04d}_{val_acc:.4f}_{val_loss:.4f}.h5',
                                            monitor='val_acc',
                                            verbose=1,
                                            mode='max',
                                            save_best_only=False,
                                            save_weights_only=True),
            keras.callbacks.CSVLogger(filename='./history/{}.csv'.format(callback_name),
                                      separator=',', append=True)
        ] if args.callbacks else []
    )
    

if __name__ == "__main__":
    main()