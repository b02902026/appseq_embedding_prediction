from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding

import argparse

def train_main(opts):
    data_generator = data_utility.get_data_generator(opts)

    model = build_model()


    # TODO: train a constant epoch, then save a visualization
    model.fit_generator()

    return


if '__name__' == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', dest='model', action='store',
                        default='skipgram',
                        help='skipgram or cbow, default skipgram')

    args = parser.parse_args()
    opts = make_option(args)

    train_main(opts)
