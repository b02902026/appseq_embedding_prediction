from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras import backend as K

from emb_model import build_model
import data_utility
import argparse
import numpy as np
import os

gpu_options = tf.GPUOptions(allow_growth=True)
KTF.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

def dump_np(np_arr, file_name):
    file_path = os.path.join('./temp/', file_name)
    with open(file_path, 'wb') as f:
        np.savetxt(f, np_arr)
    return

def train_main(opts):
    couple_list, label_list, opts = data_utility.generate_data(opts)
    print(couple_list[0])
    print(label_list[0])
    proceed_list = np.asarray([int(couple[0]) for couple in couple_list])
    context_list = np.asarray([int(couple[1]) for couple in couple_list])
    label_list = np.asarray([int(label) for label in label_list])
    dump_np(proceed_list, 'proceed_list')
    dump_np(context_list, 'context_list')
    dump_np(label_list, 'label_list')
    label_list = label_list.reshape(len(label_list), 1, 1)
    print(label_list)
    print(label_list.shape)
    print('building model...')
    model = build_model(opts)
    model.summary()
    model.compile(optimizer='sgd',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # TODO: train a constant epoch, then save a visualization
    print('starting training')
    model.fit([proceed_list, context_list], label_list, batch_size=256, 
              nb_epoch=500, verbose=1, callbacks=None, shuffle=True)

    
    model.save_weights('../model/app_embedding_weight.hd5')

    embedding_weight = model.layers[2].get_weights()
    print(embedding_weight)
    print('shape of embedding_weight is {}'.format(embedding_weight[0].shape))

    with open(os.path.join('../save_vector/', 'app_vector.npy'), 'wb') as f:
        np.savetxt(f, embedding_weight[0])

    K.clear_session()
    return

def make_option(args):
    # not important, do it later
    return {}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', dest='model', action='store',
                        default='skipgram',
                        help='skipgram or cbow, default skipgram')

    args = parser.parse_args()
    opts = make_option(args)

    train_main(opts)
