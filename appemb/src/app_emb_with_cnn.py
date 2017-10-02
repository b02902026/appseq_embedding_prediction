from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras import backend as K
from keras.optimizers import RMSprop

from process_description import *
from cnn_model import build_desc_model
from emb_model import build_model
import data_utility
import argparse
import numpy as np
import os
import json

gpu_options = tf.GPUOptions(allow_growth=True)
KTF.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

def dump_np(np_arr, file_name):
    file_path = os.path.join('./temp/', file_name)
    with open(file_path, 'wb') as f:
        np.savetxt(f, np_arr)
    return

def process_desc(app2idx, pretrain=False):
    d = load_json('../data/')
    w2v = {}
    if pretrain:
        w2v, maxlen = load_pretrain(d, app2idx)
    v, _ = make_vocab(d)
    h = word2idx(d,v,app2idx)
    maxlen, h2 = get_padding(h,v)
    return [h2,len(v),maxlen,v, w2v]

def train_main(opts):
    K.clear_session()
    couple_list, label_list, opts = data_utility.generate_data(opts)
    print(couple_list[0])
    print(label_list[0])
    proceed_list = np.asarray([int(couple[0]) for couple in couple_list])
    context_list = np.asarray([int(couple[1]) for couple in couple_list])
    label_list = np.asarray([int(label) for label in label_list])
    dump_np(proceed_list, 'proceed_list')
    dump_np(context_list, 'context_list')
    dump_np(label_list, 'label_list')
    #label_list = label_list.reshape(len(label_list), 1, 1)
    #print(label_list)
    #print(label_list.shape)
    #--------add desc-----------
    map_path = '../data/training/'
    with open(map_path + 'app_index_map','r',encoding='utf-8', errors='ignore') as f:
        app2idx = json.load(f)

    with open(map_path + 'index_app_map','r',encoding='utf-8', errors='ignore') as f:
        idx2app = json.load(f)
    use_pretrain = True 
    desc_map,desc_vocab_size,desc_maxlen, vocab, w2v = \
            process_desc(app2idx, use_pretrain)
    if use_pretrain:    
        desc_map = w2v
    print('desc vocab size:',desc_vocab_size)

    short_proceed_list,short_context_list, proceed_desc_list, context_desc_list = [],[],[],[]
    short_label_list = []
    save_idx=[]
    for p,c,l in zip(proceed_list,context_list,label_list):
        if p in desc_map and c in desc_map:
            # description
            proceed_desc_list.append(desc_map[p])
            context_desc_list.append(desc_map[c])
            save_idx.append(p)
            # app name 
            p = idx2app[str(p)]
            c = idx2app[str(c)]
            hots = lambda x: [vocab[i] for i in x]
            multihot_p = np.zeros(desc_vocab_size)
            multihot_c = np.zeros(desc_vocab_size)
            for w in hots(p.split()):
                multihot_p[w] = 1
            for w in hots(c.split()):
                multihot_c[w] = 1
                
            short_proceed_list.append(multihot_p.copy())
            short_context_list.append(multihot_c.copy())
            short_label_list.append(l)

    del proceed_list
    del context_list
    del label_list
    short_label_list = np.asarray(short_label_list)
    short_label_list = short_label_list.reshape(len(short_label_list), 1)
    print(short_label_list.shape)
    proceed_desc_list = np.asarray(proceed_desc_list)
    context_desc_list = np.asarray(context_desc_list)
    short_proceed_list = np.asarray(short_proceed_list)
    short_context_list = np.asarray(short_context_list)
    print('shape', proceed_desc_list.shape)
    #print('# train :',len(short_proceed_list))
    #print('shapes',proceed_desc_list[0],context_desc_list[0],short_proceed_list[0],short_context_list[0])
    #---------------------------
    print('building model...')
    model = build_desc_model(opts,desc_maxlen,desc_vocab_size,use_pretrain)
    model.summary()
    model.compile(optimizer=RMSprop(lr=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # TODO: train a constant epoch, then save a visualization
    print('starting training')

    model.fit([short_proceed_list, short_context_list, proceed_desc_list, context_desc_list], short_label_list, batch_size=256,
              epochs=1000, verbose=1, callbacks=None, shuffle=True)


    model.save_weights('../model/app_embedding_weight.hd5')

    #embedding_weight = model.layers[5].get_weights()
    #print(embedding_weight)
    #print('shape of embedding_weight is {}'.format(embedding_weight[0].shape))     

    model2 = build_desc_model(opts,desc_maxlen,desc_vocab_size, use_pretrain,'test')
    w = {}
    for layer in model.layers:
        w[layer.name] = layer.get_weights()
    for layer in model2.layers:
        if layer.name in w:
            layer.set_weights(w[layer.name])
    
    emb = model2.predict([short_proceed_list, short_context_list, proceed_desc_list, context_desc_list])
    print(emb[0])
    with open(os.path.join('../save_vector/', 'app_vector.npy'), 'wb') as f:
        np.savetxt(f, emb)
    print(len(save_idx))
    with open(os.path.join('../data/training/','train_index_map.npy'),'wb') as f:
        np.savetxt(f,save_idx)

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
