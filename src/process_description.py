import numpy as np
import pickle
import json
import os

def dump_dict(obj,file_name):
    prefix = './data/training/'
    with open(prefix+file_name, 'wb') as f:
        pickle.dump(obj, f)


def load_json(json_path):
    description_dict = {}
    for files in os.listdir(json_path):
        #print(files)
        files  = json_path + files
        if os.path.isfile(files):
            print(files)
            with open(files,'r', encoding='utf-8', errors='ignore') as f:
                obj = json.load(f)
                print(len(obj))
                for app in obj:
                    app_name = app['app_title'].replace('\n','')
                    if app_name not in description_dict:
                        description_dict[app_name] = app['description'].replace('\n','')

    #dump_dict(description_dict, 'app2des.pkl')
    print(len(description_dict))
    return description_dict

def make_vocab(description):

    with open('../data/training/app_index_map','r',encoding='utf-8', errors='ignore') as f:
        vocab = json.load(f)

    counter = len(vocab)
    for sentence in description.values():
        for word in sentence:
            if word not in vocab:
                vocab[word] = counter
                counter += 1

    #dump_dict(vocab, 'description_vocab.pkl')
    vocab['<PAD>'] = counter
    return vocab

def word2idx(description_dict, vocab, mapping):

    one_hot = {}
    print('map len', len(mapping))
    print('dict len', len(description_dict))
    for name, sentence in description_dict.items():
        name_id = mapping[name]
        one_hot[name_id] = []
        for word in sentence:
            one_hot[name_id].append(vocab[word])
        one_hot[name_id] = one_hot[name_id]

    return one_hot

def get_padding(app2desc, vocab):
    maxlen = 0
    for sentence in app2desc.values():
        maxlen += len(sentence)
    maxlen = int(maxlen / len(app2desc))

    for app in app2desc.keys():
        while len(app2desc[app]) < maxlen:
            app2desc[app].append(vocab['<PAD>'])

        app2desc[app] = app2desc[app][:maxlen]

    #print(app2desc[app])
    print('maxlen',maxlen)
    return maxlen, app2desc


if __name__ == "__main__":
    d = load_json('./data/')
    v = make_vocab(d)
    #print(v)
    print("v size:",len(v))
    h = word2idx (d,v)
    #print(h)
