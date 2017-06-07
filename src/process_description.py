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
        if files != 'popular.json':
            continue
        files  = json_path + files
        if os.path.isfile(files) and files.find('.json') != -1:
            with open(files,'r', encoding='utf-8', errors='ignore') as f:
                obj = json.load(f)
                #print('#app',len(obj))
                for app in obj:
                    app_name = app['app_title'].replace('\n','')
                    if app_name not in description_dict:
                        description_dict[app_name] = app['description'].replace('\n','')

    #dump_dict(description_dict, 'app2des.pkl')
    print(len(description_dict))
    return description_dict

def make_vocab(description,mode='cnn'):

    #with open('../data/training/app_index_map','r',encoding='utf-8', errors='ignore') as f:
    #    vocab = json.load(f)
    vocab = {}
    if mode == 'rnn':
        with open('../data/training/app_index_map','r',encoding='utf-8', errors='ignore') as f:
            vocab = json.load(f)
        
    _max_name_len = 0
    counter = len(vocab)
    for name, sentence in description.items():
        words = name.strip().split()
        if len(words) > _max_name_len:
            _max_name_len = len(words)
        # add name in vocab
        for w in words:
            if w not in vocab:
                vocab[w] = counter
                counter += 1

        # add description in vocab
        for word in sentence:
            if word not in vocab:
                vocab[word] = counter
                counter += 1

    #dump_dict(vocab, 'description_vocab.pkl')
    vocab['<PAD>'] = counter
    return vocab, _max_name_len

def word2idx(description_dict, vocab, mapping):

    one_hot = {}
    print('map len', len(mapping), type(mapping))
    print('dict len', len(description_dict), type(description_dict))
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

def load_pretrain(old_desc_dict, app2idx):
    VEC_DIR = 'glove_dir/glove.6B.100d.txt'
    # change the key
    desc_dict = {app2idx[k]:v for k,v in old_desc_dict.items()}
        
    # load pretrainrd Glove
    pretrain = {}
    with open(VEC_DIR,'r') as f:
        for line in f:
            line = line.strip().split()
            vec = [float(i) for i in line[1:]]
            pretrain[line[0]] = np.array(vec)
    
    # get all words' vectors
    emb_size = len(list(pretrain.values())[0])
    print('emb size is',emb_size)
    maxL = 0
    for key in desc_dict.keys():
        new_desc = []
        #if maxL < desc_dict[key]:
        maxL += len(desc_dict[key])
        for word in desc_dict[key]:
            if word in pretrain:
                new_desc.append(pretrain[word])
            elif word != ' ':
                #print(word+' not in Glove.')
                new_desc.append(np.zeros(emb_size))

        desc_dict[key] = new_desc[:]
    # get the average length
    maxL /= len(list(desc_dict.keys()))
    maxL = int(maxL)
    print("average length of description is {}".format(maxL))
    # padding
    for key in desc_dict.keys():
        desc_dict[key] = desc_dict[key][:maxL]
        while len(desc_dict[key]) < maxL:
            desc_dict[key].append(np.zeros(emb_size))
        desc_dict[key] = np.asarray(desc_dict[key]).reshape((maxL,emb_size))
    del pretrain
    return [desc_dict, maxL]

if __name__ == "__main__":
    d = load_json('../data/')
    w2v = load_pretrain(d)
    print(w2v)
    v = make_vocab(d)
    #print(v)
    print("v size:",len(v))
    h = word2idx (d,v)
    #print(h)
