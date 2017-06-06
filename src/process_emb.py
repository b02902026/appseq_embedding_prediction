import sys
import codecs
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import re, math
from adjustText import adjust_text
import json
from sklearn.manifold import TSNE

interested_list = [
    'Kik',
    'LINE: Free Calls & Messages',
    'Skype - free IM & video calls',
    'Messenger',
    'WhatsUp Messenger',
    'WeChat',
    'ZALORA Fashion Shopping',
    'Uber',
    'NBA 2K17', 
    'CCleaner', 
    'Cleaner', 
    'Firefox Browser fast & private', 
    'Opera browser - latest news', 
    'Chrome Dev', 
    'UC Browser Mini-Tiny and Fast'
]

def get_app_name_from_popular(json_file_path):
    return_list = []
    with codecs.open(json_file_path, 'r', 'utf-8') as f_in:
        obj_list = json.load(f_in)
        for obj in obj_list:
            if obj['app_title'] not in return_list:
                return_list.append(obj['app_title'])
    return return_list

#interested_list = get_app_name_from_popular('../data/popular.json')



def output_embedding_image(emb_file_path='../save_vector/app_vector.npy',
                           word_index_map='../data/training/app_index_map',
                           output_image_path='../emb_image/embedding_image.png',
                           interested_list=interested_list):
    # load embedding from file, then save the dim-reducted image.
    wv, vocabulary = load_embeddings(emb_file_path, word_index_map)
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(wv)
    x_list, y_list = [], []
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        if label in interested_list:
            x_list.append(x); y_list.append(y)
    print(Y[:, 0])
    print(x_list)
    plt.scatter(x_list, y_list, c='white')
    texts = []
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        if label in interested_list:
            print(label)
            texts.append(plt.text(x, y, label, size=15, color='blue'))
    plt.title(str(adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r',
                                                     lw=0.5)))+' iterations')
    #plt.show()
    plt.savefig(output_image_path)
    return

def save_app_emb_for_web(emb_file_path='../save_vector/app_vector.npy',
                           word_index_map='../data/training/app_index_map',
                           output_file_path='../web/app_emb_txt.js'):
    wv, vocabulary = load_embeddings(emb_file_path, word_index_map)

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(wv)
    with codecs.open(output_file_path, 'w', 'utf-8') as fout:
        fout.write('app_emb = ')
        app_emb_dict = {}
        for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
            # adhoc prevent unicode error
            app_emb_dict[label] = [x, y]
            json.dump(app_emb_dict, f_out)
    return

def main():
    embeddings_file = sys.argv[1]
    word_index_map = sys.argv[2]
    output_path = sys.argv[3]
    dict_type = 'old'
    if len(sys.argv) >= 5:
        dict_type = sys.argv[4]
    wv, vocabulary = load_embeddings(embeddings_file, word_index_map, dict_type)

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(wv)
    x_list, y_list = [], []
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        if label in interested_list:
            x_list.append(x); y_list.append(y)
    print(Y[:, 0])
    print(x_list)
    plt.scatter(x_list, y_list, c='white')
    texts = []
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        if label in interested_list:
            print(label)
            texts.append(plt.text(x, y, label, size=15, color='blue'))
    plt.title(str(adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r',
                                                     lw=0.5)))+' iterations')
    #plt.show()
    plt.savefig(output_path)
    with codecs.open('app_embedding.txt', 'w', 'utf-8') as fout:
        for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
            # adhoc prevent unicode error
            # label = label.replace('\u2122', '')
            print(label, x, y)
            fout.write('$,$'.join([label, str(x), str(y)])+'\n')
    print('finish output app_embedding.txt')

def get_app_id_to_row_map(train_index_map_path='../data/training/train_index_map.npy'):
    print('before load trian_index_map')
    with open(train_index_map_path, 'r') as f:
        train_index_map = np.loadtxt(f)
    print('after load train_index_map')
    app_id_to_row_map = {}
    added_list = []
    row_id = 0
    for app_id in train_index_map:
        if app_id not in added_list:
            added_list.append(app_id)
            app_id_to_row_map[str(int(app_id))] = int(row_id)
        row_id += 1
    return app_id_to_row_map

def load_embeddings(emb_file, voc_file, dict_type):

    return_wv = []
    return_voc = []
    with codecs.open(emb_file, 'r', 'utf-8') as f_in:
        wv = np.loadtxt(f_in)
    with codecs.open(voc_file, 'r', 'utf-8') as f_in:
        voc_dict = json.load(f_in)
        vocabulary = list(voc_dict.keys())

    if dict_type == 'new':
        app_id_to_row_list = get_app_id_to_row_map()
        print(app_id_to_row_list)
        #exit()
        for app_name in interested_list:
            try:
                return_wv.append(wv[app_id_to_row_list[str(voc_dict[app_name])]])
                return_voc.append(app_name)
            except:
                pass
    else:
        for app_name in interested_list:
            try:
                return_wv.append(wv[voc_dict[app_name]])
                return_voc.append(app_name)
            except:
                pass
    for i in range(0, 1000):
        if vocabulary[i] in interested_list:
            continue
        try:
            return_wv.append(wv[app_id_to_row_list[str(i)]])
            return_voc.append(vocabulary[i])
        except:
            pass
    #print(return_wv)
    #print(return_voc)
    return return_wv, return_voc

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('python3 process_emb.py embedding_file_path word_index_map output_path')
    main()
