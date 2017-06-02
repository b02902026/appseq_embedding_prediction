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
from sklearn.metrics.pairwise import cosine_similarity as cos

if len(sys.argv) != 4:
    print('python3 process_emb.py embedding_file_path word_index_map output_path')

interested_list = [
    'Kik', 
    'LINE: Free Calls & Messages', 
    'Skype - free IM & video calls', 
    'Messenger',
    'WhatsUp Messenger',
    'YouTube',
    'WeChat',
    'Uber', 
    'NBA 2K17', 
    'ZALORA Fashion Shopping'
]

def main():
    embeddings_file = sys.argv[1]
    word_index_map = sys.argv[2]
    output_path = sys.argv[3]
    wv, vocabulary = load_embeddings(embeddings_file, word_index_map)

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
            fout.write('\t'.join([label, str(x), str(y)])+'\n')
    print('finish output app_embedding.txt')
    


 
def load_embeddings(emb_file, voc_file):

    return_wv = []
    return_voc = []
    with codecs.open(emb_file, 'r', 'utf-8') as f_in:
        wv = np.loadtxt(f_in)
    with codecs.open(voc_file, 'r', 'utf-8') as f_in:
        voc_dict = json.load(f_in)
        vocabulary = list(voc_dict.keys())

    for app_name in interested_list:
        try:
            return_wv.append(wv[voc_dict[app_name]])
            return_voc.append(app_name)
        except KeyError:
            pass
    for i in range(0, 1000):
        if vocabulary[i] in interested_list:
            continue
        return_wv.append(wv[i])
        return_voc.append(vocabulary[i])

    return return_wv, return_voc
 
if __name__ == '__main__':
    main()
