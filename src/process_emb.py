import sys
import codecs
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import re, math

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
    'WhatsUp Messenger'
]

def main():
    embeddings_file = sys.argv[1]
    word_index_map = sys.argv[2]
    output_path = sys.argv[3]
    wv, vocabulary = load_embeddings(embeddings_file, word_index_map)

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(wv)
        
    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        if label in interested_list:
            print(label)
            plt.annotate(label, xy=(x, y), xytext=(0, 0),
                         textcoords='offset points')
    #plt.show()
    plt.savefig(output_path)
    


 
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
