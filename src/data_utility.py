import json
from sys import argv
from os import path
from collections import defaultdict
from numpy.random import choice


TRAINING_DATA_PATH = '../data/training' if len(argv) <= 2 else argv[1]

def generate_index_data():
    relation_dict = {}  # in app_name to app_name_list format
    freq_dict = defaultdict(int)
    app_index_dict, index_app_dict = {}, {}

    # load x and y
    with open(path.join(TRAINING_DATA_PATH, 'x'), 'r', encoding='utf-8',
              errors='ignore') as fx:
        with open(path.join(TRAINING_DATA_PATH, 'y'), 'r', encoding='utf-8',
                  errors='ignore') as fy:
            for line_x, line_y in zip(fx, fy):
                proceeding_app = line_x.replace('\n', '')
                relation_list = line_y.replace('\n', '').split('\t')
                relation_dict[proceeding_app] = relation_list
    # app index mapping
    counter = 1
    for app in relation_dict:
        app_index_dict[app] = counter
        index_app_dict[counter] = app
        counter += 1

    # make freq_index
    for proceeding_app in relation_dict:
        for relate_app in relation_dict[proceeding_app]:
            freq_dict[app_index_dict[relate_app]] += 1

    # make index relaiton dict
    relation_index_dict = {}
    for app in relation_dict:
        app_index = app_index_dict[app]
        relaiton_list = relation_dict[app]
        relation_index_list = []
        for relation_app in relation_list:
            relation_index_list.append(app_index_dict[relation_app])
        relation_index_dict[app_index] = relation_index_list

    # dump index_maps, index_relation_dict
    with open(path.join(TRAINING_DATA_PATH, 'index_relation_dict'), 'w',
              encoding='utf-8', errors='ignore') as fout:
        json.dump(relation_index_dict, fout)
    with open(path.join(TRAINING_DATA_PATH, 'index_app_map'), 'w',
              encoding='utf-8', errors='ignore') as fout:
        json.dump(index_app_dict, fout)
    with open(path.join(TRAINING_DATA_PATH, 'app_index_map'), 'w',
              encoding='utf-8', errors='ignore') as fout:
        json.dump(app_index_dict, fout)
    with open(path.join(TRAINING_DATA_PATH, 'freq_dict'), 'w',
              encoding='utf-8', errors='ignore') as fout:
        json.dump(freq_dict, fout)

    return 

def load_index_data():
    with open(path.join(TRAINING_DATA_PATH, 'index_relation_dict'), 'r',
              encoding='utf-8', errors='ignore') as f:
        relation_dict = json.load(f)
    with open(path.join(TRAINING_DATA_PATH, 'index_relation_dict'), 'r',
              encoding='utf-8', errors='ignore') as f:
        index_app_map = json.load(f)
    with open(path.join(TRAINING_DATA_PATH, 'index_relation_dict'), 'r',
              encoding='utf-8', errors='ignore') as f:
        app_index_map = json.load(f)
    with open(path.join(TRAINING_DATA_PATH, 'freq_dict'), 'r',
              encoding='utf-8', errors='ignore') as f:
        freq_dict = json.load(f)
    return relation_dict, index_app_map, app_index_map, freq_dict

def generate_data(opts):
    # return a tuple of lists (couple_list, label_list)
    # coupel list is a list of tuple (target_app_index, context_app_index)
    # label_list is a list of 0 or 1, with positive instance and negative instances.


    if path.isfile(path.join(TRAINING_DATA_PATH, 'index_relation_dict')) is False or \
    path.isfile(path.join(TRAINING_DATA_PATH, 'index_app_map')) is False or \
    path.isfile(path.join(TRAINING_DATA_PATH, 'app_index_map')) is False: 
        generate_index_data()
    
    relation_dict, app_index_map, index_app_map, freq_dict = load_index_data()
   
    vocab_size = len(app_index_map)
    opts['vocab_size'] = vocab_size

    # the two lists below map one to one
    # the order is gurantee in python2, not sure if python3
    # http://stackoverflow.com/questions/835092/python-dictionary-are-keys-and-values-always-the-same-order
    app_index_list = list(freq_dict.keys()) 
    app_freq_list = list(freq_dict.values())
    app_prob_list, app_freq_sum = [], 0
    for freq in app_freq_list:
        app_freq_sum += freq
    for freq in app_freq_list:
        app_prob_list.append(float(freq)/float(app_freq_sum))
    

    couple_list = []
    label_list = []
    
    if path.isfile(path.join(TRAINING_DATA_PATH, 'couple_list')) and\
    path.isfile(path.join(TRAINING_DATA_PATH, 'label_list')):
        with open(path.join(TRAINING_DATA_PATH, 'couple_list'), 'r',
                encoding='utf-8', errors='ignore') as f:
            for line in f:
                split_line = line.replace('\n', '').split(',')
                couple_list.append([int(split_line[0]), int(split_line[1])])
        with open(path.join(TRAINING_DATA_PATH, 'label_list'), 'r',
                encoding='utf-8', errors='ignore') as f:
            for line in f:
                label_list.append(int(line.replace('\n', '')))
    else:        

        counter = 0
        dict_len = len(relation_dict)
        for proceed_app, similar_list in relation_dict.items():
            counter += 1
            # this error can be ignore, should be error of formaterr
            print('\r{}/{}'.format(counter, dict_len), end="")
            # positive instance
            for similar_app in similar_list:
                couple_list.append((proceed_app, similar_app))
                label_list.append(1)
            for i in range(0, 5):
                # sample a negative similar app by wieght.
                while True:
                    sample_index = choice(app_index_list, p=app_prob_list)
                    if sample_index not in similar_list and int(sample_index) != int(proceed_app):
                        couple_list.append((proceed_app, sample_index))
                        label_list.append(0)
                        break
        
        with open(path.join(TRAINING_DATA_PATH, 'couple_list'), 'w',
                  encoding='utf-8', errors='ignore') as f:
            for couple in couple_list:
                f.write('{},{}\n'.format(couple[0], couple[1]))
        with open(path.join(TRAINING_DATA_PATH, 'label_list'), 'w',
                  encoding='utf-8', errors='ignore') as f:
            for label in label_list:
                f.write('{}\n'.format(label))

    
    return (couple_list, label_list, opts)

                


if __name__ == '__main__':
    couple_list, label_list = generate_data()
    print('lenght of couple_list = {}'.format(len(couple_list)))
    print('lenght of label_list = {}'.format(len(label_list)))
    

    







