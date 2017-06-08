# using python3

import sys
import json
import numpy as np
from sklearn.manifold import TSNE

# show default usage to simply use
if len(sys.argv) != 6:
    print('default usage:')
    print("""python3 make_web_use_app.py ../save_vector/app_vector.npy ../data/training/train_index_map.npy ../data/training/index_app_map ./app_emb_txt.js ./app_name.js""")

# setting reading and writing files
app_vector_path = sys.argv[1]
train_index_map_path = sys.argv[2] # 我猜明天要嘛用 CNN 的要嘛用 RNN 的，因此就都要這個檔案
index_app_map_path = sys.argv[3]
app_emb_text_path = sys.argv[4] # 這是 output
app_name_path = sys.argv[5]  # 這是 output

# read all necessary input file
all_app_vector_list = np.loadtxt(app_vector_path)  # all 代表全部的，row 有重複
row_to_app_id_list = np.loadtxt(train_index_map_path)
for index in range(0, len(row_to_app_id_list)):
    row_to_app_id_list[index] = int(row_to_app_id_list[index])
with open(index_app_map_path, 'r') as f_j:
    index_to_app_map = json.load(f_j)

# make app_id_to_row_map
app_id_to_row_map = {}
added_list = []
row_id = 0
for app_id in row_to_app_id_list:
    if app_id not in added_list:
        added_list.append(app_id)
        app_id_to_row_map[str(int(app_id))] = int(row_id)
    row_id += 1
del row_to_app_id_list
del added_list

# reductino and  make a app_to_vector dict
# 之所以要一起做是因為 reduction 只吃 list
app_name_list = []
app_vector_list = []
for app_index_string in index_to_app_map:
    try:
        app_name = index_to_app_map[app_index_string]
        app_vector = all_app_vector_list[app_id_to_row_map[app_index_string]]
        app_name_list.append(app_name)
        app_vector_list.append(app_vector)
    except:
        pass
print('len of app_name_list: {}, len of app_vector_list: {}'.format(len(app_name_list), len(app_vector_list)))

# dim-reduction
tsne = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
app_vector_list = tsne.fit_transform(app_vector_list)

# make app_to_vector_map
app_to_vector_map = {}
for app_name, app_vector in zip(app_name_list, app_vector_list):
    app_to_vector_map[app_name] = [app_vector[0], app_vector[1]]
del app_vector_list

# write to output
with open(app_emb_text_path, 'w') as f_text, open(app_name_path, 'w') as f_name:
    f_text.write('app_emb = ')
    f_name.write('app_name_list = ')
    json.dump(app_to_vector_map, f_text)
    json.dump(app_name_list, f_name)

