appseq_path = '/home/zychen/appseq_embedding_prediction/data/appseq_data/app_sequences.txt' 

from collections import defaultdict
import operator

app_count_dict = defaultdict(int)
with open(appseq_path, 'r') as f:
    for line in f:
        full_app_names = line.strip()
        for full_app_name in full_app_names.split(';'):
            app_count_dict[full_app_name] += 1
print('there are {} apps in data.'.format(len(app_count_dict)))
sorted_app_count_dict = sorted(app_count_dict.items(), key=operator.itemgetter(1))
print('distribution is {}'.format(sorted_app_count_dict))

