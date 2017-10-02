import sys
import json
from os import listdir
from os.path import isfile, join

# a version without description
# this script read all json file from scrapy and turn them to x and y
# x is proceeding app name
# y is similar app names
# they put in file x and y line by line

if len(sys.argv) >=2 and sys.argv[1] == '-h':
    print('Usage: python3 preporcess_data.py [DATA_DIR_PATH]')

DATA_DIR_PATH = sys.argv[1] if len(sys.argv) >= 2 else '../data'
USE_DESCRIPTION = False

json_files = [join(DATA_DIR_PATH, f) for f in listdir(DATA_DIR_PATH) if isfile(join(DATA_DIR_PATH, f))]
json_files = [join(DATA_DIR_PATH, 'popular.json')]


# key: proceeding app name
# value: a list contain similar app names, lists are in various lengths
app_relation_dict = {}

for json_file in json_files:
    if not json_file.endswith('.json'):
        continue
    with open(json_file, 'r', encoding='utf-8', errors='ignore') as f:
        print('now preceeding file {}'.format(json_file))
        data_list = json.load(f)
        for data in data_list:
            app = data['app_title']
            similar_apps = data['similar_apps']

            # clean data
            del_list = []
            for i in range(len(similar_apps)):
                similar_apps[i] = similar_apps[i].replace('\n', '')
                if len(similar_apps[i].replace(' ', '')) == 0:
                    similar_apps[i] = "###ESCAPE#"
            i = 0
            while i < len(similar_apps):
                if similar_apps[i] == "###ESCAPE#":
                    del similar_apps[i]
                    i = i - 1
                i += 1

            if USE_DESCRIPTION:
                # do something
                description = data['description']

            #print('app = {}, similar_apps = {}'.format(app, similar_apps))
            app_relation_dict[app] = similar_apps

print('finish loading json')

# checking reverse relationship
# if these peice of code is not understandable, I can write an example
apps = list(app_relation_dict.keys())
for app in apps:
    for similar_app in app_relation_dict[app]:
        if similar_app not in app_relation_dict:
            app_relation_dict[similar_app] = [app]
        elif app not in app_relation_dict[similar_app]:
            app_relation_dict[similar_app].append(app)

print('finish checking reverse relationship')
print('numbers of app: {}'.format(len(app_relation_dict)))
print('numbers of relation: {}'.format(sum( len(val) for val in app_relation_dict.values() )))

# dump to file
with open(join(DATA_DIR_PATH, 'training/x'), 'w', encoding='utf-8', errors='ignore') as fx:
    with open(join(DATA_DIR_PATH, 'training/y'), 'w', encoding='utf-8', errors='ignore') as fy:
        for app, similar_apps in app_relation_dict.items():
            if len(app) == 0:
                print(app)
                print(similar_apps)
                exit()
            fx.write(app+'\n')
            fy.write('\t'.join(similar_apps)+'\n')

print('finish dumping files {} and {}'.format(join(DATA_DIR_PATH, 'training/x'),
                                              join(DATA_DIR_PATH, 'training/y')))



