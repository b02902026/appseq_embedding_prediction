# using python3

import sys
import json

with open(sys.argv[1], 'r') as f_in:
    with open(sys.argv[2], 'w') as f_out:
        with open('app_name.js', 'w') as f_n_out:
            f_out.write('app_emb = ')
            f_n_out.write('app_name_list = ')
            app_emb_dict = {}
            app_name_list = []
            for line in f_in:
                app_name = ''.join(line.split('$,$')[0:-2])
                x = float(line.split('$,$')[-2].replace('\n', ''))
                y = float(line.split('$,$')[-1].replace('\n', ''))
                app_emb_dict[app_name] = [x, y]
                print(app_name, x, y)
                app_name_list.append(app_name)
            json.dump(app_emb_dict, f_out)
            json.dump(app_name_list, f_n_out)
print('finish making {}, {}'.format(sys.argv[2], 'app_name.js'))



