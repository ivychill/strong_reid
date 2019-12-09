import os

base_dir = '.'
list_file = 'query_a_list.txt'
query_dir = 'query_a'

f = open(os.path.join(base_dir, list_file),'w')
images = os.listdir(os.path.join(base_dir, query_dir))
pid = 20000
for image in images:
    print('{}/{} {}'.format(query_dir, image, pid))
    f.write('{}/{} {}\n'.format(query_dir, image, pid))
    pid += 1