import os



def gen_list(list_file, sub_dir, base_pid):
    f = open(os.path.join(base_dir, list_file),'w')
    images = os.listdir(os.path.join(base_dir, sub_dir))
    pid = base_pid
    for image in images:
        print('{}/{} {}'.format(sub_dir, image, pid))
        f.write('{}/{} {}\n'.format(sub_dir, image, pid))
        pid += 1

if __name__ == '__main__':
    base_dir = '.'

    # list_file = 'query_b_list.txt'
    # sub_dir = 'query_b'
    # base_pid = 20000
    # gen_list(list_file, sub_dir, base_pid)

    list_file = 'gallery_b_list.txt'
    sub_dir = 'gallery_b'
    base_pid = 40000
    gen_list(list_file, sub_dir, base_pid)