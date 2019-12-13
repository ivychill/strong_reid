import numpy as np
import os, sys
import shutil
import json
# np.random.seed(6)
np.random.seed(12)


def writeDataToDir(data_dic, img_source_dir, img_tar_dir, type_):
    dataList = dataDicToDdataList(data_dic)
    count = 0
    os.makedirs(img_tar_dir, exist_ok=True)
    save_txt = os.path.join(img_tar_dir, type_ + '.txt')
    with open(save_txt, 'w') as f:
        if os.path.exists(os.path.join(img_tar_dir, type_)):
            shutil.rmtree(os.path.join(img_tar_dir, type_))
        for data in dataList:
            img_pth_r, ind = data
            _, img_name = os.path.split(img_pth_r)
            img_name = str(ind) + '_' + img_name
            msg = "{}/{}/{} {}\n".format(type_, ind, img_name, ind)
            f.write(msg)
            count += 1
            img_pth_source = os.path.join(img_source_dir, img_pth_r)
            img_pth_dir = os.path.join(img_tar_dir, type_, str(ind))
            os.makedirs(img_pth_dir, exist_ok=True)
            shutil.copy(img_pth_source, os.path.join(img_pth_dir, img_name))

    print('All items number:', count)
    print('finish write:', save_txt)


def getDataFromTxt(train_all_txt, save_json=False):
    # 从txt中读取数据，放到dic之中
    data_dir, file_name = os.path.split(train_all_txt)
    file_name_json = os.path.join(data_dir, file_name[:-4] + '.json')
    data_all = {}
    with open(train_all_txt, 'r') as f:
        data_raw = f.readlines()
        for line_i in data_raw:
            img_name, pid = line_i.split(' ')
            if not os.path.exists(os.path.join(data_dir, img_name)):
                print('not find :', img_name)
                continue
            if int(pid) in data_all:
                data_all[int(pid)].append(img_name)
            else:
                data_all[int(pid)] = [img_name]
    print('All data len:', len(data_raw))
    printDataDicInfo(data_all)
    if save_json:
        with open(file_name_json, 'w') as f:
            json.dump(data_all, f)
    return data_all


def printDataDicInfo(data_all, data_name=''):
    info = {}
    index_dic = {}
    img_number = 0
    for key in data_all.keys():
        data_len = len(data_all[key])
        img_number += data_len
        if data_len in info:
            info[data_len] += 1
            index_dic[data_len].append(key)
        else:
            info[data_len] = 1
            index_dic[data_len] = [key]
    print('######### {} info ##############'.format(data_name))
    keys = list(info.keys())
    keys.sort()
    num_indexs = sorted(index_dic.items(),key=lambda x:x[0])
    print('-- persons number:{},image number:{}'.format(len(data_all.keys()),img_number))
    for len_pid,pids in num_indexs:
            print('{} persons has {} pics'.format(len(pids), len_pid))
    return num_indexs

def getLenToPidsDic(data_all):
    index_dic = {}
    for key in data_all.keys():
        data_len = len(data_all[key])
        if data_len in index_dic:
            index_dic[data_len].append(key)
        else:
            index_dic[data_len] = [key]
    return index_dic

def splitQueryDataGalleryData(data_all_dic, skip=1, max_len=300, ratios=0.15, remain_skip=False):
    # 从包含所有数据的字典{pid:[img_pth1,img_pth2,..],pid2:[img_pth1,img_pth2..]}的数据中
    # 分离出，如果数据量小于skip则不抽取，否则至少抽取一个。
    gallary_data = {}
    query_data = {}
    remain_data = {}
    all_pids = data_all_dic.keys()
    for pid in all_pids:
        paths = data_all_dic[pid]
        if len(paths) <= skip:
            # 不处理，直接丢掉
            if remain_skip:
                remain_data[pid] = data_all_dic[pid]
            continue
        if len(paths) > max_len:
            # 随机选择300个目标
            paths = np.random.choice(paths, int(max_len), replace=False)
        rand_num = 1 if int(len(paths) * ratios) <= 1 else int(len(paths) * ratios)
        rand_pids = np.random.randint(0, len(paths), rand_num)
        query_data[pid] = [paths[ind] for ind in rand_pids]
        gallary_data[pid] = [paths[ind] for ind in range(len(paths)) if ind not in rand_pids]

    printDataDicInfo(query_data,'query data info')
    printDataDicInfo(gallary_data,'gallery data info')
    printDataDicInfo(remain_data,'remain data info')
    return query_data, gallary_data, remain_data


def selectDataFromDataDic(inds, data_dic, skip=0):
    data_select, data_last = {}, {}
    for ind in data_dic.keys():
        if len(data_dic[ind]) <= skip:
            continue
        if ind in inds:
            data_select[ind] = data_dic[ind]
        else:
            data_last[ind] = data_dic[ind]
    return data_select, data_last


def dataDicToDdataList(data_dic, skip=0):
    data_list = []
    for ind in data_dic.keys():
        paths = data_dic[ind]
        if len(paths) <= skip:
            continue
        else:
            for img_pth in paths:
                data_list.append((img_pth, ind))
    return data_list


def splitTrainValid(data_all_dic, ratio, skip=1, max_len=300):
    # 从包含所有数据的字典{pid:[img_pth1,img_pth2,..],pid2:[img_pth1,img_pth2..]}的数据中
    # 分离出，如果数据量小于skip则不抽取，否则至少抽取一个。
    train_data = {}
    valid_data = {}

    all_inds = data_all_dic.keys()
    one_list = []
    les_then_4 = []
    mid_list = []
    biger_then_30 = []
    for ind in all_inds:
        if len(data_all_dic[ind]) == 1:
            one_list.append(ind)
            continue
        if len(data_all_dic[ind]) < 4:
            les_then_4.append(ind)
            continue
        if len(data_all_dic[ind]) >= 30:
            biger_then_30.append(ind)
            continue
        else:
            mid_list.append(ind)

    data_inds = list(all_inds)
    valid_inds = np.random.choice(data_inds, int(len(data_inds) * ratio), replace=False)
    for ind in all_inds:
        paths = data_all_dic[ind]
        if len(paths) <= skip:
            # 不处理，直接放入训练集,保留
            # train_data[ind] = data_all_dic[ind]
            continue
        if len(paths) > max_len and (ind not in valid_inds):
            # 训练数据限制最大个数，验证集不用限制
            paths = np.random.choice(paths, int(max_len), replace=False)

        if ind in valid_inds:
            valid_data[ind] = paths
        else:
            train_data[ind] = paths
    printDataDicInfo(train_data, 'train data')
    printDataDicInfo(valid_data, 'valid data')
    print('train data len:{},valid data len:{}'.format(len(train_data), len(valid_data)))
    return train_data, valid_data

def genTxtFromImageDir(img_dir, save_txt_name=None):
    res_id = []
    data_all = {}
    count = 1
    save_dir, ref_dir = os.path.split(img_dir)
    if save_txt_name is None:
        save_txt_name = os.path.join(save_dir, ref_dir + '.txt')
    else:
        save_txt_name = os.path.join(save_dir, save_txt_name)
    with open(save_txt_name, 'w') as f:
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                if file.endswith(('jpg', 'png', 'JPG', 'PNG')):
                    if '_' in file:
                        pid = file.split('_')[0]
                    else:
                        pid = count
                        count += 1
                    if not (save_dir.endswith('/')):
                        save_dir = save_dir + '/'
                    ref_dir_img = root.replace(save_dir, '')
                    f.write('{}/{} {}\n'.format(ref_dir_img, file, pid))
                    if pid in data_all:
                        data_all[pid].append(file)
                    else:
                        data_all[pid] = [file]
    printDataDicInfo(data_all)
    for key in data_all.keys():
        if len(data_all[key]) >= 60:
            res_id.append(key)
    print('more then 60 pic id:', res_id)
    return save_txt_name

def cleanDataDic(data_dic,min_len=0,max_len=100):
    data_dic_clean = {}
    if min_len>-1:
        data_all_ini_len = 0
        clean_len = 0
        for key in data_dic.keys():
            data_all_ini_len += len(data_dic[key])
            if len(data_dic[key]) > min_len:
                data_dic_clean[key] = data_dic[key]
            else:
                clean_len += len(data_dic[key])
        print('\n## initial number:{},after clean less then {} :{}({})\n'.format(data_all_ini_len, min_len,
                 data_all_ini_len - clean_len,clean_len))
    # 清理过多的数据，使得训练更加均衡
    if max_len:
        data_all_ini_len = 0
        data_all_clean_len = 0
        for key in data_dic.keys():
            data_all_ini_len += len(data_dic[key])
            if len(data_dic[key]) > max_len:
                data_dic_clean[key] = np.random.choice(data_dic[key], max_len, replace=False)
                data_all_clean_len += len(data_dic_clean[key])
            else:
                data_all_clean_len += len(data_dic[key])
        print('\n## initial number:{},after clean more then {} :{}({})\n'.format(data_all_ini_len, max_len,
            data_all_clean_len,data_all_ini_len-data_all_clean_len))
    printDataDicInfo(data_dic_clean,'after clean')
    return data_dic_clean

def meargeTxt(txt_list, save_txt='result.txt'):
    res = []
    for txt_ in txt_list:
        with open(txt_, 'r') as f:
            res.extend(f.readlines())
    with open(save_txt, 'w') as f:
        for item in res:
            f.write(item)


def replaceDic(be_replace_dic, replace_dic):
    for key in replace_dic.keys():
        if key in be_replace_dic:
            be_replace_dic[key] = replace_dic[key]
        else:
            print('key:{} not in '.format(key))
    return be_replace_dic


def writeDataDicToTxt(data_dic, save_txt_pth, shuffle=False):
    list_dic = sorted(data_dic.items(), key=lambda x: len(x[1]))
    with open(save_txt_pth, 'w') as f:
        for data in list_dic:
            key = data[0]
            for item in data[1]:
                f.write('{} {}\n'.format(item, key))
    print('save txt finish:',save_txt_pth)


def writeDataDicToTxtAndDir(data_dic, save_txt_pth, img_source_dir):
    save_dir, txt_file_name = os.path.split(save_txt_pth)
    os.makedirs(save_dir, exist_ok=True)
    with open(save_txt_pth, 'w') as f:
        for key in data_dic.keys():
            for item in data_dic[key]:
                img_name = item.split('/')[-1]
                img_name = str(key) + '_' + img_name
                f.write('{} {}\n'.format(item, key))
                img_source_pth = os.path.join(img_source_dir, item)
                img_target_pth = os.path.join(save_dir, img_name)
                os.makedirs(os.path.split(img_target_pth)[0], exist_ok=True)
                shutil.copy(img_source_pth, img_target_pth)



if __name__ == "__main__":

    # np.random.seed(4)

    all_data_dic = getDataFromTxt('/usr/zll/person_reid/data/sz_reid_aug/sz_reid_round2/train_list.txt')
    all_data_dic_clean = cleanDataDic(all_data_dic, min_len=0, max_len=100)
    writeDataDicToTxt(all_data_dic_clean, '/usr/zll/person_reid/data/sz_reid_aug/sz_reid_round2/atrain_rd_train_trip.txt')
    all_data_dic_clean = cleanDataDic(all_data_dic_clean, min_len=2, max_len=60)
    writeDataDicToTxt(all_data_dic_clean, '/usr/zll/person_reid/data/sz_reid_aug/sz_reid_round2/atrain_rd_train_soft.txt')


