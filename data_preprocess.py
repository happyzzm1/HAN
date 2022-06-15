import pickle
import time
import jieba
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from torchtext.legacy import data
import utiles


def read_data_from_csv_to_pkl(fields, train_file, validation_file, test_file, prefix):
    """
    从csv文件中读取数据，构建torchtext中的Dataset所需要的examples，并以pkl格式保存
    三个文件默认保存在data目录下
    :param prefix: 目录前缀
    :param fields: 构建Dataset所需要的fields
    :param train_file: 训练集文件名字，csv格式
    :param validation_file: 验证集文件名字，csv格式
    :param test_file: 测试集文件名字，csv格式
    """
    print("---------------开始从csv文件中读取数据---------------")
    start_time = time.time()

    train, val, test = data.TabularDataset.splits(
        path=prefix, train=train_file, validation=validation_file, test=test_file,
        format='csv', fields=fields, skip_header=True)

    train_examples = train.examples
    val_examples = val.examples
    test_examples = test.examples

    print("训练数据条目：", len(train_examples))
    print("验证数据条目：", len(val_examples))
    print("测试数据条目：", len(test_examples))

    with open(file=prefix + train_file[:-4] + '_examples' + '.pkl', mode='wb') as f:
        pickle.dump(train_examples, f)
    with open(file=prefix + validation_file[:-4] + '_examples' + '.pkl', mode='wb') as f:
        pickle.dump(val_examples, f)
    with open(file=prefix + test_file[:-4] + '_examples' + '.pkl', mode='wb') as f:
        pickle.dump(test_examples, f)
    print("读取用时：", time.time() - start_time)
    print("---------------从csv文件中读取数据成功---------------")


def read_data_from_pkl(fields, train_file, validation_file, test_file, batch_size, device, prefix):
    """
    从pkl文件中读取examples，用来构建torchtext中的Dataset，从而构建样本迭代器
    :param prefix: 目录前缀
    :param fields: 构建Dataset所需要的fields
    :param train_file: 训练集文件名字，pkl格式
    :param validation_file: 验证集文件名字，pkl格式
    :param test_file: 测试集文件名字，pkl格式
    :param batch_size: 迭代器每个batch的大小
    :param device: cpu或者cuda
    :return: 训练集、训练集迭代器、验证集迭代器、测试集迭代器
    """
    print("---------------开始从pkl文件中读取数据---------------")
    start_time = time.time()
    with open(file=prefix + train_file, mode='rb') as f:
        train_examples = pickle.load(f)
    with open(file=prefix + validation_file, mode='rb') as f:
        validation_examples = pickle.load(f)
    with open(file=prefix + test_file, mode='rb') as f:
        test_examples = pickle.load(f)

    train = data.Dataset(train_examples, fields)
    val = data.Dataset(validation_examples, fields)
    test = data.Dataset(test_examples, fields)

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train, val, test),
        sort_key=lambda x: x.TEXT,
        batch_size=batch_size, device=device, sort_within_batch=True)

    print("读取用时：", time.time() - start_time)
    print("---------------从pkl文件中读取数据成功---------------")
    return train, train_iterator, valid_iterator, test_iterator


def read_ids_from_pkl(train_file, validation_file, prefix):
    """
    从pkl文件中读取ids
    :param prefix: 目录前缀
    :param train_file: 训练集文件名字，pkl格式
    :param validation_file: 验证集文件名字，pkl格式
    :return: 训练集ids、验证集ids
    """
    print("---------------开始从pkl文件中读取数据---------------")
    start_time = time.time()
    with open(file=prefix + train_file, mode='rb') as f:
        train_ids = pickle.load(f)
    with open(file=prefix + validation_file, mode='rb') as f:
        validation_ids = pickle.load(f)

    print("读取用时：", time.time() - start_time)
    print("---------------读取ids矩阵成功---------------")
    return train_ids, validation_ids


def read_mask_from_pkl(train_file, validation_file, prefix):
    """
    从pkl文件中读取mask
    :param prefix: 目录前缀
    :param train_file: 训练集文件名字，pkl格式
    :param validation_file: 验证集文件名字，pkl格式
    :return: 训练集mask、验证集mask
    """
    print("---------------开始从pkl文件中读取数据---------------")
    start_time = time.time()
    with open(file=prefix + train_file, mode='rb') as f:
        train_mask = pickle.load(f)
    with open(file=prefix + validation_file, mode='rb') as f:
        validation_mask = pickle.load(f)

    print("读取用时：", time.time() - start_time)
    print("---------------读取mask矩阵成功---------------")
    return train_mask, validation_mask


def read_train_data_from_pkl(fields, train_file, prefix):
    """
    从pkl文件中读取examples，用来构建torchtext中的Dataset，从而构建样本迭代器
    :param prefix: 目录前缀
    :param fields: 构建Dataset所需要的fields
    :param train_file: 训练集文件名字，pkl格式
    :return: 训练集、训练集迭代器、验证集迭代器、测试集迭代器
    """
    print("---------------开始从pkl文件中读取数据---------------")
    start_time = time.time()
    with open(file=prefix + train_file, mode='rb') as f:
        train_examples = pickle.load(f)
    train = data.Dataset(train_examples, fields)
    print("读取用时：", time.time() - start_time)
    print("---------------从pkl文件中读取数据成功---------------")
    return train


def build_fields():
    ID = data.Field()
    TEXT = data.Field(include_lengths=True, tokenize=jieba.lcut)
    LOC_1, LOC_2, LOC_3 = data.LabelField(), data.LabelField(), data.LabelField()
    SER_1, SER_2, SER_3, SER_4 = data.LabelField(), data.LabelField(), data.LabelField(), data.LabelField()
    PRI_1, PRI_2, PRI_3 = data.LabelField(), data.LabelField(), data.LabelField()
    ENV_1, ENV_2, ENV_3, ENV_4 = data.LabelField(), data.LabelField(), data.LabelField(), data.LabelField()
    DISH_1, DISH_2, DISH_3, DISH_4 = data.LabelField(), data.LabelField(), data.LabelField(), data.LabelField()
    OTH_1, OTH_2 = data.LabelField(), data.LabelField()

    fields = [('ID', ID), ('TEXT', TEXT), ('LOC_1', LOC_1), ('LOC_2', LOC_2), ('LOC_3', LOC_3), ('SER_1', SER_1)
              , ('SER_2', SER_2), ('SER_3', SER_3), ('SER_4', SER_4), ('PRI_1', PRI_1), ('PRI_2', PRI_2)
              , ('PRI_3', PRI_3), ('ENV_1', ENV_1), ('ENV_2', ENV_2), ('ENV_3', ENV_3), ('ENV_4', ENV_4)
              , ('DISH_1', DISH_1), ('DISH_2', DISH_2), ('DISH_3', DISH_3), ('DISH_4', DISH_4)
              , ('OTH_1', OTH_1), ('OTH_2', OTH_2)]
    return fields


def build_vocab(train, fields, max_size):
    for field in fields:
        if field[0] == 'TEXT':
            field[1].build_vocab(train, max_size=max_size)
        else:
            field[1].build_vocab(train)


def build_mask_from_ids(ids, pad_idx):
    """从ids构建mask矩阵，输入的ids是一个list"""
    mask_list = []
    for i in range(len(ids)):
        mask = [[1 if id_ != pad_idx else 0 for id_ in line]for line in ids]
        mask_list.append(mask)
    return mask_list


def build_mask_from_ids_to_pkl(data_file, prefix, ids_list, pad_idx):
    print("---------------开始构建mask矩阵---------------")
    start_time = time.time()
    mask_list = []
    for i in range(len(ids_list)):
        ids = ids_list[i]
        mask = [[1 if id_ != pad_idx else 0 for id_ in line] for line in ids]
        mask_list.append(mask)
    with open(file=prefix + data_file[:-4] + '_mask' + '.pkl', mode='wb') as f:
        pickle.dump(mask_list, f)
    print("构建用时：", time.time() - start_time)
    print("---------------构建mask矩阵成功---------------")


def build_ids_from_csv_to_pkl(data_file, prefix, vocab):
    print("---------------开始构建ids矩阵---------------")
    start_time = time.time()
    dataframe = pd.read_csv(prefix + data_file, header=0, encoding='utf-8')
    ids_list = []
    mask_list = []
    for index in range(len(dataframe)):
        text = str(dataframe.content[index])[1:-1]  # 去掉前后的引号
        words_list = jieba.lcut(text)  # 将文档切分成单词
        sentences_list, lengths = utiles.cut_sentences_by_list(words_list, max_len=64, max_sen_num=8)  # 将文档切分成句子
        ids = [[vocab.stoi[word] for word in sen] for sen in sentences_list]  # (sen_num, sen_len)
        ids_list.append(ids)
        # mask_list.append(build_mask_from_ids(ids, 0))
    with open(file=prefix + data_file[:-4] + '_ids' + '.pkl', mode='wb') as f:
        pickle.dump(ids_list, f)
    print("构建用时：", time.time() - start_time)
    print("---------------构建ids矩阵成功---------------")


class DatasetForHAN(Dataset):
    def __init__(self, data_file, ids, mask, tag_str):
        self.dataframe = pd.read_csv(data_file, header=0, encoding='utf-8')
        self.ids = ids
        self.len = len(self.dataframe)
        self.mask = mask
        self.tag_str = tag_str

    def __len__(self):
        return self.len

    def __getitem__(self, index) -> T_co:
        ids = self.ids[index]
        mask = self.mask[index]
        return {
            'ids': ids,  # (sen_num, sen_len)
            'mask': mask,  # (sen_num, sen_len)，padding处为0，其余为1
            'sen_num': len(ids),  # 实际的句子数量
            'sen_len': len(ids[0]),  # 最大句子长度
            # 'lengths': lengths,  # 实际句子长度
            self.tag_str: torch.tensor(
                self.dataframe['location_traffic_convenience'][index], dtype=torch.long).unsqueeze(0) + 2
            # +2的目的是把标签从[-2, -1, 0, 1]映射到[0, 1, 2, 3]
        }


def build_data_loader_for_han(dataset, batch_size):
    def build_tensor_for_han(data_list):
        new_dict = {}
        ids_list = []
        mask_list = []
        sen_nums = []
        max_sen_len = max([data_list[i]['sen_len'] for i in range(len(data_list))])
        max_sen_num = max([data_list[i]['sen_num'] for i in range(len(data_list))])
        pad_sen = [0] * max_sen_len
        for i in range(len(data_list)):
            # data_list[i]是从dataset返回的字典, data_list[i]['ids']是二维数组——（sen_num, sen_len)
            data_han = data_list[i]
            new_ids = []
            new_mask = []
            sen_len = data_han['sen_len']
            sen_num = data_han['sen_num']
            # 首先扩充句子长度
            sen_nums.append(sen_num)
            for j in range(sen_num):
                new_ids.append(data_han['ids'][j] + [0] * max(0, max_sen_len - sen_len))
                new_mask.append(data_han['mask'][j] + [0] * max(0, max_sen_len - sen_len))
            # 接下来扩充句子数量
            for k in range(max_sen_num - sen_num):
                new_ids.append(pad_sen)
                new_mask.append(pad_sen)
            # 转换为tensor
            ids_list.append(torch.tensor(new_ids, dtype=torch.long).unsqueeze(0))
            mask_list.append(torch.tensor(new_mask, dtype=torch.long).unsqueeze(0))
            # (1, max_sen_num, max_sen_len)
        new_dict['ids'] = torch.cat(ids_list, dim=0)  # (batch_size, max_sen_num, max_sen_len)
        new_dict['mask'] = torch.cat(mask_list, dim=0)  # (batch_size, max_sen_num, max_sen_len)
        new_dict[dataset.tag_str] = torch.cat([d[dataset.tag_str] for d in data_list])  # (batch_size,)
        new_dict['sen_nums'] = sen_nums
        return new_dict

    params = {'batch_size': batch_size,
              'collate_fn': lambda d: build_tensor_for_han(d),
              'shuffle': True,
              'num_workers': 0
              }
    data_loader = DataLoader(dataset, **params)
    return data_loader


