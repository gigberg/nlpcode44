import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import prettytable as pt
from gensim.models import KeyedVectors
from transformers import AutoTokenizer
import os
import utils
import requests
os.environ["TOKENIZERS_PARALLELISM"] = "false"

dis2idx = np.zeros((1000), dtype='int64') # Region embedding 2的0-9次方指数阶梯式距离
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'

    def __init__(self):
        self.label2id = {self.PAD: 0, self.SUC: 1}
        self.id2label = {0: self.PAD, 1: self.SUC}

    def add_label(self, label): #存实体entity["type"],存pad,存suc,,,Tail-Head-Word-*: t,,“*” indicates the entity type.,,所以实际上n*n矩阵元素的标签分类不止2个
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label

        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.label2id)

    def label_to_id(self, label):
        label = label.lower()
        return self.label2id[label]

    def id_to_label(self, i):
        return self.id2label[i]

def collate_fn(data):
    bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text = map(list, zip(*data))

    max_tok = np.max(sent_length) # 不含cls,sep的sent word length max
    sent_length = torch.LongTensor(sent_length) # 在这里,而没在先前的get_item中变成tensor, 这里效果没区别, 建议放get_item
    max_pie = np.max([x.shape[0] for x in bert_inputs]) #x.shape=[9], 取shape[0]相当于item()
    bert_inputs = pad_sequence(bert_inputs, batch_first=True)
    batch_size = bert_inputs.size(0)

    def fill(data, new_data): # batch个2d padding container
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data
    # max_tok  # 不含cls,sep的sent word length max
    dis_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    dist_inputs = fill(dist_inputs, dis_mat)
    labels_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_labels = fill(grid_labels, labels_mat)
    mask2d_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.bool)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)
    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool) # max_pie
    pieces2word = fill(pieces2word, sub_mat) #max_pie是加了cls_和_sep, 另外原先的pieces2word中就预留了cls_=0的pad位置,也即已经pad过cls了,这里只需pad sep和pad max_subword length

    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text #未处理 entity_text list[str]


class RelationDataset(Dataset):
    def __init__(self, bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text):
        self.bert_inputs = bert_inputs # Dataset初始化时已经全部tokenizer好了, getitem无需再tok
        self.grid_labels = grid_labels
        self.grid_mask2d = grid_mask2d
        self.pieces2word = pieces2word
        self.dist_inputs = dist_inputs
        self.sent_length = sent_length
        self.entity_text = entity_text

    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
               torch.LongTensor(self.grid_labels[item]), \
               torch.LongTensor(self.grid_mask2d[item]), \
               torch.LongTensor(self.pieces2word[item]), \
               torch.LongTensor(self.dist_inputs[item]), \
               self.sent_length[item], \
               self.entity_text[item] #sent_length已经是LongTensor, entity_text是str数据

    def __len__(self):
        return len(self.bert_inputs)


def process_bert(data, tokenizer, vocab):

    bert_inputs = []
    grid_labels = []
    grid_mask2d = []
    dist_inputs = []
    entity_text = []
    pieces2word = []
    sent_length = []

    for index, instance in enumerate(data):
        if len(instance['sentence']) == 0: # 去除空数据, 这个没放到ipynb数据预处理中
            continue

        tokens = [tokenizer.tokenize(word) for word in instance['sentence']] # instance['sentence']] = list[word or punctuation] #逐个word->subword
        pieces = [piece for pieces in tokens for piece in pieces]
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces) # 单个bert输入
        _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])

        length = len(instance['sentence']) # 单个输入长度[不含cls和sep,注意目前猪油bert_input中含cls和sep,用于lstm pack, 和dis emb
        _grid_labels = np.zeros((length, length), dtype=np.int)
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool)
        _dist_inputs = np.zeros((length, length), dtype=np.int)
        _grid_mask2d = np.ones((length, length), dtype=np.bool) # length*length, 不含cls/sep,初始化为1,用于生成上下三角类型矩阵

        if tokenizer is not None: # 如果未进行分词? 就不用获得piece2word矩阵
            start = 0
            for i, pieces in enumerate(tokens): #tokens = [['headache', '##s'], [','], ['pain'], ['in'], ['throat'], ...]
                if len(pieces) == 0: # 什么情况下word会不含subword?, word本身为空?
                    continue
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1 # _pieces2word, 0留给cls(wordcount_or_tokencount*subwordcount_or_piececount),,,其中pieces[0] + 1是统一空出第一列给cls[因为subwordcount_or_piececount中包含了cls和sep],,,其中:pieces[-1] + 2是因为一则左边+1右边也要+1才能保持长度不变, 二则slice操作的end位置是取不到的,要想取到需要end+1才能保证end被取到...不直接start=1的原因是空数据?测试发现没什么区别
                start += len(pieces)

        for k in range(length):  # 单个输入长度,用于lstm pack, 和dis emb, 不含cls和sep
            _dist_inputs[k, :] += k, # 第k行 0 - k, 相当于每个位置值为i-j(范围±len),上三角全负,对角线全0,下三角全正 写的这么复杂
            #右边(含)加k左边减k,每个元素等于全部左边(含)元素相加再减去全部右边元素
            _dist_inputs[:, k] -= k # 第k列 0- k, 相当于左上两边沿着对角线往右下滑动[注意是十字而不是厂字滑动]

        for i in range(length): #不含cls和sep
            for j in range(length):
                if _dist_inputs[i, j] < 0: #翻了翻了,,右上角i-j<0
                    _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9 #右上角log_2(|i-j|) + 9
                else:
                    _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
        _dist_inputs[_dist_inputs == 0] = 19 #左下角1-9,对角线19,右上角10-18

        for entity in instance["ner"]: #单个数据,全部实体mention, 不含cls和sep
            index = entity["index"]
            for i in range(len(index)): # _grid_labels zero初始化,默认矩阵元素标签0(none)
                if i + 1 >= len(index):
                    break #不要最后一个,没有后驱
                _grid_labels[index[i], index[i + 1]] = 1 #每行指向实体mention中的后驱,矩阵元素标签
            _grid_labels[index[-1], index[0]] = vocab.label_to_id(entity["type"]) # 每行下三角指向head,下三角不是存bool值,而是存实体类型??任务更复杂了,矩阵元素标签2

        _entity_text = set([utils.convert_index_to_text(e["index"], vocab.label_to_id(e["type"])) for e in instance["ner"]]) # _entity_text=[index_list_text + "-#-" + ent_typeid_text,,,,],,,entity_type 0和1标签分别被pad[0]/suc[NNW]占用了,其它n个标签是随不同数据集的train+dev+test情况灵活增长的

        sent_length.append(length)
        bert_inputs.append(_bert_inputs)
        grid_labels.append(_grid_labels)
        grid_mask2d.append(_grid_mask2d)
        dist_inputs.append(_dist_inputs)
        pieces2word.append(_pieces2word)
        entity_text.append(_entity_text)

    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text


def fill_vocab(vocab, dataset):
    entity_num = 0
    for instance in dataset:
        for entity in instance["ner"]:
            vocab.add_label(entity["type"])
        entity_num += len(instance["ner"])
    return entity_num # dataset上的总实体数


def load_data_bert(config):
    with open('./data/{}/train.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('./data/{}/dev.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    with open('./data/{}/test.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache/")

    vocab = Vocabulary()
    train_ent_num = fill_vocab(vocab, train_data) #总实体数,不去重,主要用于统计
    dev_ent_num = fill_vocab(vocab, dev_data)
    test_ent_num = fill_vocab(vocab, test_data)

    table = pt.PrettyTable([config.dataset, 'sentences', 'entities'])
    table.add_row(['train', len(train_data), train_ent_num])
    table.add_row(['dev', len(dev_data), dev_ent_num])
    table.add_row(['test', len(test_data), test_ent_num])
    config.logger.info("\n{}".format(table)) #prettytable通过logging或print打印

    config.label_num = len(vocab.label2id) # 实体类型去重后的标签数,含pad, suc,作为n*n矩阵的2-n的标签值,pad0,nnw1, THW-*(2-n)
    config.vocab = vocab

    train_dataset = RelationDataset(*process_bert(train_data, tokenizer, vocab))
    dev_dataset = RelationDataset(*process_bert(dev_data, tokenizer, vocab))
    test_dataset = RelationDataset(*process_bert(test_data, tokenizer, vocab))
    return (train_dataset, dev_dataset, test_dataset), (train_data, dev_data, test_data)

