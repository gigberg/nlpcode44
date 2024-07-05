import torch
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import json
from collections import defaultdict
import numpy as np
import prettytable as pt
from tqdm import tqdm

from utils import get_edge_matrix

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def collate_fn(data):
    bert_input, word_subword, sent_entity, cls_index, edge_matrix, consist_label = map(list, zip(*data))

    def pad_plane(batch_word_subword):
        batch_size= len(batch_word_subword)
        max_word_len = max([x.size(-2) for x in batch_word_subword])
        subword_size = max([x.size(-1) for x in batch_word_subword])
        new_word_subword = torch.zeros((batch_size,*(batch_word_subword[0].shape[:-2]), max_word_len, subword_size), dtype=torch.bool)
        for dia, _word_subword in enumerate(batch_word_subword):
            new_word_subword[dia,...,:_word_subword.shape[-2], :_word_subword.shape[-1]] = _word_subword
        return new_word_subword

    bert_input = pad_sequence(bert_input, batch_first=True)
    cls_index = pad_sequence(cls_index, batch_first=True)
    word_subword = pad_plane(word_subword)
    sent_entity = pad_plane(sent_entity)
    edge_matrix = pad_plane(edge_matrix) #[bat,edge_type,n,n] -> [bat*edge_type,n,n]
    consist_label = torch.tensor(consist_label).long()

    return bert_input, word_subword, sent_entity, cls_index, edge_matrix, consist_label


class MyDataset(Dataset):
    def __init__(self, bert_input, word_subword, sent_entity, cls_index, edge_matrix, consist_label):
        self.bert_input = bert_input
        self.word_subword = word_subword
        self.sent_entity = sent_entity
        self.cls_index = cls_index
        self.edge_matrix = edge_matrix
        self.consist_label = consist_label

    def __getitem__(self, index):
        return torch.tensor(self.bert_input[index]).long(), \
               torch.tensor(self.word_subword[index]).long(), \
               torch.tensor(self.sent_entity[index]).long(), \
               torch.tensor(self.cls_index[index]).long(), \
               torch.tensor(self.edge_matrix[index]).long(), \
               self.consist_label[index]

    def __len__(self):
        return len(self.bert_input)


def process_bert(data, tokenizer, config):
    def traverse_entity(sent, is_ans):
        entity_index = []
        for ent in config.global_entities:
            _entity_index = (np.array(sent)==ent).nonzero()[0].tolist()
            entity_index.extend(_entity_index)
        # assert entity_index if i%2 else True # 发现并非每个ans必定有entity, ['they', 'both', 'have', 'none', '[CLS]'] 此时用随机词代替
        if is_ans and not entity_index:
            entity_index = [len(sent) - 1] # 没有entity的一般句子较短, 所以取[CLS]当entity
        return entity_index

# 0. extract dialogue sentence, and amr/adj src, tgt index
    with open('data/citod/citod_amr2adj.json') as f_amr, open('data/citod/citod_coref2adj.json') as f_coref:
        coref_json, amr_json = json.load(f_coref), json.load(f_amr)
    dia_dict = defaultdict(list)
    consist_label, all_edge = [], []

    for index, item in enumerate(data):
        if len(item['dialogue']) % 2:
            continue
        for utt in item['dialogue']:
            utt_word_list = utt['utterance'].split()
            dia_dict[index].append(utt_word_list) #处理成bert tokenizer不会拆分的,不带标点的word list, 不移实体词下划线
        consist_label.append(1 if int(item['scenario']["qi"]) + int(item['scenario']["hi"]) > 0 else 0) # 历史一致性标签，融合Qi、Hi
        id = item['id'] # In JSON (JavaScript Object Notation), keys can only be strings.
        assert coref_json[str(id)]['dia_word'] == amr_json[str(id)]['dia_word']
        assert coref_json[str(id)]['dia_len'] == sum([len(x) for x in dia_dict[index]]) + len(dia_dict[index])
        _all_edge = dict()
        _all_edge['coref_edge'] = coref_json[str(id)]['coref_adj']
        _all_edge['amr_edge'] = amr_json[str(id)]['amr_adj']
        _all_edge['utt_edge'] = (np.array(coref_json[str(id)]['dia_word']) == '[CLS]').nonzero()[0].tolist()
        all_edge.append(_all_edge)

    # 1.get bert index# get sentence token ids <最慢的放在最后>
    utt_word = []
    bert_input = []
    cls_index = []
    word_subword = []
    sent_entity= []
    for dia in tqdm(dia_dict.values(), desc='Processing Tokenizition'):
        sent_word = []
        _utt_word = []
        for utt in dia:
            _sent_word = [tokenizer.tokenize(word) for word in utt]
            _sent_word.append([tokenizer.cls_token])
            _utt_word.extend(utt + [tokenizer.cls_token])
            sent_word.append(_sent_word) # one dialogue, list[sent] of list[word] of list[subword]
        _dia_word = [word for sent in sent_word for word in sent]
        subword_flattened = [subword for word in _dia_word for subword in word]
        _bert_input = tokenizer.convert_tokens_to_ids(subword_flattened)

        sent_count = len(dia)
        word_count = len(_dia_word)
        subword_count= len(_bert_input)

        _word_subword = np.zeros((word_count, subword_count), dtype=np.bool)
        _sent_entity = np.zeros((sent_count, word_count), dtype=np.bool)

        start = 0
        for i, subwords in enumerate(_dia_word):
            if len(subwords) == 0:
                continue
            subwords = list(range(start, start + len(subwords)))
            _word_subword[i, subwords[0]:subwords[-1] + 1] = 1
            start += len(subwords)
        for i, sent in enumerate(sent_word):
            sent = [''.join(x) for x in sent]
            entity_index = traverse_entity(sent, i%2)
            _sent_entity[i, entity_index] = 1
        _cls_index = np.where(_utt_word == np.array(tokenizer.cls_token))[0]

        utt_word.append(_utt_word)
        bert_input.append(_bert_input)
        word_subword.append(_word_subword)
        cls_index.append(_cls_index)
        sent_entity.append(_sent_entity)
    config.logger.info("-----Constructing Graph Edge-----")
    edge_matrix = get_edge_matrix(all_edge, utt_word)

    return bert_input, word_subword, sent_entity, cls_index, edge_matrix, consist_label


def load_data_bert(config):
    with open('./data/{}/{}_train.json'.format(config.dataset, config.dataset), 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('./data/{}/{}_dev.json'.format(config.dataset, config.dataset), 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    with open('./data/{}/{}_test.json'.format(config.dataset, config.dataset), 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache/", use_fast=True, local_files_only=True)
    config.global_entities = load_entity_vocab('./data/citod/entities.json')

    table = pt.PrettyTable([config.dataset, 'dialogues']) #column
    table.add_row(['train', len(train_data)])
    table.add_row(['dev', len(dev_data)])
    table.add_row(['test', len(test_data)])
    config.logger.info("\n{}".format(table))


    train_dataset = MyDataset(*process_bert(train_data, tokenizer, config))
    dev_dataset = MyDataset(*process_bert(dev_data, tokenizer, config))
    test_dataset = MyDataset(*process_bert(test_data, tokenizer, config))
    return (train_dataset, dev_dataset, test_dataset)


def load_entity_vocab(path):
    with open(path) as f:
        data = json.load(f)

    global_entities = []
    for key in data.keys():
        global_entities.append(key)
        global_entities.extend([str(x).lower().replace(" ", "_") for x in data[key]])
    return sorted(list(set(global_entities) - {'-'}))