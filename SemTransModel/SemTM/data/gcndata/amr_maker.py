# -*- coding:utf-8 -*-
import json
import amrlib
import penman
from tqdm import tqdm

import spacy
import re

file_path = ['citod_train.json', 'citod_dev.json', 'citod_test.json']
output_path = 'citod_amr2adj.json'


stog = amrlib.load_stog_model()
spacy_nlp = spacy.load('../en_core_web_sm-2.3.0/en_core_web_sm/en_core_web_sm-2.3.0')
spacy_nlp.tokenizer = spacy_nlp.tokenizer.tokens_from_list # pretokenized data
def amr_parse(sentences, stog=stog):
    #sent graph转为penman node, penman.instances(显示lemma_后的amr树节点的节点标识和节点lemma_token[可能当前只是假的中间节点不存在token属性]), penman.edges(显示amr树的边, src->tgt均为节点标识), 于是algin amr节点的方法就是手动对原始sent使用spacy en模型的nlp(默认包含全部理器/也即含lemma,通过for token in Doc: token.lemm_访问)或nltk的wordnet模型的WordNetLemmatizer().lemmatize(word)得到lemma后的单词string
    dia_word = [tok for sent in sentences for tok in sent.split() + ['.']] # 先用标点占位, 后边替换成[CLS]
    dia_sent = [sent + ' .' for sent in sentences] # 先用标点占位, 后边替换成[CLS]
    graphs = stog.parse_sents(dia_sent) #list of str, 自己下载的预训练模型中,内部调用的是transformer的bart分词? model_stog -> model_parse_xfm_bart_base-v0_1_0/
    # envs/SGDCI/lib/python3.8/site-packages$ ll amrlib/data/,, 缺点传list of sent返回的是list of graph, 传入单个sent str,传入的则是list of graph
    # 实测似乎stog用什么bart分词不重要,毕竟amr图中本来就不是原本的单词,而是词干还原后的token

    # get sent amr graph list, and sent string list
    amr_graph_lines = []
    for line in graphs:
        amr_graph_lines.extend(line.split('\n')) # print('\n'.join(amr_graph_lines))
    amr, sent_graph, sent ='',  [], [] #amr store amr graph line temp # sent_graph is list of sent amr string, # sent store the original sents, #sent is list of sent string
    for line in amr_graph_lines:
        if line[:7] == '# ::snt':     # used to get the original sents
            if amr != '':
                sent_graph.append(amr)
            amr = ''
            # sent.append(line[8:]) #sent 其实没用到
        elif line[0] != '#':            # get amr sentences
            amr = amr + line
    sent_graph.append(amr)#artile is list of sent amr string


    # construct all the words as AMR_Node
    sent_penman_node = [penman.decode(_sent_graph) for _sent_graph in sent_graph]

    # nltk.download('wordnet', '../nldk_data')
    # nltk.download('omw-1.4')
    # from nltk.stem import WordNetLemmatizer
    # import nltk
    # nltk.data.path.append("../nltk_data")
    # lemmatizer = WordNetLemmatizer()
    # spacy_nlp = spacy.load('../en_core_web_sm-2.3.0/en_core_web_sm/en_core_web_sm-2.3.0')
    # spacy_nlp.tokenizer = spacy_nlp.tokenizer.tokens_from_list # pretokenized data

    amrnode_wordid = dict()
    src_node, tgt_node = [], []
    for penman_node in sent_penman_node:        # for every sentence in the article
        # sent_spacyed = spacy_nlp(" ".join(dia_sent))
        sent_spacyed = spacy_nlp(dia_word)
        word_lemmaed = [word.lemma_ for word in sent_spacyed]
        assert len(word_lemmaed) == len(dia_word) # 假定spacy不产生多余的分词行为, 以便将word_lemmaed和dia_word一一对应, (话说本来nltk的lemmatizer.lemmatize(word)就是逐个单词进行的lemma,不会产生分词行为, 只是说用的amr bart模型内部的amr 节点的lemma值,似乎用的spacy?不不不懂?)

        # 目前已经有原始word: dia_word, lemma_后的word:word_lemmaed, 现在将word_lemmaed和amr penman node(一个sent的全部node)的instances属性基于字符规则匹配
        for node in penman_node.instances(): #[Instance(source='h', role=':instance', target='hotel'),,,]
            if node.target == None: # instance.target, 也即amr penman图节点中的lemma后的token, 也即待匹配的token
                continue
            amr_token = re.sub(r'[0-9-]', '', node.target) #比如node.target字符'amr-unknow'
            amrnode_wordid[node.source] = word_lemmaed.index(amr_token) if amr_token in word_lemmaed else -1 # 匹配规则? 直接判断lemma后的的字符是否"全等"

        for edges in penman_node.edges(): #[Edge(source='h', role=':quant', target='a'),,,]
            if amrnode_wordid[edges.source] == -1 or amrnode_wordid[edges.target] == -1:
                continue
            src_node.append(amrnode_wordid[edges.source]) #amrnode_wordid[edges.source/target]可能是-1, 表明为未匹配到源word字符的假amr节点
            tgt_node.append(amrnode_wordid[edges.target])

    head_set, tail_set = [], []
    dia_len = len(dia_word)
    for i in range(len(src_node)):
        if src_node[i] < dia_len and tgt_node[i] < dia_len:
            head_set.extend([src_node[i], tgt_node[i]]) # 双向连接?
            tail_set.extend([tgt_node[i], src_node[i]])
    dia_word = [tok for sent in sentences for tok in sent.split() + ['CLS']]
    return [head_set, tail_set], dia_len, dia_word


if __name__ == '__main__':
    Data = dict()
    for path in file_path:
        with open(path) as f:
            raw_data = json.load(f)

        for dialog_item in tqdm(raw_data):
            sentences = [speak_turn["utterance"] for speak_turn in dialog_item["dialogue"]]
            amr_adj, dia_len, dia_word = amr_parse(sentences)
            Data[dialog_item['id']] = dict()
            Data[dialog_item['id']]['amr_adj'] = amr_adj
            Data[dialog_item['id']]['dia_word'] = dia_word
            Data[dialog_item['id']]['dia_len'] = dia_len

    with open(output_path, 'w', encoding='utf-8') as f: # In JSON (JavaScript Object Notation), keys can only be strings. 所以int id 或变成str id
        json.dump(Data, f, indent=2, ensure_ascii=False) #json list(with [] outter). not json line，or one json dict


