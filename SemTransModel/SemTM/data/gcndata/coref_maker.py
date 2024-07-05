# -*- coding:utf-8 -*-
import json
from tqdm import tqdm

# Load your usual SpaCy model (one of SpaCy English models)
import spacy
spacy_nlp = spacy.load('../en_core_web_sm-2.3.0/en_core_web_sm/en_core_web_sm-2.3.0')
spacy_nlp.tokenizer = spacy_nlp.tokenizer.tokens_from_list # pretokenized data

# Add neural coref to SpaCy's pipe
# import neuralcoref
import sys
sys.path.insert(0, '../neuralcoref_master') # pip仓库的neuralcoref似乎会报错,必须用github的,所以手动添加到import搜索路径
import neuralcoref
neuralcoref.add_to_pipe(spacy_nlp)

# You're done. You can now use NeuralCoref as you usually manipulate a SpaCy document annotations.
# doc = spacy_nlp('i need to find out the date and time for my swimming_activity. i have two which one i have one for the_14th at 6pm and one for the_12th at 7pm')

# print(doc._.has_coref)
# print(doc._.coref_clusters)
# print(doc._.coref_clusters[1].main)
# print(doc._.coref_clusters)
# print(doc._.coref_clusters[1].mentions)
# print(doc._.coref_clusters[1].mentions[-1].start)
# print(doc._.coref_clusters[1].mentions[-1].end)

# print(doc._.coref_clusters[1].mentions[-1]._.coref_cluster.main)
file_path = ['citod_train.json', 'citod_dev.json', 'citod_test.json']

output_path = 'citod_coref2adj.json'

def core_parse(sentences, nlp=spacy_nlp):
    dia_word = [tok for sent in sentences for tok in sent.split() + ['.']] # 先用标点占位, 后边替换成[CLS]
    doc = spacy_nlp(dia_word) # spacy_nlp()支持doc str, 或者doc list, 返回, pipe()则支持list of doc或list of doc list, doc 可以是一个sent或者一个文档, 因为会自动调用Sentence detection莫快会每个token预测.is_sent_start属性值,
    # print(doc._.has_coref)
    # print(doc._.coref_clusters)
    # print(len(doc._.coref_clusters))
    head_set, tail_set = [], []

    dia_len = len(dia_word)
    span_headtail_list = []
    for i in range(len(doc._.coref_clusters)): # [neither route: [neither route, the fastest route]]
        index_tuple = []
        for j in range(len(doc._.coref_clusters[i].mentions)): #mentions: [neither route, the fastest route]
            if doc._.coref_clusters[i].mentions[j].start < dia_len and doc._.coref_clusters[i].mentions[j].end < dia_len:
                index_tuple.append((doc._.coref_clusters[i].mentions[j].start,doc._.coref_clusters[i].mentions[j].end)) # [(22, 24), (38, 41)]
        span_headtail_list.append(index_tuple) # [[(22, 24), (38, 41)], ]

    for _coref_index in range(len(span_headtail_list)):
        _coref_span = span_headtail_list[_coref_index]
        for _span_index in range(len(_coref_span)): #第i组mention, 的第j个mention tuple, 和第k个mention tuple, 两两建立头尾连接(n*n-1)*2条连接【可能因为共指是两两互相的】,, 有点离谱,相当于只关注头头连接,尾尾连接
            for _span_index_other in range(len(_coref_span)):
                if _span_index != _span_index_other:
                    head_set.extend([_coref_span[_span_index][0], _coref_span[_span_index][1]]) #头头连接,尾尾连接
                    tail_set.extend([_coref_span[_span_index_other][0], _coref_span[_span_index_other][1]]) # 另外和amr一样是双向的，只不过双向放到j/k循环次数中没放到循环体中
    dia_word = [tok for sent in sentences for tok in sent.split() + ['CLS']]
    return [head_set, tail_set], dia_len, dia_word


if __name__ == '__main__':
    Data = dict()
    for path in file_path:
        with open(path) as f:
            raw_data = json.load(f)

        for dialog_item in tqdm(raw_data):
            sentences = [speak_turn["utterance"] for speak_turn in dialog_item["dialogue"]]

            coref_adj, dia_len, dia_word = core_parse(sentences)

            Data[dialog_item['id']] = dict()
            Data[dialog_item['id']]['coref_adj'] = coref_adj
            Data[dialog_item['id']]['dia_word'] = dia_word
            Data[dialog_item['id']]['dia_len'] = dia_len

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(Data, f, indent=2, ensure_ascii=False)
