import os
import logging
import time
import numpy as np
import pickle


def get_logger(pathname):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - [line:%(lineno)d] - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')
    os.makedirs(os.path.dirname(pathname), exist_ok=True)
    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def save_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def get_edge_matrix(all_edge, utt_word, edge_types=4):
    edge_matrix= []
    for i, dia in enumerate(utt_word):
        dia_len = len(dia)

        _edge_matrix = np.zeros((edge_types, dia_len, dia_len))
        _all_edge = all_edge[i]

        # coref matrix
        for _coref_index in range(len(_all_edge['coref_edge'][0])):
            _edge_type = 0
            _src_index, _tgt_index = _all_edge['coref_edge'][0][_coref_index], _all_edge['coref_edge'][1][_coref_index]
            if _src_index + 1 < dia_len and _tgt_index + 1 < dia_len:
                _edge_matrix[_edge_type, _tgt_index, _src_index] = 1
        # amr matrix
        for _amr_index in range(len(_all_edge['amr_edge'][0])):
            _edge_type = 1
            _src_index, _tgt_index = _all_edge['amr_edge'][0][_amr_index], _all_edge['amr_edge'][1][_amr_index]
            if _src_index + 1 < dia_len and _tgt_index + 1 < dia_len:
                _edge_matrix[_edge_type, _tgt_index, _src_index] = 1
        # utt matrix
        for _cls_index in _all_edge['utt_edge']:
            _edge_type = 2
            for _cls_index_other in  _all_edge['utt_edge']:
                if _cls_index != _cls_index_other:
                    _edge_matrix[_edge_type, _cls_index, _cls_index_other] = 1
        # uni matrix
        for _word_index in range(len(dia)): # 逐行汇聚邻居信息的(含自身), 所以pre->word的单向信息流在下三角(word_tgt, pre_src)
            _edge_type = 3
            for _word_index_pre in range(_word_index+1):
                _edge_matrix[_edge_type, _word_index, _word_index_pre] = 1
        edge_matrix.append(_edge_matrix)
    return edge_matrix