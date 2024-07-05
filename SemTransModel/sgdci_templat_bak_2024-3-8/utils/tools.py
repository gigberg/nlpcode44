import json
import os
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class DatasetTool(object):
    @staticmethod
    def get(args, shuffle=True):
        """
        Get the train, dev, test data in a inner format of infos, last_responses, consistencys.
        ** infos means kb+history
        :param args:
        :return: train, dev, test data
        """
        train_paths = [os.path.join(args.dir.dataset, train_path) for train_path in args.dataset.train.split(' ')]
        dev_paths = [os.path.join(args.dir.dataset, dev_path) for dev_path in args.dataset.dev.split(' ')]
        test_paths = [os.path.join(args.dir.dataset, test_path) for test_path in args.dataset.test.split(' ')]
        train, dev, test = [], [], []
        [train.extend(DatasetTool.load_data(train_path)) for train_path in train_paths]
        [dev.extend(DatasetTool.load_data(dev_path)) for dev_path in dev_paths]
        [test.extend(DatasetTool.load_data(test_path)) for test_path in test_paths]
        if shuffle:
            np.random.shuffle(train)
            np.random.shuffle(dev)
        entities = DatasetTool.load_entity(args)
        return train, dev, test, entities
    @staticmethod
    def load_data(data_path):
        """
        Load the data from data path.
        :param data_path: the json file path
        :return: infos and consistency_tuples
        each <info> is the constructed format of dialogue.
        each <consistency_tuple> is a tuple of (qi,hi,kbi)
        """
        with open(data_path) as f:
            raw_data = json.load(f)
        domain = os.path.split(data_path)[-1].split("_")[0]
        data = []
        with open('data/amr2adj.json') as ff:
            adj_json = json.load(ff)
        with open('data/core2adj.json') as fff:
            core_json = json.load(fff)

        for dialogue_components_item in raw_data:
            data_item = dict()
            id = dialogue_components_item['id']
            sent_lenth = adj_json[0][data_path[5:]][str(id)]['lenth']
            constructed_info, intent_dict, last_response, consistency = DatasetTool.agg_dialogue_info(dialogue_components_item, domain, sent_lenth)

            data_item["constructed_info"] = constructed_info
            data_item["last_response"] = last_response
            data_item["consistency"] = consistency
            # data_item['adj'] = adj_json[0][data_path[5:]][str(id)]['adj']
            # data_item['core_adj'] = core_json[0][data_path[5:]][str(id)]['adj']
            data_item.update(intent_dict) # inplace update
            data.append(data_item)
        return data

    @staticmethod
    def agg_dialogue_info(dialogue_components_item, domain, sent_lenth):
        """
        Transfer a dialogue item from the data
        :param dialogue_components_item: a dialogue(id, dialogue, kb, (qi,hi,kbi)) from data file (json item)
        :param domain: the domain of the data file
        :return: constructed_info: the constructed info which concat the info and format as
        the PhD. Qin mentioned.
                consistency: (qi,hi,kbi)
        """
        dialogue = dialogue_components_item["dialogue"] # dialogue：turn，requested，slots；scenario：kb，qi，hi，kbi；HIPositon（空）

        sentences = []
        history_sentences = []
        last_response = ''
        intent_dict = dict()
        intent_dict['requested'] = []
        intent_dict['slots'] = []

        for speak_turn in dialogue:
            sentences.append(speak_turn["utterance"])
            if speak_turn['turn'] == 'driver':
                intent_dict['requested'].append(speak_turn["requested"])
                intent_dict['slots'].append(speak_turn["slots"])
        if len(sentences) % 2 == 0:
            history_sentences.extend(sentences[:-1])
            last_response = sentences[-1]
        else:
            history_sentences.extend(sentences)

        knowledge_base = dialogue_components_item["scenario"]["kb"]['items']
        kb_expanded = DatasetTool.expand_kb(knowledge_base, domain)

        consistency = [float(x) for x in
                       [dialogue_components_item["scenario"]["qi"], dialogue_components_item["scenario"]["hi"],
                        dialogue_components_item["scenario"]["kbi"]]]


        # head_set, tail_set, sent_lenth = DatasetTool.amr_parse(history_sentences, last_response)
        constructed_info = DatasetTool.construct_info(kb_expanded, history_sentences, sent_lenth) # info包含kb和h二合一

        return constructed_info, intent_dict, last_response, consistency

    @staticmethod
    def expand_kb(knowledge_base, domain):
        """
        Expand the kb into (subject, relation, object) representation.
        :param knowledge_base: kb a list of dict.
        :param domain: the domain of the data
        :return: a list of list each item is a (subject, relation, object) representation.
        """
        expanded = []
        if domain == "navigate":
            for kb_row in knowledge_base:
                kb_row_list = []
                entity = kb_row['poi']
                for attribute_key in kb_row.keys():
                    kb_row_list.append((entity, attribute_key, kb_row[attribute_key]))
                expanded.append(kb_row_list)
        elif domain == "calendar":
            if knowledge_base == None:
                return []
            for kb_row in knowledge_base:
                kb_row_list = []
                entity = kb_row['event']
                for attribute_key in kb_row.keys():
                    if kb_row[attribute_key] == "-":
                        continue
                    kb_row_list.append((entity, attribute_key, kb_row[attribute_key]))
                expanded.append(kb_row_list)
        elif domain == "weather":
            for kb_row in knowledge_base:
                kb_row_list = []
                entity = kb_row['location']
                for attribute_key in kb_row.keys():
                    kb_row_list.append((entity, attribute_key, kb_row[attribute_key]))
                expanded.append(kb_row_list)
        else:
            print("Dataset is out of range(navigate, weather, calendar). Please recheck the path you have set.")
            assert False
        return expanded

    @staticmethod
    def construct_info(kb_expanded, history_sentences, dialogue_len):
        """
        Concatenate the kb_expanded and history_sentences
        :param kb_expanded: the (subject, relation, object) representation expanded kb.
        :param history_sentences: history sentences.
        :return: the concatenated string.
        """
        construct_info_kb = []

        for row in kb_expanded: # 单个对话的list[tuple]
            kb_temp = '[SOK] ' # kb_temp单个tuple
            kb_temp += ' '.join([triple[1] + " " + triple[2] for triple in row])
            # 三元变为二元？减少token数量，毕竟weather的token基本700+token
            kb_temp = kb_temp.replace('  ',' ')
            # if len(kb_temp.split(' ')) > 500 - dialogue_len: # 没啥用，kb又没有和句子公用一个bert
            #     break
            kb_temp += ' [EOK]'

            construct_info_kb.append(kb_temp)


        construct_info_hist = ''
        for i, sentence in enumerate(history_sentences):
            if i % 2 == 0:
                construct_info_hist += "[USR] " + sentence + ' '
            else:
                construct_info_hist += "[SYS] " + sentence + ' '

        # construct_info += 'The next generated [SYS] response is [MASK1]  with knowleadge base [SOK] , [MASK2] with history [EOK] and [MASK3] with last question [USR]. '

        return (construct_info_kb, construct_info_hist) # 二合一



    @staticmethod
    def load_entity(args):
        entities = []
        for entity_path in args.dataset.entity.split(' '):
            with open(os.path.join(args.dir.dataset, entity_path)) as f:
                global_entity = json.load(f)
            entities.extend(DatasetTool.generate_entities(global_entity))
        return entities

    @staticmethod
    def generate_entities(global_entity):
        words = []
        for key in global_entity.keys():
            words.extend([str(x).lower().replace(" ", "_") for x in global_entity[key]])
            if '_' in key:
                words.append(key)
        return sorted(list(set(words)))



class EvaluateTool(object):

    @staticmethod
    def evaluate(pred_qkh, pred_inte, dataset, args):
        # dataset 为整个数据集, 而非单个batch
        pred_qi = [pred_i[0] for pred_i in pred_qkh]
        pred_hi = [pred_i[1] for pred_i in pred_qkh]
        pred_kbi = [pred_i[2] for pred_i in pred_qkh]

        gold_qi = [gold_i['consistency'][0] for gold_i in dataset]
        gold_hi = [gold_i['consistency'][1] for gold_i in dataset]
        gold_kbi = [gold_i['consistency'][2] for gold_i in dataset]
        tmp_summary = {}

        if not os.path.exists(args.dir.output):
            os.makedirs(args.dir.output)
        tmp_summary["precision_qi"], tmp_summary["precision_hi"], tmp_summary["precision_kbi"] = precision_score(y_pred=pred_qi, y_true=gold_qi), precision_score(y_pred=pred_hi, y_true=gold_hi), precision_score(y_pred=pred_kbi, y_true=gold_kbi)
        tmp_summary["recall_qi"], tmp_summary["recall_hi"], tmp_summary["recall_kbi"] = recall_score(y_pred=pred_qi,y_true=gold_qi), recall_score(y_pred=pred_hi, y_true=gold_hi), recall_score(y_pred=pred_kbi, y_true=gold_kbi)
        tmp_summary["f1_qi"], tmp_summary["f1_hi"], tmp_summary["f1_kbi"] = f1_score(y_pred=pred_qi, y_true=gold_qi), f1_score(y_pred=pred_hi, y_true=gold_hi), f1_score(y_pred=pred_kbi, y_true=gold_kbi)
        tmp_summary["overall_acc_qkh"] = accuracy_score(pred_qkh, np.array([gold_qi, gold_hi, gold_kbi]).T)

        gold_inte = intent2onehot(dataset)
        tmp_summary["precision_inte"] = precision_score(y_pred=pred_inte, y_true=gold_inte, average='samples')
        tmp_summary["recall_inte"] = recall_score(y_pred=pred_inte,y_true=gold_inte, average='samples')
        tmp_summary["f1_inte"] = f1_score(y_pred=pred_inte, y_true=gold_inte, average='samples')
        tmp_summary["overall_acc_inte"] = accuracy_score(pred_inte, gold_inte)

        return tmp_summary


class Batch(object):
    @staticmethod
    def to_list(source, batch_size):
        """
        Change the list to list of lists, which each list contains a batch size number of items.
        :param source: list
        :param batch_size: batch size
        :return: list of lists
        """
        batch_list = []
        idx = 0
        while idx < len(source):
            next_idx = idx + batch_size
            if next_idx > len(source):
                next_idx = len(source)
            batch_list.append(source[idx: next_idx])
            idx = next_idx
        return batch_list


def apply_each(source, method):
    """
    In each is a iterator function which you can employ the method
    in every item in source.
    :param source: a list of items
    :param method: the method you want to employ to the items
    :return: the new items
    """
    return [method(x) for x in source]


def intent2onehot(batch):
    """
    Convert the dataset batch into flattened label, then into one-hot vector
    :param dataset_batch: list[dict***]
    :return: list[list]
    """
    labels = ["traffic", "address", "distance", "poi_type", "poi", "date", "low_temperature", "weather_attribute", "high_temperature", "location", "date", "time", "event", "agenda", "room", "party"]

    requested_flatten = []
    [requested_flatten.extend(item['requested']) for item in batch] # extend是inplace操作, requested和slots也和batch解耦, 整个batch的list[list]处理后flatten为一维
    # slots_flatten = []
    # [slots_flatten.extend(item['slots']) for item in batch]

    intent_onehot = np.zeros((len(requested_flatten), len(labels)))

    for row, item in enumerate(requested_flatten):
            for col, aspect in enumerate(labels):
                if aspect in item and item[aspect] == True:
                    intent_onehot[row][col] = 1
    return intent_onehot