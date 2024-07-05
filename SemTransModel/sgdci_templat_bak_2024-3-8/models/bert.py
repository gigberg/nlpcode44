import logging
from tqdm import tqdm
import pprint
import random
import fitlog


import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


from transformers import BertTokenizer, BertModel
import models.base
# from models.GCNLayer import GraphConvolution, HighWay

import utils.tools as tools
from utils.tools import Batch
from utils.tools import DatasetTool, EvaluateTool

class BERTTool(object):
    def init(args):
        BERTTool.bert = BertModel.from_pretrained(args.bert.location, return_dict=False)
        BERTTool.tokenizer = BertTokenizer.from_pretrained(args.bert.location)
        BERTTool.pad = BERTTool.tokenizer.pad_token
        BERTTool.sep = BERTTool.tokenizer.sep_token
        BERTTool.cls = BERTTool.tokenizer.cls_token
        BERTTool.pad_id = BERTTool.tokenizer.pad_token_id
        BERTTool.sep_id = BERTTool.tokenizer.sep_token_id
        BERTTool.cls_id = BERTTool.tokenizer.cls_token_id
        BERTTool.special_tokens = ["[SOK]", "[EOK]", "[SOR]", "[EOR]", "[USR]", "[SYS]", '[MASK1]', '[MASK2]', '[MASK3]']
        # SOK: start of knowledge base
        # EOK: end of knowledge base
        # SOR: start of row
        # EOR: end of row
        # USR: start of user turn
        # SYS: start of system turn


class Model(models.base.Model):
    def __init__(self, args, inputs):

        np.random.seed(args.train.seed)
        torch.manual_seed(args.train.seed)
        random.seed(args.train.seed)

        super().__init__(args, inputs)
        _, _, _, entities = inputs
        BERTTool.init(self.args)
        self.bert = BERTTool.bert
        self.tokenizer = BERTTool.tokenizer
        self.DatasetTool = DatasetTool
        self.EvaluateTool = EvaluateTool

        special_tokens_dict = {'additional_special_tokens': BERTTool.special_tokens+entities}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.bert.resize_token_embeddings(len(self.tokenizer))

        self.hidden_size = 768

        # 2024-03-03 15:52:41
        self.w_intent = nn.Linear(self.hidden_size, 5 + 5 + 6)

        self.w_hi = nn.Linear(768 * 2, 2) # 由于上一层输入是gcn和kb两个隐藏层横向拼接，所以输入维度为768*2

        self.w_qh = nn.Linear(768 * 2, 2)
        self.w_qi = nn.Linear(768 * 2, 2)
        self.w_kbi = nn.Linear(768 * 2, 2)
        # self.criterion = nn.BCELoss()



    def forward(self, batch):
        tokenized_hist, tokenized_kb, kb_index, sep_starts= self.get_tokenized(batch) # kb_index 记录batch中每条数据的知kb长度(kb第1维是长度184而非batch8), adj_4d为最后两维度是5个子图的左右端点
        utt_h, utt_cls = self.bert(**tokenized_hist.to(self.device)) # utt_h (8*62*768), utt_cls (8*768)

        sep_cls = torch.cat([utt_h[i][seps,:] for i, seps in enumerate(sep_starts)], dim=0) #整个batch的sep_cls(17*768)

        _, kb_cls = self.bert(**tokenized_kb.to(self.device)) # kb_cls[184*768],184是整个batch的kb放一起bert,未分成batch来bert

        # 解耦kb的token和bert过程,整个batch的kb拼接成n>batch组句子放到批量tokenizer中, 然后再拆分成n->batch组token,放到bert中. 直接拼接成batch组来tokenizer会导致kb过长在tokenizer时就被截断
        kb_batch, _ = self.kb_reshape(kb_index,kb_cls) # kb_batch是batch内所有kb词向量([batch_8, item_184, _hidden_768],第二维有padding), kb_mask是用来提取单个对话的掩码
        kb_batch_sum = torch.max(kb_batch, dim=1)[0] # 每个对话,kb句子的词向量压缩为的一个词袋模型(8*768), 相当将不同batch维的bert输出(h和kb),手动处理成同形状的tensor

        utt_cls = torch.cat((utt_cls, kb_batch_sum),dim=-1) # 8*1536

        out_intent = self.w_intent(sep_cls) # [17, 16]

        out_qi = self.w_qi(utt_cls) # 8*2
        out_hi = self.w_hi(utt_cls)
        out_kbi = self.w_kbi(utt_cls)

        loss = torch.Tensor([0]) # 添加此行应对self.eval的forward时候没有loss变量,但是却return loss导致的报错UnboundLocalError: local variable 'loss' referenced before assignment

        # --2024-03-04 12:33:46 requested的one-hot标签
        label_intent = tools.intent2onehot(batch)
        label_intent_tensor = torch.Tensor(label_intent).to(self.device) # float32
        # --

        if self.training:
            loss_consis =  F.cross_entropy(out_qi,
                                   torch.Tensor(tools.apply_each(batch, lambda x: x["consistency"][0])).long().to(
                                       self.device)) \
                   + F.cross_entropy(out_hi,
                                     torch.Tensor(tools.apply_each(batch, lambda x: x["consistency"][1])).long().to(
                                         self.device)) \
                   + F.cross_entropy(out_kbi,
                                     torch.Tensor(tools.apply_each(batch, lambda x: x["consistency"][2])).long().to(
                                         self.device))
            loss_intent = F.binary_cross_entropy_with_logits(out_intent, label_intent_tensor)

            loss = loss_consis + 0.4 * loss_intent

        out_qkh = []
        for qi, hi, kbi in zip(out_qi, out_hi, out_kbi):
            out_qkh.append([qi.argmax().data.tolist(), hi.argmax().data.tolist(), kbi.argmax().data.tolist()]) #.data可以从计算图中剪枝,移除梯度,使得本行为只涉及标量运算,不过这里也用不到?因为loss已经计算完了

        # sigmoid > 0.5则取该标签
        out_intent_ids = (torch.sigmoid(out_intent) > 0.5).int().tolist()

        return loss, out_qkh, out_intent_ids

    def run_train(self, train, dev, test):
        self.set_optimizer()
        iteration = 0
        best = {}
        for epoch in range(self.args.train.epoch):
            self.train()
            logging.info("Starting training epoch {}".format(epoch))
            summary = self.get_summary(epoch, iteration)
            loss, iter = self.run_batches(train) #只有这里调用了base中的方法run_batches
            fitlog.add_loss({"train_loss": loss}, step=epoch) # 叠加intent任务的总体loss
            iteration += iter
            summary.update({"loss": loss}) # 这个loss只在终端标准输出中，没有通过add_metric()保存到fitlog中
            # ds = {"train": train, "dev": dev, "test": test}
            ds = {"test": test} #--2024-03-05 10:25:10值跑测试集上的f1就行>,免去验证集选择最优模型,并且免去最终的最优模型是的eval跑分
            if not self.args.train.not_eval: # 并没有设置训练时不评估模型f1值
                for set_name, dataset in ds.items():
                    tmp_summary, _, _ = self.run_test(dataset) #没有调用run_batches(), 而是手动通过Class Batch实现批量验证
                    # self.record(pred, pred_intent, dataset, set_name, self.args) # pass
                    summary.update({"eval_{}_{}".format(set_name, k): v for k, v in tmp_summary.items()})
                    fitlog.add_metric({"eval_{}_{}".format(set_name, k): v for k, v in tmp_summary.items()}, step=epoch)
            best = self.update_best(best, summary, epoch) #基于dev的overall_acc, 改成基于test的f1_hI

            best_f1, summary_f1 = {}, {}
            for key in ["epoch", "eval_test_f1_qi", "eval_test_f1_hi", "eval_test_f1_kbi", "eval_test_f1_inte", "eval_test_overall_acc_qkh", 'eval_test_overall_acc_inte']:
                best_f1[key] = best.get(key, 0)
                summary_f1[key] = summary.get(key, 0)
            fitlog.add_best_metric(best_f1) # 用此方法添加的值，会被显示在表格中的 metric 列及其子列中。
            fitlog.add_to_line(best_f1) # 测试用途
            logging.info(f'epoch:{epoch}\nbest_fi:\n {pprint.pformat(best_f1)}' + f'\nsummary_fi:\n {pprint.pformat(summary_f1)}') # 所以每次打印两个日志
            logging.info(f'best_epoch: {best["epoch"]}' + f'\nbest_test_f1_hi: {best["eval_test_f1_hi"]}' + f'\ncurrent_test_f1_hi: {summary["eval_test_f1_hi"]}')


    def run_batches(self, dataset):
        all_loss = 0
        all_size = 0
        iteration = 0
        for batch in tqdm(Batch.to_list(dataset, self.args.train.batch)[0 : self.get_max_train_batch(dataset)]):
            loss, _, _ = self.forward(batch)
            self.zero_grad()
            loss.backward()
            self.optimizer.step()
            all_loss += loss.item()
            iteration += 1
            # if self.args.train.iter_save is not None: # 没设置,默认为None
            #     if iteration % self.args.train.iter_save == 0:
            #         if self.args.train.max_save > 0:
            #             self.save_model('epoch={epoch},iter={iter}'.format(epoch = epoch, iter = iteration))
            #             self.clear_saves()
            all_size += len(batch)
        return all_loss / all_size, iteration


    def run_test(self, dataset):
        self.eval()
        all_out_qkh, all_out_intent = [], []
        for batch in tqdm(Batch.to_list(dataset, self.args.train.batch)[0 : self.get_max_train_batch(dataset)]):
            loss, out_qkh, out_intent_ids = self.forward(batch)
            all_out_qkh += self.pred_flatten(out_qkh)
            all_out_intent += self.pred_flatten(out_intent_ids)

        tmp_summary = self.EvaluateTool.evaluate(all_out_qkh, all_out_intent, dataset[0 : self.get_max_train_batch(dataset) * self.args.train.batch], self.args) # 评估时传入dataset 而非单个batch, 所以需要乘以batch数
        return tmp_summary, all_out_qkh, all_out_intent


    # def run_eval(self, train, dev, test):
    #     logging.info("Starting evaluation")
    #     self.load("saved/best_model.pkl")
    #     self.eval()
    #     summary = {}
    #     ds = {"test": test}
    #     for set_name, dataset in ds.items():
    #         tmp_summary, pred, pred_intent = self.run_test(dataset)
    #         self.record(pred, dataset, set_name, self.args)
    #         summary.update({"eval_{}_{}".format(set_name, k): v for k, v in tmp_summary.items()})
    #     logging.info(pprint.pformat(summary))

    def sent2word(self, constructed_info, last_response):
        batch_size = len(constructed_info)
        sen1, sen2 = [], []
        sep_starts = []
        for item in range(batch_size):
            sentence1 =  constructed_info[item].split(' ')
            # sentence1.remove('')
            word_list = ['[CLS]'] + sentence1 + ['[SEP]']

            sentence2 = last_response[item].split(' ')
            word_list += sentence2 + ['[SEP]']
            # word_list.remove('')

            sep_tmp = np.where(np.array(word_list) == '[USR]')
            sep_starts.append(sep_tmp[0].tolist())

            sen1.append(sentence1)
            sen2.append(sentence2)

        return  sen1, sen2, sep_starts


    def get_tokenized(self, batch):
        construced_infos = [item['constructed_info'][1] for item in batch]
        kb_list = [item['constructed_info'][0] for item in batch]
        knowbase = [[],[]]
        for kb in kb_list:
            knowbase[0].extend(kb)
            # 索引
            knowbase[1].append(len(kb))
        for i in range(len(knowbase[1])):
            if i == 0:
                pass
            else:
                knowbase[1][i] += knowbase[1][i-1] # cumsum，可能是应对weather的kb超过512，所以对kb精细处理
        kb_index = knowbase[1]
        last_responses = [item['last_response'] for item in batch]
        sen1, sen2, sep_starts= self.sent2word(construced_infos, last_responses) # dialogue word pad后的长度(此时还没bert分词为token只用于图节点对齐，h_list，a_list, h字数list,分隔token的id_list
        # adj_4d = self.get_adj(amr_adj_list,core_adj_list,utt_adj_list, w_max, kb_lenth, w_lenth_list) # 它原来可能想吧图的节点扩展到kb中的token上,后来发现没必要

        tokenized_hist = self.tokenizer(sen1, sen2, truncation='only_first', padding=True, return_tensors='pt', max_length = self.tokenizer.max_model_input_sizes['bert-base-uncased'], return_token_type_ids=True)
        # get sep_starts id for intend classifier
        tokenized_kb = self.tokenizer(knowbase[0], truncation='only_first', padding=True, return_tensors='pt',
                                   max_length=self.tokenizer.max_model_input_sizes['bert-base-uncased'],
                                   return_token_type_ids=True)


        # tokenized_hist BatchEncoding类型,UserDict的子类; tokenized_hist.data,也即UserDict.data,为普通dict of tensor类型, 只有前者实现了to(device)方法[内部是逐个将value存入cuda中],普通dict则没有to(device)方法
        return tokenized_hist, tokenized_kb, kb_index, sep_starts


    def kb_reshape(self, kb_index, kb_cls):
        max_len = np.max(kb_index)
        m = torch.zeros(len(kb_index),max_len, kb_cls.size(-1)).to(self.device) # len(kb_index)=batch_8，max_len=184?所有batch的数据库项数,搭配mask掩码使用，kb_cls[batch_8*hidden_768]
        mask = torch.zeros(len(kb_index),max_len).to(self.device)
        for i in range(len(kb_index)): # batch_8
            if i == 0:
                kb_rep = kb_cls[:kb_index[i],:] # 截取单条数据的kb词向量
            else:
                kb_rep = kb_cls[kb_index[i-1]:kb_index[i],:]

            j = len(kb_rep)
            mask[i][:j] = 1
            m[i, :j, :] = kb_rep
            # for j in range(len(kb_rep)):
            #     mask[i][j] = 1
            #     for z in range(len(kb_rep[j])):
            #         m[i][j][z] = kb_rep[j][z]
        return m, mask # m是batch内所有kb词向量, mask是用来提取单个对话的掩码


    def start(self, inputs):
        train, dev, test, _ = inputs
        if self.args.model.resume is not None:
            self.load(self.args.model.resume)
        if not self.args.model.test:
            self.run_train(train, dev, test)
        # self.run_eval(train, dev, test) # 一次行为,用最后一个epoch的模型,不不,用最优模型去在train/dev/test三个数据集上跑分一次
