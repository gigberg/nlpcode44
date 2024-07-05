import argparse
import json
import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import transformers
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader

import config # 本地模块,主要用于封装args.config=config.json中的配置文件,一并放到args对象中:
import data_loader
import utils
from model import Model


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]

        self.optimizer = transformers.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,  num_warmup_steps=config.warm_factor * updates_total, num_training_steps=updates_total)

    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        pred_result = []
        label_result = []

        for i, data_batch in enumerate(data_loader):
            data_batch = [data.cuda() for data in data_batch[:-1]] # 除去entity_text

            bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

            #sent_length由于lstm pack, outputs # [8, 7, 7, 3]
            outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)

            grid_mask2d = grid_mask2d.clone() #无inplace操作,可以不clone, [8, 7, 7])
            loss = self.criterion(outputs[grid_mask2d], grid_labels[grid_mask2d]) # 多分类0-n
            # [8, 7, 7, 3] -> 392,3, [8, 7, 7] -> 392 #交叉熵的输入个时是N,C,*,*,,,要么flatten/view,要么permute
            #类似mask_seleted返回faltten结果(值在maskedso涉及的维度上全局flatten,在其它维度上对该操作独立重复), 只不过由于[8, 7, 7, 3]和[8, 7, 7]总维数不匹配,自动广播导致重复3次[392, 3]
            #因为交叉熵是逐位置操作,和是flatten还是shape无关(另外reduce默认=mean,也即batch内全部sent的全部word标量值位置上求均值), 另外交叉熵可以传label id,也可传same shape one hot

            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clip_grad_norm)
            # self.optimizer.step()
            # self.optimizer.zero_grad()

            loss_list.append(loss.cpu().item())

            outputs = torch.argmax(outputs, -1) # [8, 7, 7, 3] -> [8, 7, 7]
            grid_labels = grid_labels[grid_mask2d].contiguous().view(-1) # [8, 7, 7] -> 392
            outputs = outputs[grid_mask2d].contiguous().view(-1) # [8, 7, 7] -> [392],没再进一步必要view(-1)

            label_result.append(grid_labels.cpu()) # [[392],,,,]
            pred_result.append(outputs.cpu())
            # 由于学习率的warmup是逐batch的(updates_total)
            # self.scheduler.step() #学习率调度, 可以每个batch改变一次学习率,也可以每个epoch改变一次

        label_result = torch.cat(label_result) # [[392],,,,], how many batches
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(), #多分类0-n
                                                      pred_result.numpy(),
                                                      average="macro")

        table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
        table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] +
                      ["{:3.4f}".format(x) for x in [f1, p, r]])
        logger.info("\n{}".format(table))
        return f1

    def eval(self, epoch, data_loader, is_test=False):
        self.model.eval()

        pred_result = []
        label_result = []

        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0
        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                entity_text = data_batch[-1] # 从index构建单词
                data_batch = [data.cuda() for data in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1) #[8, 7, 7, 3] -> [8, 7, 7]
                ent_c, ent_p, ent_r, _ = utils.decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy()) # _entity_text=[index_list_text + "-#-" + ent_typeid_text,,,,],,length=sent_length

                total_ent_r += ent_r # ent_c, ent_p, ent_r = 0,3,1, 似乎是样本数统计
                total_ent_p += ent_p # 真假真负例真值表的样本数, c是 左上角, p是左侧p的分母, r是上册r的分母
                total_ent_c += ent_c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1) # grid_labels,(THW和NNW)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r) # entity type 传入真假真负例真值表的样本数, c是 左上角, p是左侧p的分母, r是上册r的分母

        title = "EVAL" if not is_test else "TEST"
        logger.info('{} Label F1 {}'.format(title, f1_score(label_result.numpy(),
                                                            pred_result.numpy(),
                                                            average=None)))

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])

        logger.info("\n{}".format(table))
        return e_f1

    def predict(self, epoch, data_loader, data):
        self.model.eval() #和eval的区别是ent_index_list转为句子中相应的text,以便人类可读,同时用于论文编写的样本错误分析

        pred_result = []
        label_result = []

        result = []

        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0

        i = 0 #i用来截取original sentences, 以便将ent index_id_list 转为原句的mentions texts,, 注意由于dataloader drop_last,以及shuffle存在, train_data上可能没法对应(original sample和dataloader_sample),好在这里是test data上
        with torch.no_grad():
            for data_batch in data_loader: #直接用zip(data_loader, data)?
                sentence_batch = data[i:i+config.batch_size] #original data
                entity_text = data_batch[-1] # [{'0-1-#-2', '3-4-5-#-2'},,,]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, decode_entities = utils.decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy())

                for ent_list, sentence in zip(decode_entities, sentence_batch):
                    sentence = sentence["sentence"] # ent_list, sentence一一对应
                    instance = {"sentence": sentence, "entity": []}
                    for ent in ent_list: # ent[0]是index list, ent[1]是ent_type_id
                        instance["entity"].append({"text": [sentence[x] for x in ent[0]],
                                                   "type": config.vocab.id_to_label(ent[1])})
                    result.append(instance) #一个instance为一个sample, 含有多个ent,每个ent是由 mentions_texts和type_text构成

                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())
                i += config.batch_size

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)

        title = "TEST"
        logger.info('{} Label F1 {}'.format("TEST", f1_score(label_result.numpy(),
                                                            pred_result.numpy(),
                                                            average=None)))

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])

        logger.info("\n{}".format(table))

        with open(config.predict_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return e_f1

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/example.json') #./config/conll03.json
    parser.add_argument('--save_path', type=str, default='./model.pt')
    parser.add_argument('--predict_path', type=str, default='./output.json')
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--type_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--conv_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--ffnn_hid_size', type=int)
    parser.add_argument('--biaffine_size', type=int)

    parser.add_argument('--dilation', type=str, help="e.g. 1,2,3")

    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--conv_dropout', type=float)
    parser.add_argument('--out_dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)

    parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")

    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    config = config.Config(args)

    logger = utils.get_logger(config.dataset)
    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    # random.seed(config.seed) # 以下下都是为了模型结果可复现
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = False # CNN, False when debugging less overhead
    torch.backends.cudnn.deterministic = True # CNN, True for reproducible in debugging or benchmarking

    logger.info("Loading Data")
    datasets, ori_data = data_loader.load_data_bert(config)

    train_loader, dev_loader, test_loader = (
        DataLoader(dataset=dataset,
                   batch_size=config.batch_size,
                   collate_fn=data_loader.collate_fn,
                   shuffle= i == 0,
                   num_workers=4,
                   drop_last= i == 0)
        for i, dataset in enumerate(datasets)
    )

    updates_total = len(datasets[0]) // config.batch_size * config.epochs

    logger.info("Building Model")
    model = Model(config)

    model = model.cuda()

    trainer = Trainer(model)

    best_f1 = 0
    best_test_f1 = 0
    for i in range(config.epochs):
        logger.info("Epoch: {}".format(i))
        trainer.train(i, train_loader)
        f1 = trainer.eval(i, dev_loader)
        test_f1 = trainer.eval(i, test_loader, is_test=True)
        if f1 > best_f1:
            best_f1 = f1
            best_test_f1 = test_f1
            trainer.save(config.save_path)
    logger.info("Best DEV F1: {:3.4f}".format(best_f1))
    logger.info("Best TEST F1: {:3.4f}".format(best_test_f1))
    # trainer.load(config.save_path)
    trainer.predict("Final", test_loader, ori_data[-1]) #return e_f1
