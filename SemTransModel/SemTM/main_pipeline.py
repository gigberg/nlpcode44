import argparse
import time
import numpy as np
import prettytable as pt
from tqdm import tqdm
# import os
# os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score
from torch.utils.data import DataLoader

import config
import dataload
import utils
from models.SentTrans_pipeline import SentTransModel


class Trainer(object):
    def __init__(self, model, config):
        self.model = model.cuda()

        self.consist_criterion = nn.CrossEntropyLoss()

        bert_params = set(self.model.embedding.bert.parameters())
        consist_params = set(self.model.consist_classifier.parameters())
        other_params = list(set(self.model.parameters()) - bert_params - consist_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.embedding.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.embedding.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': self.model.consist_classifier.parameters(),
             'lr': config.consist_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': other_params,
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]

        self.optimizer = AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        # self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=config.warm_factor * config.updates_total, num_training_steps=config.updates_total) # 有warmup增加过程的话, 可以使用大学习率1e-3而非1e-5

    def train(self, epoch, data_loader):
        self.model.train()
        loss_array = []
        pred_result = []
        label_result = []
        import builtins
        builtins.train_epoch = epoch
        for i, data_batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Train Step Starts:"):
            builtins.train_step = i
            data_batch = [data.cuda() for data in data_batch]
            consist_label = data_batch[-1]

            consist_out, spk_loss, flow_loss, spk_output, spk_label = self.model(*data_batch)

            # consist loss
            # consist_loss = self.consist_criterion(consist_out, consist_label)
            loss = spk_loss# + flow_loss

            loss.backward()

            # torch.cuda.empty_cache() # 发现主要是cnn的cache占用太多10~19gb，当然cnn的allocated在不同时间步也能累积？9gb
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_array.append(loss.item())
            consist_pred = torch.argmax(spk_output, -1) # mask一下？
            pred_result.append(consist_pred.cpu())
            label_result.append(spk_label.cpu())
            # self.scheduler.step() # num_warmup_steps=config.warm_factor * updates_total

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)
        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(), #numpy for sklearn
                                                      pred_result.numpy(),
                                                      average='macro')

        loss = np.mean(loss_array)
        table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
        table.add_row(["Label", "{:.4f}".format(loss)] +
                      ["{:3.4f}".format(x) for x in [f1, p, r]])
        logger.info("\n{}".format(table))

        return f1


    def train_consist(self, epoch, data_loader):
        self.model.train()
        loss_array = []
        pred_result = []
        label_result = []
        import builtins
        builtins.train_epoch = epoch
        for i, data_batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Train Step Starts:"):
            builtins.train_step = i
            data_batch = [data.cuda() for data in data_batch]
            consist_label = data_batch[-1]

            consist_out, spk_loss, flow_loss, spk_output, spk_label = self.model(*data_batch)

            # consist loss
            consist_loss = self.consist_criterion(consist_out, consist_label)
            # loss = spk_loss + flow_loss + 3 * consist_loss

            # loss.backward()
            consist_loss.backward()

            # torch.cuda.empty_cache() # 发现主要是cnn的cache占用太多10~19gb，当然cnn的allocated在不同时间步也能累积？9gb
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_array.append(loss.item())
            consist_pred = torch.argmax(consist_out, -1) # mask一下？
            pred_result.append(consist_pred.cpu())
            label_result.append(consist_label.cpu())
            # self.scheduler.step() # num_warmup_steps=config.warm_factor * updates_total

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)
        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(), #numpy for sklearn
                                                      pred_result.numpy(),
                                                      average='macro')

        loss = np.mean(loss_array)
        table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
        table.add_row(["Label", "{:.4f}".format(loss)] +
                      ["{:3.4f}".format(x) for x in [f1, p, r]])
        logger.info("\n{}".format(table))

        return f1


    def eval(self, epoch, data_loader, is_test=False):
        self.model.eval()

        loss_array = []
        pred_result = []
        label_result = []

        with torch.no_grad():
            for _, data_batch in enumerate(data_loader):
                data_batch = [data.cuda() for data in data_batch]
                consist_label = data_batch[-1]

                consist_out, spk_loss, flow_loss, spk_output, spk_label = self.model(*data_batch)

                # #consist loss
                # consist_loss = self.consist_criterion(consist_out, consist_label)
                # loss = spk_loss + flow_loss + consist_loss
                # loss_array.append(consist_loss.item())

                if hasattr(config, 'flow_pretrain'):
                    consist_pred = torch.argmax(spk_output, -1) # mask一下？
                    pred_result.append(consist_pred.cpu())
                    label_result.append(spk_label.cpu())
                consist_pred = torch.argmax(consist_out, -1) # mask一下？
                pred_result.append(consist_pred.cpu())
                label_result.append(consist_label.cpu())

                # consist_pred[consist_out.ge(0.5)] = 1

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)
        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(), #numpy for sklearn
                                                      pred_result.numpy(),
                                                      average='macro') # default

        title = "EVAL" if not is_test else "TEST" # average=macro,返回list用来筛选
        logger.info('{} Label F1 {}'.format(title, f1_score(label_result.numpy(),
                                                            pred_result.numpy(),
                                                            average=None))) # 不reduce， 查看0和1各自f1情况

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        logger.info("\n{}".format(table))

        return f1

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='example.cfg')

    # 数据集和模块选择
    parser.add_argument('--dataset', type=str, default="navigate")
    parser.add_argument('--bert_name', type=str, default="bert-base-uncased")
    parser.add_argument('--dia_enc', type=str, default="MultiUniAtten", help="MultiUniAtten, MultiSpanRNN, MultidilaCNN, MultiUniGCN")
    parser.add_argument('--sent_enc', type=str, default="SentIdentity", help='SentUniAtten, SentIdentity')
    parser.add_argument('--flow_type', type=str, default="FlowCellQE", help="FlowCellQA, FlowCellQQ, FlowCellQE, FlowRemove")
    parser.add_argument('--classifier', type=str, default="SoftmaxClassifier", help="SoftmaxClassifier, MLPClassifier, BiaffineClassifier")

    parser.add_argument('--save_path', type=str, default="./model.pt")

    parser.add_argument('--hidden_size', type=int, default=552) #lstm
    parser.add_argument('--bert_hid_size', type=int, default=768)
    parser.add_argument('--pad_token_idx', type=int, default=0)
    parser.add_argument('--cln_hid_size', type=int, default=552*2)
    parser.add_argument('--ffnn_hid_size', type=int, default=552*2)
    parser.add_argument('--head_count', type=int, default=12)
    parser.add_argument('--num_hidden_layers', type=int, default=2)

    # parser.add_argument('--lstm_hid_size', type=int, default=512) # 2
    parser.add_argument('--sent_hid_size', type=int, default=552//2) # 2, 普通除法会返回浮点数
    parser.add_argument('--flow_size', type=int, default= 100) # 2
    parser.add_argument('--use_bert_last_4_layers', action='store_true')

    parser.add_argument('--max_position_embeddings', type=int, default=2000)
    parser.add_argument('--position_emb_size', type=int, default=20)
    parser.add_argument('--type_vocab_size', type=int, default=4)
    parser.add_argument('--type_emb_size', type=int, default=20)

    parser.add_argument('--dilation', type=int, nargs='+', help="e.g. 1 2 3", default=[1,2,3]) # default=[1,2,3]不能传list,,不不同时写多个就是list(需要设置nargs='+',然后默认空格分隔)

    parser.add_argument('--attn_dropout', type=float)
    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--conv_dropout', type=float)
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12)

    # 训练相关
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--log', type=str, default='')
    # parser.add_argument('--retoken', action='store_true')
    parser.add_argument('--retoken', type=bool, default=0)

    # 优化器权重衰减
    parser.add_argument('--bert_learning_rate', type=float, default=1e-5)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--warm_factor', type=float, default=0.1)

    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0)

    # -----------------------------
    parser.add_argument('--consist_epochs', type=int, default=20)
    parser.add_argument('--consist_learning_rate', type=float, default=1e-3)

    # other
    parser.add_argument('--noise_rate', type=float, default=0.0)
    parser.add_argument('--flow_bias', action='store_true')

    # 除了--config参数外, 基本args和config一致

    args = parser.parse_args()
    config = config.Config(args)

    logger = utils.get_logger("./log/{}_{}_{}_{}_{}.txt".format(config.dataset, args.dia_enc, args.flow_type, config.log, time.strftime("%m-%d_%H-%M-%S")))
    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(config.device)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True

    if config.retoken: # if not to switch, the reason is tokenize word by word is a little slow
        logger.info("Loading data from tokenizerfast...")
        datasets = dataload.load_data_bert(config) # train_set, dev_set, test_set
        logger.info("Saving data to tokennizered pickle...")
        torch.save((datasets, config), f'./checkpoint/{config.dataset}_datasets.pkl')
    else:
        logger.info("Loading data from tokenizered pickle...")
        datasets, old_config = torch.load(f'./checkpoint/{config.dataset}_datasets.pkl')
        old_config.__dict__.update(config.__dict__)
        config = old_config


    if config.evaluate:
        dev_loader, test_loader = (
        DataLoader(dataset=dataset,
                   batch_size=config.batch_size,
                   collate_fn=dataload.collate_fn,
                   shuffle= index == 0,
                #    num_workers=4,
                   drop_last= True)
        for index, dataset in enumerate(datasets) if index != 0
    )
    else:
        train_loader, dev_loader, test_loader = (
        DataLoader(dataset=dataset,
                   batch_size=config.batch_size,
                   collate_fn=dataload.collate_fn,
                   shuffle= index == 0,
                #    num_workers=4,
                   drop_last= True)
        for index, dataset in enumerate(datasets)
    )

    model = SentTransModel(config)
    config.updates_total = len(datasets[0]) // config.batch_size * config.epochs

    logger.info("Building Model")
    # torch.cuda.memory._record_memory_history()
    if config.evaluate:
        config.updates_total = len(datasets[0]) // config.batch_size * config.consist_epochs

        trainer = Trainer(model, config)
        trainer.load(f"./checkpoint/model_{config.dataset}.pkl")
        dev_loss, dev_f1 = trainer.eval(dev_loader)
        logger.info("Dev Loss: {:.4f} F1: {:.4f}".format(dev_loss, dev_f1))
        test_loss, test_f1 = trainer.eval(test_loader)
        logger.info("Test Loss: {:.4f} F1: {:.4f}".format(test_loss, test_f1))
    else:
        # pretrain flow
        config.flow_pretrain = True

        trainer = Trainer(model, config)
        for epoch in range(config.epochs):
            trainer.train(epoch, train_loader)
        trainer.save(f"./checkpoint/model_pipeline_{config.dataset}.pkl")

        # consist model
        best_f1 = 0.0
        best_test_f1 = 0.0

        logger.info("Building Consist Model")
        config.updates_total = len(datasets[0]) // config.batch_size * config.consist_epochs
        trainer = Trainer(model, config)
        trainer.load(f"./checkpoint/model_pipeline_{config.dataset}.pkl")

        for epoch in range(config.consist_epochs):
            train_f1 = trainer.train(epoch, train_loader)
            dev_f1 = trainer.eval(epoch, dev_loader)
            test_f1 = trainer.eval(epoch, test_loader, is_test=True)
            if best_f1 < dev_f1:
                best_f1 = dev_f1
                best_test_f1 = test_f1
                trainer.save(config.save_path)

        logger.info("Best Dev F1: {:3.4f}".format(best_f1))
        logger.info("Best TEST F1: {:3.4f}".format(best_test_f1))
