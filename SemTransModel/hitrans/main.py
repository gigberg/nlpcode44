import argparse
import time
import utils
import dataset
import torch
import torch.autograd
import torch.nn as nn
import torch.optim as optim
from models.HiTrans import HiTrans
from models.Loss import MultiTaskLoss
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import random
import numpy as np


class Trainer(object):
    def __init__(self, model):
        self.model = model.cuda() # model to device (means the model's parameters)

        self.emo_criterion = nn.CrossEntropyLoss()
        self.spk_criterion = nn.CrossEntropyLoss()

        self.multi_loss = MultiTaskLoss(2).cuda()
        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params) #用set是为了便于做集合加减法
        no_decay = ['bias', 'LayerNorm.weight'] # 遵循hugginface的和bert源码的bert预训练微调设置, 偏置和归一化的权重不衰减
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)], #bert中需要衰减的参数的名字
             'lr': args.bert_lr,
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)], # bert中不用衰减的, 用named_parameters来过滤不需要却终衰减的参数
             'lr': args.bert_lr,
             'weight_decay': 0.0},
            {'params': other_params, #模型中除去bert外, 所有自己设计的层的参数, 都进行权重衰减
             'lr': args.lr,
             'weight_decay': args.weight_decay},
            {"params": self.multi_loss.parameters(), # loss的参数,由于使用了多任务带参的loss权重, 都进行权重衰减
             'lr': args.lr,
             "weight_decay": args.weight_decay}
        ] # 不同参数支持不同学习率

        self.optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay) # param_groups: list[dict](包含params_list,lr,weight_decay), Adam的后两个参数用作缺省值
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.alpha) # 用了学习率调度, 每个epoch结束时降低优化器的学习率α, Decays the learning rate of each parameter group by gamma every epoch. When last_epoch=-1, sets initial lr as lr.

    def train(self, data_loader):
        self.model.train() # self.model来自Trainter.__init_()
        loss_array = []
        emo_gold_array = []
        emo_pred_array = []
        for dia_input, emo_label, spk_label, cls_index, emo_mask, spk_mask in data_loader: # 1个input,两个output,1个index用来提取sep表示,两个mask用来应对padding
        # dia_input[8*212], emo_label[8*17], spk_label[8*17*17], cls_index[8*17], emo_mask[8*17], spk_mask[8*17*17]
            dia_input = dia_input.cuda() # input to device(逐batch)
            emo_label = emo_label.cuda()
            spk_label = spk_label.cuda()
            cls_index = cls_index.cuda()
            emo_mask = emo_mask.cuda()
            spk_mask = spk_mask.cuda()
            # emo_output: batch8*utt_count17*label7, spk_output: 8*17*17*2丨dia_input: batch8*dialogue_utt_word_count212, cls_index:8*17, emo_mask: 8*17
            emo_output, spk_output = self.model(dia_input, cls_index, emo_mask) #模型训练, 为什么要传emo_mask(两层transformer中能用到吗?), 不传bert mask(因为后面会手动计算)
            # cls_index用来提取utt的sep vector, 作为两个解码器的输入, (emo_mask用来【传给第二层句间transformer的self attention】作为sent hidden的padding attention mask)
            emo_output = emo_output[emo_mask] #emo和spk两个输入batch mask的作用,提取模型output层的有效数据,用于不污染Tensor的交叉熵loss计算,从而在loss反向传播梯度时忽略mask区域外权重值. PS: 以bert的tokenizer padding和transformer decoder的attention mask为例, mask区域外的无效的权重值只参与前向的Tensor张量计算对模型训练没什么影响(看具体的模型运算公式和mask调用的时机吧...bert的tokenizer padding的mask应该是在模型训练的不同层中多次被调用,而不是想此文一样在训练完后的计算loss环节才调用, 而且好像发现这里没有保存tokenizer padding的attention mask, 不过在后文代码找到了通过input token id自行反推计算的attention mask)???
            emo_label = emo_label[emo_mask] #mask为布尔tensor, 用于批量索引[保留]子区域，，，这里为什么emo_output不做mask处理？？？？【上一行代码做了】
            emo_loss = self.emo_criterion(emo_output, emo_label) # 8*17*7, 8*17非one-hot(交叉熵也支持one-hot), 多分类和二匪类都是交叉熵, 多标签分类(不互斥)则是one-hot二进制交叉熵

            spk_output = spk_output[spk_mask] #两个mask的作用都是去除padding影响
            spk_label = spk_label[spk_mask]
            spk_loss = self.spk_criterion(spk_output, spk_label) # 8*17*17*2, 8*17*17, 结果为tensor(0.7413, device='cuda:0'), 属于tensor scalar(0维), 使用.item(只适合单元素的标量或张量)或者.tolist(都适合)提取为非Tensor的普通python数值, 注意.data提取的仍为tensor,而且是共用内存的浅拷贝(用于share权重但是不share梯度?),只是requires_grad=False.

            loss = self.multi_loss(emo_loss, spk_loss) # 多任务loss合并

            loss.backward() #交叉熵loss反向梯度传播
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm) # 梯度裁剪(缩放)放在反向传播后/节点更新前,计算的norm超过给定阈值则全局梯度值缩放变小,使得再次计算的norm等于预设阈值

            self.optimizer.step() # 优化器step(),根据反向传播后每层网络每个权重向量的梯度值(导数值),更新权重向量的权重值,
            self.model.zero_grad() # 模型所有层梯度重新置为零(更推荐放到loss.backward()反向传播梯度之前.),以便下一个train batch step重新正向训练+loss反向传播梯度+权重值逐层优化/更新

            loss_array.append(loss.item()) # 记录每个batch的loos以便求每个epoch的平均loss,作为训练收敛的评价指标打印到日志

            emo_pred = torch.argmax(emo_output, -1)
            emo_pred_array.append(emo_pred.cpu().numpy()) #记录每个batch的emo预测标签,用于整个epoch上的模型f1值评估
            emo_gold_array.append(emo_label.cpu().numpy()) # 每个batch的emo真实标签, # TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

        self.scheduler.step() # 优化器.step()在batch/step内改变权重参数值, 调度器.step()在每个epoch结束时降低优化器的学习率α
        emo_gold_array = np.concatenate(emo_gold_array)
        emo_pred_array = np.concatenate(emo_pred_array)

        f1 = f1_score(emo_gold_array, emo_pred_array, average='weighted') # 直接在训练集上评估f1值(epoch? 不评估辅助任务spk的f1值, weighted(类似):根据数据量中的真实标签比例来对不同标签的f1值加权

        loss = np.mean(loss_array) # 全部训练不的loss平均

        return loss, f1

    def eval(self, data_loader):
        self.model.eval() #冻结激活层和批量归一化全局均值
        loss_array = []
        emo_gold_array = []
        emo_pred_array = []
        with torch.no_grad(): #不记录梯度,加快推理速度
            for dia_input, emo_label, spk_label, cls_index, emo_mask, spk_mask in data_loader:
                dia_input = dia_input.cuda()
                emo_label = emo_label.cuda()
                spk_label = spk_label.cuda()
                cls_index = cls_index.cuda()
                emo_mask = emo_mask.cuda()
                spk_mask = spk_mask.cuda()

                emo_output, spk_output = self.model(dia_input, cls_index, emo_mask)

                emo_output = emo_output[emo_mask]
                emo_label = emo_label[emo_mask]
                emo_loss = self.emo_criterion(emo_output, emo_label)

                spk_output = spk_output[spk_mask]
                spk_label = spk_label[spk_mask]
                spk_loss = self.spk_criterion(spk_output, spk_label)

                loss = self.multi_loss(emo_loss, spk_loss)

                loss_array.append(loss.item())

                emo_pred = torch.argmax(emo_output, -1)
                emo_pred_array.append(emo_pred.cpu().numpy())
                emo_gold_array.append(emo_label.cpu().numpy())

        emo_gold_array = np.concatenate(emo_gold_array)
        emo_pred_array = np.concatenate(emo_pred_array)

        f1 = f1_score(emo_gold_array, emo_pred_array, average='weighted')
        loss = np.mean(loss_array)

        return loss, f1

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--d_ff', type=int, default=768)
    parser.add_argument('--heads', type=int, default=6)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--input_max_length', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--bert_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()

    logger = utils.get_logger("./log/HiTrans_{}.txt".format(time.strftime("%m-%d_%H-%M-%S")))
    logger.info(args)

    torch.cuda.set_device(args.device)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    logger.info("Loading data...")

    (train_set, dev_set, test_set), vocab = dataset.load_data(args.input_max_length)
    if args.evaluate:
        dev_loader = DataLoader(dataset=dev_set, batch_size=args.batch_size, collate_fn=dataset.collate_fn)
        test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, collate_fn=dataset.collate_fn)
    else:
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, collate_fn=dataset.collate_fn,
                                  shuffle=True)
        dev_loader = DataLoader(dataset=dev_set, batch_size=args.batch_size, collate_fn=dataset.collate_fn)
        # test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, collate_fn=dataset.collate_fn)

    model = HiTrans(args.hidden_dim,
                    len(vocab.label2id),
                    d_model=args.d_model,
                    d_ff=args.d_ff,
                    heads=args.heads,
                    layers=args.layers,
                    dropout=args.dropout)

    trainer = Trainer(model)

    if args.evaluate:
        trainer.load("./checkpoint/model.pkl")
        dev_loss, dev_f1 = trainer.eval(dev_loader) # 也打印了验证集
        logger.info("Dev Loss: {:.4f} F1: {:.4f}".format(dev_loss, dev_f1))
        test_loss, test_f1 = trainer.eval(test_loader)  # 测试集上只验证不更新梯度(全部训练结束后只进行一次)
        logger.info("Test Loss: {:.4f} F1: {:.4f}".format(test_loss, test_f1))
    else:
        best_f1 = 0.0
        for epoch in range(args.epochs):
            train_loss, train_f1 = trainer.train(train_loader)
            logger.info("Epoch: {} Train Loss: {:.4f} F1: {:.4f}".format(epoch, train_loss, train_f1))
            dev_loss, dev_f1 = trainer.eval(dev_loader) # 验证集上只验证不更新梯度(每个batch step)
            logger.info("Epoch: {} Dev Loss: {:.4f} F1: {:.4f}".format(epoch, dev_loss, dev_f1))
            # test_loss, test_f1 = trainer.eval(test_loader)
            # logger.info("Test Loss: {:.4f} F1: {:.4f}".format(test_loss, test_f1))
            logger.info("---------------------------------")
            if best_f1 < dev_f1: # dev_f1.tolist(), ndarray -> float
                best_f1 = dev_f1
                trainer.save("./checkpoint/model.pkl")

        logger.info("Best Dev F1: {:.4f}".format(best_f1))
