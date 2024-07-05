import logging
import random

import numpy as np
import torch

import utils.tool
from utils.configue import Configure
import fitlog


def set_seed(args):
    # 读取数据的过程采用了随机预处理, PyTorch 的可重复性问题 （如何使实验结果可复现）
    np.random.seed(args.train.seed)
    random.seed(args.train.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train.seed)  # gpu
        torch.cuda.manual_seed(args.train.seed)  # gpu

    torch.manual_seed(args.train.seed)  # cpu
    torch.random.manual_seed(args.train.seed)  # 读取数据的过程采用了随机预处理


def get_args_dict(args):
    ans = {}
    for x, y in args:
        if isinstance(y, (int, float, str)):  # 判断y是Args对象, 还是(int, float, str)等叶子层的模型参数
            ans[x] = y
        else:
            ans[x] = get_args_dict(y)  # 递归Args对象, 得到嵌套词典
    return ans


def start():
    fitlog.set_log_dir('./logs/') # 可以通过将log_dir设置为具体的log文件夹名 fitlog.set_log_dir('logs/log_20190417_140311') 
    # # 发现fitlog字文件名或者说id名只能以特定格式logs/log_20190417_140311(改名后无法在命令行fitlog log logs/中被识别)，所以需要专设column来显著标记实验名称/实验备注等信息
    logging.basicConfig(level=logging.INFO)
    args = Configure.Get() ## args分别来自：1)命令行参数，2)KBRetriver_DC_INTERACTIVE.cfg配置文件，3) configure.py.Get()中手动添加的变量args.dir.(model/exp/dataset/configure/output)
    # 获取命令行参数--cfg KBRetriver_DC/KBRetriver_DC_BART.cfg和KBRetriver_DC_BART.cfg中的模型训练参数
    set_seed(args)
    x = get_args_dict(args)  # 参数嵌套Args对象 -> 参数嵌套词典
    fitlog.add_hyper(x)  # 用于fitlog添加超参数到文件hyper.log(json)。用此方法添加到值，会被放置在表格中的 hyper 列及其子列中
    loader = utils.tool.get_loader(args.dataset.tool)
    evaluator = utils.tool.get_evaluator()
    inputs = loader.get(args)  # return train, dev, test, entities
    # 1)inputs为返回的train_data, dev_data, test_data三个部分的内部结构为：
    # train_data：list of m dicts(navigate or calendar or weather):
    # every dict is a dialogue(navigate or calendar or weather): {constructed_dialogue_info_string: 'kb+history', last_response_string: 'last', consistency:[qi,hi,kbi]}
    # return construct_dialogue_info_string: a concatenated long string："[SOK] "kb_entities_1_info_1_key 1_value xx xx xx xx ... ; kb_entities_2_info_1_key 1_value xx xx ... ;  [EOK]  [USR] xx [SYS] yy [USR] xx..."
    # 2)返回的entities部分的内部结构为：an list of entities's string in all entity.json files, 其中会有一些特殊元素：'-'

    Model = utils.tool.get_model(args.model.name)  # 导入bert.py模块并返回model/bert.py中的Model类

    model = Model(args, loader, evaluator, inputs)
    # pytorch_quick_start: train(train_dataloader, model, loss_fn, optimizer) need these 4 arguments, and other hyperparameters.
    if args.train.gpu:
        model.cuda()  # Moves all model parameters and buffers to the GPU.
    model.start(inputs) # inputs结构见上面若干行 #---!!!正式训练时记得修改代码中"# modified-debug 减少调试耗时部分"---# 主要改动了1）base.py中run_batches和run_test只2个batch，update_best中最优模型文件名称只含dev_overall_acc；2）bert.py中run_train和run_eval的验证集只dev数据集，run_train中的epoch只2
    # models.KBRetriever_DC.bert.bert.Model -继承自-> models.KBRetriever_DC.base.Model  -继承自-> models.base.Model -继承自-> torch.nn.modules.module.Module -继承自-> object
    # model = Model(args, loader, evaluator, inputs) # Model.__init__() # 模型初始化时设置模型loss函数self.criterion = nn.BCELoss(), 最后forward()中实际使用损失函数的是F.cross_entropy()
    # 0>models.KBRetriever_DC.bert.bert.start(inputs)
    # -1> # 模型训练和验证 # models.KBRetriever_DC.bert.run_train(train, dev, test): # 模型训练前设置模型opt函数 opt = AdamW(params) # 在每个epoch开始时需要设置模型状态：epoch_for -> self_model.train(切换模式，保证可重复性) -> batch_for(run_batches/batch_train)
    # --2> #--- 模型训练(共n epoch) ---# tqdm*1_train # loss, iter = models.base.run_batches(train, epoch) in models.KBRetriever_DC.bert.py # train：list of m dicts(navigate or calendar or weather), # every dict is a dialogue(navigate or calendar or weather): {constructed_dialogue_info_string: 'kb+history', last_response_string: 'last', consistency:[qi,hi,kbi]}
    # ---3> for batch in tqdm([[],[],...n batches]): in models.base.py(train)
    # ----4> models.KBRetriever_DC.bert.forward(batch)        # batch: list of num<=batch dicts(navigate or calendar or weather), # for batch in "[] -> [[],[],...n batches]" #
    # -----5> forward-1-tokenize: token_ids, type_ids, mask_ids = models.KBRetriever_DC.bert.get_info(batch)
    # ------6> get_info(batch) 使用tokenizer实例来对输入句子tokenize, 内部实际调用的是transformers.PreTrainedTokenizer.__call__(), 返回值类型是BatchEncoding(继承自dict类型)，通过.data属性得到输入句子tokenize后的dict（ BatchEncoding().data ）
    # ------6> get_info(batch) 内部代码：tokenized = self.tokenizer(construced_infos, last_responses).data
    # ------6> # tokenize后的数据结构tokenized: <class 'dict'> {input_ids: [8[512]] , token_type_ids: [8[512]], attention_mask: [8[512]]}
    # ------6> get_info(batch)返回值结构(顺便转移到gpu中) # return tokenized['input_ids'].to(self.device), tokenized['token_type_ids'].to(self.device), tokenized['attention_mask'].to(self.device)
    # -----5> forward-2-modeltrain_bert_(batchsize=8): h[8bat*512tok*768hide], utt[8,768] = self.bert(input_ids = token_ids, token_type_ids = type_ids, attention_mask = mask_ids) # 在hungging下载好的bert-base-uncased文件bert_config.json中配置了隐藏层大小"hidden_size": 768,
    # -----5> forward-2-modeltrain_nn.Linerar768.2_(batchsize=8): out_qi[8,2] = self.w_qi(utt),out_hi[8,2] = self.w_hi(utt),out_kbi[8,2] = self.w_kbi(utt)
    # -----5> forward-2-lossfn: loss = F.cross_entropy(out_qi_N8*C2, target_consistency[0]_LongTensor_N8) + F.cross_entropy(out_hi,consistency[1]) + F.cross_entropy(out_kbi,consistency[2])
    # ----4> models.KBRetriever_DC.bert.forward(batch) # forward最后返回 loss, out #其中 loss: 交叉熵损失, out(batch_8*3不是tensor类型而是普通list类型): list of list[qi_预测值0或1, hi_预测值0或1, kbi_预测值0或1]
    # ----4> models.base.zero_grad() -> loss.backward() ---> self.optimizer.step() 丨（自动梯度+参数优化）all in models.base.py run_batches()
    # --2> loss, iter = models.base.run_batches(train, epoch) ：# return all_loss / all_size, iteration # 返回：单个epoch内各batch的平均损失(每batch一次，loss自动梯度更新参数一次)，和单个epoch的迭代次数320
    # --2> fitlog.add_loss({"train_loss": loss}, step=epoch) in models.KBRetriever_DC.bert.py # 用于fitlog添加loss.log(step+dict)文件。用此方法添加的值不会显示在表格中，但可以在单次训练的详情曲线图中查看
    # --2> for dataset in [train, dev, test]: in models.base.py # (比训练时多一层外层的数据集类型遍历for循环)注意: 这里把训练集验证集、测试集都用来作为输入，测试模型性能？
    # ---3> #--- 模型验证(每个epoch内1次，after run_train()) ---# tqdm*3_train/dev/test这里有点多余,实际只需输入dev数据集就行 # tmp_summary, pred = models.base.run_test(dataset) in models.base.py 
    # ----4> models.base.eval() 切换模型为eval状态
    # ----4> for batch in tqdm([[],[],...n_320 batches]) in models.base.py(eval)
    # -----5> loss, out = self.forward(batch) # 同模型训练时步骤, # forward最后返回 loss, out #其中 loss: 交叉熵损失, out(batch_8*3,不是tensor而且不在gpu中): list of list[qi_预测值0或1, hi_预测值0或1, kbi_预测值0或1]      
    # -----5> all_out += models.KBRetriever_DC.base.get_pred(out) in models.base.run_test() # [[],[]] -> [[],[]]（该函数基本没作用) # all_out == out(total_data2553*3)
    # -----5> utils.evaluate.EvaluateTool.evaluate(all_out, dataset, self.args) # # 使用sklearn.metrics工具包precision_score, recall_score, f1_score，计算预测值y_pred和目标值y_true的精度、召回率和f1值、全局准确度
    # -----5> utils.evaluate.EvaluateTool.evaluate(all_out, dataset, self.args) # return summary返回数据结构# dict of 3*3+1 element: {precision_qi:x, precision_hi:x, precision_kbi:x, recall_qi:x,.., f1_qi:x,.., overall_acc:x}
    # ---3> tmp_summary, pred = models.base.run_test(dataset) # return summary, all_out # # run_test返回：模型验证指标值dict(3*3+1element), 模型预测标签listoflist(非tensor: total_data2553*3)
    # ---3> self.DatasetTool.record(pred, dataset, set_name, self.args) # DatasetTool.record函数为空，用于调参？(位于utils.process.KBRetriever_DC.loader.py)
    # ---3> summary.update( ...tmp_summary.items()) --->  fitlog.add_metric({eval_{}_{} in tmp_summary.items()}, step=epoch) #  用于添加metric.log(step+dict) 
    # --2> best = self.update_best(best, summary, epoch)  # 依据第epoch轮的训练结果summary_dict，更新最优结果best_dict，保存当前最优模型，并删除已保存的多于max_save=2的次优模型
    # --2> logging.info(pprint.pformat(best)) ---> logging.info(pprint.pformat(summary))
    # -1> #--- 模型测试(共1次) ---# models.KBRetriever_DC.bert.run_eval # 返回：logging.info(pprint.pformat(summary))
    #0>models.KBRetriever_DC.bert.bert.start(inputs) end! 
    # fitlog.finish() 告知fitlog当前实验已经正确结束或者被中断

    # fitlog的调用顺序：
    # fitlog.set_log_dir('logs/log_20190417_140311')
    # fitlog.add_hyper(x) : hyper.log table <-- other.log耗时，meta.log启动/结束状态
    # fitlog.add_loss({"train_loss": loss}, step=epoch) : loss.log cahrt
    # fitlog.add_metric({"eval_{}_{}".format(set_name, k): v for k, v in tmp_summary.items()}, step=epoch) : metric.log chart
    # fitlog.finish() 告知fitlog当前实验已经正确结束，meta.log启动/结束状态
    # run: fitlog log logs/, open: http://localhost:5000/table, 有些字段不在表格中显示可能是因为logs文件夹的某个子文件夹中的日志没有缺少该字段
    # （①只有行间存在差异的字段列才会在表格中显示,所有行取值都一样的字段隐藏在Con.cols按钮内，②某行日志如果如果缺少某字段,则默认显示'-'以表示不含该字段）

    # 测评五个预训练模型的代码区别：
    # bert/roberta/longformer: 基本一致，from_pretrain()配置有点区别，forward调用时输入输出格式一致，只需一行
    # --> h, utt = self.bert(input_ids=token_ids, token_type_ids=type_ids, attention_mask=mask_ids)
    # bart：基本一致，from_pretrain()配置有点区别，forward调用时输入输出有三行才能得到
    # --> eos_mask = token_ids.eq(self.tokenizer.eos_token_id)
    # --> x = self.bert(input_ids=token_ids, token_type_ids=type_ids, attention_mask=mask_ids)[0]
    # --> utt = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]
    # xlnet：基本一致，from_pretrain()配置有点区别，forward调用时输入输出一行但是有点区别
    # --> utt = self.bert(input_ids=token_ids, token_type_ids=type_ids, attention_mask=mask_ids)[0][:, -1, :].squeeze(1)

    fitlog.finish()


if __name__ == "__main__":
    # tmux capture-pane -pS - > ./pane-history-1 # shell命令：用于导出模型训练时tmux终端内的tqdm+logging.info输出记录
    start() # modified-debug （该类注释都是为了"减少调试耗时"）
