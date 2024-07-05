import logging
import random
import os

import numpy as np
import torch


from utils.configue import Configure
from utils.tools import DatasetTool


from models.bert import Model
import fitlog

def set_seed(args):
    np.random.seed(args.train.seed)
    random.seed(args.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train.seed)
        torch.cuda.manual_seed(args.train.seed)

    torch.manual_seed(args.train.seed)
    torch.random.manual_seed(args.train.seed)

def get_args_dict(args):
    ans = {}
    for x, y in args:
        if isinstance(y, (int, float, str)):
            ans[x] = y
        else:
            ans[x] = get_args_dict(y)
    return ans


def start():
    logging.basicConfig(level=logging.INFO) # 放到第一个logging.info()语句之前, 该语句位于Configure.Get()内
    # logging.getLogger().setLevel(logging.INFO) #万能的,顺序无关的,不会被覆盖的版本

    args = Configure.Get()
    set_seed(args)
    x = get_args_dict(args)

    if not os.path.exists('./logs/'):
        os.mkdir('./logs/')
    fitlog.set_log_dir('./logs/')

    fitlog.add_hyper(x)

    # # --2024-03-07 15:41:31测试fitlog其它功能
    # fitlog.add_other({'task': '实验任务名称'}) # 直接添加到命令行参数中了
    # fitlog.add_progress(19) # 根据loss.log中的epoch数(从0开始来判断的current_step = Math.max(current_step, loss_data[loss_data.length-1]['step']);
    # # [参见代码/home/zhoujiaming/anaconda3/envs/SGDCI/lib/python3.8/site-packages/fitlog/fastserver/templates/chart.html],
    # # 由于本代码中是每个epoch记录一次fitlog.add_loss,所以这里的total_steps应该设置为epoch数-1(转换为从0开始)

    inputs = DatasetTool.get(args) # 预处理数据
    # evaluator = utils.tools.get_evaluator() # evaluate评估 在model中有设置

    if args.train.gpu and torch.cuda.is_available():
        args.train.gpu = True
    else:
        args.train.gpu = False

    model = Model(args, inputs)

    if args.train.gpu:
        model.cuda()

    model.start(inputs)

    fitlog.finish()


if __name__ == "__main__":
    start()
