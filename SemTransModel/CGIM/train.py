import logging
import random

import numpy as np
import torch

from models.bert_interact import CGIM
from utils.loader import DatasetTool
from utils.tools.configure import Configure
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
    fitlog.commit(__file__, 'CGIM_bart_20')   
    fitlog.set_log_dir('./logs/') 
    # # 发现fitlog字文件名或者说id名只能以特定格式logs/log_20190417_140311(改名后无法在命令行fitlog log logs/中被识别)，所以需要专设column来显著标记实验名称/实验备注等信息
    # # 目前暂时在前端界面手动编辑fit_msg列并点右上角保存，用来暂时记录日志对应的实验名称
    logging.basicConfig(level=logging.INFO)
    args = Configure.Get() # args分别来自：1)命令行参数，2)KBRetriver_DC_INTERACTIVE.cfg配置文件，3) configure.py.Get()中手动添加的变量args.dir.(model/exp/dataset/configure/output)
    set_seed(args)
    x = get_args_dict(args)
    fitlog.add_hyper(x)
    inputs = DatasetTool.get(args) #return train2553, dev319, test318, entities
    # inputs = loader.get(args) #bert.py
    # bert_interact.py中的get()和bert.py中的get()相比: 
    # 共同点: 返回的都是train, dev, test, entities
    # 区别: 内部数据结构(train, dev, test)
    # # bert.p中train_data结构为：[list of n dict_dialogue, every dialogue is a dict of 3key-value ]: constructed_concatenate_dialogue_info_string(kb+h), last_response, consistency
    # # bert_interact.py本代码中interact_train_data结构为：[list of n dict_dialogue, dialogue is dict of 5key-value]:  kb,history,query, last_response, consistency
    model = CGIM(args, inputs)
    if args.train.gpu:
        model.cuda()

    # ！！！注意input.train_data/inputs.batch的格式，内部的数据时dialogue dict of 5 key-value
    # ！！！也即本模型CGIM和CI-ToD尽管都是多重二分类器训练任务，但两者的输入格式不同，输出则是一样qi/hi/kbi类别的f1值

    model.start(inputs)
    fitlog.finish()

if __name__ == "__main__":
    start()


# CGIM和CI-ToD代码的区别  (其中相同名称的py文件内，也可能有细微区别，比如处理数据的loader.py)
# alt+shift # vscode多行竖列编辑
# CI-ToD vs CGIM                             # CGIM
# .                                          # 
# ├── configure                              # 
# │   ├── __console__.cfg                    #  
# │   └── KBRetriver_DC                      # └── KBRetriver_DC_INTERACTIVE
# │       ├── KBRetriver_DC_BART.cfg         #     ├── KBRetriver_DC_INTERACTIVE.cfg
# │       ├── KBRetriver_DC_BERT.cfg         #     └── test.cfg
# │       ├── KBRetriver_DC_Longformer.cfg   # 
# │       ├── KBRetriver_DC_RoBERTa.cfg      # 
# │       └── KBRetriver_DC_XLNet.cfg        # 
# ├── data                                   # 
# │   └── KBRetriever_DC                     # 
# │       ├── calendar_dev.json              # 
# │       ├── calendar_test.json             # 
# │       ├── calendar_train.json            # 
# │       ├── entities.json                  # 
# │       ├── kvret_entities.json            # 
# │       ├── navigate_dev.json              # 
# │       ├── navigate_test.json             # 
# │       ├── navigate_train.json            # 
# │       ├── weather_new_dev.json           # 
# │       ├── weather_new_test.json          # 
# │       └── weather_new_train.json         # 
# ├── img                                    # 
# │   ├── model.png                          # 
# │   ├── pytorch.png                        # 
# │   ├── SCIR_logo.png                      # 
# │   └── triple_inconsistency.png           # └── Columbia_logo.png
# ├── models                                 # 
# │   ├── base.py                            # 
# │   ├── __init__.py                        # 
# │   └── KBRetriever_DC                     # 
# │       ├── bart.py                        # 
# │       ├── base.py                        # ├── interactive_block.py
# │       ├── bert.py                        # └── bert_interact.py
# │       ├── __init__.py                    # 
# │       ├── longformer.py                  # 
# │       ├── roberta.py                     # 
# │       └── xlnet.py                       # 
# ├── py3.6pytorch1.1_.yaml                  # 
# ├── README.md                              # 
# ├── train.py                               # 
# └── utils                                  # └── utils
#     ├── configue.py                        #     ├── """loader.py
#     ├── evaluate.py                        #     ├── 
#     ├── __init__.py                        #     ├── 
#     ├── process                            #     └── tools
#     │   ├── __init__.py                    #         ├── """configure.py
#     │   └── KBRetriever_DC                 #         ├── __init__.py
#     │       ├── __init__.py                #         ├── manager.py  ## 从bert.py中拆分出来的两个功能模块：①返回预训练模型PretrainedModelManager，②返回tokenized后的适合于bert模型输入的tensor(gpu),batch.get_info()
#     │       └── loader.py                  #         ├── recorder.py
#     └── tool.py                            #         └── """tool.py
                                             # ── train_bert.sh
# 10 directories, 40 files                   # 

