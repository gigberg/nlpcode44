import json


class Config:
    def __init__(self, args):
        with open(args.config, 'r', encoding="utf-8") as f:
            json_data = f.read()
            json_data_without_comments = self.remove_comments(json_data)
            config = json.loads(json_data_without_comments)

        # 数据集和模块选择
        self.dataset = config["dataset"]
        self.bert_name = config["bert_name"]
        self.dia_enc = config["dia_enc"]
        self.sent_enc = config["sent_enc"]
        self.flow_type = config["flow_type"]
        self.classifier = config["classifier"]

        self.save_path = config["save_path"]

        # 模型形状
        self.hidden_size = config["hidden_size"]
        self.max_position_embeddings = config["max_position_embeddings"]
        self.pad_token_idx = config["pad_token_idx"]
        self.type_vocab_size = config["type_vocab_size"]
        self.cln_hid_size = config["cln_hid_size"]
        self.ffnn_hid_size = config["ffnn_hid_size"]
        self.head_count = config["head_count"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.flow_size = config["flow_size"]

        self.dilation = config["dilation"]

        self.attn_dropout = config["attn_dropout"]
        self.emb_dropout = config["emb_dropout"]
        self.conv_dropout = config["conv_dropout"]
        self.layer_norm_eps = config["layer_norm_eps"]
        # self.biaffine_dropout = config["biaffine_dropout"]

        # 训练相关
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.seed = config["seed"]
        self.device = config["device"]
        self.evaluate = config["evaluate"]
        self.retoken = config["retoken"]

        # 优化器权重衰减
        self.bert_learning_rate = config["bert_learning_rate"]
        self.learning_rate = config["learning_rate"]
        self.warm_factor = config["warm_factor"]

        self.clip_grad_norm = config["clip_grad_norm"]
        self.weight_decay = config["weight_decay"]

        # other
        self.noise_rate = config["noise_rate"]
        self.flow_bias = config["flow_bias"]

        for k, v in args.__dict__.items():
            if v is not None: # --config
                self.__dict__[k] = v

    def __repr__(self):
        return "{}".format(self.__dict__.items())

    def remove_comments(self, json_str):
        lines = json_str.splitlines()
        lines = [line for line in lines if not line.strip().startswith('#')] #单行
        lines = [line.split('#')[0].strip() for line in lines] #行内
        return '\n'.join(lines)
