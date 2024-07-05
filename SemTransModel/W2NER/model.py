import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel


class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)): # [8, 7, 1, 512] - [8, 7, 512]
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1), [8, 1, 7, 512], cond和input的两个1在不同位置,从而支持两个维度上的同时广播(逐位置乘法),,cln相比注意力打分矩阵(尽管两者都是单向的),有个缺点就是同一个wordj对其它word的gate是同一个值(或同一个[feature个标量值构成的逐位置向量]),因为cln用的广播机制实现交互(广播意味着不独立的重复),,,注意力打分则是正常矩阵乘法的交互(矩阵乘法是独立的重复),,不过注意力打分的缺点是不同feature上用同一个分值(恰恰是cln所解决的),所以cln和注意力交互各自关注点不同吧[类似层归一化和批量归一化一个(batchnorm/attention)关注同一feature位置不同word交互,一个(layernorm/cln毕竟cln就是layernorm)关注统一word不同feature位置交互]

            if self.center:
                beta = self.beta_dense(cond) + self.beta # [8, 1, 7, 512],未改变维度
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1) # [8, 7, 1] -> [8, 7, 1, 1]
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std # [8, 7, 1, 512]), [8, 7, 1, 1]
            outputs = outputs * gamma # 缩放校正 [8, 7, 1, 512] * [8, 1, 7, 512], 相当于每行用相同的左值(当前word)和不同的右值列表(不同word对该行的贡献),另外发现(由于广播机制)同一word对不同word的贡献是同一个缩放gamma值,,毕竟每个word只是通过线性dens映射计算得到一个gamma值(in_input_size, out_input_size[每个维度值对于所有其它word 的贡献都只有一个值, 类似自身线性映射产生gate?,,毕竟计算cond时并没有词向量两两交互行为(双仿射/或是自注意力)])
        if self.center:
            outputs = outputs + beta #偏差校正 [8, 7, 1, 512] + [8, 1, 7, 512]

        return outputs


class ConvolutionLayer(nn.Module):
    def __init__(self, input_size, channels, dilation, dropout=0.1): #lstm512+20+20, chann128
        super(ConvolutionLayer, self).__init__()
        self.base = nn.Sequential( # 1*1卷积核不卷积只降维521+20+20->128(相当于逐位置线性映射,cnn还能表示线性映射!!)
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU(), # 李京烨比较喜欢gelu代替relu(毕竟bert用gelu,transformer用relu)
        )

        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d) for d in dilation]) #dilation = [1, 2, 3, 4], in_128, out_128

    def forward(self, x): # [8, 7, 7, 20+20+512])
        x = x.permute(0, 3, 1, 2).contiguous() # n,[c],h,w, permute接contiguous避免后续出错
        x = self.base(x) # 先跑个1*1卷积核降维,

        outputs = []
        for conv in self.convs:
            x = conv(x) # [8, 128, 7, 7])
            x = F.gelu(x) # 又是gelu, 这里没有1*1卷积和的dropout
            outputs.append(x) # append for concat
        outputs = torch.cat(outputs, dim=1) #dim=feature_channel
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1)

        return s


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x) 
        x = self.linear(x)
        x = self.activation(x)
        return x


class CoPredictor(nn.Module): #(config.label_num|0/1/2,none/thw/nnw, config.lstm_hid_size|512, config.biaffine_size|512,config.conv_hid_size * len(config.dilation)|多核chann, config.ffnn_hid_size|mlp384,config.out_dropout|0_biaffine 2个linear和mlp的1个linear都要dropout)
    def __init__(self, cls_num, hid_size, biaffine_size, channels, ffnn_hid_size, dropout=0):
        super().__init__()
        self.mlp1 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout) # biaffine前的两个投影和dropout
        self.mlp2 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.biaffine = Biaffine(n_in=biaffine_size, n_out=cls_num, bias_x=True, bias_y=True)
        self.mlp_rel = MLP(channels, ffnn_hid_size, dropout=dropout) #chann128*4# ffn384
        self.linear = nn.Linear(ffnn_hid_size, cls_num) #cls_num
        self.dropout = nn.Dropout(dropout)# 李京烨喜欢在linear后drouout

    def forward(self, x, y, z):
        h = self.dropout(self.mlp1(x))
        t = self.dropout(self.mlp2(y))
        o1 = self.biaffine(h, t)

        z = self.dropout(self.mlp_rel(z))
        o2 = self.linear(z)
        return o1 + o2


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.use_bert_last_4_layers = config.use_bert_last_4_layers

        self.lstm_hid_size = config.lstm_hid_size
        self.conv_hid_size = config.conv_hid_size

        lstm_input_size = 0 # ?

        self.bert = AutoModel.from_pretrained(config.bert_name, cache_dir="./cache/", output_hidden_states=True)
        lstm_input_size += config.bert_hid_size

        self.dis_embs = nn.Embedding(20, config.dist_emb_size) # 0没用到,话说一般embed都将0当成pad位置中(同事vocab.text2idx也要让<pad>对应到0,别让有用的text词被掩码掩盖了),以便后续直接用ne(0)计算掩码
        self.reg_embs = nn.Embedding(3, config.type_emb_size) # 0没用到

        self.encoder = nn.LSTM(lstm_input_size, config.lstm_hid_size // 2, num_layers=1, batch_first=True, bidirectional=True) # #lstm将bert 768变长了lstm512(双向268)

        conv_input_size = config.lstm_hid_size + config.dist_emb_size + config.type_emb_size
        # lstm512 + dist20 + type20
        self.convLayer = ConvolutionLayer(conv_input_size, config.conv_hid_size, config.dilation, config.conv_dropout)
        self.dropout = nn.Dropout(config.emb_dropout)
        self.predictor = CoPredictor(config.label_num, config.lstm_hid_size, config.biaffine_size,config.conv_hid_size * len(config.dilation), config.ffnn_hid_size,config.out_dropout) # config.label_num = pad0+nnw1+thw*(实体类型数),,在data_loader.y中有config.label_num = len(vocab.label2id)

        self.cln = LayerNorm(config.lstm_hid_size, config.lstm_hid_size, conditional=True)

    def forward(self, bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length):
        '''
        :param bert_inputs: [B, L']
        :param grid_mask2d: [B, L, L] # # length*length, 不含cls/sep, , 目前还是初始化的1
        :param dist_inputs: [B, L, L]
        :param pieces2word: [B, L, L']
        :param sent_length: [B]
        :return:
        '''
        bert_embs = self.bert(input_ids=bert_inputs, attention_mask=bert_inputs.ne(0).float()) # bert_inputs.ne(0).float()手动获取bert mask(前提是前面能确保有效区间都是非0值,且0值都是无效数据,必须完全一一对应)
        if self.use_bert_last_4_layers:
            bert_embs = torch.stack(bert_embs[2][-4:], dim=-1).mean(-1)
        else:
            bert_embs = bert_embs[0]

        length = pieces2word.size(1) # bat * word_len L * subword_len L' (不含cls/sep)

        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2) #subword dim, max自动squeeze降维

        word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False) #rnn常用的两项参数batch_first=True, enforce_sorted=False
        packed_outs, (hidden, _) = self.encoder(packed_embs) #lstm将bert 768变长了lstm512, 另外双向的out是同一个词正反向叠加(out也是hid,而非label,相当于h1-hn的另一种组组织形式),而非同一个step正反向叠加
        word_reps, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=sent_length.max())

        cln = self.cln(word_reps.unsqueeze(2), word_reps) # [8, 7, 1, 512], [8, 7, 512])

        dis_emb = self.dis_embs(dist_inputs)
        tril_mask = torch.tril(grid_mask2d.clone().long()) # grid_mask2d length*length, 不含cls/sep, 目前还是初始化的1
        reg_inputs = tril_mask + grid_mask2d.clone().long() #2d的2下三角和1上三角(grid_mask2d初始值为1), [8, 7, 7]
        reg_emb = self.reg_embs(reg_inputs)

        conv_inputs = torch.cat([dis_emb, reg_emb, cln], dim=-1) # [8, 7, 7, 20+20+512])
        conv_inputs = torch.masked_fill(conv_inputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        # 卷积前先mask一下(主要是三层词嵌入pad位置目前都非0), grid_mask2d.eq(0).unsqueeze(-1) 使用逐位置mask_filled, 或者使用逐位置矩阵乘法,效果一样
        conv_outputs = self.convLayer(conv_inputs)
        # 卷积后再mask一下,保持输入输出一致,然后传入分类器
        conv_outputs = torch.masked_fill(conv_outputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        outputs = self.predictor(word_reps, word_reps, conv_outputs) # [8, 7, 512]biaffine(biaffin之前通常接两个mlp), [8, 7, 7, 512] position mlp

        return outputs
