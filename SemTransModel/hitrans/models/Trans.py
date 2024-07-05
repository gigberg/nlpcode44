import math

import torch
import torch.nn as nn


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module): # 所谓position-wis是指相对cnn来说不用flatten最后两维度,才去再接全连接层
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x): #8*17*768
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x)))) #这里transformer原文用的是relu(唯一一处relu,其它都是gelu), layernorm怎么放在这而非为add后
        output = self.dropout_2(self.w_2(inter))
        return output + x #8*17*768 element-wise, 怎么不norm,原来norm放到TransformerEncoderLayer(sub-layer)开头去了


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1): # 6, 768
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count # 128
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head) # 768, 768=6*128? 那计算dim_per_head有什么意义(用在view中避免计算前将head_count转为batch维,计算后避免concat)? 这里弄错了不该乘head_count
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head) # 不不,还是该乘,因为这里也是按照哈弗transformer文章中的一样,利用view机制将head转为batch维执行并行批量计算(所以qkv的形状是合并后的而非拆分成head_count个qkv),避免单独执行head_count个自注意力
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim) # 多头输出cat之后形状能变回input hidden,  self.dim_per_head = model_dim // head_count是为了保证这个

    def forward(self, key, value, query, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2) # batch, seq, hidden -> (batch_size, -1[seq], head_count, dim_per_head) - > (batch_size, head_count, -1[seq], dim_per_head) # 将head_count提到batch维,执行批矩阵运算(沿着非后两维的其它维度独立重复),和哈佛文章中一样
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        query = query / math.sqrt(dim_per_head) # 缩放
        scores = torch.matmul(query, key.transpose(2, 3)) # transpose(-2,-1)

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores) # (8*17 ->) 8*17*1 -> 8*1*17*1 # mask维数要待遮盖的数据和一致(纬度长可以不一会,为1能支持广播就行)
            scores = scores.masked_fill(mask, -1e10) #mask==0的位置置为负无穷,而非相乘置为0,因为mask_fill在softmax前,0的softmax值并非非负数

        attn = self.softmax(scores)

        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value).transpose(1, 2).contiguous().view(batch_size, -1, #transpose(1head, 2seq) head从batch维换到倒数第二维[其中倒数第一维是待cat的维度,放到他身边,且放到他前一个,表明待会view沿着前一个的方向cat],用于view实现cat)
                                                                                   head_count * dim_per_head)
        output = self.linear(context)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0) # 512*768 -> 1*512*768,第一维广播到batch维大大小
        self.register_buffer('pe', pe) # regeister buffer, 以便不训练梯度但能通过self.pe引用
    def forward(self, x):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        x = x + pos_emb # 8*512[474]*768 + 1*512[474]*768
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout): # hidden size, self-attention heads, the feed-forward hidden size
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask): # 接受一个query参数但是没用到
        if (iter != 0): # 第0层embedding不norm,其它层的norm本该放到ffn中,但是这里移到sub-layer开始位置了
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1) # 8*17 -> 8*17*1 # mask维数要待遮盖的数据和一致(纬度长可以不一会,为1能支持广播就行)
        context = self.self_attn(input_norm, input_norm, input_norm, # 8*17*768,qkv
                                 mask=mask)
        out = self.dropout(context) + inputs # add(no norm, norm放到下一步ffn代码中了)
        return self.feed_forward(out) #


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, layers, dropout=0.1): # hidden size, the feed-forward hidden size, self-attention heads, tansformer layers
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask): # [8, 18, 768], mask=8*24
        x = self.pos_emb(x)
        x = self.dropout(x)
        for i in range(self.layers): # transformer本身有多层, 比如bert是12层
            x = self.transformer_inter[i](i, x, x, mask.eq(0)) # 传i是因为要判断,让除第1次外transformer做layer_norm(ps:一般都是一个trans层在快结束时layernorm,这里放刚开始时?), 传的第一个x是 # 接受一个query参数但是没用到, 传递第二个x是input_ids, mask取反?1表示padding?,,原因是用的data.mask_fill(mask==0,-inf寻找padding位置置为1)而非data*(mask==1,寻找有有效数据位置置为0)
        return x
