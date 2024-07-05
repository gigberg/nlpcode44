import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel
from models.GCNLayer import GraphConvolution, HighWay
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, d_out=2, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_out = d_out
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_out = nn.LayerNorm(d_out)
        self.actv = F.gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x): #8*17*768
        if self.d_out != self.d_model:
            inter = self.actv(self.w_1(self.layer_norm(x)))
            return self.w_2(inter)
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, config):
        super(TransformerEncoderLayer, self).__init__()
        self.module_dict = {
            'MultiUniAtten': MultiUniAtten(d_model, head_count=config.head_count, dropout=0.1),
            'MultiSpanRNN': MultiSpanRNN(d_model, batch_size=config.batch_size, span_count=3, para_pooling_span=4, skip_span=6, para=2, layer=2, dropout=0.1),
            'MultidilaCNN': MultidilaCNN(d_model, dilation=config.dilation, dropout=0.1),
            'MultiUniGCN': MultiUniGCN(d_model, edge_type=3, dropout=0.1),
            'DiaRemove': nn.Identity()
            }
        self.dia_enc = config.dia_enc
        self.layers = config.num_hidden_layers
        self.self_attn = self.module_dict[config.dia_enc]
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, d_model)
        self.feed_forward_out = PositionwiseFeedForward(d_model, config.sent_hid_size, config.flow_size)
        self.layer_norm = nn.LayerNorm(d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.attn_dropout)

        if self.dia_enc == 'MultidilaCNN':
            self.cln_low = ConditionalLayerNorm(config.hidden_size, config.hidden_size, conditional=True, hidden_units=config.cln_hid_size)
            self.cln_up = ConditionalLayerNorm(config.hidden_size, config.hidden_size, conditional=True, hidden_units=config.cln_hid_size)

    def forward(self, iter, inputs, pad_mask, edge_matrix): #mask[bat, word, 1], 只有attn和cnn需要pad
        inputs = inputs * pad_mask
        __builtins__["train_iter"] = iter
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs
        if self.dia_enc == 'MultidilaCNN':
            tri_low = self.cln_low(input_norm).transpose(-1,-2).tril().transpose(-1,-2).contiguous()
            tri_up = self.cln_up(input_norm).transpose(-1,-2).triu(diagonal=1).transpose(-1,-2).contiguous()
            input_norm = tri_low + tri_up
            if pad_mask is not None:
                grid_mask2d = torch.einsum('bxk,byk->bxy', pad_mask, pad_mask)
                input_norm = input_norm.masked_fill(grid_mask2d.eq(0).unsqueeze(-1), 0.0)

        context = self.self_attn(input_norm, mask=pad_mask, edge_matrix=edge_matrix) # 8*17*768,qkv
        out = self.dropout(context) + inputs # add(no norm, norm放到下一步ffn代码中了)
        if iter == self.layers - 1:
            return self.feed_forward_out(out)
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    def __init__(self, config, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = config.hidden_size
        self.d_ff = config.ffnn_hid_size
        self.layers = config.num_hidden_layers
        self.dia_enc = config.dia_enc

        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(self.d_model, self.d_ff, config)
             for _ in range(self.layers)])

    def forward(self, word_reps, edge_matrix, pad_mask): # [bat,word,hid],, pad_mask=[bat,word,1]
        for i in range(self.layers):
            word_reps = self.transformer_inter[i](i, word_reps, pad_mask, edge_matrix)
        return word_reps


class BertStyleEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.use_bert_last_4_layers = config.use_bert_last_4_layers
        self.lstm_hid_size = config.hidden_size #要求是6的倍数（2和3）# 552 // 3 = 184
        self.lstm_input_size = config.bert_hid_size + config.position_emb_size + config.type_emb_size # 768 + 20 +20

        self.bert = AutoModel.from_pretrained(config.bert_name, cache_dir="./cache/", output_hidden_states=True, local_files_only=True)
        self.position_embeddings  = PositionalEncoding(config.max_position_embeddings, config.position_emb_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.type_emb_size)
        self.lstm_encoder = nn.LSTM(self.lstm_input_size, self.lstm_hid_size // 2, num_layers=1, batch_first=True, bidirectional=True)

        self.LayerNorm = nn.LayerNorm(self.lstm_input_size, eps=config.layer_norm_eps) # 别，三种不是一个空间
        self.dropout = nn.Dropout(config.emb_dropout) #bert的dropout

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
        self, input_ids, token_type_ids, word_subword, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        # seq_length = input_shape[1]
        seq_length = word_subword.shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device) # position_ids已经在buffer设备中

        bert_embs = self.bert(input_ids=input_ids, attention_mask=input_ids.ne(0)) # pad_mask, [bat,word,1]， 可以不加1
        if self.use_bert_last_4_layers:
            bert_embs = torch.stack(bert_embs[2][-4:], dim=-1).mean(-1)
        else:
            bert_embs = bert_embs[0]

        # piece to word
        length = word_subword.size(1)
        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, word_subword.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2) #subword dim, max自动squeeze降维
        # word_reps = word_reps * pad_mask #排除pad_hidden_dim!=0的影响(pad的三种embedding都变为0)
        word_reps = self.dropout(word_reps)

        # word level concat type + position
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids).repeat(input_ids.size(0), 1, 1)
        embeddings = torch.cat([word_reps, token_type_embeddings, position_embeddings], dim = -1)

        #lstm embed encoder
        sent_length = word_subword.sum(-1).ne(0).sum(-1) #[bat,word,subword] -> bat
        packed_embs = pack_padded_sequence(embeddings, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.lstm_encoder(packed_embs)
        embeddings, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=sent_length.max())
        # embeddings = embeddings * pad_mask # pad_packed_sequence中已经pad过
        return embeddings


class PositionalEncoding(nn.Module):

    def __init__(self, max_len=512, dim=768):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0) # 512*768 -> 1*512*768,第一维广播到batch维大大小
        self.register_buffer('pe', pe)
    def forward(self, x):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        return pos_emb # 1*512*768


class MultiUniAtten(nn.Module):
    def __init__(self, model_dim, head_count=12, dropout=0.1): # 6, 768
        super().__init__()
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count # 128
        self.model_dim = model_dim

        self.head_count = head_count

        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, input, mask=None, **kwargs): #mask[bat, word]
        batch_size = input.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        key = self.linear_k(input).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(input).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(input).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        query = query / math.sqrt(dim_per_head) # 缩放
        scores = torch.matmul(query, key.transpose(2, 3)) # transpose(-2,-1)

        if mask is not None:
            mask = torch.tril(mask.eq(0).unsqueeze(1).expand_as(scores)) #下三角注意力(前对后) # 提取和填充0值
            scores = scores.masked_fill(mask, -1e10)

        attn = self.softmax(scores)

        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value).transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)
        output = self.linear(context)
        return output


class MultiSpanRNN(nn.Module):
    """ 统一用bert的embedding, entity也用bert的分词作max pooling
    Args: input_size: word2vec=100, hidden_size: bert=768
    Input: x (batch_size, seq_len, feature_hid)
    Output: x (batch_size, seq_len, feature_hid)
    """
    def __init__(self, input_size, batch_size=8, span_count=3, para_pooling_span=4, skip_span=6, para=2, layer=2, dropout=0.1):
        super().__init__()
        assert input_size % span_count == 0 # 768 // 3 = 256
        self.dim_per_head = input_size // span_count # 128, per_output_size, sum=input_size
        self.input_size = input_size # input_size
        self.para_pooling_span = para_pooling_span
        self.skip_span = skip_span

        self.rnn_span = nn.ModuleList(
            [nn.GRU(input_size, hidden_size=self.dim_per_head, num_layers=layer, batch_first=True, dropout=dropout) for _ in range(para * span_count)])

        self.h_0 = nn.Parameter(torch.zeros(para, layer, batch_size, self.dim_per_head)) #num_layer = 2
        self.gate = nn.Linear(self.dim_per_head * 2, self.dim_per_head) # 两层同一个gate
        self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax(dim=-1)
        # self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_size, input_size)


    def forward_span(self, input, span, model_index):

        _index = list(range(input.shape[1]))
        reshape_index = [_index[i::span] for i in range(span)]
        # pivot_len = len(_index) % span * (span + 1) #0-21/4=5.2: 0,5,10,14,18(pivot10)
        start_index = [0] + torch.tensor([len(x) for x in reshape_index]).cumsum(0).tolist()[:-1]
        reshape_index = [_x for x in reshape_index for _x in x]
        reshape_back_index = torch.tensor(reshape_index).sort()[1].tolist()
        reshape_form = torch.arange(len(input)).unsqueeze(1)
        input = input[reshape_form, reshape_index]

        h_t_combined = []  # List to store combined hidden states at each time step
        hidden_span_one, hidden_span_two = self.h_0 # 2*h
        self.h_pre_one, self.h_pre_two = self.h_0 #2*bat*h
        for i in range(input.size(1)): # seq_len
            if i in start_index: # 2024-4-14 这里有点问题, 没有重置skip和pool的连接计数
                hidden_span_one, hidden_span_two = self.h_0 # 2*h # 每个分支起点都重置为0
            _, hidden_span_one = self.rnn_span[model_index](input[:, i:i+1, :], hidden_span_one) #2*bat*h
            _, hidden_span_two = self.rnn_span[model_index+1](input[:, i:i+1, :], hidden_span_two)
            # if 判断放到rnn之后,确保最后一个rnn step的output也被处理
            if not (i+1) % self.skip_span: # 冲突时, 先skip连接再同伴交流
                hidden_span_one = hidden_span_one + self.h_pre_one #逐位置 #2*bat*h
                self.h_pre_one = hidden_span_one
                hidden_span_two = hidden_span_two + self.h_pre_two #逐位置 #2*bat*h
                self.h_pre_two = hidden_span_two
            if (i+1) % self.para_pooling_span: # 同伴交流, 目前只适配了para_pooling_span=1的情形
                combined_hidden = torch.cat((hidden_span_one, hidden_span_two), dim=-1)
                gate_output = self.sigmoid(self.gate(combined_hidden))
                h_t = gate_output * hidden_span_one + (1 - gate_output) * hidden_span_two
                hidden_span_one = hidden_span_one = h_t # 同伴交流
            h_t_combined.append(hidden_span_one[1]) # list of bat*h, 假定同伴交流前后信息都汇聚在one
        output_span = torch.stack(h_t_combined, dim=1) #bat*step*h
        output_span = output_span[reshape_form, reshape_back_index]
        return output_span


    def forward(self, input, **kwargs):
        # rnn with span 1, 3, 5
        out_span = []
        for index, span in enumerate((1, 3, 5)):
            _out_span = self.forward_span(input, span, index * 2)
            out_span.append(_out_span)
        output = self.linear(torch.cat(out_span, dim=-1)) #bat*step*h
        return output


class MultidilaCNN(nn.Module):
    """"unidirection cnn"""
    def __init__(self, input_size, dilation=[1,2,3], dropout=0.1):
        super().__init__() #lstm512+20+20, chann128
        assert input_size % len(dilation) == 0 # 768 // 3 = 256
        self.input_size = input_size # input_size
        self.dim_per_head = input_size // len(dilation)
        self.dilation = dilation

        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(self.input_size, self.dim_per_head, kernel_size=1),
            nn.GELU()
        )

        self.convs = nn.ModuleList([
                nn.Sequential(
                # nn.ZeroPad2d((2*d,0,2*d,0)), # 右下角汇聚卷积 # left/right/top/bottom
                nn.ZeroPad2d((2*d,0,2*d,0)), # 2024-5-4 发现弄错了公式卷积核的大小为dila*(kernel-1) + 1,所以零填充应该是dila*(kernel-1),好像没错,(kernel-1)就是2
                nn.Conv2d(self.dim_per_head, self.dim_per_head, kernel_size=3, groups=self.dim_per_head, dilation=d, padding=0)
                ) for d in dilation
                ])

    def forward(self, input, **kwargs): # [8, 7, 7, 20+20+512])
        # cnn with dilation 1, 2, 3
        # if __builtins__["train_step"] == 8 and __builtins__["train_iter"] == 8:
        #     torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
        input = input.permute(0, 3, 1, 2).contiguous()
        input = self.base(input)

        output = []
        for conv in self.convs:
            input = conv(input) # [8, 128, 7, 7])
            input = F.gelu(input)
            output.append(input)
        output = torch.cat(output, dim=1) #dim=feature_channel, [8, 128, 7, 7]
        diag_mask = torch.eye(output.size(-1)).to(output.device).ne(0)
        output = output.masked_select(diag_mask).view(output.shape[:-1]).transpose(-1,-2) #bat*step*h
        # torch.cuda.empty_cache()
        return output# 汇聚到对角线 n*n -> n


# 受到w2ner启发,cln由于是影响传递方向是顺时针direction)默认上三角是后对前任务, 下三角是前对后任务,,, 这里有两个不同direction的三角cln,使得上三角也变长前对后影响,,另外为了区分两个cln,这里用上了self.hidden_dense /self.hidden_units来对cond=input先各自做个投影,然后各自进一步投影回input_size(两个mlp. w2ner只用了后边的一个mlp)
class ConditionalLayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super().__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center #均值
        self.scale = scale #标准差
        self.conditional = conditional
        self.hidden_units = hidden_units # 条件hj要不要先投影一下, #条件投影的中间维度
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12 #方差上的噪声
        self.input_dim = input_dim
        self.cond_dim = cond_dim # 条件的input维度,一般等于input_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim)) #条件b
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim)) #条件W

        if self.conditional:
            if self.hidden_units is not None: # 也即条件先过个投影
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
                cond_dim = self.hidden_units
            if self.center:
                self.beta_dense = nn.Linear(in_features=cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights() # 类似bert的predict mlm任务头中的init_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None: # hidden_unit_size = 256 ; input(512)
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal_(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight) #下划线表示inplace操作

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0) # 缩放校正应该初始化1

    def forward(self, inputs, cond=None):
        cond = inputs
        inputs = inputs.unsqueeze(2) # input[8, 7, 1, 512]
        if self.conditional: #重新计算条件w(gamma和条件b(beta)
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)): # 兼容不同cond,这段代码应该是他复制过来的
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1), # cond[8, 1, 7, 512]

            if self.center:
                beta = self.beta_dense(cond) + self.beta # self.beta并没更新,却依然设置了self.beta参数
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma # self.beta并没更新,估计是为了兼容各种需求
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
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma #缩放校正
        if self.center:
            outputs = outputs + beta # 标准化和条件b必须同时存在(偏差校正)
        return outputs


class MultiUniGCN(nn.Module):
    def __init__(self, input_size, edge_type=3, dropout=0.1) -> None:
        super().__init__()
        self.num_layers = 3

        # GCN
        self.gcn = nn.ModuleList()
        for _layer_index in range(self.num_layers):
            gcn_layer = GraphConvolution(in_features=input_size,
                                     out_features=input_size, ##input feature = input_size_wv100, #output feature = hidden_size_768
                                     edge_types=edge_type,
                                     dropout=0.5 if _layer_index != self.num_layers - 1 else 0)
            self.gcn.append(gcn_layer)

        # HighWay
        self.highway = nn.ModuleList()
        for _layer_index in range(self.num_layers):
            hw_layer = HighWay(size=input_size, dropout_ratio=0.5) #input/output feature = input_size_wv100, 也即highway残差mlp由于add行为导致无法改变,导致不改变input的特征维度
            self.highway.append(hw_layer)

        # GCN_UNI
        self.gcn_uni = nn.ModuleList()
        edge_type_uni = 1
        for _layer_index in range(self.num_layers):
            gcn_layer_uni = GraphConvolution(in_features=input_size,
                                     out_features=input_size, ##input feature = input_size_wv100, #output feature = hidden_size_768
                                     edge_types=edge_type_uni,
                                     dropout=0.5 if _layer_index != self.num_layers - 1 else 0)
            self.gcn_uni.append(gcn_layer_uni)

    def forward(self, input, *, edge_matrix, **kwargs): #位置参数+位置默认参数+可变参数+命名关键字参数+命名关键字默认参数+关键字参数
        edge_matrix = edge_matrix.to(input.device)
        sent_edge = edge_matrix[:,:-1]
        uni_edge = edge_matrix[:,-1:] #保持shape不降维
        for _layer_index in range(self.num_layers):
            input = self.gcn[_layer_index](input, sent_edge) + self.highway[_layer_index](input)
            self.gcn_uni[_layer_index](input, uni_edge)
        output = input
        return output