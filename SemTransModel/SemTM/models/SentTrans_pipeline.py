import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Trans import BertStyleEmbeddings, TransformerEncoder, MultiUniAtten, PositionwiseFeedForward
from models.Loss import RMSLELoss


class SentTransModel(nn.Module):
    """loss,adaw和f1都放在模型外边的Trainer类中, bert_ids(word2vec) -> sent_hidden(masked) -> sent_emb(masked) -> consist_out, loss

    Input: bert_ids or word2vec
    Output: consist_out, loss
    """
    def __init__(self, config):
        super().__init__()
        self.embedding = BertStyleEmbeddings(config)
        self.dialog_encoder = DialogueEncoder(config)
        self.sent_encoder = SentenceEncoder(config)
        self.spk_output_layer = SpeakerMatchLayer(config.flow_size)
        self.dialog_flow = DialogueFlow(config)
        self.consist_classifier = ConsistClassifier(config)
        self.spk_criterion = nn.CrossEntropyLoss(reduction='mean')
        # self.consist_criteria = nn.CrossEntropyLoss()
        # self.consist_criteria =  nn.BCELoss()

    def pretrained_embedding(self, bert_input, word_subword, cls_index):
        token_type_ids = torch.zeros_like(word_subword.long())[...,0]
        for i, _cls in enumerate(cls_index):
            k = 0
            for j in torch.split(_cls, 2):
                token_type_ids[i, k:j[0]+1] = 1
                k=j[1]+1
                token_type_ids[i, j[0]+1:k] = 2
            start, end = _cls[_cls.ne(0)][-2:] #start_倒数第2个cls, end_倒数第1个cls
            token_type_ids[i, start:end+1] = 3

        pad_mask = word_subword.sum(-1).ne(0).unsqueeze(-1) #pad_mask = [bat,word,1]
        word_reps = self.embedding(bert_input, token_type_ids, word_subword)

        return word_reps, pad_mask


    def shuffled_spk(self, sent_emb, cls_index):
        shuffled_indices = torch.randperm(sent_emb.size(1)).to(sent_emb.device)
        # shuffled_indices = torch.arange(sent_emb.size(1)).to(sent_emb.device) # no shuffle
        shuffled_sent_emb = sent_emb.index_select(dim=1, index=shuffled_indices)

        _label = cls_index.clone().detach()
        _label = _label[:,shuffled_indices]
        _label[:,0::2], _label[:,1::2] = 1, 2
        _label = _label * cls_index.ne(0)
        _label_matrix = torch.einsum('bx,by->bxy', _label, _label).long() # 外积, 1,2,4,0

        spk_mask = _label_matrix.ne(0)
        spk_label = _label_matrix % 3 % 2  # 1, 4 ->1, 2->0

        return shuffled_sent_emb, spk_label, spk_mask


    def forward(self, bert_input, word_subword, sent_entity, cls_index, edge_matrix, consist_label):
        word_reps, pad_mask = self.pretrained_embedding(bert_input, word_subword, cls_index)
        sent_hidden, entity_emb = self.dialog_encoder(word_reps, sent_entity, cls_index, edge_matrix, pad_mask) #pad_mask = [bat,word,1]
        sent_emb = self.sent_encoder(sent_hidden)
        assert not sent_emb.isnan().any()

        # spk loss
        shuffled_sent_emb, spk_label, spk_mask = self.shuffled_spk(sent_emb, cls_index)
        spk_output = self.spk_output_layer(shuffled_sent_emb) # pask masked

        spk_output = spk_output[spk_mask] # flatten_n*2 # flatten, cross entropy flatten_n batch, c=2
        spk_label = spk_label[spk_mask]
        spk_loss = self.spk_criterion(spk_output, spk_label)

        # flow loss
        sent_a_hat_n, sent_a_n, flow_loss = self.dialog_flow(sent_emb, entity_emb) # both [bat, sent/entity, hid]

        # consist loss
        consist_out = self.consist_classifier(sent_a_n.clone().detach(), sent_a_hat_n.clone().detach())
        # consist_out = self.consist_classifier(sent_a_n, sent_a_hat_n)
        assert consist_out.shape[0] == consist_label.shape[0] #由于qq_a中的单turn数据被丢弃了,导致这边交叉熵出问题, 没有的时候就直接用qa预测a_hat吧

        return consist_out, spk_loss, flow_loss, spk_output, spk_label


class DialogueEncoder(nn.Module):
    """
    Input: bert_ids or word2vec_vec
    Output: bert_hidden or rnn hidden (dialogue hidden, masked), sent_mask, entity_emb"""
    def __init__(self, config): # word2vec.input_dim = 100
        super().__init__()
        self.dia_encoder = TransformerEncoder(config)

    def entity_pooling(self, dialog_hidden, sent_entity):
            sent_length = sent_entity.size(1) # sent_entity[bat, sent, ent/subword]
            sent_word = dialog_hidden.unsqueeze(1).expand(-1, sent_length, -1, -1)
            sent_entity = sent_word * sent_entity.unsqueeze(-1)
            entity_emb, _ = torch.max(sent_entity, dim=2) # multi entity to one
            return entity_emb

    def forward(self, word_reps, sent_entity, cls_index, edge_matrix, pad_mask):
        word_reps = word_reps * pad_mask
        dialog_hidden = self.dia_encoder(word_reps, edge_matrix, pad_mask) #dialog_hidden has been masked
        entity_emb = self.entity_pooling(dialog_hidden, sent_entity)

        sent_hidden =  dialog_hidden[torch.arange(len(word_reps))[:, None], cls_index] #cls_index中的pad值会选择到非0的向量，所以需要另外mask一下
        sent_hidden = sent_hidden * cls_index.ne(0).unsqueeze(-1) #确保后边能计算出mask(也可以直接传mask参数, 不过会增加函数参数数量)

        return sent_hidden, entity_emb


class SentenceEncoder(nn.Module):
    """
    Input: sent_hidden
    Output: sent_emb
    """
    def __init__(self, config): # word2vec.input_dim = 100
        super().__init__()
        d_model = config.hidden_size
        d_ff = config.ffnn_hid_size
        self.sent_enc = config.sent_enc
        self.module_dict = {
            'SentIdentity': nn.Identity(),
            'SentUniAtten': MultiUniAtten(d_model, head_count=config.head_count, dropout=0.1)
            }
        self.sent_encoder = self.module_dict[config.sent_enc]
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.w_2 = nn.Linear(d_model, config.flow_size)
        self.layernorm = nn.LayerNorm(config.flow_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.attn_dropout)

    def forward(self, sent_hidden): # [bat,word,1]
        sent_mask = sent_hidden.sum(-1, keepdim=True).ne(0) # # masked
        if self.sent_enc == 'SentIdentity':
            return sent_hidden * sent_mask
        sent_emb = self.sent_encoder(sent_hidden, sent_mask)
        assert not sent_emb.shape[1] % 2 # 确保对号长度为为偶数,这都是数据预处理阶段没提出奇数个对话,以及没剔除不足4句的对话导致的

        out = self.w_2(F.gelu(self.feed_forward(sent_emb))) #外边接个relu确保都在第一象限
        out = out * sent_mask
        return self.layernorm(out)


class DialogueFlow(nn.Module):
    """
    Input: sent_emb_paded, entity_emb_paded
    Output: sent_a_hat_n, sent_a_n, flow_loss
    """
    def __init__(self, config):
        super().__init__()
        self.module_dict = {
            'FlowCellQA': FlowCellQA(config),
            'FlowCellQQ': FlowCellQQ(config),
            'FlowCellQE': FlowCellQE(config),
            'FlowRemove': FlowRemove(config)
            }
        self.dia_encoder = self.module_dict[config.flow_type]

    def forward(self, sent_emb, entity_emb): # both [bat, sent/entity, hid]
        sent_a_hat_n, sent_a_n, flow_loss = self.dia_encoder(sent_emb, entity_emb)
        return sent_a_hat_n, sent_a_n, flow_loss


class FlowRemove(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, sent_emb, entity_emb):
        sent_mask = sent_emb[:,1::2,0].ne(0)
        sent_a_hat_n = sent_emb[range(len(sent_mask)), sent_mask.sum(-1)-2]
        sent_a_n = sent_emb[range(len(sent_mask)), sent_mask.sum(-1)-1]
        flow_loss = torch.tensor(0.)
        return sent_a_hat_n, sent_a_n, flow_loss


class FlowCellBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.flow_trans = nn.Linear(config.flow_size, config.flow_size) # d_model, d_ff, dropout=0.1
        self.flow_criteria = nn.MSELoss(reduction='sum') # Loss层一般都不含待训练的参数


class FlowCellQA(FlowCellBase): #原论文中W=ones
    """WQ -> A, W^-1A -> Q[也即WA->WWQ], assume Q and A in different coordinate, orthogonal rotation matrix
    Input: sent_emb_masked, entity_emb_masked, masked_zero because need to compute mask tensor
    Output: sent_a_hat_n, sent_a_n, flow_loss
    """
    def __init__(self, config):
        super().__init__(config)

    def forward(self, sent_emb, entity_emb): # both [bat, sent/entity, hid]
        _q_in, _q_target, qa_last = [], [], []
        _a_in, _a_target = [], []
        for dia in sent_emb:
            dia = dia[dia.sum(-1).ne(0)] #去除pad语句
            _q_in.extend(dia[0:-2:2])
            _q_target.extend(dia[1:-2:2])
            _a_in.extend(dia[1:-2:2])
            _a_target.extend(dia[2::2])
            qa_last.append(dia[-2:])
        #a->q
        sent_q_hat = torch.stack(_a_in)
        target = self.flow_trans(torch.stack(_a_target))
        flow_loss_a_in = self.flow_criteria(sent_q_hat, target.clone().detach()) # sum, tensor scalar
        #q->a
        sent_a_hat = self.flow_trans(torch.stack(_q_in))
        target = torch.stack(_q_target)
        flow_loss_q_in = self.flow_criteria(sent_a_hat, target.clone().detach())
        #consist data
        qa_last = torch.stack(qa_last)
        sent_a_hat_n = self.flow_trans(qa_last[:,0,:])
        sent_a_n = qa_last[:,1,:]
        flow_loss= flow_loss_q_in + flow_loss_a_in
        return sent_a_hat_n, sent_a_n, flow_loss


class FlowCellQQ(FlowCellBase):
    """Q  + WA -> Q, A + W^-1Q -> A, assume Q and A in different coordinate, orthogonal rotation matrix
    尝试一下不用矩阵掩码的方式,直接rnncell类似的for循环,不过要用batch内的pack(这里不用unpack), 不追求速度但追求代码可读性"""
    def __init__(self, config):
        super().__init__(config)
        # 正交矩阵旋转平移太复杂, 而且限制了模型表达能力, 先用线性映射代替一下

    def forward(self, sent_emb, entity_emb): # both [bat, sent/entity, hid]
        # sent_mask = sent_emb[:,::2,0].ne(0)
        # assert sent_mask.shape[-1] > 1 # 一个batch中至少有一个两句话，否则flow_sent_mask=[全false], 从而没有flow_loss导致,必然报错

        # 先处理sent_a_hat_n, 没有的直接去掉, 所以返回长度可能小于batch
        _q_in, _q_a, _q_target = [], [], [] # 为flow模型重新组织数据集结构
        _a_in, _a_q, _a_target, aqa_last = [], [], [], [] #a_in, a_in_q, a_hat = aqa_last
        for index, dia in enumerate(sent_emb):
            sent_lenght = dia[:,0].ne(0).sum() #去除pad语句
            for index in range(0, sent_lenght, 2):
                if not index+2 < sent_lenght: #也即只有一轮对话时跳过该条dialogue数据集, 因为连a_hat_n都生成不了,没法做一致性任务. 此举保证一定有q_hat,a_hat_n,但3turn才有a_hat_flow
                    #由于qq_a中的单turn数据被丢弃了,导致这边交叉熵出问题, 没有的时候就直接用qa预测a_hat吧
                    aqa_last.append(torch.stack([dia[0], dia[1], dia[1]]))
                    break
                # q1+a1->q2, 单条数据
                if index+3 < sent_lenght: #index+3,表明有两轮对话q,a,q,a, 有两轮对话才有必要处理
                    q1, a1, q2, a2= dia[index:index+4]
                    _q_in.append(q1)
                    _q_a.append(a1)
                    _q_target.append(q2)
                    # ----
                    if index+3 == sent_lenght - 1: # 表明a2就是an, an单独保存要传给历史一致性
                        aqa_last.append(torch.stack([a1, q2, a2]))
                        break
                    _a_in.append(a1)
                    _a_q.append(q2)
                    _a_target.append(a2)

        sent_q_hat = self.flow_trans((torch.stack(_q_in))) + torch.stack(_q_a) # expression to modele the dialogue flow, 可以输入前n轮的q,然后单独mask提取最后一轮的q
        _q_target = self.flow_trans(torch.stack(_q_target))
        flow_loss_q_in = self.flow_criteria(sent_q_hat, _q_target.clone().detach())

        flow_loss_a_in = flow_loss_q_in.new_zeros(()) # 0, tensor scalar
        if _a_in:
            sent_a_hat = torch.stack(_a_in) + self.flow_trans(torch.stack(_a_q))
            flow_loss_a_in = self.flow_criteria(sent_a_hat, torch.stack(_a_target).clone().detach())

        aqa_last = torch.stack(aqa_last)
        sent_a_hat_n = aqa_last[:,0,:] + self.flow_trans(aqa_last[:,1,:])
        sent_a_n = aqa_last[:,2,:]

        flow_loss= flow_loss_q_in + flow_loss_a_in
        return sent_a_hat_n, sent_a_n, flow_loss


class FlowCellQE(FlowCellBase):
    """Q + WE -> A_hat, assume Q and A in the same coordinate, no need to retote coordinate"""
    def __init__(self, config):
        super().__init__(config)
        # self.We = nn.Linear(config.flow_size, config.flow_size) #bias存在导致self.We(entity_a)中mask=0位置的输出不畏0, 从而没有移除mask的loss贡献

    def forward(self, sent_emb, entity_emb): # both [bat, sent/entity, hid]
        sent_q = sent_emb[:, 0::2, :] #逢单取q
        sent_a = sent_emb[:, 1::2, :]
        entity_a = entity_emb[:, 1::2, :] # dataload.py中已经确保answer都有entity
        sent_mask = sent_emb[:,::2].sum(-1).ne(0)

        last_sent_mask = torch.zeros_like(sent_mask)
        last_sent_mask[range(len(sent_mask)), sent_mask.sum(-1)-1] = True
        flow_sent_mask = sent_mask > last_sent_mask

        sent_a_hat = sent_q + self.flow_trans(entity_a) # expression to modele the dialogue flow
        assert sent_a_hat.shape == sent_a.shape
        flow_loss = self.flow_criteria(sent_a_hat * flow_sent_mask.unsqueeze(-1), sent_a.clone().detach() * flow_sent_mask.unsqueeze(-1)) #

        sent_a_hat_n = sent_a_hat.masked_select(last_sent_mask.unsqueeze(-1)).reshape(-1, sent_a_hat.shape[-1])
        sent_a_n = sent_a.masked_select(last_sent_mask.unsqueeze(-1)).reshape(-1, sent_a.shape[-1])

        return sent_a_hat_n, sent_a_n, flow_loss


class ConsistClassifier(nn.Module):
    """
    Input: sent_a_n, sent_a_hat_n
    Output: consist_out
    """
    def __init__(self, config): # word2vec.input_dim = 100
        super().__init__()
        self.module_dict = {
            'MLPClassifier': MLPClassifier(config.flow_size, n_out=2),
            'BiaffineClassifier': BiaffineClassifier(config.flow_size, n_out=2),
            'SoftmaxClassifier': SoftmaxClassifier(config.flow_size)
            }
        self.classifier = self.module_dict[config.classifier]
    def forward(self, sent_a_n, sent_a_hat_nb): # both [bat, sent/entity, hid]注除法运算
        consist_out = self.classifier(sent_a_n, sent_a_hat_nb)
        return consist_out # bat, 2


class BiaffineClassifier(nn.Module):
    """x, y must be 2d tensor[bat*hid]"""
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        # nn.init.xavier_uniform_(self.linear.weight) inplace, replace but not modify
        self.weight = nn.Parameter(weight, requires_grad=True)


    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1) # 增广权重矩阵
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)

        if len(x.shape) == 2:
            return torch.einsum('bi,oij,bj->bo', x, self.weight, y)

        s = torch.einsum('bxi,oij,byj->bxyo', x, self.weight, y)
        return s # 8, 17, 17, 2(这种输出格式很适合两两分类任务)


class MLPClassifier(nn.Module):
    """x, y must be 2d tensor[bat*hid]"""
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super().__init__()
        self.mlp = nn.Linear(n_in*2, n_out)

    def forward(self, x, y):
        s = self.mlp(torch.cat((x,y), dim=-1))
        return s


class SoftmaxClassifier(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.fc1 = nn.Linear(n_in * 2, n_in)  # Input size: 3 (vector dimension), Output size: 2
        self.fc2 = nn.Linear(n_in, 2)  # Input size: 2, Output size: 1

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)  # Concatenate the two input vectors
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = F.softmax(self.fc2(x)) #cross entropy包含softmax, softmax导数梯度较小,两个一起可能会梯度消失
        return x


class SpeakerMatchLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0):
        super(SpeakerMatchLayer, self).__init__()
        self.mpl1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.1), # 非线性
        )

        self.mpl2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.biaffine = BiaffineClassifier(hidden_dim // 2, 2)

    def forward(self, x):
        x1 = self.mpl1(x) #使用两个mpl增加了一些网络复杂度拟合能力,另外似乎不除以二也行
        x2 = self.mpl2(x)
        output = self.biaffine(x1, x2)
        return output