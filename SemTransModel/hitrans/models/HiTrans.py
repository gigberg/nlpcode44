import torch
import torch.nn as nn
from transformers import BertModel
from models.Trans import TransformerEncoder


class HiTrans(nn.Module):
    def __init__(self, hidden_dim, emotion_class_num, d_model, d_ff, heads, layers, dropout=0, input_max_length=512):
        super(HiTrans, self).__init__()
        self.input_max_length = input_max_length
        self.bert = BertModel.from_pretrained('/home/zhoujiaming/.pretrained/bert-base-uncased')
        self.encoder = TransformerEncoder(d_model, d_ff, heads, layers, 0.1)

        self.emo_output_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emotion_class_num)
        )

        self.spk_output_layer = SpeakerMatchLayer(hidden_dim, dropout)

    def forward(self, dia_input, cls_index, mask): # 8*462, 8*24, 8*24(emo_mask用来传给第二层句间的self attention作为sent hidden的padding attention mask)
        bert_outputs = []
        for i in range((dia_input.size(1) - 1) // self.input_max_length + 1): # 拼接第一层两个bert的输出
            cur_input = dia_input[:, i * self.input_max_length:(i + 1) * self.input_max_length]
            cur_mask = cur_input.ne(0)
            bert_output, _ = self.bert(cur_input, cur_mask)
            bert_outputs.append(bert_output)
        bert_outputs = torch.cat(bert_outputs, dim=1) # 8*462*768

        bert_outputs = bert_outputs[torch.arange(bert_outputs.size(0)).unsqueeze(1), cls_index.long()] # 8*462*768[8*1(自动广播到8*24), 8*24] -> 8*24*768, 相当于给出所有要提取的元素的行坐标子集和列坐标子集(一一对应,子集可以flatten成一维,也可保持二维不处理),指定要slice提取出的位置(此处每行要提取的列不同,不能直接行用:冒号就行)
        bert_outputs = bert_outputs * mask[:, :, None].float() # 输入执行padding操作(也可不?毕竟有attention_mask的话pading无所谓是否为0)

        bert_outputs = self.encoder(bert_outputs, mask) #调用第二层层transformer模型, 8*24*768, 8*24 -> 8*24*768
        emo_output = self.emo_output_layer(bert_outputs)
        spk_output = self.spk_output_layer(bert_outputs)

        return emo_output, spk_output


class SpeakerMatchLayer(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(SpeakerMatchLayer, self).__init__()
        self.mpl1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.mpl2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.biaffine = Biaffine(hidden_dim // 2, 2)

    def forward(self, x):
        x1 = self.mpl1(x) #使用两个mpl增加了一些网络复杂度拟合能力,另外似乎不除以二也行
        x2 = self.mpl2(x)
        output = self.biaffine(x1, x2)
        return output


class Biaffine(nn.Module):

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True): # n_in = 768 //2 = 384
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y))) #由于权重shape随参数变化,所以不能直接调用需要固定长度的nn.Linear,而是自己使用nn.Parameter构建权重tensor, 并且双线性并不在pytorch的预定义nn.Module中（是预定义的，李京烨在手动实现torch源码）,从这个角度来看也需要自行编写权重部分的代码
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    # def extra_repr(self): # 未用到
    #     s = f"n_in={self.n_in}, n_out={self.n_out}"
    #     if self.bias_x:
    #         s += f", bias_x={self.bias_x}"
    #     if self.bias_y:
    #         s += f", bias_y={self.bias_y}"

    #     return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1) # 增广权重矩阵
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y) # batch8, utt_count17, hidden768//2+1
        s = s.permute(0, 2, 3, 1)
        return s # 8, 17, 17, 2(这种输出格式很适合两两分类任务)
