import torch
import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks):
        super(MultiTaskLoss, self).__init__()
        self.sigma = nn.Parameter(torch.ones(num_tasks), requires_grad=True) # shape=(2,), σ标准差,实际操作中只变为随机初始化的参数?因为没有让CE输出raw loss向量以便求标准差

    def forward(self, *losses): # 变长参数*args, 形如tuple(tensor[一个数], tensor[一个数]),losses[2.2435, 0.7413]
        losses = torch.cat([loss.unsqueeze(0) for loss in losses]) # unsqueeze()没啥效果,对于shaep(1,)输入来说. 此处输出形如tensor[一个数, 一个数]
        loss = (0.5 / torch.pow(self.sigma, 2)) * losses
        return loss.sum() + self.sigma.log().sum() # 对数公式中的乘法化为加法sum()


class RMSLELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, pred, actual): # log不能处理负数坐标值, 要么过一趟relu(没啥用,q->a的旋转行行为也会导致负数), 要么移动log到平方外边
        # return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
        assert not pred.isnan().any()
        return torch.log(self.mse(pred, actual) + 1)