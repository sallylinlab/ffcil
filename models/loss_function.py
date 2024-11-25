import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.autograd import Variable
import math

class focal_loss(nn.Module):    
    def __init__(self, args, alpha=0.25, gamma=3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)      
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """

        super(focal_loss,self).__init__()
        num_classes = args.episode_way + args.low_way
        self.size_average = size_average
        # if isinstance(alpha,list):
        #     assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
        #     print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
        #     self.alpha = torch.Tensor(alpha)
        # else:
        #     assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
        #     print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
        #     self.alpha = torch.zeros(num_classes)
        #     self.alpha[0] += alpha
        #     self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.alpha = torch.zeros(num_classes)
        self.alpha[:args.episode_way] += alpha
        self.alpha[args.episode_way:] += (1-alpha)
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算        
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数        
        :param labels:  实际类别. size:[B,N] or [B]        
        :return:
        """        
        # assert preds.dim()==2 and labels.dim()==1        
        preds = preds.view(-1,preds.size(-1))        
        self.alpha = self.alpha.to(preds.device)        
        preds_softmax = F.softmax(preds, dim=1) # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)        
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )        
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))        
        self.alpha = self.alpha.gather(0,labels.view(-1))        
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())        
        if self.size_average:        
            loss = loss.mean()        
        else:            
            loss = loss.sum()        
        return loss

class LGMLoss(nn.Module):
    """
    Refer to paper:
    Weitao Wan, Yuanyi Zhong,Tianpeng Li, Jiansheng Chen
    Rethinking Feature Distribution for Loss Functions in Image Classification. CVPR 2018
    re-implement by yirong mao
    2018 07/02
    """
    def __init__(self, num_classes, feat_dim, alpha):
        super(LGMLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.alpha = alpha

        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.log_covs = nn.Parameter(torch.zeros(num_classes, feat_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        log_covs = torch.unsqueeze(self.log_covs, dim=0)


        covs = torch.exp(log_covs) # 1*c*d
        tcovs = covs.repeat(batch_size, 1, 1) # n*c*d
        diff = torch.unsqueeze(feat, dim=1) - torch.unsqueeze(self.centers, dim=0)
        wdiff = torch.div(diff, tcovs)
        diff = torch.mul(diff, wdiff)
        dist = torch.sum(diff, dim=-1) #eq.(18)


        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.alpha)
        y_onehot = y_onehot + 1.0
        margin_dist = torch.mul(dist, y_onehot)

        slog_covs = torch.sum(log_covs, dim=-1) #1*c
        tslog_covs = slog_covs.repeat(batch_size, 1)
        margin_logits = -0.5*(tslog_covs + margin_dist) #eq.(17)
        logits = -0.5 * (tslog_covs + dist)

        cdiff = feat - torch.index_select(self.centers, dim=0, index=label.long())
        cdist = cdiff.pow(2).sum(1).sum(0) / 2.0

        slog_covs = torch.squeeze(slog_covs)
        reg = 0.5*torch.sum(torch.index_select(slog_covs, dim=0, index=label.long()))
        likelihood = (1.0/batch_size) * (cdist + reg)

        return logits, margin_logits, likelihood

class LGMLoss_(nn.Module):
    def __init__(self, num_classes, feat_dim, alpha=0.1, lambda_=0.01):
        super(LGMLoss_, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.lambda_ = lambda_
        self.means = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.means, gain=math.sqrt(2.0))

    def forward(self, feat, labels=None):
        batch_size= feat.size()[0]

        XY = torch.matmul(feat, torch.transpose(self.means, 0, 1))
        XX = torch.sum(feat ** 2, dim=1, keepdim=True)
        YY = torch.sum(torch.transpose(self.means, 0, 1)**2, dim=0, keepdim=True)
        neg_sqr_dist = -0.5 * (XX - 2.0 * XY + YY)

        if labels is None:
            psudo_labels = torch.argmax(neg_sqr_dist, dim=1)
            means_batch = torch.index_select(self.means, dim=0, index=psudo_labels)
            likelihood_reg_loss = self.lambda_ * (torch.sum((feat - means_batch)**2) / 2) * (1. / batch_size)
            return neg_sqr_dist, likelihood_reg_loss, self.means

        labels_reshped = labels.view(labels.size()[0], -1)

        if torch.cuda.is_available():
            ALPHA = torch.zeros(batch_size, self.num_classes).cuda().scatter_(1, labels_reshped, self.alpha)
            K = ALPHA + torch.ones([batch_size, self.num_classes]).cuda()
        else:
            ALPHA = torch.zeros(batch_size, self.num_classes).scatter_(1, labels_reshped, self.alpha)
            K = ALPHA + torch.ones([batch_size, self.num_classes])

        logits_with_margin = torch.mul(neg_sqr_dist, K)
        means_batch = torch.index_select(self.means, dim=0, index=labels)
        likelihood_reg_loss = self.lambda_ * (torch.sum((feat - means_batch)**2) / 2) * (1. / batch_size)
        return self.means, logits_with_margin, likelihood_reg_loss

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss



def GCE(x, y):
        """
        The function implements the forward pass for the GB loss.
        :param x: Predictions
        :param y: Ground truth labels
        """
        x1 = x.clone()
        x1[range(x1.size(0)), y] = -float("Inf")
        x_gt = x[range(x.size(0)), y].unsqueeze(1)
        x_topk = torch.topk(x1, 5, dim=1)[0]  # k=5 Negative classes to focus on, its a hyperparameter
        x_new = torch.cat([x_gt, x_topk], dim=1)

        return F.cross_entropy(x_new, torch.zeros(x_new.size(0)).cuda().long())




    
class RingLoss(nn.Module):
    """
    Refer to paper
    Ring loss: Convex Feature Normalization for Face Recognition
    """
    def __init__(self, type='L2', loss_weight=1.0):
        super(RingLoss, self).__init__()
        self.radius = nn.Parameter(torch.Tensor(1)).cuda()
        self.radius.data.fill_(-1)
        self.loss_weight = loss_weight
        self.type = type

    def forward(self, x):
        x = x.pow(2).sum(dim=1).pow(0.5)
        if self.radius.item() < 0: # Initialize the radius with the mean feature norm of first iteration
            self.radius.data.fill_(x.mean().item())
        if self.type == 'L1': # Smooth L1 Loss
            loss1 = F.smooth_l1_loss(x, self.radius.expand_as(x)).mul_(self.loss_weight)
            loss2 = F.smooth_l1_loss(self.radius.expand_as(x), x).mul_(self.loss_weight)
            ringloss = loss1 + loss2
        elif self.type == 'auto': # Divide the L2 Loss by the feature's own norm
            diff = x.sub(self.radius.expand_as(x)) / (x.mean().detach().clamp(min=0.5))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        else: # L2 Loss, if not specified
            diff = x.sub(self.radius.expand_as(x))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        return ringloss