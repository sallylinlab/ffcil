import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.Network import MYNET as Net
import numpy as np
import tensorboardX as tbx

class MYNET(Net):

    def __init__(self, args, mode=None):
        super().__init__(args,mode)


        hdim=self.num_features
        self.slf_attn = MultiHeadAttention(3, hdim, hdim, hdim, dropout=0.5)
        # self.classifier = nn.Sequential(
        #     nn.Conv2d(hdim*2, 64, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Conv2d(64, 64, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Conv2d(64, 1, 1),
        # )
        self.classifier = RelationNetwork(hdim)

        # self.projection = nn.Sequential(
        #     nn.Linear(self.num_features, self.num_features*2),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(self.num_features*2, self.num_features*2),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(self.num_features*2, self.num_features, bias=False)
        # )

    def forward(self, input):
        if self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            support_idx, query_idx = input
            logits = self._forward(support_idx, query_idx)
            return logits

    def forward_(self, support, query):
        emb_dim = support.size(-1)
        # get mean of the support
        proto = support.mean(dim=1)
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = query.shape[1]*query.shape[2]#num of query*way

        proto = proto.view((1, num_proto, emb_dim, 1, 1))
        proto = proto.repeat((num_query, 1, 1, 1, 1))

        query = query.view((num_query, 1, emb_dim, 1, 1))
        query = query.repeat((1, num_proto, 1, 1, 1))

        features = torch.cat([query, proto], dim=2)
        features = features.view((-1, emb_dim*2, 1, 1))

        output = self.classifier(features)
        output = output.view((num_query, num_proto))
        output = F.softmax(output, dim=1)

        return output

    def _forward(self, support, query):

        emb_dim = support.size(-1)
        # get mean of the support
        proto = support.mean(dim=1)
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = query.shape[1]*query.shape[2]#num of query*way

        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        query = query.view(-1, emb_dim).unsqueeze(1)

        proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
        proto = proto.view(num_batch*num_query, num_proto, emb_dim)


        # logits=F.cosine_similarity(query, proto, dim=-1)
        # logits=logits*self.args.temperature
        # return logits

        combined = torch.cat([proto, query], 1) # Nk x (N + 1) x d, batch_size = NK
        combined = self.slf_attn(combined, combined, combined)
        # a_proto = self.slf_attn(proto, proto, proto)

        # compute distance for all batches
        proto, query = combined.split(num_proto, 1)

        logits=F.cosine_similarity(query, proto, dim=-1)
        # logits=F.cosine_similarity(query, a_proto, dim=-1)
        logits=logits*self.args.temperature

        # return F.softmax(logits, dim=1)
        return logits

    def forward_metric_(self, support, query):
        support = support.view(-1, self.num_features)
        query = query.view(-1, self.num_features)
        # print(support.shape, query.shape)
        x = F.linear(F.normalize(query, p=2, dim=-1), F.normalize(support, p=2, dim=-1))
        x = self.args.temperature * x

        return x



class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        # print(attn.shape, v.shape)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, feat_size, hidden_size=8):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(feat_size*2, feat_size, kernel_size=1, padding=0),
                        nn.BatchNorm2d(feat_size, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(1))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(feat_size, feat_size, kernel_size=1, padding=0),
                        nn.BatchNorm2d(feat_size, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(1))
        self.fc1 = nn.Linear(feat_size, hidden_size*hidden_size)
        self.fc2 = nn.Linear(hidden_size*hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))
        return out