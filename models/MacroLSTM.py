import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd

class Pro_for_CCL(nn.Module):
    """
    Projection layer of the Cyclic Consistence Loss function
    """
    def __init__(self, in_dim=256, hidden_dim=512,out_dim=256):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,out_dim))

    def forward(self,x):
        return self.fc(x)


class Feq_Cross_Correlation(nn.Module):
    """
    Cross_Correlation based Frequency Enhanced module
    """
    def __init__(self, S_s, I_s, K, seq_len):
        super(Feq_Cross_Correlation, self).__init__()
        self.seq_len = seq_len

        self.group_list = [16, 31, 10, 10, 14, 22, 19, 5]
        self.group_projections = nn.ModuleList(nn.Linear(i, 1) for i in self.group_list)
        self.group_projections_v = nn.ModuleList(nn.Linear(i, 1) for i in self.group_list)

        self.I_s =I_s+len(self.group_list)
        self.query_projection = nn.Linear(S_s, S_s)
        self.key_projection = nn.Linear(self.I_s, self.I_s)
        self.value_projection = nn.Linear(self.I_s, self.I_s)
        self.K = K

    def group_Agg(self, queries1, keys, values,L):
        """
        Cross_Correlation based Group Aggregate module
        """
        group_list = [0]+self.group_list
        corr_group = []
        queries = queries1.repeat(1,1,8)
        values_groups = []
        for i in range(0, len(self.group_list)):
            keys_group=self.group_projections[i](keys[...,sum(group_list[:i+1]):sum(group_list[:i+1+1])])
            values_groups.append(self.group_projections_v[i](values[...,sum(group_list[:i+1]):sum(group_list[:i+1+1])]))
            q_fft = torch.fft.rfft(queries[...,i].unsqueeze(-1).permute(0, 2, 1).contiguous(), dim=-1)
            k_fft = torch.fft.rfft(keys_group.permute(0, 2, 1).contiguous(), dim=-1)
            res = q_fft * torch.conj(k_fft)
            corr = torch.fft.irfft(res, dim=-1)
            corr_group.append(corr)
        corr_group = torch.cat(corr_group,1)
        scale = 1. / math.sqrt(len(self.group_list))
        weight = torch.softmax(torch.mean(corr_group, -1) * scale, dim=-1)
        values_groups = torch.cat(values_groups,-1)
        values_g = torch.mul(weight.unsqueeze(1).repeat(1, L, 1), values_groups)

        return values_g

    def forward(self, queries, keys, values):
        org_queries = queries
        B, L, _ = queries.shape
        _, S, E = keys.shape
        scale = 1. / math.sqrt(E)

        queries = self.query_projection(queries)
        # Group Aggregate ===========================================================================#
        values_g = self.group_Agg(queries, keys, values, L)
        # ===========================================================================================#
        queries = queries.repeat(1, 1, self.I_s)
        keys = self.key_projection(torch.cat([keys,values_g],-1))
        values = self.value_projection(torch.cat([values, values_g],-1))

        # Cross-Correlation =========================================================================#
        q_fft = torch.fft.rfft(queries.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 1).contiguous(), dim=-1)
        v_fft = torch.fft.rfft(values.permute(0, 2, 1).contiguous(), dim=-1)

        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        # FeqEN======================================================================================#
        qk_ftt = torch.einsum("bel,bes->bsl", q_fft, k_fft)
        qk_ftt = torch.softmax(abs(qk_ftt), dim=-1)
        qk_ftt = torch.complex(qk_ftt, torch.zeros_like(qk_ftt))
        Vftt = torch.einsum("bsl,bds->bdl", qk_ftt, v_fft)
        values_ftt_out = torch.fft.irfft(Vftt / self.I_s / self.I_s, n=queries.size(1)).permute(0, 2, 1)

        # TopK =====================================================================================#
        topk, index = torch.topk(torch.mean(torch.mean(corr, -1), 0), self.K-1, -1)
        weight_k = torch.softmax(topk * scale, dim=-1)
        topk_V_ft = torch.zeros((B, L, index.shape[0])).to(index.device)
        topk_V = torch.zeros((B, L, index.shape[0])).to(index.device)
        for i in range(index.shape[0]):
            topk_V_ft[:,:,i] = torch.mul(weight_k[i].unsqueeze(0).unsqueeze(0).repeat(B,L), values_ftt_out[:,:,index[i]])
            topk_V[:,:,i] = torch.mul(weight_k[i].unsqueeze(0).unsqueeze(0).repeat(B,L), values[:,:,index[i]])
        # Full =====================================================================================#
        weight = torch.softmax(torch.mean(corr, -1) * scale, dim=-1)
        values_out = torch.mul(weight.unsqueeze(1).repeat(1,L,1), values)
        # ==========================================================================================#
        out = torch.cat([org_queries, values_out], -1)
        topk_V_out = torch.cat([org_queries, topk_V, topk_V_ft], -1)

        return out, topk_V_out, weight_k

class MacroLSTM_part(nn.Module):
    def __init__(self, configs, K, group_size, hidden_size=128, num_layers=2, dropout=0.0):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.driven_size = configs.driven_size
        self.K = K
        self.group_size = group_size

        self.Relation = Feq_Cross_Correlation(1, self.driven_size, self.K, self.seq_len)

        self.rnn = nn.LSTM(
            input_size=self.driven_size+1+self.group_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.rnn_K = nn.LSTM(
            input_size=(self.K-1)*2+1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.fc_out = nn.Linear(hidden_size, 1)
        self.fc_out_k = nn.Linear(hidden_size, 1)
        #===================================================================#

    def forward(self, driven, target, init):
        #=======loop 1=======#
        DA_out, topK_out, weight_k1 = self.Relation(target, driven, driven)

        DA_out = torch.cat([DA_out, init], dim=1)
        topK_out = torch.cat([topK_out, init[:,:,:topK_out.shape[-1]]], dim=1)
        out, (out_hn, out_cn) = self.rnn(DA_out)
        topK_out, (topK_hn, topK_cn) = self.rnn_K(topK_out)

        pre =  self.fc_out(torch.cat([out[:, -self.pred_len:, :]]))
        pre_K = self.fc_out_k(topK_out[:, -self.pred_len:, :])
        pre = pre+pre_K
        # =======loop 2=======#
        up_in = torch.cat([target,pre], 1)[:, self.pred_len:, :]

        DA_out, topK_out, weight_k2 = self.Relation(up_in, driven, driven)
        out, _ = self.rnn(DA_out, (out_hn, out_cn))
        topK_out, _ = self.rnn_K(topK_out, (topK_hn, topK_cn))

        pre = self.fc_out(torch.cat([out[:, -self.pred_len:, :]]))
        pre_K = self.fc_out_k(topK_out[:, -self.pred_len:, :])
        pre = pre+pre_K
        #===================================================#

        return pre, weight_k1, weight_k2

class LSTMModel(nn.Module):
    def __init__(self, configs, hidden_size=128, num_layers=2, dropout=0.0):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.driven_size = configs.driven_size
        self.group_size = 8
        self.K = (configs.driven_size + 1) // 4
        self.MacroLSTM = MacroLSTM_part(configs, self.K, self.group_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

        self.projector = Pro_for_CCL(in_dim=self.K-1,
                                             hidden_dim=hidden_size * 2,
                                             out_dim=hidden_size)

    def forward(self, driven, target, target_label):
        #=======init=========#
        zeros = torch.zeros([driven.shape[0], self.pred_len, driven.shape[2]+target.shape[2]+self.group_size], device=driven.device)

        pre, weight_k1, weight_k2 = self.MacroLSTM(driven, target, zeros)
        pro_loop2 = self.projector(weight_k2.view(1,-1))
        pro_loop1 = self.projector(weight_k1.view(1,-1))
        temp = 0.01
        a = (pro_loop2 / temp).softmax(dim=-1)
        b = (pro_loop1 / temp).softmax(dim=-1)

        divergence = F.kl_div(a.log(), b, reduction='sum')
        MSE = torch.nn.MSELoss()
        loss = MSE(pre, target_label)+divergence

        return loss, pre