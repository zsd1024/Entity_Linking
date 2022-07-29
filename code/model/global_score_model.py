import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import re
import datetime
import gensim
import math
import gc
import pickle
import heapq
import sys

from utils.nn import LSTM, Linear

np.set_printoptions(threshold=sys.maxsize)
EMBEDDING_DIM = 300

class Fai_score(nn.Module):
    def __init__(self, args,filter_num, filter_window, doc, context, title, embedding, lamda):
        super(Fai_score, self).__init__()
        # self.embed = nn.Embedding(629937, EMBEDDING_DIM)
        #self.embed = nn.Embedding(5053, EMBEDDING_DIM)
        # for p in self.parameters():p.requires_grad=False
        self.dim_compared_vec = filter_num  # 卷积核个数
        self.num_words_to_use_conv = filter_window  # 卷积窗口大小
        self.sur_num_words_to_use_conv = 2
        self.lamda = lamda
        self.document_length = doc
        self.context_length = context * 2
        self.title_length = title
        self.embedding_len = embedding  # 词向量长度
        self.surface_length = 10
        self.num_indicator_features = 623

        self.softmax_layer = nn.Softmax(dim=1)

        self.layer_local = nn.Linear(6, 1, bias=True)
        self.layer_sensefeat = nn.Linear(self.num_indicator_features, 1, bias=True)
        self.layer_local_combine1 = nn.Linear(self.num_indicator_features + 1, 1, bias=True)

        self.cos_layer = nn.CosineSimilarity(dim=1, eps=1e-6)


        self.relu_layer = nn.ReLU(inplace=True)

        self.args = args

        # local_score 1.embedding
        self.char_emb = nn.Embedding(5053, EMBEDDING_DIM, padding_idx=1)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)

        #local_score  1.embedding_charcnn
        self.char_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input_height
                out_channels=args.char_channel_size,  # n_filters
                kernel_size=[EMBEDDING_DIM, args.char_channel_width]),  # filter_size)
            nn.ReLU()
        )

        # 3. Contextual Embedding Layer
        self.context_LSTM = LSTM(input_size=args.hidden_size * 2,
                                 hidden_size=args.hidden_size,
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=args.dropout)


        # 4. Attention Flow Layer
        # self.att_weight_c = Linear(args.hidden_size * 2, 1)
        self.att_weight_c = Linear(args.hidden_size, 1)
        self.att_weight_q = Linear(args.hidden_size, 1)
        self.att_weight_cq = Linear(args.hidden_size, 1)

        # 4.1(z = wx+b)
        self.linear_z = Linear(args.hidden_size * 4, 1)

        # 5 Output Layer
        self.p1_weight= Linear(1, 1, dropout=args.dropout)
        self.p2_weight = Linear(1, 1, dropout=args.dropout)

        self.dropout = nn.Dropout(p=args.dropout)

    def cos(self, x, y):
        cos = (x * y).sum() / (math.pow((x * x).sum(), 0.5) * math.pow((y * y).sum(), 0.5))
        return cos


    def sloppyMathLogSum(self, vals):
        m = float(vals.max().cpu().data)
        r = torch.log(torch.exp(vals - m).sum())
        r = r + m
        return r

    def global_softmax(self, x, e2e_mask, n):
        x = x.cpu().data
        # x = math.e**x
        # sum_x = x.cpu().data
        sum_x = torch.mm(x, e2e_mask)
        for i in range(n):
            sum_x[0][i] = 1 / sum_x[0][i]
        # sum_x=Variable(sum_x,requires_grad=False).cuda()
        x = x * sum_x
        x = Variable(x, requires_grad=False).cuda()
        return x

    def uniform_avg(self, x, n):
        for i in range(n):
            if abs(x[i].sum() - 0) < 1.0e-6: continue
            x[i] = x[i] / x[i].sum()

        return x

    def local_score(self, args, batch, pos_embed_dict):
        def char_emb_layer(x):
            """
            :param x: (batch, seq_len, word_len)
            :return: (batch, seq_len, char_channel_size)
            """
            x = x.unsqueeze(0)
            batch_size = x.size(0)
            # (batch, seq_len, word_len, char_dim)
            # 1，9，10，300
            x = self.dropout(x)
            # (batch， seq_len, char_dim, word_len)
            #    1，9，300，10
            x = x.transpose(2, 3)
            # (batch * seq_len, 1, char_dim, word_len)
            x = x.view(-1, int(self.embedding_len), x.size(3)).unsqueeze(1)
            # (batch * seq_len, char_channel_size, 1, conv_len) -> (batch * seq_len, char_channel_size, conv_len)
            x = self.char_conv(x).squeeze(2)
            # (batch * seq_len, char_channel_size, 1) -> (batch * seq_len, char_channel_size)
            x = F.max_pool1d(x, x.size(2)).squeeze()
            # (batch, seq_len, char_channel_size)
            x = x.view(batch_size, -1, self.args.char_channel_size)
            #print('x',x.size())
            return x

        def highway_network(x1, x2):
            """
            :param x1: (batch, seq_len, char_channel_size)
            :param x2: (batch, seq_len, word_dim)
            :return: (batch, seq_len, hidden_size * 2)
            """
            # (batch, seq_len, char_channel_size + word_dim)
            x = torch.cat([x1, x2], dim=-1)
            for i in range(2):
                h = getattr(self, 'highway_linear{}'.format(i))(x)
                g = getattr(self, 'highway_gate{}'.format(i))(x)
                x = g * h + (1 - g) * x
            # (batch, seq_len, hidden_size * 2)
            return x

        def att_flow_layer(c, q):
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
            """
            c_len = c.size(1)
            q_len = q.size(1)

            # (batch, c_len, q_len, hidden_size * 2)
            # c_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            # q_tiled = q.unsqueeze(1).expand(-1, c_len, -1, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            # cq_tiled = c_tiled * q_tiled
            # cq_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1) * q.unsqueeze(1).expand(-1, c_len, -1, -1)

            cq = []
            for i in range(q_len):
                # (batch, 1, hidden_size * 2)
                qi = q.select(1, i).unsqueeze(1)
                # (batch, c_len, 1)
                ci = self.att_weight_cq(c * qi).squeeze()
                cq.append(ci)
            # (batch, c_len, q_len)
            cq = torch.stack(cq, dim=-1)

            # (batch, c_len, q_len)
            s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                cq

            # (batch, c_len, q_len)
            a = F.softmax(s, dim=2)
            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)
            # (batch, 1, c_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2c_att = torch.bmm(b, c).squeeze(1)
            # (batch, c_len, hidden_size * 2) (tiled)
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # (batch, c_len, hidden_size * 8)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            return x

        def output_later(p1,p2):
            p = (self.p1_weight(p1) + self.p2_weight(p2)).squeeze(2)
            return p

        e_embed = []
        m_embed = []
        pos_embed = []

        # 1. Character Embedding Layer
        title2vec = self.char_emb(batch['title_vec'][0].cuda())
        body2vec = self.char_emb(batch['body_vec'][0].cuda())
        mention2vec = self.char_emb(batch['mention_vec'][0].cuda())
        context2vec = self.char_emb(batch['context_vec'][0].cuda())
        docment2vec = self.char_emb(batch['doc_vec'][0].cuda())
        for i in range(batch['m']):
            candi = 0
            candi_list = []
            for j in range(batch['n']):
                if int(batch['mention_entity'][0][i][j]) == 1:
                    a = title2vec[j].unsqueeze(0)
                    b = body2vec[j].unsqueeze(0)
                    m = mention2vec[i].unsqueeze(0)
                    c = context2vec[i].unsqueeze(0)
                    d = docment2vec.unsqueeze(0)
                    tt = batch['sfeats'][0][str(i) + '|' + str(j)]
                    x = Variable(torch.Tensor(tt), requires_grad=False).cuda()
                    x = x.unsqueeze(0)
                    #print('d',d.size())

                    pos_embed.append(pos_embed_dict[i])
                   #combain = torch.cat((m,c),1)

                    if candi == 0:
                        m2title_vec = a
                        body_vec = b
                        #en_con_vec = combain
                        mention_vec = m
                        context_vec = c
                        docment_vec = d
                        ss_vec = x
                    else:
                        m2title_vec = torch.cat((m2title_vec, a), 0)
                        body_vec = torch.cat((body_vec, b), 0)
                        #men_con_vec = torch.cat((men_con_vec, combain), 0)
                        mention_vec = torch.cat((mention_vec, m), 0)
                        context_vec = torch.cat((context_vec, c), 0)
                        docment_vec = torch.cat((docment_vec, d), 0)
                        ss_vec = torch.cat((ss_vec,x),0)
                        #ss_vec = ss_vec.transpose(0,1)
                    candi += 1
                    candi_list.append(j)
                
            #print('ss_vec', ss_vec.size())
            #==========================1.(context,title)(docment,body)===========================
            #context_char = char_emb_layer(context_vec)
            #title_char = char_emb_layer(m2title_vec)
            #docment_char = char_emb_layer(docment_vec)
            #body_char = char_emb_layer(body_vec)

            # 2. Contextual Embedding Layer
            #context = self.context_LSTM(args, context_char)
            #title = self.context_LSTM(args, title_char)

            #docment = self.context_LSTM(args, docment_char)
            #body = self.context_LSTM(args, body_char)

            # 3. Attention Flow Layer
            #g = att_flow_layer(context, title)
            #p1 = self.linear_z(g)
            #h = att_flow_layer(docment,body)
            #p2 = self.linear_z(h)

            #==========================2.(mention,title)(context,body)===========================
            title_char = char_emb_layer(m2title_vec)
            mention_char = char_emb_layer(mention_vec)
            
            #context_char = char_emb_layer(context_vec)
            #body_char = char_emb_layer(body_vec)
            #2. Contextual Embedding Layer
            title = self.context_LSTM(args,title_char)
            mention = self.context_LSTM(args,mention_char)
            #context = self.context_LSTM(args,context_char)
            #body = self.context_LSTM(args,body_char)
            #3. Attention Flow Layer
            g = att_flow_layer(title,mention)
            #print('g',g)
            p1 = self.linear_z(g)
            p1_f = F.relu(p1)
            p1_f = p1_f.squeeze(2)
            #print('p1_f',p1_f.size())
            #print('p1',p1.size())

            #h = att_flow_layer(context,body)
            #p2 = self.linear_z(h)
            #print('p2',p2)
            #==========================================================
            # 4. Output Layer
            #C_score = output_later(p1,p2)
            #print('score',C_score.size())
           
            ss_vec = ss_vec.transpose(0,1)
            #print("ss_vec",ss_vec.size())
            F_local = torch.cat((ss_vec,p1_f),0) 
            #F_local = torch.cat((ss_vec, C_score), 0)
            F_local = F_local.transpose(0,1)
            #print('F_local',F_local.size())
            true_output = self.layer_local_combine1(F_local)
            true_output = true_output.transpose(0,1)
            #print('true_output',true_output.size())
            #print('true_output',true_output)

            if len(true_output) == 1:
                true_output_softmax = true_output
                true_output_temp = true_output
                true_output_uniform = true_output
            else:
                true_output_softmax = (true_output - torch.mean(true_output)) / torch.std(true_output)
                true_output_temp = (true_output - torch.mean(true_output)) / torch.std(true_output)
                true_output_uniform = (true_output + 1 - true_output.min()) / (
                        true_output.max() - true_output.min())

            true_output_softmax = torch.exp(true_output_softmax - self.sloppyMathLogSum(true_output_softmax))
            true_output_softmax_s = self.softmax_layer(true_output_softmax)
            true_output_uniform = true_output_uniform / true_output_uniform.sum()

            mask_2 = torch.zeros(candi, batch['n'])
            for can_ii in range(candi): mask_2[can_ii][candi_list[can_ii]] = 1
            true_output = torch.mm(true_output, Variable(mask_2).cuda())
            true_output_uniform = torch.mm(true_output_uniform, Variable(mask_2).cuda())
            true_output_softmax = torch.mm(true_output_softmax, Variable(mask_2).cuda())
            true_output_temp = torch.mm(true_output_temp, Variable(mask_2).cuda())
            true_output_softmax_s = torch.mm(true_output_softmax_s, Variable(mask_2).cuda())

            if i == 0:
                local_score_temp = true_output_temp
                local_score_softmax = true_output_softmax
                local_score_uniform = true_output_uniform
                local_score_softmax_s = true_output_softmax_s
            else:
                local_score_temp = torch.cat((local_score_temp, true_output_temp), 0)
                local_score_softmax = torch.cat((local_score_softmax, true_output_softmax), 0)
                local_score_softmax_s = torch.cat((local_score_softmax_s, true_output_softmax), 0)
                local_score_uniform = torch.cat((local_score_uniform, true_output_uniform), 0)
            #print('local_score_temp',local_score_temp.size())
        return local_score_temp, local_score_softmax, local_score_uniform, m_embed, e_embed, pos_embed, local_score_softmax_s


    def global_score(self, batch, local_score_norm, random_k,
                     lamda, flag, pos_embed, entity_embed_dict, fai_local_score_softmax_s):
        n = batch['n']
        m = batch['m']
        SR = batch['SR'][0]
        mention_entity = batch['mention_entity'][0]
        entity_entity = batch['entity_entity'][0]
        flag_entity = int(flag.split(":")[0])
        flag_sr = int(flag.split(":")[1])
        concat_embed = []
        entity_dis = []
        combine_dis = []

        for i in range(n):
            #concat_embed.append(torch.cat((entity_embed_dict[i], pos_embed[i].unsqueeze(0)), 1))
            concat_embed.append(entity_embed_dict[i]+pos_embed[i].unsqueeze(0))
        for i in range(batch['n']):
            entity_dis_tmp = []
            combine_dis_tmp = []
            for j in range(n):
                entity_dis_tmp.append(self.cos(entity_embed_dict[i], entity_embed_dict[j]))
                combine_dis_tmp.append(self.cos(concat_embed[i], concat_embed[j]))
            entity_dis.append(entity_dis_tmp)
            combine_dis.append(combine_dis_tmp)
        entity_dis = torch.Tensor(entity_dis)
        combine_dis = torch.Tensor(combine_dis)

        # 每个mention的前n个候选项之间相互传播
        candidate = []
        for i in range(m):
            t_local = local_score_norm[i].cpu().data.numpy().tolist()
            temp_max = list(map(t_local.index, heapq.nlargest(flag_entity, t_local)))
            candidate += temp_max

        e2e_mask = torch.ones(n, n)
        for i in range(n):
            for j in range(n):
                if (int(entity_entity[i][j]) == 0) or (j not in candidate): SR[i][j] = 0
                if (int(entity_entity[i][j]) == 1): e2e_mask[i][j] = 0
                if abs(SR[i][j] - 0.0) < 1.0e-6: continue
                SR[i][j] = SR[i][j] * 10
                SR[i][j] = math.e ** SR[i][j]
        if flag_sr == 1:
            for i in range(n):
                for j in range(n):
                    if abs(SR[i][j] - 0.0) < 1.0e-6:
                        entity_dis[i][j] = 0
            SR = SR + entity_dis

        if flag_sr == 2:
            for i in range(n):
                for j in range(n):
                    if abs(SR[i][j] - 0.0) < 1.0e-6:
                        combine_dis[i][j] = 0
            SR = SR + combine_dis
        if flag_sr == 3:
            for i in range(n):
                for j in range(n):
                    if abs(SR[i][j] - 0.0) < 1.0e-6:
                        combine_dis[i][j] = 0
            SR = combine_dis

        if flag_sr == 4:
            SR = torch.rand(n, n)
        SR = self.uniform_avg(SR, n)

        SR = Variable(SR, requires_grad=True).cuda()
        s = torch.ones(1, m)
        s = Variable(s, requires_grad=False).cuda()
        s = torch.mm(s, local_score_norm)
        fai_global_score = s
        for i in range(random_k):
            fai_global_score = (1 - lamda) * torch.mm(fai_global_score, SR) + lamda * s
        global_score = fai_global_score
        m2e = Variable(mention_entity).cuda()
        for iiii in range(m - 1):
            global_score = torch.cat((global_score, fai_global_score), 0)
        global_score = m2e * global_score
        fai_global_score = self.global_softmax(fai_global_score, e2e_mask, n)
        return s, fai_global_score, global_score

    def forward(self, args, batch, random_k, lamda, flag, pos_embed_dict, entity_embed_dict):
        fai_local_score, fai_local_score_softmax, fai_local_score_uniform, m_embed, e_embed, pos_embed, fai_local_score_softmax_s = self.local_score( args, batch, pos_embed_dict)

        s, fai_global_score, global_score = self.global_score(batch, fai_local_score_softmax, random_k, lamda, flag,
                                                              pos_embed, entity_embed_dict, fai_local_score_softmax_s)
        return s, fai_global_score, fai_local_score, fai_local_score_softmax, global_score
