import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from utils.nn import LSTM, Linear

import re
import datetime
import gensim
import math
import gc
import pickle
import sys

# numpy.set_printoptions(threshold=numpy.NaN)
# from code.data_process import SQuAD

np.set_printoptions(threshold=sys.maxsize)

EMBEDDING_DIM = 300


class Local_Fai_score(nn.Module):
    def __init__(self, args, filter_num, filter_window, doc, context, title, embedding, lamda):
        super(Local_Fai_score, self).__init__()

        self.dim_compared_vec = filter_num  # 卷积核个数 64
        self.num_words_to_use_conv = filter_window  # 卷积窗口大小3
        self.sur_num_words_to_use_conv = 2
        self.lamda = lamda
        self.document_length = doc
        self.context_length = context * 2
        self.title_length = title
        self.embedding_len = embedding  # 词向量长度 300
        # self.word_embedding=nn.Embedding()
        self.surface_length = 10
        self.num_indicator_features = 623

        self.relu_layer = nn.ReLU(inplace=True)

        #self.softmax_layer = nn.Softmax(dim=1)

        #self.layer_local = nn.Linear(6, 1, bias=True)
        #self.layer_sensefeat = nn.Linear(self.num_indicator_features, 1, bias=True)
        #self.layer_local_combine1 = nn.Linear(self.num_indicator_features + 6, 1, bias=True)

        #self.cos_layer = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.args = args
 
        # 1. Character Embedding Layer   100
        # self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=1)
        # self.char_emb = nn.Embedding(5053, EMBEDDING_DIM, padding_idx=1)

        self.char_emb = nn.Embedding(5053, EMBEDDING_DIM, padding_idx=1)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)
        # self.char_emb = self.context_vec

        # in_channels:1  out_channels:100,kernn_size:(8,5)
        self.char_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input_height
                out_channels=args.char_channel_size,  # n_filters
                kernel_size=[EMBEDDING_DIM, args.char_channel_width]),  # filter_size)
            nn.ReLU()
        )

        # 2. Word Embedding Layer
        # initialize word embedding with GloVe
        # self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)

        # highway network
        # assert self.args.hidden_size * 2 == (self.args.char_channel_size + self.args.word_dim)
        # for i in range(2):
        #     setattr(self, 'highway_linear{}'.format(i),
        #             nn.Sequential(Linear(args.hidden_size * 2, args.hidden_size * 2),
        #                           nn.ReLU()))
        #     setattr(self, 'highway_gate{}'.format(i),
        #             nn.Sequential(Linear(args.hidden_size * 2, args.hidden_size * 2),
        #                           nn.Sigmoid()))

        # 3. Contextual Embedding Layer
        self.context_LSTM = LSTM(input_size=args.hidden_size * 2,
                                 hidden_size=args.hidden_size,
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=args.dropout)

        # 4. Attention Flow Layer
        #self.att_weight_c = Linear(args.hidden_size * 2, 1)
        self.att_weight_c = Linear(args.hidden_size, 1)
        self.att_weight_q = Linear(args.hidden_size, 1)
        self.att_weight_cq = Linear(args.hidden_size, 1)

        # 4.1(z = wx+b)
        self.linear_z = Linear(args.hidden_size * 4, 1)
  
        # 5. Output Layer
        self.p1_weight = Linear(1, 1, dropout=args.dropout)
        self.p2_weight = Linear(1, 1, dropout=args.dropout)

        self.dropout = nn.Dropout(p=args.dropout)

        # 构建词向量
        # data = SQuAD(args,mention_vec,doc_vec, context_vec,body_vec,title_vec)
        # self.mention_vec = self.embed(mention_vec)
        # self.doc_vec = self.embed(doc_vec)
        # self.context_vec = self.embed(context_vec)
        # self.body_vec = self.embed(body_vec)
        # self.title_vec = self.embed(title_vec)
    def sloppyMathLogSum(self, vals):
        m = float(vals.max().cpu().data)
        r = torch.log(torch.exp(vals - m).sum())
        r = r + m
        return r
    def forward(self, args, batch):
        # 都是已经进行过embedding
        def char_emb_layer(x):
            """
            :param x: (batch, seq_len, word_len)
            :return: (batch, seq_len, char_channel_size)
            """
            #x = x.unsqueeze(0)
            #print('x1',x.size())
            x = x.unsqueeze(0)
            batch_size = x.size(0)
            # (batch, seq_len, word_len, char_dim)
            # 1，9，10，300
            x = self.dropout(x)
            # (batch， seq_len, char_dim, word_len)
            #    1，9，300，10
            x = x.transpose(2, 3)
            # (batch * seq_len, 1, char_dim, word_len)
            x = x.view(-1, int(self.args.embedding), x.size(3)).unsqueeze(1)
            # (batch * seq_len, char_channel_size, 1, conv_len) -> (batch * seq_len, char_channel_size, conv_len)
            x = self.char_conv(x).squeeze(2)
            # (batch * seq_len, char_channel_size, 1) -> (batch * seq_len, char_channel_size)
            x = F.max_pool1d(x, x.size(2)).squeeze()
            #print('xF',x.size())
            # (batch, seq_len, char_channel_size)
            x = x.view(batch_size, -1, self.args.char_channel_size)
            #print('x2',x.size())
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
        
        def output_layer(p1, p2):
            """
            :param g: (batch, c_len, hidden_size * 8)
            :param m: (batch, c_len ,hidden_size * 2)
            :return: p1: (batch, c_len), p2: (batch, c_len)
            """
            # (batch, c_len)
            p1 = (self.p1_weight(p1) + self.p2_weight(p2)).squeeze(2)
            # (batch, c_len, hidden_size * 2)
            #print('p1',p1.size())
            return p1       
 
        print('m',batch['m'])
        print('n',batch['n'])
        #print('docment',batch['doc_vec'][0])
        # 1. Character Embedding Layer
        title2vec = self.char_emb(batch['title_vec'][0].cuda())
        body2vec = self.char_emb(batch['body_vec'][0].cuda())
        mention2vec = self.char_emb(batch['mention_vec'][0].cuda())
        context2vec = self.char_emb(batch['context_vec'][0].cuda())
        docment2vec = self.char_emb(batch['doc_vec'][0].cuda())
        #print('docment',docment2vec.size())
        #print('body',body2vec.size())
        print('mention',mention2vec.size())
        #print('context',context2vec.size()) 
        for i in range(batch['m']):
            #print('mention2entity',batch['mention_entity'][0])
            #print('mention_vec[1]',mention[1].size())
            #print('mention_vec[2]',mention[2].size())
            candi = 0
            candi_list = []
            #print('m2e',batch['mention_entity'][0])
            for j in range(batch['n']):
                if int(batch['mention_entity'][0][i][j]) == 1:
                    a = title2vec[j].unsqueeze(0)
                    b = body2vec[j].unsqueeze(0)
                    m = mention2vec[i].unsqueeze(0)
                    c = context2vec[i].unsqueeze(0)
                    d = docment2vec.unsqueeze(0)
                    #print('d',d.size())
                    
                    if candi == 0:
                        m2title_vec = a
                        body_vec = b
                        mention_vec = m
                        context_vec = c
                        docment_vec = d
                    else:
                        m2title_vec = torch.cat((m2title_vec, a), 0)
                        body_vec = torch.cat((body_vec, b), 0)
                        mention_vec = torch.cat((mention_vec, m), 0)
                        context_vec = torch.cat((context_vec, c), 0)
                        docment_vec = torch.cat((docment_vec, d), 0)
                       
                    candi += 1
                    candi_list.append(j)
          
            #print('docment_vec',docment_vec.size())
            #print('body_vec',body_vec.size())
            #t = m2title_vec
            #m = mention_vec
            #if i == 0:
            #    m2title_vec_doc = t
            #    mention_vec_doc = m
            #else:
            #mention_char = char_emb_layer(mention_vec)
            title_char = char_emb_layer(m2title_vec)
            context_char = char_emb_layer(context_vec)
            print('title_char',title_char.size())
            print('context_char',context_char.size())
            docment_char = char_emb_layer(docment_vec)
            body_char = char_emb_layer(body_vec)


            ## 2. Contextual Embedding Layer
            #mention = self.context_LSTM(args,mention_char)
            title = self.context_LSTM(args,title_char)
            context = self.context_LSTM(args,context_char)

            docment = self.context_LSTM(args, docment_char)
            body = self.context_LSTM(args, body_char)

            # 3. Attention Flow Layer
            g = att_flow_layer(context,title)
            print('g',g)
            p1 = self.linear_z(g)
            print('p1',p1)
            p1_f = F.relu(p1)
            #print('p1',p1.size()) #1,7,1
            #print('p1_f',p1_f)  #1,7,1
            #true_output = F.relu(p1).squeeze(2)
            #print('output.shape:\n', true_output.size())  # 1,7  

            h = att_flow_layer(docment,body)
            p2 = self.linear_z(h)
            #p2_f = F.relu(p2)
            #print('p2',p2)
            #print('p2',p2.size())
           
            
            # 6. Output Layer
            true_output = output_layer(p1, p2)
            print('score',true_output.size())
            print('score',true_output)
            
            #r = r.squeeze(2)
           
            if len(true_output) == 1:
                true_output_softmax = true_output
                true_output_uniform = true_output
            else:
                true_output_softmax = (true_output - torch.mean(true_output)) / torch.std(true_output)
                true_output = (true_output - torch.mean(true_output)) / torch.std(true_output)
                true_output_uniform = (true_output + 1 - true_output.min()) / (true_output.max() - true_output.min())

            true_output_softmax = torch.exp(true_output_softmax - self.sloppyMathLogSum(true_output_softmax))
            true_output_uniform = true_output_uniform / true_output_uniform.sum()
            mask_2 = torch.zeros(candi, batch['n'])
            for can_ii in range(candi): mask_2[can_ii][candi_list[can_ii]] = 1
            true_output = torch.mm(true_output, Variable(mask_2).cuda())
            true_output_uniform = torch.mm(true_output_uniform, Variable(mask_2).cuda())
            true_output_softmax = torch.mm(true_output_softmax, Variable(mask_2).cuda())

            if i == 0:
                local_score = true_output
                local_score_softmax = true_output_softmax
                local_score_uniform = true_output_uniform
            else:
                local_score = torch.cat((local_score, true_output), 0)
                local_score_softmax = torch.cat((local_score_softmax, true_output_softmax), 0)
                local_score_uniform = torch.cat((local_score_uniform, true_output_uniform), 0)
            print('local',local_score.size())
        return local_score, local_score_softmax, local_score_uniform

        # men_vec_char = char_emb_layer(batch['mention_vec'][0][i].cuda())
        # con_vec_char = char_emb_layer(batch['context_vec'][0][i].cuda())


        # 3. Contextual Embedding Layer
        # context = self.context_LSTM(args, con_vec_char)
        # title = self.context_LSTM(args, title_char)
        # 4. Attention Flow Layer
        # g = att_flow_layer(context, title)
        # 4.1 g是上下文嵌入和关注向量组合在一起以产生G，其中每个列向量可以被视为每个上下文单词的查询感知表示，通过多层感知机
        # hidden= self.linear_z(g)
        # print('z.weight.shape:\n ', self.z.weight.shape)
        # print('z.bias.shape:\n', self.z.bias.shape)
        # print('output.shape:\n', hidden.shape)
        # f_output = F.relu(hidden)
        #
        # return f_output



