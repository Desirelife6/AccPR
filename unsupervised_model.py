import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import random


class BatchTreeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_dim, batch_size, use_gpu, pretrained_weight=None):
        super(BatchTreeEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        self.activation = F.relu
        self.stop = -1
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.node_list = []
        self.th = torch.cuda if use_gpu else torch
        self.batch_node = None
        self.max_index = vocab_size
        # pretrained  embedding
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            # self.embedding.weight.requires_grad = False

    def create_tensor(self, tensor):
        if self.use_gpu:
            return tensor.cuda()
        return tensor

    def traverse_mul(self, node, batch_index):
        size = len(node)
        if not size:
            return None
        batch_current = self.create_tensor(Variable(torch.zeros(size, self.embedding_dim)))

        index, children_index = [], []
        current_node, children = [], []
        for i in range(size):
            # if node[i][0] is not -1:
            index.append(i)
            current_node.append(node[i][0])
            temp = node[i][1:]
            c_num = len(temp)
            for j in range(c_num):
                # 使用j作为第一个轴对相同位置的children做了一个聚类，比如第一个来的有四个children
                # e.g. 则j = 1,2，3，4都与index相等，index变为[[0],[0],[0],[0]]
                # 很快啊！出现了一个j = 2的,那么就直接在对应维度的数组里加就好了
                if temp[j][0] is not -1:
                    if len(children_index) <= j:
                        children_index.append([i])
                        children.append([temp[j]])
                    else:
                        children_index[j].append(i)
                        children[j].append(temp[j])
        # else:
        #     batch_index[i] = -1

        batch_current = self.W_c(batch_current.index_copy(0, Variable(self.th.LongTensor(index)),
                                                          self.embedding(Variable(self.th.LongTensor(current_node)))))

        for c in range(len(children)):
            zeros = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tmp, tree = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                batch_current += zeros.index_copy(0, Variable(self.th.LongTensor(children_index[c])), tree)
        # batch_index = [i for i in batch_index if i is not -1]
        b_in = Variable(self.th.LongTensor(batch_index))
        self.node_list.append(self.batch_node.index_copy(0, b_in, batch_current))
        return b_in, batch_current

    def forward(self, x, bs):
        self.batch_size = bs
        self.batch_node = self.create_tensor(Variable(torch.zeros(self.batch_size, self.encode_dim)))
        self.node_list = []
        a, b = self.traverse_mul(x, list(range(self.batch_size)))
        self.node_list = torch.stack(self.node_list)
        return torch.max(self.node_list, 0)[0]


class BatchProgramCC(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, label_size, batch_size, use_gpu=True,
                 pretrained_weight=None, decode_dim=64):
        super(BatchProgramCC, self).__init__()
        self.stop = [vocab_size - 1]
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.decode_dim = decode_dim
        self.label_size = label_size
        self.encoder = BatchTreeEncoder(self.vocab_size, self.embedding_dim, self.encode_dim,
                                        self.batch_size, self.gpu, pretrained_weight)
        self.root2label = nn.Linear(self.encode_dim, self.label_size)
        # gru encode
        self.bigru_encode = nn.GRU(self.encode_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                                   batch_first=True)
        # gru decode
        self.bigru_decode = nn.GRU(self.hidden_dim * 2, self.decode_dim, num_layers=self.num_layers,
                                   bidirectional=True,
                                   batch_first=True)
        # linear
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.label_size)
        # hidden
        self.hidden = self.init_hidden()
        self.hidden_decode = self.init_hidden_decode()
        self.dropout = nn.Dropout(0.2)

    def init_hidden(self):
        if self.gpu is True:
            if isinstance(self.bigru_encode, nn.LSTM):
                h0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                c0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                return h0, c0
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim)).cuda()
        else:
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim))

    def init_hidden_decode(self):
        if self.gpu is True:
            if isinstance(self.bigru_decode, nn.LSTM):
                h0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.decode_dim).cuda())
                c0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.decode_dim).cuda())
                return h0, c0
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.decode_dim)).cuda()
        else:
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.decode_dim))

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.encode_dim))
        if self.gpu:
            return zeros.cuda()
        return zeros

    def encode(self, x):
        lens = [len(item) for item in x]
        max_len = max(lens)
        # max_len = 60

        encodes = []
        for i in range(self.batch_size):
            for j in range(lens[i]):
                encodes.append(x[i][j])

        encodes = self.encoder(encodes, sum(lens))
        seq, start, end = [], 0, 0
        for i in range(self.batch_size):
            end += lens[i]
            if max_len - lens[i]:
                seq.append(self.get_zeros(max_len - lens[i]))
            seq.append(encodes[start:end])
            start = end
        encodes = torch.cat(seq)
        encodes = encodes.view(self.batch_size, max_len, -1)

        # return encodes

        # batch_size=True, [batch_size, seq_len, input_dim],[32,60,128]
        # gru_encode_out [32, 60, 64] [batch_size, seq_len, hidden_size * 2]
        # hidden [2, 32, 32] [num_layers, batch_size, hidden_size]
        gru_encode_out, hidden = self.bigru_encode(encodes, self.hidden)

        gru_decode_out, hidden_decode = self.bigru_decode(gru_encode_out, self.hidden_decode)

        # gru_out = gru_out[:,-1]

        encodes = torch.transpose(encodes, 1, 2)
        # pooling
        encodes = F.max_pool1d(encodes, encodes.size(2)).squeeze(2)

        gru_encode_out = torch.transpose(gru_encode_out, 1, 2)
        gru_encode_out = F.max_pool1d(gru_encode_out, gru_encode_out.size(2)).squeeze(2)

        gru_decode_out = torch.transpose(gru_decode_out, 1, 2)
        gru_decode_out = F.max_pool1d(gru_decode_out, gru_decode_out.size(2)).squeeze(2)

        return encodes, gru_encode_out, gru_decode_out

    def forward(self, x1, x2):
        encodes, gru_encode_out, gru_decode_out = self.encode(x1)

        encodes2, gru_encode_out2, gru_decode_out2 = self.encode(x2)
        # lvec, rvec = self.encode(x1), self.encode(x2)

        abs_dist = torch.abs(torch.add(gru_encode_out, -gru_encode_out2))

        y = torch.sigmoid(self.hidden2label(abs_dist))

        return encodes, gru_encode_out, gru_decode_out, y
