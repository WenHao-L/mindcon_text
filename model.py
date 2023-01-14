import math
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore.common.initializer import Uniform, HeUniform
from mindspore import Parameter, Tensor, ops


class TextCNN(nn.Cell):
    def __init__(self, sequence_length, num_classes, filter_sizes,
                 num_filters, embeddings, pad_idx):
        super(TextCNN, self).__init__()
        vocab_size, embedding_dim = embeddings.shape
        self.num_filters_total = num_filters * len(filter_sizes)
        self.filter_sizes = filter_sizes
        self.sequence_length = sequence_length
        self.W = nn.Embedding(vocab_size, embedding_dim, embedding_table=ms.Tensor(embeddings), 
                              padding_idx=pad_idx)
        self.Weight = nn.Dense(self.num_filters_total, num_classes, has_bias=False)
        self.Bias = Parameter(Tensor(np.ones(num_classes), ms.float32), name='bias')
        self.filter_list = nn.CellList()
        self.sigmoid = ops.Sigmoid()
        for size in filter_sizes:
            seq_cell = nn.SequentialCell([
                nn.Conv2d(1, num_filters, (size, embedding_dim), pad_mode='valid'),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(sequence_length - size + 1, 1))
            ])
            self.filter_list.append(seq_cell)

    def construct(self, X):
        embedded_chars = self.W(X)
        embedded_chars = embedded_chars.expand_dims(1)
        pooled_outputs = []
        for conv in self.filter_list:
            pooled = conv(embedded_chars)
            pooled = pooled.transpose((0, 3, 2, 1))
            pooled_outputs.append(pooled)
            
        h_pool = ops.concat(pooled_outputs, len(self.filter_sizes))
        h_pool_flat = h_pool.view(-1, self.num_filters_total)
        model = self.Weight(h_pool_flat) + self.Bias
        return self.sigmoid(model).squeeze()

    
class RNN(nn.Cell):
    def __init__(self, embeddings, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()
        vocab_size, embedding_dim = embeddings.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, 
                                      embedding_table=ms.Tensor(embeddings), padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)
        weight_init = HeUniform(math.sqrt(5))
        bias_init = Uniform(1 / math.sqrt(hidden_dim * 2))
        self.fc = nn.Dense(hidden_dim * 2, output_dim, weight_init=weight_init, bias_init=bias_init)
        self.dropout = nn.Dropout(1 - dropout)
        self.sigmoid = ops.Sigmoid()

    def construct(self, inputs):
        embedded = self.dropout(self.embedding(inputs))
        _, (hidden, _) = self.rnn(embedded)
        hidden = self.dropout(mnp.concatenate((hidden[-2, :, :], hidden[-1, :, :]), axis=1))
        output = self.fc(hidden)
        return self.sigmoid(output).squeeze()