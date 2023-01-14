import os
import sys
import time
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.common import set_seed
from mindspore import Parameter, Tensor, ops
from model import RNN, TextCNN
from args import get_config
from data import load_data


def train_one_epoch(model, train_dataset):
    model.set_train()
    loss_total = 0
    step_total = 0

    for i in train_dataset.create_tuple_iterator():
        loss = model(*i)
        loss_total += loss.asnumpy()
        step_total += 1
    
    return loss_total / step_total


def binary_accuracy(preds, y):
    """
    计算每个batch的准确率
    """

    # 对预测值进行四舍五入
    rounded_preds = np.around(preds)
    correct = (rounded_preds == y).astype(np.float32)
    acc = correct.sum() / len(correct)
    return acc


def evaluate(model, test_dataset, criterion):
    epoch_loss = 0
    epoch_acc = 0
    step_total = 0
    model.set_train(False)

    for i in test_dataset.create_tuple_iterator():
        predictions = model(i[0])
        loss = criterion(predictions, i[1])
        epoch_loss += loss.asnumpy()

        acc = binary_accuracy(predictions.asnumpy(), i[1].asnumpy())
        epoch_acc += acc

        step_total += 1

    return epoch_loss / step_total, epoch_acc / step_total



def train(args):
    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }
    context.set_context(mode=mode[args.graph_mode], device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)
    
    set_seed(args.seed)

    comment_train, comment_valid, vocab, embeddings = load_data(args.dataset_url, args.word2vec_url,
                                                      args.batch_size,args.train_split_ratio, args.sequence_length)
    pad_idx = vocab.tokens_to_ids('<pad>')

    # net = TextCNN(args.sequence_length, args.output_size, args.filter_sizes, 
    #               args.num_filters, embeddings, pad_idx)
    
    net = RNN(embeddings, args.hidden_size, args.output_size, args.num_layers, 
              args.bidrectional, args.dropout, pad_idx)

    # loss = nn.CrossEntropyLoss()
    loss = nn.BCELoss(reduction='mean')
    net_with_loss = nn.WithLossCell(net, loss)
    
    # 定义递减的学习率
    step_size = comment_train.get_dataset_size()
    lr = nn.cosine_decay_lr(min_lr=0.0001, max_lr=args.lr, total_step=args.epochs * step_size,
                            step_per_epoch=step_size, decay_epoch=args.epochs)
    # 定义优化器
    optimizer = nn.Adam(net.trainable_params(), lr, args.momentum)
    
    train_one_step = nn.TrainOneStepCell(net_with_loss, optimizer)

    best_valid_acc = 0
    os.makedirs(args.ckpt_url, exist_ok=True)
    ckpt_file_path = os.path.join(args.ckpt_url, 'best_model.ckpt')

    for epoch in range(args.epochs):
        train_epoch_loss = train_one_epoch(train_one_step, comment_train)
        valid_epoch_loss, valid_epoch_acc = evaluate(net, comment_valid, loss)
            
        print('Train | Epoch {}: loss = {}'.format(epoch, train_epoch_loss), end='  ||  ')
        print('Valid | Epoch {}: loss = {}, acc = {}'.format(epoch, valid_epoch_loss, valid_epoch_acc))
        
        if valid_epoch_acc > best_valid_acc:
            best_valid_acc = valid_epoch_acc
            ms.save_checkpoint(net, ckpt_file_path)


"""控制台输出记录到文件"""
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if 	__name__ == '__main__':
    args = get_config()
    
    # 自定义目录存放日志文件
    log_path = args.log_url 
    os.makedirs(log_path, exist_ok=True)
    
    # 日志文件名按照程序运行时间设置
    log_file_name = os.path.join(log_path, 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log')
    
    # 记录正常的 print 信息
    sys.stdout = Logger(log_file_name)
    # 记录 traceback 异常信息
    sys.stderr = Logger(log_file_name)
    
    # 开始训练
    train(args)
    