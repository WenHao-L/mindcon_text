import os
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from model import RNN, TextCNN
from args import get_config
from data import load_data


def inference():
    args = get_config()

    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }
    context.set_context(mode=mode[args.graph_mode], device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)

    comment_test, vocab, embeddings = load_data(args.dataset_url, args.word2vec_url, 
                                                args.batch_size, args.train_split_ratio, 
                                                args.sequence_length, mode="test")
    pad_idx = vocab.tokens_to_ids('<pad>')

    # net = TextCNN(args.sequence_length, args.output_size, args.filter_sizes, 
    #               args.num_filters, embeddings, pad_idx)
    
    net = RNN(embeddings, args.hidden_size, args.output_size, 
              args.num_layers, args.bidrectional, args.dropout, pad_idx)
    
    ckpt_file = os.path.join(args.ckpt_url, 'best_model.ckpt')
    param_dict = ms.load_checkpoint(ckpt_file)
    ms.load_param_into_net(net, param_dict)
    net.set_train(False)
    
    result_list = []
    for i in comment_test.create_tuple_iterator():
        prediction = net(i[0])
        pred = int(np.round(prediction.asnumpy()))
        result_list.append(str(pred))
    
    with open(args.inference_result_url, "w") as f:
        for item in result_list:
            f.writelines(item)
            f.writelines('\n')
        f.close()

if 	__name__ == '__main__':
    inference()
