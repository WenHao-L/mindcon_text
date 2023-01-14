"""global args"""
import argparse


def parse_arguments():
    """parse_arguments"""
    parser = argparse.ArgumentParser(description="MindSpore Training")

    # path
    parser.add_argument('--dataset_url', default="./dataset", help='Location of data.')
    parser.add_argument('--ckpt_url', default='./model_ckpt', help='model to save/load')
    parser.add_argument('--fine_tune_ckpt_url', default='./fine_tune_ckpt', help='fine tune model to save/load')
    parser.add_argument('--log_url', default='./log', help='log path')
    parser.add_argument('--word2vec_url', default='./dataset/word2vec.txt', help='word2vec file')
    parser.add_argument('--inference_result_url', default='./result.txt', help='inference result')

    # dataset 
    parser.add_argument("--train_split_ratio", default=0.9, type=float, help="The percentage of training sets")
    parser.add_argument("--sequence_length", default=32, type=int, help="sequence length")    

    # device
    parser.add_argument("--device_id", default=0, type=int, help="Device Id")
    parser.add_argument("--device_num", default=1, type=int, help="device num")
    parser.add_argument("--device_target", default="Ascend", choices=["GPU", "Ascend", "CPU"], type=str)
    parser.add_argument("--graph_mode", default=0, type=int, help="graph mode with 0, python with 1")

    # model
    parser.add_argument("--output_size", default=1, type=int, help="output size")
    
    # TextCNN
    parser.add_argument("--filter_sizes", default=[2, 3, 4], type=list, help="filter sizes")
    parser.add_argument("--num_filters", default=16, type=int, help="filters number")
    
    # LSTM
    parser.add_argument("--hidden_size", default=256, type=int, help="hidden size")
    parser.add_argument("--num_layers", default=2, type=int, help="num of layers")
    parser.add_argument("--bidrectional", default=True, type=bool, help="bidrectional")
    parser.add_argument("--dropout", default=0.5, type=float, help="dropout")

    # training
    parser.add_argument("--epochs", default=15, type=int, help="number of total epochs to run")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--seed", default=42, type=int, help="seed for initializing training. ")
    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")

    # optimiter
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    
    args = parser.parse_args()

    return args


def get_config():
    """get_config"""

    args = parse_arguments()
    return args
