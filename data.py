import os
import jieba
import re
import string
import zhon.hanzi as hanzi
import numpy as np
import mindspore as ms
import mindspore.dataset as ds


class CommentData():
    """Comment数据集加载器

    加载Comment数据集并处理为一个Python迭代对象。

    """
    def __init__(self, path, mode="train"):
        self.mode = mode
        self.path = path
        self.comments, self.labels = [], []
        self._load_data()


    def _load_data(self):
                            
        stopwords = ['的', '了', '我', '就', '很', '是', '就是']

        if self.mode == "train":
            data_path = os.path.join(self.path, 'train', 'data.txt')

            with open (data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    sentence = line[line.find(',')+1:].rstrip("\n\r")
                    sentence = re.sub('[{}]'.format(hanzi.punctuation), '', sentence)
                    sentence = re.sub('[{}]'.format(string.punctuation), '', sentence)
                    sentence = [s for s in jieba.cut(sentence)]
                    
                    word_list = []
                    for word in sentence:  # for循环遍历分词后的每个词语
                        if word not in stopwords:     #判断分词后的词语是否在停用词表内
                            word_list.append(word)
                            
                    label = int(line[0])

                    self.comments.append(word_list)
                    self.labels.append(label)
        else:
            data_path = os.path.join(self.path, 'test', 'test.txt')
            with open (data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    sentence = line.rstrip("\n\r")
                    sentence = re.sub('[{}]'.format(hanzi.punctuation), '', sentence)
                    sentence = re.sub('[{}]'.format(string.punctuation), '', sentence)
                    sentence = [s for s in jieba.cut(sentence)]
                    
                    word_list = []
                    for word in sentence:  # for循环遍历分词后的每个词语
                        if word not in stopwords:     #判断分词后的词语是否在停用词表内
                            word_list.append(word)
                            
                    self.comments.append(word_list)
                    self.labels.append(-1)

    def __getitem__(self, idx):
        return self.comments[idx], self.labels[idx]

    def __len__(self):
        return len(self.comments)


def load_word2vec(word2vec_path):

    embeddings = []
    tokens = []
    with open(word2vec_path, encoding='utf-8') as f:
        for line in f:
            word, embedding = line.split(maxsplit=1)
            tokens.append(word)
            embeddings.append(np.fromstring(embedding, dtype=np.float32, sep=' '))

    # 添加 <unk>, <pad> 两个特殊占位符对应的embedding
    embeddings.append(np.random.rand(300))
    embeddings.append(np.zeros((300,), np.float32))

    vocab = ds.text.Vocab.from_list(tokens, special_tokens=["<unk>", "<pad>"], special_first=False)
    embeddings = np.array(embeddings).astype(np.float32)
    return vocab, embeddings


def load_data(dataset_url, word2vec_path, batch_size, train_split_ratio, sequence_length, mode="train"):
    
    vocab, embeddings = load_word2vec(word2vec_path)
    
    lookup_op = ds.text.Lookup(vocab, unknown_token='<unk>')
    pad_op = ds.transforms.PadEnd([sequence_length], pad_value=vocab.tokens_to_ids('<pad>'))
    type_cast_op = ds.transforms.TypeCast(ms.float32)
    
    
    if mode == "train":
        comment_train = ds.GeneratorDataset(CommentData(dataset_url, mode), column_names=["comment", "class"], shuffle=True)
        comment_train = comment_train.map(operations=[lookup_op, pad_op], input_columns=['comment'])
        comment_train = comment_train.map(operations=[type_cast_op], input_columns=['class'])
        comment_train, comment_valid = comment_train.split([train_split_ratio, 1-train_split_ratio])
        comment_train = comment_train.batch(batch_size, drop_remainder=True)
        comment_valid = comment_valid.batch(batch_size, drop_remainder=False)
        
        return comment_train, comment_valid, vocab, embeddings
    
    else:
        comment_test = ds.GeneratorDataset(CommentData(dataset_url, mode), column_names=["comment", "class"], shuffle=False)
        comment_test = comment_test.map(operations=[lookup_op, pad_op], input_columns=['comment'])
        comment_test = comment_test.batch(1, drop_remainder=False)

        return comment_test, vocab, embeddings
