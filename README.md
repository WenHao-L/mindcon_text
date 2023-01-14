# MindCon-外卖评论文本分类

本赛题任务是对外卖评论的文本进行分类。数据集为外卖评论数据集，包含约4000条正向评论、8000条负向评论。 正向评价为1，负向评价为0.



## 数据集

数据集：https://xihe.mindspore.cn/datasets/drizzlezyk/mindcon_text_classification

数据集下载后按以下文件目录放置：

```
└─dataset
   ├─train  # 解压后的训练数据集
   └─test   # 解压后的测试数据集
```



## 分词、词向量

- 分词采用 jieba

- 词向量：https://github.com/Embedding/Chinese-Word-Vectors，（Sogou News 搜狗新闻的Word）

​		下载解压后放在 `dataset` 目录，并通过 `preprocessing.ipynb` 的词向量预处理生成 `word2vec.txt`



## 环境要求

- 硬件（Ascend910）
- 框架（MindSpore1.8.1）

- 环境依赖 `pip install -r requirements.txt`



## 脚本说明

```
├── mindcon_text
  ├── README.md                           // mindcon_text相关说明
  ├── dataset                             // 数据集
      ├── train 						  // 训练数据集
      ├── test                            // 测试数据集
      ├── word2vec.txt                    // 词向量
  ├── log                                 // 训练日志
  ├── model_ckpt                          // 模型保存文件夹
  ├── result                              // 推理结果
  ├── args.py                             // 配置文件
  ├── data.py                             // 数据处理
  ├── model.py                            // 模型保存文件夹
  ├── train.py                            // 训练文件
  ├── inference.py                        // 推理文件
  ├── result.txt                          // 推理结果
```



## 训练和推理

- 训练

```shell
python train.py
```

- 推理

```shell
python inference.py
```

推理结果保存在 `./result.txt`

