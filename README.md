# 前沿深度学习代码复现与优化

## 简介：
这里对学术界2014-2018年State Of The Art papers进行了代码集成与复现，便于进行对比实验验证分析。
目前已集成文本分类的相关代码，供大家学习，有任何写得不好的地方，欢迎大家及时指正。

## 环境配置:
Python:  3.67
tensorflow:	1.90

## 代码运行:
cd ～/research/classify/sentiment_classfy
python sentiment_trainer.py

## 数据格式:
Json格式存储。每一行Json数据对象应包含query字段,intent为分类标签,tags为序列标注标签,该数据格式支持
分类和序列标注的联合模型实用,intent和tags均为可选字段(不可同时缺失)

## 参数说明:
由于本项目中对多个模型进行了集成,可能需要您选定指定的参数去调用不同的embedding或encoder。本项目将所
有参数集中在Json格式的文件中(config/sentimentConfig),常见的调用参数列表如下:
* 1 corpus_info: 这里存储的主要是数据集相关文件的路径配置文件参数，如:
					intent_id_file: 为分类标签和id的映射关系文件
					dev_res: 交叉验证结果的存储文件(便于badcase分析)
* 2 model_parameters: 模型的一些超参数, 您可能需要常修改的参数如:
					**embedding_type**: embedding的类别, 项目里支持多种embedding类型，可选值包括:
									"word_embedding": word_embedding
									"win_pool_embedding": facebook fastText region embedding
									"scalar_region_embedding" facebook region embedding(scalar)
									"word_context_embedding": baidu wordContextEmbedding
									"context_word_embedding": baidu ContextWordEmbedding
									"multi_region_embedding": facebook region embedding(multi region)
									异常情况，默认使用"word_embedding"参数
					vocab_size: 词表的长度
					out_size: 分类类别数
					ckpt_file_path: ckpt存储路径
* 3 loss_parameters：分类损失函数的一些超参数列表, 您可能需要常修改的参数如：
					**loss_type**: loss 的类别，项目中针对loss 进行了优化，可选参数值包括:
								"cross_entropy":  常见的交叉熵损失函数
								"focal_loss": focal loss (He Kaiming)
								"uni_loss": 针对样本不平衡的一种优化loss
								"hard_cross_entropy": 针对二分类任务，仅对较难区分的样本进行计算交叉熵损失
								异常情况，默认实用"cross_entropy"参数
					gamma: focal_loss 参数， 默认2.0
					alpha: focal_loss 参数, 默认0.25
					lamda：uni_loss 参数, 默认0.05
					binary_margin: hard_cross_entropy参数,默认0.8

* 4	encoder__parameters: 模型结构的配置参数列表，您可能需要常修改的参数如:
					**encoder_type**: 网络结构适配模块参数,可选参数如:
									"rnn_encoder":RNN系列模型结构，可通过修改basic_cell实现RNN,LSTM,GRU, indRNN
												等RNN网络结构
									"cnn_encoder":TextCNN网络结构                                         
									"dpcnn_encoder":腾讯的DPCNN深度CNN网络结构                                     
									"vsum_encoder":facebook FastText
									"weighted_sum_encoder":facebook FastText                        
									"idcnn_encoder":IDCNN(IDCNN+CRF做序列标注的)
									"dcnn_encoder":DCNN, k-max-pooling
									"attention_encoder":attention is all you need                             
									"han_encoder": HAN网络结构，级联的Attention机制
					**basic_cell**: RNN网络结构基本神经元结构, 可选参数如: RNN， LSTM, GRU, indRNN
					need_highway: 是否加入highway网络结构, 默认为false(不加入highway结构)

## 实验结果:
(待整理)

## 参考文献:
(周末整理)


		

