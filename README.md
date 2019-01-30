# 前沿深度学习代码复现与优化

## 简介：
这里对学术界2014-2018年**State Of The Art papers**进行了代码集成与复现，便于进行对比实验验证分析。
目前已集成文本分类的相关代码，供大家学习，有任何写得不好的地方，欢迎大家及时指正。

## 环境配置:
<br/> Python:  3.67 </br>
tensorflow:  1.90

## 代码运行:
<br/> cd ～/research/classify/sentiment_classfy </br>
python sentiment_trainer.py

## 数据格式:
Json格式存储。每一行Json数据对象应包含query字段,intent为分类标签,tags为序列标注标签,该数据格式支持
分类和序列标注的联合模型实用,intent和tags均为可选字段(不可同时缺失), 如:  
{"query": "今天很开心", "intent": "positive"}   
{"query": "张三今天很开心", "intent": "positive", "tags": "B_PERSON E_PERSON O O O O O"}   

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
                      异常情况，默认使用"cross_entropy"参数           
             gamma: focal_loss 参数， 默认2.0                
             alpha: focal_loss 参数, 默认0.25               
             lamda：uni_loss 参数, 默认0.05                
             binary_margin: hard_cross_entropy参数,默认0.8       

* 4 encoder__parameters: 模型结构的配置参数列表，您可能需要常修改的参数如:    
             **encoder_type**: 网络结构适配模块参数,可选参数如:    
                     "rnn_encoder":RNN系列模型结构，可通过修改basic_cell实现RNN,LSTM,GRU, indRNN等RNN网络结构        
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
             **activation_type**: 激活函数类型选择参数，这里集成了18种不同的激活函数，包括tf种已经集成的和2017年   
                                 Ramachandran et提出的swish激活函数等等，可选参数如:  
                        "relu":tf.nn.relu   
                        "sigmoid":tf.nn.sigmoid   
                        "tanh":tf.nn.tanh  
                        "leaky_relu":leakyRelu  
                        "elu":tf.nn.elu   
                        "selu":tf.nn.selu   
                        "swish":swish   
                        "sin":tf.sin  
                        "cube":cube  
                        "penalized_tanh":penalized_tanh   
                        "cosper":cosper  
                        "minsin":minsin  
                        "tanhrev":tanhrev   
                        "maxsig":maxsig   
                        "maxtanh":maxtanh  
                        "softplus":tf.nn.softplus  
                        "softsign":tf.nn.softsign   
                        "linear":linear   
                        默认值为**swish**, 建议使用**swish**或**penalized_tanh**

## 实验结果:
### embedding:     

| embedding类型 | 模型结构 | loss | 准确率 |
| ---------  | :---------:   | :---------: | :---------: |
| word_embedding  | CNN | cross_entropy | 91.2%  |
| win_pool_embedding | CNN | cross_entropy | **92.4%** |
| scalar_region_embedding | CNN | cross_entropy | 92.1% |
| word_context_embedding | CNN | cross_entropy | 92.3% |
| context_word_embedding | CNN | cross_entropy | 92.0% |
| multi_region_embedding | CNN | cross_entropy | 91.8% |

###  loss:
| loss类型 | 模型结构 | embedding类型 | 准确率 |
| --------- | :---------: | :----------: | :----------: |
| cross_entropy | CNN | word_embedding | 91.2% |
| focal_loss | CNN | word_embedding | **91.4%** |
| uni_loss | CNN | word_embedding | 91.3% |
| hard_cross_entropy | | | |

### encoder:
| encoder结构 | embedding类型 | loss类型 | 准确率 |
| --------- | :---------: | :----------: | :----------: |
| CNN | word_embedding | cross_entropy | 91.2% |
| RNN(rnn_cell) | word_embedding | cross_entropy | 91.6% |
| RNN(lstm_cell) | word_embedding | cross_entropy | 91.9% |
| RNN(gru_cell) | word_embedding | cross_entropy | 91.7% |
| RNN(indRnn_cell) | word_embedding | cross_entropy | 91.9% |
| DCNN | word_embedding | cross_entropy | 91.8% |
| DPCNN | word_embedding | cross_entropy | 91.7% |
| VSUM | word_embedding | cross_entropy | 88.7% |
| Weighted_SUM | word_embedding | cross_entropy | 89.2% |
| ATTENTION | word_embedding | cross_entropy | 90.6% |
| HAN | word_embedding | cross_entropy | 92.2% |
| RNN(rnn_cell) + Attention | word_embedding | cross_entropy | 92.1% |
| RNN(lstm_cell) + Attention | word_embedding | cross_entropy | **92.4%** |
| RNN(gru_cell) + Attention | word_embedding | cross_entropy | 92.2% |
| RNN(indRnn_cell) + Attention | word_embedding | cross_entropy | 92.1% |
| CNN + highway | word_embedding | cross_entropy | 91.4% |
| DCNN + highway | word_embedding | cross_entropy | 91.7% |
| DPCNN + highway | word_embedding | cross_entropy | 91.4% |
| RNN(lstm_cell) + highway | word_embedding | cross_entropy | 91.6% |
| RNN(lstm_cell) + Attention + highway | word_embedding | cross_entropy | 92.2% |
| HAN + highway | word_embedding | cross_entropy | 92.1% |
| ATTENTION + highway | word_embedding | cross_entropy | 91.2% |

### activation_type:  
| activation_type | encoder结构 | embedding类型 | 准确率 |
| ---------- | :----------: | :----------: | :---------: |
| relu | CNN | word_embedding | 91.2% |
| sigmoid | CNN | word_embedding | 90.8 |
| tanh | CNN | word_embedding | 90.7 |
| leaky_relu | CNN | word_embedding | 91.0 |
| elu | CNN | word_embedding | 90.8 |
| selu | CNN | word_embedding | 90.8 |
| swish | CNN | word_embedding | **91.5** |
| sin | CNN | word_embedding | 90.2 |
| cube | CNN | word_embedding | 90.3 |
| penalized_tanh | CNN | word_embedding | **91.5%** |
| cosper | CNN | word_embedding | 90.8% |
| minsin | CNN | word_embedding | 90.7% |
| tanhrev | CNN | word_embedding | 90.6% |
| maxsig | CNN | word_embedding | 90.4% |
| maxtanh | CNN | word_embedding | 90.6% |
| softplus | CNN | word_embedding | 90.7% |
| softsign | CNN | word_embedding | 90.4% |
| linear | CNN | word_embedding | 90.7% |
| swish | DCNN | word_embedding | 91.6% |
| swish | DPCNN | word_embedding | 91.6% |
| swish | CNN + highway | word_embedding | 91.6% |
| swish | DCNN + highway | word_embedding | 91.7% |
| swish | CNN | scalar_region_embedding | **91.8%** |
| swish | CNN | win_pool_embedding | **91.8%** |
| swish | CNN | word_context_embedding | **91.8%** |
| penalized_tanh | DCNN | word_embedding | 91.6% |
| penalized_tanh | DPCNN | word_embdedding | 91.7% |
| penalized_tanh | CNN | scalar_region_embedding | 91.7% |
| penalized_tanh | CNN | win_pool_embedding | 91.7 % |
| penalized_tanh | CNN | word_context_embedding | **91.8%** |

## 致谢
感谢在校期间顾进广老师团队的各位师兄(姐)弟(妹)领我入门NLP, 感谢徐芳芳(芳芳姐)对我的关照，   
感谢陪我一路走过的朋友的鼓励与支持，感谢一路遇到的每一位同事对我工作的指导和Code Review，  
最后感谢我最亲爱的女王大人和尚未出生的孩子，感谢你们能给我不断前行的动力！

## 说明
如果您觉得该代码好用, 引用该代码时请注明代码出处，记得留下你的小星星哦.  
有技术交流，求职或技术合作的需求可联系博主进行沟通(635401873@qq.com)。

## 参考文献:
1. Yoom Kim, 2014, Convolutional Neural Networks for Sentence Classification.  
2. Nal Kalchbrenner, 2014, A Convolutional Neural Network for Modelling Sentences.  
3. Peng Zhou, 2016, ACL, Attention-Based Bidirectional Long Short-Term Memory Networks for
Relation Classification.  
4. Ashish Vaswani, 2017, NIPS, Attention is All you Need.  
5. Shuai Li, 2018, CVPR, Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN.  
6. Zichao Yang, 2016, ACL, Hierarchical Attention Networks for Document Classification.  
7. Rie Johnson, ACL, Tencent, Deep Pyramid Convolutional Neural Networks for Text Categorization.  
8. Chao Qiao, 2018, ACL, baidu, A NEW METHOD OF REGION EMBEDDING FOR TEXT CLASSIFICATION.  
9. He Kaiming, 2017, ICCV, Focal Loss for Dense Object Detection.  
10. Steffen Eger, 2018, EMNLP, Is it Time to Swish? Comparing Deep Learning Activation Functions Across NLP tasks.
11. Prajit Ramachandran, 2017, CVPR, Google, SEARCHING FOR ACTIVATION FUNCTIONS.
