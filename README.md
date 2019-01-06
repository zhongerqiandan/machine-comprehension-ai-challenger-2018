# machine-comprehension-ai-challenger-2018
## 简介
本数据集针对阅读理解中较为复杂的，需要利用整篇文章中多个句子的信息进行综合才能得到正确答案的观点型问题，构造了30万组由问题、篇章、候选答案组成的训练和测试集合。是目前为止全球难度最大的中文阅读理解公开数据集，全球最大的观点型机器阅读理解公开数据集。训练集：25万
验证集：3万
测试集A：1万
测试集B：1万
## 数据说明
每条数据为<问题，篇章，候选答案> 三元组组成。每个问题对应一个篇章（500字以内），以及包含正确答案的三个候选答案。
* 问题：真实用户自然语言问题，从搜索日志中随机选取并由机器初判后人工筛选
* 篇章：与问题对应的文本段，从问题相关的网页中人工选取
* 候选答案：人工生成的答案，提供若干（三个）选项，并标注正确答案
## 数据预览
数据以JSON格式表示如下样例：

{
            “query_id”:1,
            “query”:“维生c可以长期吃吗”,
            “url”: “https://wenwen.sogou.com/z/q748559425.htm”,
            “passage”: “每天吃的维生素的量没有超过推荐量的话是没有太大问题的。”,
            “alternatives”:”可以|不可以|无法确定”,
            “answer”:“可以”
        }
        
训练集给出上述全部字段，测试集不给answer字段
# 模型与实现
## 总体思路
本次比赛的题型和sQuAD不同。sQuAD是给出原文和问题，然后答案在原文中，模型需要给出答案在原文中的开始和结束位置。本次比赛是给出原文和问题，给出三个候选答案，模型需要给出一个最佳选项。模型总体借鉴BIDAF，修改output layer，让其输出一个词向量，然后用输出的词向量与候选答案中的三个词向量做相似度计算，然后将三个相似度归一化，看成三个概率，用cross entropy 做损失函数训练模型。这样做在验证集上训练第二轮后模型准确率达到67以上，但是后面就开始过拟合。后来我将BIDAF中contextual embed layer和modeling layer中的lstm全部换成了google在attention is all you need里的transformer的结构，但是验证集准确率依然没有上升。后来我发现我没有用position embedding，如果用了pe效果应该会好，但是最近都没什么时间再做。下面是BIDAF的结构介绍和几个我觉得重要地方的实现。参考2017年的在ICLR会议上发表的论文《BI-DIRECTIONAL ATTENSION FLOW FOR MACHINE COMPREHENSION》
## 关键结构和实现
该模型的结构图如下：
![image](https://github.com/zhongerqiandan/machine-comprehension-ai-challenger-2018/blob/master/1.png)

* 输入模块：从上图可以清楚的看出，在该模型中的输入编码模块，首先采用word Embedding 和character Embedding策略生成对应每个单词的词向量。其中，word Embedding使用的算法是Glove，而character Embedding采用的是类似于yoon kim提出的卷积神经网络架构，只不过输入时每一个character Embedding，然后通过卷积和max-pooling生成词向量。接下来，将Character embedding和word embedding进行拼接，一起输入到双向LSTM中，这个部分被称之为Contextual Embedding layer，假设原文本的长度为T，单向LSTM的输出维度为d∗T，那么双向LSTM的输出则为2d∗T。对于汉字来说，可以将其中的character Embedding换成字向量。word_embedding很简单，直接用tf.nn.embedding_lookup实现就可以，character Embedding实现如下：


            def char_embed_layer(self,input_char,num_filters=3,filter_size=2):
                    with tf.name_scope('char_embedding'):
                        embedded_chars = tf.nn.embedding_lookup(self.char_W,input_char)
                        embedded_chars_expanded = tf.expand_dims(embedded_chars,-1)

                        filter_shape = [filter_size, self.char_embed_size, 1, num_filters]
                        W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='W')
                        b = tf.Variable(tf.constant(0.1,shape=[num_filters]),name='b')
                        conv = tf.nn.conv2d(embedded_chars_expanded,W,strides=[1,1,1,1],padding='VALID',name='conv')
                        h = tf.nn.relu(tf.nn.bias_add(conv,b),name='relu')
                        pooled = tf.nn.max_pool(h,ksize=[1,self.word_length - filter_size + 1,1,1],strides=[1,1,1,1],padding='VALID',name='pool')

                        pooled_shape = pooled.get_shape()
                        total_size = None
                        for idx in list(range(len(pooled_shape)))[1:]:
                            if total_size is None:
                                total_size = int(pooled_shape[-1 * idx])
                            else:
                                total_size *= int(pooled_shape[-1 * idx])

                        print('total size of char_embed_layer:{}'.format(total_size))

                        return tf.reshape(pooled,[-1,int(total_size)])
                        
* 交互模块：接下来在context内容和query内容交互模块部分，采用了context-to-query和query-to-context的double attention机制。假设context为H，query为U。首先针对context中的每一个单词和query中的每一个单词进行相似度的计算，这样就能生成一个相似度的矩阵Stj=α(Ht,Uj)=WTs[h;u;h⨀u]，其中Ws是一个维度为6d的向量，它也是模型参数的一部分，随着模型一起进行训练。这个相似度矩阵S是用来辅助context-to-query和query-to-context attention系数的生成。S的实现代码如下：

                def similarity_layer(self,H,U):
                    d = int(H.get_shape()[-1])
                    h_dim = int(H.get_shape()[-2])
                    u_dim = int(U.get_shape()[-2])
                    print('Similarity layer h_dim:{},u_dim:{}'.format(h_dim,u_dim))

                    Sw = tf.get_variable('Sw',shape=[3 * d,1],initializer=tf.contrib.layers.xavier_initializer())
                    H = tf.reshape(tf.transpose(H,[0,2,1]),[-1,h_dim])
                    U = tf.reshape(tf.transpose(U,[0,2,1]),[-1,u_dim])

                    fH = tf.tile(H,[u_dim,1])
                    fU = tf.tile(tf.expand_dims(tf.concat(tf.unstack(U,axis=-1),axis=0),-1),[1,h_dim])
                    fHU = fH * fU

                    fH = tf.reshape(fH,[-1,u_dim,int(1 * d),h_dim])
                    fU = tf.reshape(fU,[-1,u_dim,int(1 * d),h_dim])
                    fHU = tf.reshape(fHU,[-1,u_dim,int(1 * d),h_dim])

                    f = tf.concat([fH,fU,fHU],axis=2)
                    f = tf.reshape(tf.transpose(f,[0,1,3,2]),[-1,int(3 * d)])

                    #[batch_num,T,J]
                    return tf.transpose(tf.reshape(tf.squeeze(tf.matmul(f,Sw)),[-1,u_dim,h_dim]),[0,2,1])
