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

context-to-query Attention：
简单来说，就是用query中所有的加权和来表征context中的每一个词向量，这个加权的系数就是通过对上述生成的S矩阵中的每一个行来做一个softmax归一化得到。这样得到的表征U∗则是维度为2d长度为T的矩阵。

            self.A = tf.nn.softmax(self.S,dim=-1,name='A')
                    d = self.U.get_shape()[-1]
                    self.U_bar_list = []
                    for i in range(self.batch_num):
                        A = tf.squeeze(tf.slice(self.A,[i,0,0],[1,-1,-1]))
                        U = tf.squeeze(tf.slice(self.U,[i,0,0],[1,-1,-1]))
                        self.U_bar_list.append(tf.expand_dims(tf.matmul(A,U),0))

                    self.U_bar = tf.concat(self.U_bar_list,axis=0)

query-to-context Attention：
这个就是针对context中的每一个词，把它和query词语中相似性最大的取出来作为其权重，然后针对context中每一个词语的权重进行softmax生成归一化的权重，然后使用这个归一化的权重对context中的词向量进行加权求和，生成唯一的query-to-context Attention机制下的词向量，把这个词向量复制T次，同样生成了维度为2d长度为T的矩阵H∗。接下来将生成的H∗和 U∗以及原始的context表征H一起输入函数G=β(H∗,U∗,H)=[h;u∗;h⨀u∗;h⨀h∗]，很显然这个输出矩阵的维度是8d∗T。其实这个β函数可以有很多种的表现形式，这里面例子给出的是最简单的直接拼接的方式，同时还可以尝试multi-layer perceptron 等方式。上述生成的矩阵G在原文中被描述为：“ encodes the query-aware representations of context words”。

    self.b = tf.nn.softmax(tf.reduce_max(self.S,axis=-1),dim=-1,name='b')
        self.h_bar_list = []
        
        for i in range(self.batch_num):
            b = tf.slice(self.b,[i,0],[1,-1])
            H = tf.squeeze(tf.slice(self.H,[i,0,0],[1,-1,-1]))
            self.h_bar_list.append(tf.matmul(b,H))
            
        self.h_bar = tf.concat(self.h_bar_list,axis=1)
        self.H_bar = tf.reshape(tf.tile(self.h_bar,[1,self.T]),[-1,self.T,int(d)])

接下来生成的矩阵G被输入到双向LSTM之中，这个在原文中被称之为“Modeling layer”，目的是“captures the interaction among the context words conditioned on the query”。生成的M矩阵维度为2d∗T。
接下来是transformer的实现，该实现来自https://github.com/Kyubyong/transformer，需要调整输入输出，使其和模型其他部分接口适配。

    def attention(self,inputs,num_blocks,num_units=512,num_heads=8,dropout_rate=0.0,is_training=True,causality=False,reuse=None):
        def normalize(inputs, 
                  epsilon = 1e-8,
                  scope="ln",
                  reuse=None):
            '''
            Applies layer normalization.
            
            Args:
              inputs: A tensor with 2 or more dimensions, where the first dimension has
                `batch_size`.
              epsilon: A floating number. A very small number for preventing ZeroDivision Error.
              scope: Optional scope for `variable_scope`.
              reuse: Boolean, whether to reuse the weights of a previous layer
                by the same name.
              
            Returns:
              A tensor with the same shape and data dtype as `inputs`.
            '''
            with tf.variable_scope(scope, reuse=reuse):
                inputs_shape = inputs.get_shape()
                params_shape = inputs_shape[-1:]
        
                mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
                beta= tf.Variable(tf.zeros(params_shape))
                gamma = tf.Variable(tf.ones(params_shape))
                normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
                outputs = gamma * normalized + beta
            
            return outputs

        def multihead_attention(queries, 
                            keys, 
                            num_units=512, 
                            num_heads=8, 
                            dropout_rate=0.0,
                            is_training=True,
                            causality=False,
                            scope="multihead_attention", 
                            reuse=None):
            '''Applies multihead attention.
            
            Args:
              queries: A 3d tensor with shape of [N, T_q, C_q].
              keys: A 3d tensor with shape of [N, T_k, C_k].
              num_units: A scalar. Attention size.
              dropout_rate: A floating point number.
              is_training: Boolean. Controller of mechanism for dropout.
              causality: Boolean. If true, units that reference the future are masked. 
              num_heads: An int. Number of heads.
              scope: Optional scope for `variable_scope`.
              reuse: Boolean, whether to reuse the weights of a previous layer
                by the same name.
                
            Returns
              A 3d tensor with shape of (N, T_q, C)  
            '''
            with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
                if num_units != int(queries.get_shape()[-1]):
                    num_units = int(queries.get_shape()[-1])
            
            # Linear projections
                Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
                K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
                V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
            
            # Split and concat
                Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
                K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
                V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

            # Multiplication
                outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
            
            # Scale
                outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            
            # Key Masking
                key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
                key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
                key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
            
                paddings = tf.ones_like(outputs)*(-2**32+1)
                outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
      
            # Causality = Future blinding
                if causality:
                    diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
                    tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k)
                    masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
       
                    paddings = tf.ones_like(masks)*(-2**32+1)
                    outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
      
            # Activation
                outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
             
            # Query Masking
                query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
                query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
                query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
                outputs *= query_masks # broadcasting. (N, T_q, C)
              
            # Dropouts
                outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
                   
            # Weighted sum
                outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
            
            # Restore shape
                outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
                  
            # Residual connection
                outputs += queries
                  
            # Normalize
                outputs = normalize(outputs) # (N, T_q, C)
     
            return outputs

        def feedforward(inputs, 
                    num_units=[2048, 512],
                    scope="multihead_attention", 
                    reuse=None):
            '''Point-wise feed forward net.
            
            Args:
              inputs: A 3d tensor with shape of [N, T, C].
              num_units: A list of two integers.
              scope: Optional scope for `variable_scope`.
              reuse: Boolean, whether to reuse the weights of a previous layer
                by the same name.
                
            Returns:
              A 3d tensor with the same shape and dtype as inputs
            '''
            with tf.variable_scope(scope, reuse=reuse):
            # Inner layer
                params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
                outputs = tf.layers.conv1d(**params)
            
            # Readout layer
                params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                      "activation": None, "use_bias": True}
                outputs = tf.layers.conv1d(**params)
            
            # Residual connection
                outputs += inputs
            
            # Normalize
                outputs = normalize(outputs)
        
            return outputs

        for i in range(num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
            ### Multihead Attention
                inputs = multihead_attention(queries=inputs,keys=inputs,num_units=num_units,num_heads=num_heads,dropout_rate=dropout_rate,is_training=is_training,causality=False,reuse=reuse)
                            
            ### Feed Forward
                inputs = feedforward(inputs,num_units=[4*num_units, num_units],reuse=reuse)
        return inputs
