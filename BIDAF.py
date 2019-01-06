import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


class BIDAF(object):
    '''
    char_embed_size:embed size for single character
    char_size:num of char
    word_length:max single word length
    '''
    def __init__(self,char_embed_size,char_size,word_length,
                 word_size,batch_num,word_embed_size,T=150,J=30,K=6):
        self.char_embed_size = char_embed_size
        self.char_size = char_size
        self.word_length = word_length
        self.word_size = word_size
        self.word_embed_size = word_embed_size
        
        self.d = char_embed_size + word_embed_size
        self.num_units = self.d
        self.num_blocks = 1

        self.num_heads = 1

        self.loss = None
        self.p1_accuracy = None
        self.p2_accuracy = None
        self.accuracy = None
        
        self.batch_num = batch_num
        
        with tf.name_scope('char_embedding'):
            self.char_W = tf.Variable(tf.random_uniform([self.char_size,self.char_embed_size],-1.0,1.0),name='char_W')
        with tf.name_scope('word_embedding'):
            self.word_W = tf.Variable(tf.random_uniform([self.word_size,self.word_embed_size],-1.0,1.0),name='word_W')
        
        self.word_length = word_length
        
        self.T = T
        self.J = J
        self.K = K
        
        self.c_char = tf.placeholder(dtype=tf.int32,shape=[None,T,word_length],name='c_char')
        self.c_word = tf.placeholder(dtype=tf.int32,shape=[None,T],name='c_word')
        
        self.q_char = tf.placeholder(dtype=tf.int32,shape=[None,J,word_length],name='q_char')
        self.q_word = tf.placeholder(dtype=tf.int32,shape=[None,J],name='q_word')
        
        self.a_char = tf.placeholder(dtype=tf.int32,shape=[None,K,word_length],name='a_char')
        self.a_word = tf.placeholder(dtype=tf.int32,shape=[None,K])
        
        self.p1_seq = tf.placeholder(dtype=tf.int32,shape=[None,T],name='p1_seq')
        self.p2_seq = tf.placeholder(dtype=tf.int32,shape=[None,T],name='p2_seq')
        
        self.label = tf.placeholder(dtype=tf.float32,shape=[None,3],name='label')
        
        self.model_construct()
        self.opt_construct()
        
    def model_construct(self):
        #context embed
        self.c_char_embed_flat = self.char_embed_layer(tf.reshape(self.c_char,[-1,self.word_length]),num_filters=self.char_embed_size)
        self.c_char_embed = tf.reshape(self.c_char_embed_flat,[-1,self.T,int(self.c_char_embed_flat.get_shape()[-1])],name='c_char_embed')
        print('c_char_embed',self.c_char_embed)
        self.c_word_embed = self.word_embed_layer(self.c_word)
        print('c_word_embed',self.c_word_embed)
        self.c_embed = tf.concat([self.c_char_embed,self.c_word_embed],axis=-1,name='c_embed')
        c_embed_last_dim = int(self.c_embed.get_shape()[-1])
        print(c_embed_last_dim)
        
        #query embed
        self.q_char_embed_flat = self.char_embed_layer(tf.reshape(self.q_char,[-1,self.word_length]),num_filters=self.char_embed_size)
        self.q_char_embed = tf.reshape(self.q_char_embed_flat,[-1,self.J,int(self.q_char_embed_flat.get_shape()[-1])],name='q_char_embed')
        self.q_word_embed = self.word_embed_layer(self.q_word)
        self.q_embed = tf.concat([self.q_char_embed,self.q_word_embed],axis=-1,name='q_embed')
        q_embed_last_dim = int(self.q_embed.get_shape()[-1])
        
        #answer embed [batch*3,K,d]
        self.a_char_embed_flat = self.char_embed_layer(tf.reshape(self.a_char,[-1,self.word_length]),num_filters=self.char_embed_size)
        self.a_char_embed = tf.reshape(self.a_char_embed_flat,[-1,self.K,int(self.a_char_embed_flat.get_shape()[-1])],name='q_char_embed')
        self.a_word_embed = self.word_embed_layer(self.a_word)
        self.a_embed = tf.concat([self.a_char_embed,self.a_word_embed],axis=-1,name='a_embed')
        a_embed_last_dim = int(self.a_embed.get_shape()[-1])
        
        #high way scope
        with tf.variable_scope('high_way_layer') as scope:
            self.X = tf.reshape(self.high_way_layer(tf.reshape(self.c_embed,[-1,c_embed_last_dim])),[-1,self.T,c_embed_last_dim])
            scope.reuse_variables()
            self.Q = tf.reshape(self.high_way_layer(tf.reshape(self.q_embed,[-1,q_embed_last_dim])),[-1,self.J,q_embed_last_dim])
        '''
        with tf.variable_scope('test') as scope:
            test = tf.Variable(tf.truncated_normal([32,150,512]))
            test = self.attention(test,self.num_blocks,self.num_units,self.num_heads)
            print('tested')
        '''
        #first bilstm layer
        with tf.variable_scope('first_attention_layer_context') as scope:
            print(self.X)
            self.H = self.attention(self.X,self.num_blocks,self.num_units,self.num_heads)
            #scope.reuse_variables()
        with tf.variable_scope('first_attention_layer_query') as scope:
            self.U = self.attention(self.Q,self.num_blocks,self.num_units,self.num_heads)
            
        #similarity layer
        #[batch_num,T,J]
        self.S = self.similarity_layer(self.H,self.U)
        
        #context to query attention
        self.A = tf.nn.softmax(self.S,dim=-1,name='A')
        d = self.U.get_shape()[-1]
        self.U_bar_list = []
        for i in range(self.batch_num):
            A = tf.squeeze(tf.slice(self.A,[i,0,0],[1,-1,-1]))
            U = tf.squeeze(tf.slice(self.U,[i,0,0],[1,-1,-1]))
            self.U_bar_list.append(tf.expand_dims(tf.matmul(A,U),0))
            
        self.U_bar = tf.concat(self.U_bar_list,axis=0)
        
        #query to context attention
        self.b = tf.nn.softmax(tf.reduce_max(self.S,axis=-1),dim=-1,name='b')
        self.h_bar_list = []
        
        for i in range(self.batch_num):
            b = tf.slice(self.b,[i,0],[1,-1])
            H = tf.squeeze(tf.slice(self.H,[i,0,0],[1,-1,-1]))
            self.h_bar_list.append(tf.matmul(b,H))
            
        self.h_bar = tf.concat(self.h_bar_list,axis=1)
        self.H_bar = tf.reshape(tf.tile(self.h_bar,[1,self.T]),[-1,self.T,int(d)])
        print('H',self.H)
        print('H_bar',self.H_bar)
        print('U_bar',self.U_bar)
        #G layer
        self.G = self.G_layer(self.H,self.H_bar,self.U_bar)
        print('G',self.G)
        #second lstm layer
        with tf.variable_scope('second_attention_layer') as scope:
            self.M = self.attention(inputs=self.G,num_blocks=self.num_blocks,num_units=int(self.G.get_shape()[-1]),num_heads=self.num_heads)
        print('M',self.M)
        '''
        #third lstm layer
        self.M2 = self.third_bilstm_layer(self.M)
        '''
        #answer lstm layer [batch,3,d]
        with tf.variable_scope('answer_attention_layer') as scope:
            self.A_lstm = self.answer_lstm_layer(self.a_embed)
        
        
        #classification layer
        GM = tf.concat([self.G,self.M],axis=-1)
        print('GM',GM)
        '''
        answer_W = tf.get_variable('answer_W',shape=[5*d*self.T,self.d],initializer=tf.contrib.layers.xavier_initializer())
        answer_b = tf.get_variable('answer_b',shape=[self.d],initializer=tf.contrib.layers.xavier_initializer())
        '''
        #[batch,d,1]

        #pre_word_embed = tf.expand_dims(tf.matmul(tf.reshape(GM,[-1,int(5*d*self.T)]),answer_W) + answer_b,axis=-1)
        pre_word_embed = tf.expand_dims(self.GM_conv_layer(GM,self.d),-1)
        print('pre_word_embed',pre_word_embed)
        #[batch,3]
        self.answer_logit = tf.squeeze(tf.matmul(self.A_lstm,pre_word_embed))
        print('answer_logit',self.answer_logit)
        
        
    def opt_construct(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=self.answer_logit))
        self.opt = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = self.opt.minimize(self.loss)
        
        self.pred = tf.argmax(self.answer_logit,axis=1,name='pred')
        
        #Accuracy
        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred,tf.argmax(self.label,axis=1)),tf.float32))
    
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
    def GM_conv_layer(self,inputs,num_filters=512,filter_size=3):
        with tf.name_scope('GM_conv_layer'):

            filter_shape = [filter_size, int(inputs.get_shape()[-1]), 1, num_filters]
            inputs = tf.expand_dims(inputs,-1)

            W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='W')
            b = tf.Variable(tf.constant(0.1,shape=[num_filters]),name='b')

            conv = tf.nn.conv2d(inputs,W,strides=[1,1,1,1],padding='VALID',name='conv')
            h = tf.nn.relu(tf.nn.bias_add(conv,b),name='relu')
            d = int(h.get_shape()[1])
            pooled = tf.nn.max_pool(h,[1,d,1,1],strides=[1,1,1,1],padding='VALID')
            return tf.reshape(pooled,[-1,num_filters])

    def word_embed_layer(self,input_word):
        with tf.name_scope('word_embedding'):
            embedded_word = tf.nn.embedding_lookup(self.word_W,input_word)
        return embedded_word
        
    def high_way_layer(self,input):
        input_shape = input.get_shape()
        last_dim = int(input_shape[-1])
            
        HW = tf.get_variable('HW',shape=[last_dim,last_dim],initializer=tf.contrib.layers.xavier_initializer())
        Hb = tf.get_variable('Hb',shape=[last_dim],initializer=tf.contrib.layers.xavier_initializer())
            
        H = tf.nn.xw_plus_b(input,HW,Hb)
            
        TW = tf.get_variable('TW',shape=[last_dim,last_dim],initializer=tf.contrib.layers.xavier_initializer())
        Tb = tf.get_variable('Tb',shape=[last_dim],initializer=tf.contrib.layers.xavier_initializer())
            
        T = tf.nn.xw_plus_b(input,TW,Tb)
            
        return H * T + input * (1 - T)
    ''' 
    def first_bilstm_layer(self,input):
        input_shape = input.get_shape()
        d = int(input_shape[-1])
            
        fw_cell = rnn.BasicLSTMCell(d,forget_bias=1.,state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
        bw_cell = rnn.BasicLSTMCell(d,forget_bias=1.,state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
            
        rnn_outputs,_ = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,input,scope='first-bi-lstm',dtype=tf.float32)
            
        return tf.concat(rnn_outputs,axis=2,name='first_bilstm_output')
    '''  
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
    
    def G_layer(self,H,H_bar,U_bar):
        d = int(H.get_shape()[-1])
        
        def g(h,h_bar,u_bar):
            return tf.concat([h,u_bar,h * u_bar,h * h_bar],axis=-1)
        
        h_list = tf.unstack(H,axis=1,name='h_list')
        h_bar_list = tf.unstack(H_bar,axis=1,name='h_bar_list')
        u_bar_list = tf.unstack(U_bar,axis=1,name='u_bar_list')
        
        g_list = []
        for idx in range(len(h_list)):
            h = h_list[idx]
            h_bar = h_bar_list[idx]
            u_bar = u_bar_list[idx]
            g_ele = g(h,h_bar,u_bar)
            g_list.append(g_ele)
            
        return tf.transpose(tf.reshape(tf.concat(g_list,axis=-1),[-1,int(4 * d),self.T]),[0,2,1],name='G')
    '''
    def second_bilstm_layer(self,input):
        input_shape = input.get_shape()
        d = int(int(input_shape[-1]) / 8)
        
        with tf.name_scope('second_lstm_layer'):
            fw_cell = rnn.BasicLSTMCell(d,forget_bias=1.,state_is_tuple=True)
            bw_cell = rnn.BasicLSTMCell(d,forget_bias=1.,state_is_tuple=True)
            
            rnn_outputs,_ = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,input,scope='second-bi-lstm',dtype=tf.float32)
            
        return tf.concat(rnn_outputs,axis=2,name='second_bilstm_output')
    
    def third_bilstm_layer(self,input):
        input_shape = input.get_shape()
        d = int(int(input_shape[-1]) / 2)
        
        with tf.name_scope('third_lstm_layer'):
            fw_cell = rnn.BasicLSTMCell(d,forget_bias=1.,state_is_tuple=True)
            bw_cell = rnn.BasicLSTMCell(d,forget_bias=1.,state_is_tuple=True)
            
            rnn_outputs,_ = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,input,scope='third-bi-lstm',dtype=tf.float32)
        return tf.concat(rnn_outputs,axis=2,name='third_bilstm_output')
    '''
    def answer_lstm_layer(self,input):
        input_shape = input.get_shape()
        #batch = int(input_shape[0])
        #assert batch == self.batch_num * 3
        batch = 3 * self.batch_num
        with tf.name_scope('answer_lstm_layer'):
            cell = tf.nn.rnn_cell.BasicRNNCell(self.d)
            initial_state = cell.zero_state(batch, dtype=tf.float32)
            outputs,state = tf.nn.dynamic_rnn(cell,input,initial_state=initial_state,dtype=tf.float32)
            state = tf.reshape(state,[-1,3,self.d])
        return state
    
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


