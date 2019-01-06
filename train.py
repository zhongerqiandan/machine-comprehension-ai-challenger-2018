import tensorflow as tf
import numpy as np
import os

from BIDAF import BIDAF
from dataPre import *

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string('checkpoint_dir_train','train/','Directory for saving and loading model checkpoints.')
#tf.flags.DEFINE_string('checkpoint_dir_max_acc','','Directory for saving and loading model checkpoints with max accurate.')
tf.flags.DEFINE_integer('word_vocabulary_size',10002,'Number of words in the dictory.')
tf.flags.DEFINE_integer('c_vocabulary_size',3012,'Number of characters in the dictory.')
tf.flags.DEFINE_integer('word_embed_size',50,'Size of embedding layer of word')
tf.flags.DEFINE_integer('char_embed_size',50,'Size of embedding layer of char')

tf.flags.DEFINE_integer('word_length',6,'The max length of word.')
#tf.flags.DEFINE_float('initial_learning_rate',0.00001,'Initial learning rate.')
tf.flags.DEFINE_integer('T',150,'context length')
tf.flags.DEFINE_integer('J',30,'question length')
tf.flags.DEFINE_integer('K',6,'alter length')
tf.flags.DEFINE_integer('batch',6,'batch number')
tf.flags.DEFINE_integer('epochs',10,'epochs')
#tf.flags.DEFINE_float('reg_rate',0.001,'Regularation rate.')



checkpoint_dir_train = FLAGS.checkpoint_dir_train
word_vocabulary_size = FLAGS.word_vocabulary_size
c_vocabulary_size = FLAGS.c_vocabulary_size
word_embed_size = FLAGS.word_embed_size
char_embed_size = FLAGS.char_embed_size
word_length = FLAGS.word_length
T = FLAGS.T
J = FLAGS.J
K = FLAGS.K
batch = FLAGS.batch
epochs = FLAGS.epochs

def train():
    
    #from time import time
    '''
    train_set_list = read_data('train')
    print('training data prepared')
    val_set_list =read_data('validation')
    print('validation data prepared')

    train_set_list = data_transformation(train_set_list)
    print('training data transformed')
    val_set_list = data_transformation(val_set_list)
    print('validation data transformed')
    
    words,characters = words_generator(train_set_list)
    
    w2id,c2id = build_vocabulary(word_vocabulary_size=word_vocabulary_size,c_vocabulary_size=c_vocabulary_size,words=words,characters=characters)
    
    del words
    del characters
    print('vocabulary prepared')
    
    c_word_list_train,c_char_list_train,q_word_list_train,q_char_list_train,a_word_list_train,a_char_list_train,label_list_train = data_generator(train_set_list,w2id,c2id,batch=batch,T=T,J=J,K=K,word_length=word_length)
    print('training batches prepared')

    c_word_list_val,c_char_list_val,q_word_list_val,q_char_list_val,a_word_list_val,a_char_list_val,label_list_val = data_generator(val_set_list,w2id,c2id,batch=batch,T=T,J=J,K=K,word_length=word_length)
    
    print('validation batches prepared')
'''    

    c_word_list_train = [np.random.randint(0,10002,[batch,T]) for i in range(1000)]
    c_char_list_train = [np.random.randint(0,3012,[batch,T,word_length]) for i in range(1000)]
    q_word_list_train = [np.random.randint(0,10002,[batch,J]) for i in range(1000)]
    q_char_list_train = [np.random.randint(0,3012,[batch,J,word_length]) for i in range(1000)]
    a_word_list_train = [np.random.randint(0,10002,[3*batch,K]) for i in range(1000)]
    a_char_list_train = [np.random.randint(0,3012,[3*batch,K,word_length]) for i in range(1000)]
    label_list_train = [np.array([[1.0,0.0,0.0] for i in range(batch)]) for i in range(1000)]

    c_word_list_val = [np.random.randint(0,10002,[batch,T]) for i in range(500)]
    c_char_list_val = [np.random.randint(0,3012,[batch,T,word_length]) for i in range(500)]
    q_word_list_val = [np.random.randint(0,10002,[batch,J]) for i in range(500)]
    q_char_list_val = [np.random.randint(0,3012,[batch,J,word_length]) for i in range(500)]
    a_word_list_val = [np.random.randint(0,10002,[3*batch,K]) for i in range(500)]
    a_char_list_val = [np.random.randint(0,3012,[3*batch,K,word_length]) for i in range(500)]
    label_list_val = [np.array([[1.0,0.0,0.0] for i in range(batch)]) for i in range(500)]

    
    
    bidaf_ext = BIDAF(char_embed_size=char_embed_size,char_size=c_vocabulary_size,word_length=word_length,
                 word_size=word_vocabulary_size,batch_num=batch,word_embed_size=word_embed_size,T=T,J=J,K=K)
    print('Model constructed')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if not os.path.exists(checkpoint_dir_train):
            os.makedirs(checkpoint_dir_train)
            saver.save(sess,checkpoint_dir_train + 'model.ckpt')
        else:
            saver.restore(sess,tf.train.latest_checkpoint(checkpoint_dir_train))
        loss_list = []
        acc_list = []
        for epoch in range(epochs):
            for i in range(len(c_word_list_val)):
                c_word,c_char,q_word,q_char,a_word,a_char,label = c_word_list_val[i],c_char_list_val[i],q_word_list_val[i],q_char_list_val[i],a_word_list_val[i],a_char_list_val[i],label_list_val[i]
                acc = sess.run(bidaf_ext.accuracy,feed_dict={bidaf_ext.c_word:c_word,bidaf_ext.c_char:c_char,bidaf_ext.q_word:q_word,bidaf_ext.q_char:q_char,bidaf_ext.a_word:a_word,bidaf_ext.a_char:a_char,bidaf_ext.label:label})
                acc_list.append(acc)
            print('epoch%d,accuracy in validation set is %f'%(epoch,np.mean(np.array(acc_list))))
            acc_list = []
            for step in range(len(c_word_list_train)):
                c_word,c_char,q_word,q_char,a_word,a_char,label = c_word_list_train[step],c_char_list_train[step],q_word_list_train[step],q_char_list_train[step],a_word_list_train[step],a_char_list_train[step],label_list_train[step]

                _,loss = sess.run([bidaf_ext.train_op,bidaf_ext.loss],feed_dict={bidaf_ext.c_word:c_word,bidaf_ext.c_char:c_char,bidaf_ext.q_word:q_word,bidaf_ext.q_char:q_char,bidaf_ext.a_word:a_word,bidaf_ext.a_char:a_char,bidaf_ext.label:label})
                loss_list.append(loss)
                if step % 20 == 0:
                    print('epoch%d,step %d - %d,average loss is %f'%(epoch+1,step-20,step,np.mean(np.array(loss_list))))
                    loss_list = []
            saver.save(sess,checkpoint_dir_train + 'model.ckpt')
        print('Done')

if __name__ == '__main__':
    train()
            

