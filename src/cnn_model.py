from keras.layers import Dropout, Input, Flatten, Lambda,Reshape, Embedding, Dense, merge, Activation
from keras.models import Model
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.merge import concatenate,dot, multiply
from keras import backend as K
import tensorflow as tf
import numpy as np
window_size = [2,3,5]

def build_desc_model(opts,desc_maxlen,vocab_size,batch_size, mode='train'):

    #embedding_layer = Embedding(output_dim=100, input_dim=opts['vocab_size'])
    name_embedding = Dense(100,use_bias=False, name='name_embedding')
    proceed_input = Input(shape=(vocab_size,), dtype='float32', name='proceed_input')
    context_input = Input(shape=(vocab_size,), dtype='float32', name='context_input')
    proceed_emb = name_embedding(proceed_input)
    context_emb = name_embedding(context_input)
    #proceed_emb = Flatten()(proceed_emb)
    #context_emb = Flatten()(context_emb)
    # incorporate desc information (using cnn)
    embedding_desc = Embedding(output_dim=100, input_dim=vocab_size, name='desc_embedding')
    proceed_desc_input = Input(shape=(desc_maxlen,),name='desc_proceed')
    context_desc_input = Input(shape=(desc_maxlen,),name='desc_context')
    proceed_desc_emb = embedding_desc(proceed_desc_input)
    context_desc_emb = embedding_desc(context_desc_input)
    # feed desc to cnn
    proc_conv, context_conv = [], []
    for window in window_size:
        # procced app
        conv_p = Conv1D(128,kernel_size=window, strides=1,
                        activation='relu',name='conv'+str(window))(proceed_desc_emb)
        pool_p = MaxPooling1D(pool_size=2)(conv_p)
        pool_p = Flatten()(pool_p)
        proc_conv.append(pool_p)
        # context app
        conv_c = Conv1D(128,kernel_size=window, strides=1, activation='relu')(context_desc_emb)
        pool_c = MaxPooling1D(pool_size=2)(conv_c)
        pool_c = Flatten()(pool_c)
        context_conv.append(pool_c)
    
    fully_connected = Dense(100, activation='relu',name='cnn')
    cnn_p = concatenate(proc_conv) if len(proc_conv) > 1 else proc_conv[0]
    cnn_p = Dropout(0.3)(cnn_p)
    cnn_p_out = fully_connected(cnn_p)
    cnn_c = concatenate(context_conv) if len(context_conv) > 1 else context_conv[0]
    cnn_c = Dropout(0.3)(cnn_c)
    cnn_c_out = fully_connected(cnn_c)
    # concat CNN output and app embedding

    p = concatenate([cnn_p_out, proceed_emb], axis=-1)
    c = concatenate([cnn_c_out, context_emb], axis=-1)

    # merge two apps
    output = dot([p, c],axes=-1,normalize=True)
    #dot_result = merge([p,c], mode='mul',dot_axes = -1)
    #dot_result = Lambda(lambda x:K.sum(x, axis=-1))(dot_result)
    #output = Reshape((1,))(dot_result)
    #output = Activation('tanh')(dot_result)
    if mode == 'train' :
        model = Model(input=[proceed_input,
                         context_input,proceed_desc_input,context_desc_input],
                  output=output)

    elif mode == 'test' :
        model = Model(input=[proceed_input,
                         context_input,proceed_desc_input,context_desc_input],
                  output=p)
    return model


