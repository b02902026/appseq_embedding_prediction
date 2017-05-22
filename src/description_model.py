from keras.layers import Input, Lambda,Reshape, Embedding, Dense, merge, Activation
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers.merge import concatenate,dot, multiply
from keras import backend as K

def build_desc_model(opts,desc_maxlen,vocab_size):

    embedding_layer = Embedding(output_dim=100, input_dim=opts['vocab_size'])
    proceed_input = Input(shape=(1,), dtype='int32', name='proceed_input')
    context_input = Input(shape=(1,), dtype='int32', name='context_input')
    proceed_emb = embedding_layer(proceed_input)
    context_emb = embedding_layer(context_input)
    # incorporate desc information
    embedding_desc = Embedding(output_dim=100, input_dim=vocab_size, name='desc_embedding')
    proceed_desc_input = Input(shape=(desc_maxlen,),name='desc_proceed')
    context_desc_input = Input(shape=(desc_maxlen,),name='desc_context')
    proceed_desc_emb = embedding_desc(proceed_desc_input)
    context_desc_emb = embedding_desc(context_desc_input)
    proceed_lstm_input = concatenate([proceed_emb,proceed_desc_emb],axis=-2)
    context_lstm_input = concatenate([context_emb,context_desc_emb],axis=-2)
    #proceed_lstm_input = merge([proceed_emb,proceed_desc_emb],mode='concat',concat_axis=-2)
    #context_lstm_input = merge([context_emb,context_desc_emb],mode='concat',concat_axis=-2)
    p = LSTM(128, dropout=0.4,return_sequences=False)(proceed_lstm_input)
    c = LSTM(128, dropout=0.4,return_sequences=False)(context_lstm_input)
    # merge two apps
    dot_result = dot([p, c],axes=-1,normalize=True)
    #dot_result = merge([p,c], mode='mul',dot_axes = -1)
    #dot_result = Lambda(lambda x:K.sum(x, axis=-1))(dot_result)
    #output = Reshape((1,))(dot_result)
    output = Activation('tanh')(dot_result)

    model = Model(input=[proceed_input, context_input,proceed_desc_input,context_desc_input], output=output)

    return model


