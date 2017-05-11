from keras.layers import Input, Embedding, Dense, merge, Activation
from keras.models import Model

def build_model(opts):
    embedding_layer = Embedding(output_dim=100, input_dim=opts['vocab_size'])

    proceed_input = Input(shape=(1,), dtype='int32', name='proceed_input')
    context_input = Input(shape=(1,), dtype='int32', name='context_input') 

    proceed_emb = embedding_layer(proceed_input)
    context_emb = embedding_layer(context_input)

    dot_result = merge([proceed_emb, context_emb], mode='dot', dot_axes=-1)
    output = Activation('tanh')(dot_result)

    model = Model(input=[proceed_input, context_input], output=output)

    return model
