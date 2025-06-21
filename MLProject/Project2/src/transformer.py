import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
import numpy as np


def positional_encoding(seq_len,d_model):
    angles=np.arange(seq_len)[:,np.newaxis]/(np.power(10000,(2*(np.arange(d_model)[np.newaxis,:]//2))/np.float32(d_model)))
    pos_encoding=np.zeros_like(angles)
    pos_encoding[:,0::2]=np.sin(angles[:,0::2])
    pos_encoding[:,1::2]=np.cos(angles[:,1::2])
    return tf.cast(pos_encoding[np.newaxis,...],dtype=tf.float32)

def build_multi_output_model(seq_len=60,d_model=64,n_features=25,num_heads=4):
    seq_input=Input(shape=(seq_len,1),name='sequence_Input')
    x=layers.Dense(d_model)(seq_input)
    x+=positional_encoding(seq_len,d_model)
    x=layers.MultiHeadAttention(num_heads=num_heads,key_dim=d_model)(x,x)
    x=layers.LayerNormalization()(x)
    x=layers.GlobalAveragePooling1D()(x)

    feature=Input(shape=(n_features,),name='feature_input')
    y=layers.Dense(64,activation='elu')(feature)
    y=layers.BatchNormalization()(y)
    y=layers.Dense(32,activation='elu')(y)

    combined=layers.Concatenate()([x,y])
    combined=layers.Dense(64,activation='elu')(combined)
    combined=layers.Dropout(0.3)(combined)

    y_price=layers.Dense(1,name='price_output')(combined)
    y_trend=layers.Dense(3,activation='softmax',name='trend_output')(combined)
    y_signal=layers.Dense(3,activation='softmax',name='signal_output')(combined)

    model=Model(inputs=[seq_input,feature],outputs=[y_price,y_trend,y_signal])
    optimizer=Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss={
            'price_output': 'mse',
            'trend_output': 'sparse_categorical_crossentropy',
            'signal_output': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'price_output':3,
            'trend_output':1,
            'signal_output':1
        },
        metrics={
            'price_output':'mae',
            'trend_output': 'accuracy',
            'signal_output': 'accuracy'
        }
    )
    return model