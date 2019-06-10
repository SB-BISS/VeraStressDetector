import numpy
from keras.models import Sequential, Model
from keras.layers import Dense, RepeatVector, Activation,Convolution2D, GlobalMaxPool1D, GlobalAvgPool1D, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.layers import TimeDistributed,MaxPooling2D, Flatten, Input, Permute
from keras.layers import Dropout, merge, Lambda
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import RMSprop
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.regularizers import l2
from keras.callbacks import *
# from visualizer import *
from keras.models import *
from keras.optimizers import *
from keras.utils.np_utils import to_categorical
from keras.layers.core import *
from keras.layers import Input, Embedding, LSTM, Dense, merge, TimeDistributed
#import theano
    

    
class Structures:

   
    def __init__(self, CONTEXT_LENGTH,embedding_size, Labs, NEURONS):
            self.model = Sequential()
            self.CONTEXT_LENGTH = CONTEXT_LENGTH
            self.NEURONS = NEURONS
            self.embedding_size = embedding_size
            self.Labels_one_hot_len = Labs
    
    #For attention model

    def get_H_n(self,X):
        ans = X[:, -1, :]  # get last element from time dim
        return ans


    def get_Y(self,X, xmaxlen):
        return X[:, :xmaxlen, :]  # get first xmaxlen elem from time dim


    def get_R(self,X):
        Y, alpha = X[0], X[1]
        ans = K.T.batched_dot(Y, alpha)
        return ans
    

    
   
    def structure_11_simple_attention_dot(self):
        #keras.layers.merge.Dot(axes, normalize=False)
        L = self.CONTEXT_LENGTH
        NEURONS = self.NEURONS
        encoding_input = Input(shape=(L,self.embedding_size))
        drop_out = Dropout(0.1, name='dropout')(encoding_input)
        lstm_fwd = LSTM(NEURONS, return_sequences=True, name='lstm_fwd')(drop_out)
        lstm_bwd = LSTM(NEURONS, return_sequences=True, go_backwards=True, name='lstm_bwd')(drop_out)
        bilstm = merge([lstm_fwd, lstm_bwd], name='bilstm', mode='concat')
        drop_out = Dropout(0.1)(bilstm)
        #
        # compute importance for each step
        attention = Dense(1, activation='tanh')(drop_out)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        #attention = RepeatVector(NEURONS*2)(attention) #bidirectional attention model
        #attention = Permute([2, 1])(attention)#what is this step doing?
        sent_representation = merge([attention,drop_out], dot_axes=1, mode='dot')
        #sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(NEURONS*2,))(sent_representation)
        out_relu =  Dense(NEURONS, activation = "relu")(sent_representation)
        out = Dense(self.Labels_one_hot_len, activation='softmax')(out_relu)
        output = out
        model = Model(input=[encoding_input], output=output)
        return model
    
        
    
   
    def structure_11_cnn_attention_dot(self):
        #keras.layers.merge.Dot(axes, normalize=False)
        L = self.CONTEXT_LENGTH
        NEURONS = self.NEURONS
        encoding_input = Input(shape=(L,self.embedding_size))
        drop_out = Dropout(0.1, name='dropout')(encoding_input)
        modconv = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(drop_out)
        maxp = MaxPooling1D(pool_size=1)(modconv)
        lstm_fwd = LSTM(NEURONS, return_sequences=True, name='lstm_fwd')(maxp)
        lstm_bwd = LSTM(NEURONS, return_sequences=True, go_backwards=True, name='lstm_bwd')(maxp)
        bilstm = merge([lstm_fwd, lstm_bwd], name='bilstm', mode='concat')
        drop_out = Dropout(0.1)(bilstm)
        #
        # compute importance for each step
        attention = Dense(1, activation='tanh')(drop_out)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        #attention = RepeatVector(NEURONS*2)(attention) #bidirectional attention model
        #attention = Permute([2, 1])(attention)#what is this step doing?
        sent_representation = merge([attention,drop_out], dot_axes=1, mode='dot')
        #sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(NEURONS*2,))(sent_representation)
        out_relu =  Dense(NEURONS, activation = "relu")(sent_representation)
        out = Dense(self.Labels_one_hot_len, activation='softmax')(out_relu)
        output = out
        model = Model(input=[encoding_input], output=output)
        return model
    
    
    def structure_11_cnn_attention_dot2(self):
        #keras.layers.merge.Dot(axes, normalize=False)
        from keras.layers.merge import Concatenate        
        from keras.layers.merge import Dot

        
        L = self.CONTEXT_LENGTH
        NEURONS = self.NEURONS
        encoding_input = Input(shape=(L,self.embedding_size))
        drop_out = Dropout(0.1, name='dropout')(encoding_input)
        modconv = Conv1D(filters=64, kernel_size=2, padding='same', activation='relu')(drop_out)
        maxp = MaxPooling1D(pool_size=2)(modconv)
        lstm_fwd = LSTM(NEURONS, return_sequences=True, name='lstm_fwd')(maxp)
        lstm_bwd = LSTM(NEURONS, return_sequences=True, go_backwards=True, name='lstm_bwd')(maxp)
        bilstm = Concatenate(name='bilstm')([lstm_fwd, lstm_bwd])
        drop_out = Dropout(0.1)(bilstm)
        #
        # compute importance for each step
        attention = Dense(1, activation='tanh')(drop_out)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        #attention = RepeatVector(NEURONS*2)(attention) #bidirectional attention model
        #attention = Permute([2, 1])(attention)#what is this step doing?
        sent_representation = Dot(axes=1)([attention,drop_out])
        #sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(NEURONS*2,))(sent_representation)
        out_relu =  Dense(NEURONS, activation = "relu")(sent_representation)
        out = Dense(self.Labels_one_hot_len, activation='softmax')(out_relu)
        output = out
        model = Model(inputs=[encoding_input], outputs=output)
        return model
    
    
    def structure_11_cnn_attention_dot4(self):
        #keras.layers.merge.Dot(axes, normalize=False)
        from keras.layers.merge import Concatenate
        from keras.layers import Bidirectional
        from keras.layers.merge import Dot
        import keras
        from keras_self_attention import SeqSelfAttention
        
        L = self.CONTEXT_LENGTH
        NEURONS = self.NEURONS
        encoding_input = Input(shape=(L,self.embedding_size))
        drop_out = Dropout(0.1, name='dropout')(encoding_input)
        modconv = Conv1D(filters=64, kernel_size=2, padding='same', activation='relu')(drop_out)
        maxp = MaxPooling1D(pool_size=2)(modconv)
       
        bilstm = keras.layers.Bidirectional(keras.layers.LSTM(units=NEURONS,
                                                    return_sequences=True))(maxp)
        
        drop_out = Dropout(0.1)(bilstm)
        attention = SeqSelfAttention(
                       attention_type=SeqSelfAttention.ATTENTION_TYPE_ADD,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_activation = 'softmax',
                       attention_regularizer_weight=1e-4,
                       name='Attention')(drop_out)
        
        
        #
        # compute importance for each step
        #attention = Dense(1, activation='tanh')(att)
        attention = Flatten()(attention)
        
        #attention = Activation('softmax')(attention)
        #attention = RepeatVector(NEURONS*2)(attention) #bidirectional attention model
        #attention = Permute([2, 1])(attention)#what is this step doing?
        #sent_representation = Dot(axes=1)([attention,drop_out])
        #sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(NEURONS*2,))(sent_representation)
        #attention = GlobalAvgPool1D()(attention)
        #attention = BatchNormalization(axis=-1)(attention)

        out_relu =  Dense(NEURONS, activation = "relu")(attention)
        out = Dense(self.Labels_one_hot_len, activation='softmax')(out_relu)
        out2 = Dense(1, activation='sigmoid')(out_relu)

        output = [out,out2]
        model = Model(inputs=[encoding_input], outputs=output)
        return model
    
    
    
    def structure_11_cnn_attention_dot3(self):
        #keras.layers.merge.Dot(axes, normalize=False)
        from keras.layers.merge import Concatenate
        from keras.layers import Bidirectional
        from keras.layers.merge import Dot
        import keras
        from keras_self_attention import SeqSelfAttention
        
        L = self.CONTEXT_LENGTH
        NEURONS = self.NEURONS
        encoding_input = Input(shape=(L,self.embedding_size))
        drop_out = Dropout(0.1, name='dropout')(encoding_input)
        modconv = Conv1D(filters=64, kernel_size=2, padding='same', activation='relu')(drop_out)
        maxp = MaxPooling1D(pool_size=2)(modconv)
       
        bilstm = keras.layers.Bidirectional(keras.layers.LSTM(units=NEURONS,
                                                    return_sequences=True))(maxp)
        #maxpoo = GlobalMaxPool1D()(bilstm)

        drop_out = SpatialDropout1D(0.1)(bilstm)
        attention = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_activation = 'softmax',
                       attention_regularizer_weight=1e-4,
                       name='Attention')(drop_out)
        
        #
        # compute importance for each step
        #attention = Dense(1, activation='tanh')(att)
        attention = Flatten()(attention)
        #attention = Activation('softmax')(attention)
        #attention = RepeatVector(NEURONS*2)(attention) #bidirectional attention model
        #attention = Permute([2, 1])(attention)#what is this step doing?
        #sent_representation = Dot(axes=1)([attention,drop_out])
        #sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(NEURONS*2,))(sent_representation)
        
        #attention = keras.layers.concatenate([attention,])
        x_1 = DropConnect(Dense(32, activation="relu"), prob=0.2)(attention)
    
        x_2 = Capsule(num_capsule=10, dim_capsule=10, routings=4, share_weights=True)(bilstm)
        x_2 = Flatten()(x_2)
        x_2 = DropConnect(Dense(32, activation="relu"), prob=0.2)(x_2)
        
        conc = concatenate([x_1, x_2])
        
        out_relu =  Dense(NEURONS, activation = "relu")(conc)
        out = Dense(self.Labels_one_hot_len, activation='softmax')(out_relu)
        
        out2 = Dense(1, activation='sigmoid')(out_relu)

        output = out
        model = Model(inputs=[encoding_input], outputs=[output,out2])
        return model
    
    def structure_11_cnn_attention_capsule_1obj(self):
        #keras.layers.merge.Dot(axes, normalize=False)
        from keras.layers.merge import Concatenate
        from keras.layers import Bidirectional
        from keras.layers.merge import Dot
        import keras
        from keras_self_attention import SeqSelfAttention
        
        L = self.CONTEXT_LENGTH
        NEURONS = self.NEURONS
        encoding_input = Input(shape=(L,self.embedding_size))
        drop_out = Dropout(0.1, name='dropout')(encoding_input)
        modconv = Conv1D(filters=64, kernel_size=2, padding='same', activation='relu')(drop_out)
        maxp = MaxPooling1D(pool_size=2)(modconv)
       
        bilstm = keras.layers.Bidirectional(keras.layers.LSTM(units=NEURONS,
                                                    return_sequences=True))(maxp)
        #maxpoo = GlobalMaxPool1D()(bilstm)

        drop_out = SpatialDropout1D(0.1)(bilstm)
        attention = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_activation = 'softmax',
                       attention_regularizer_weight=1e-4,
                       name='Attention')(drop_out)
        
        #
        # compute importance for each step
        #attention = Dense(1, activation='tanh')(att)
        attention = Flatten()(attention)
        #attention = Activation('softmax')(attention)
        #attention = RepeatVector(NEURONS*2)(attention) #bidirectional attention model
        #attention = Permute([2, 1])(attention)#what is this step doing?
        #sent_representation = Dot(axes=1)([attention,drop_out])
        #sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(NEURONS*2,))(sent_representation)
        
        #attention = keras.layers.concatenate([attention,])
        x_1 = DropConnect(Dense(32, activation="relu"), prob=0.2)(attention)
    
        x_2 = Capsule(num_capsule=10, dim_capsule=10, routings=4, share_weights=True)(bilstm)
        x_2 = Flatten()(x_2)
        x_2 = DropConnect(Dense(32, activation="relu"), prob=0.2)(x_2)
        
        conc = concatenate([x_1, x_2])
        
        out_relu =  Dense(NEURONS, activation = "relu")(conc)
        out = Dense(self.Labels_one_hot_len, activation='softmax')(out_relu)
        
        output = out
        model = Model(inputs=[encoding_input], outputs=[output])
        return model
    
    
    def structure_11_LCA_attention_dot3(self):
        #keras.layers.merge.Dot(axes, normalize=False)
        from keras.layers.merge import Concatenate
        from keras.layers import Bidirectional
        from keras.layers.merge import Dot
        import keras
        from keras_self_attention import SeqSelfAttention
        
        L = self.CONTEXT_LENGTH
        NEURONS = self.NEURONS
        encoding_input = Input(shape=(L,self.embedding_size))
        drop_out = Dropout(0.1, name='dropout')(encoding_input)
        
        bilstm = keras.layers.Bidirectional(keras.layers.LSTM(units=NEURONS,
                                                    return_sequences=True))(drop_out)
        #maxpoo = GlobalMaxPool1D()(bilstm)

        drop_out = SpatialDropout1D(0.1)(bilstm)
        attention = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_activation = 'softmax',
                       attention_regularizer_weight=1e-4,
                       name='Attention')(drop_out)
        
        #
        # compute importance for each step
        #attention = Dense(1, activation='tanh')(att)
        attention = Flatten()(attention)
        #attention = Activation('softmax')(attention)
        #attention = RepeatVector(NEURONS*2)(attention) #bidirectional attention model
        #attention = Permute([2, 1])(attention)#what is this step doing?
        #sent_representation = Dot(axes=1)([attention,drop_out])
        #sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(NEURONS*2,))(sent_representation)
        
        #attention = keras.layers.concatenate([attention,])
        x_1 = DropConnect(Dense(32, activation="relu"), prob=0.2)(attention)
    
        x_2 = Capsule(num_capsule=10, dim_capsule=10, routings=4, share_weights=True)(bilstm)
        x_2 = Flatten()(x_2)
        x_2 = DropConnect(Dense(32, activation="relu"), prob=0.2)(x_2)
        
        conc = concatenate([x_1, x_2])
        
        out_relu =  Dense(NEURONS, activation = "relu")(conc)
        out = Dense(self.Labels_one_hot_len, activation='softmax')(out_relu)
        out2 = Dense(1, activation='sigmoid')(out_relu)

        output = out
        model = Model(inputs=[encoding_input], outputs=[output,out2])
        return model
    
    
    def structure_11_LCA_attention_dot3_one_obj(self):
        #keras.layers.merge.Dot(axes, normalize=False)
        from keras.layers.merge import Concatenate
        from keras.layers import Bidirectional
        from keras.layers.merge import Dot
        import keras
        from keras_self_attention import SeqSelfAttention
        
        L = self.CONTEXT_LENGTH
        NEURONS = self.NEURONS
        encoding_input = Input(shape=(L,self.embedding_size))
        drop_out = Dropout(0.1, name='dropout')(encoding_input)
        
        bilstm = keras.layers.Bidirectional(keras.layers.LSTM(units=NEURONS,
                                                    return_sequences=True))(drop_out)
        #maxpoo = GlobalMaxPool1D()(bilstm)

        drop_out = SpatialDropout1D(0.1)(bilstm)
        
        attention = Attention(L)(drop_out)
        
        '''attention = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_activation = 'softmax',
                       attention_regularizer_weight=1e-4,
                       name='Attention')(drop_out)
        '''
        #
        # compute importance for each step
        #attention = Dense(1, activation='tanh')(att)
        #attention = Flatten()(attention)
        #attention = Activation('softmax')(attention)
        #attention = RepeatVector(NEURONS*2)(attention) #bidirectional attention model
        #attention = Permute([2, 1])(attention)#what is this step doing?
        #sent_representation = Dot(axes=1)([attention,drop_out])
        #sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(NEURONS*2,))(sent_representation)
        
        #attention = keras.layers.concatenate([attention,])
        x_1 = DropConnect(Dense(32, activation="relu"), prob=0.2)(attention)
    
        x_2 = Capsule(num_capsule=10, dim_capsule=10, routings=4, share_weights=True)(bilstm)
        x_2 = Flatten()(x_2)
        x_2 = DropConnect(Dense(32, activation="relu"), prob=0.2)(x_2)
        
        conc = concatenate([x_1, x_2])
        
        out_relu =  Dense(NEURONS, activation = "relu")(conc)
        out = Dense(self.Labels_one_hot_len, activation='softmax')(out_relu)
        
        output = out
        model = Model(inputs=[encoding_input], outputs=[output])
        return model
    
    
    
    def structure_11_cnn_attention_dot2_multi(self):
        #keras.layers.merge.Dot(axes, normalize=False)
        from keras.layers.merge import Concatenate        
        from keras.layers.merge import Dot

        
        L = self.CONTEXT_LENGTH
        NEURONS = self.NEURONS
        encoding_input = Input(shape=(L,self.embedding_size))
        drop_out = Dropout(0.1, name='dropout')(encoding_input)
        modconv = Conv1D(filters=64, kernel_size=2, padding='same', activation='relu')(drop_out)
        maxp = MaxPooling1D(pool_size=2)(modconv)
        lstm_fwd = LSTM(NEURONS, return_sequences=True, name='lstm_fwd')(maxp)
        lstm_bwd = LSTM(NEURONS, return_sequences=True, go_backwards=True, name='lstm_bwd')(maxp)
        bilstm = Concatenate(name='bilstm')([lstm_fwd, lstm_bwd])
        drop_out = Dropout(0.1)(bilstm)
        #
        # compute importance for each step
        attention = Dense(1, activation='tanh')(drop_out)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        #attention = RepeatVector(NEURONS*2)(attention) #bidirectional attention model
        #attention = Permute([2, 1])(attention)#what is this step doing?
        sent_representation = Dot(axes=1)([attention,drop_out])
        #sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(NEURONS*2,))(sent_representation)
        out_relu =  Dense(NEURONS, activation = "relu")(sent_representation)
        out = Dense(self.Labels_one_hot_len, activation='sigmoid')(out_relu)
        output = out
        model = Model(inputs=[encoding_input], outputs=output)
        return model
    
       
    def structure_convolutions_attention(self):
        L = self.CONTEXT_LENGTH
        NEURONS = self.NEURONS
        encoding_input = Input(shape=(self.embedding_size,L))
        
        conv1 = Conv1D(NEURONS, 5,padding='same',input_shape=(self.embedding_size,L))(encoding_input)
        act1 = Activation('relu')(conv1)
        conv2= Conv1D(NEURONS, 5,padding='same')(act1)
        act2 = Activation('relu')(conv2)
        drp1 = Dropout(0.1)(act2)
        maxp=  MaxPooling1D(pool_size=(8))(drp1)
        
        
        
        conv3 = Conv1D(NEURONS, 5,padding='same',)(maxp)
        act3= Activation('relu')(conv3)
        conv4= Conv1D(NEURONS, 5,padding='same',)(act3)
        act4= Activation('relu')(conv4)
        conv5= Conv1D(NEURONS, 5,padding='same',)(act4)
        act5 =Activation('relu')(conv5)
        drp2= Dropout(0.2)(act5)
        conv6=Conv1D(NEURONS, 5,padding='same',)(drp2)
        act6=Activation('relu')(conv6)
        
        lstm_fwd = LSTM(NEURONS, return_sequences=True, name='lstm_fwd')(act6)
        lstm_bwd = LSTM(NEURONS, return_sequences=True, go_backwards=True, name='lstm_bwd')(act6)
        bilstm = merge([lstm_fwd, lstm_bwd], name='bilstm', mode='concat')
        drop_out = Dropout(0.1)(bilstm)
        #
        # compute importance for each step
        attention = Dense(1, activation='tanh')(drop_out)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        
        sent_representation = merge([attention,drop_out], dot_axes=1, mode='dot')
       
        dns= Dense(self.Labels_one_hot_len)(sent_representation)
        outcv = Activation('softmax')(dns)
        model = Model(input=[encoding_input], output=outcv)
        #opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
        #model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
        return model
    
    
    
    def structure_convolutions(self):
        L = self.CONTEXT_LENGTH
        NEURONS = self.NEURONS
        model = Sequential()
        model.add(Conv1D(NEURONS, 5,padding='same',input_shape=(self.embedding_size,L)))
        model.add(Activation('relu'))
        model.add(Conv1D(NEURONS, 5,padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))
        model.add(MaxPooling1D(pool_size=(8)))
        model.add(Conv1D(NEURONS, 5,padding='same',))
        model.add(Activation('relu'))
        model.add(Conv1D(NEURONS, 5,padding='same',))
        model.add(Activation('relu'))
        model.add(Conv1D(NEURONS, 5,padding='same',))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Conv1D(NEURONS, 5,padding='same',))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(self.Labels_one_hot_len))
        model.add(Activation('softmax'))
        #opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
        #model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
        return model
    
    
    
    
    
    
    
    def structrue_11_simple_attention(self):
        L = self.CONTEXT_LENGTH
        NEURONS = self.NEURONS
        encoding_input = Input(shape=(L,self.embedding_size))
        drop_out = Dropout(0.1, name='dropout')(encoding_input)
        lstm_fwd = LSTM(NEURONS, return_sequences=True, name='lstm_fwd')(drop_out)
        lstm_bwd = LSTM(NEURONS, return_sequences=True, go_backwards=True, name='lstm_bwd')(drop_out)
        bilstm = merge([lstm_fwd, lstm_bwd], name='bilstm', mode='concat')
        drop_out = Dropout(0.1)(bilstm)
    
        # compute importance for each step
        attention = Dense(1, activation='tanh')(drop_out)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(NEURONS*2)(attention) #bidirectional attention model
        attention = Permute([2, 1])(attention)#what is this step doing?
        sent_representation = merge([drop_out, attention], mode='mul')
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(NEURONS*2,))(sent_representation)

        out = Dense(self.Labels_one_hot_len, activation='softmax')(sent_representation)
        output = out
        model = Model(input=[encoding_input], output=output)
        return model
    
    


    #simple bidirectional model.    
    def structrue_11_no_attention(self):
        L = self.CONTEXT_LENGTH
        NEURONS = self.NEURONS
        encoding_input = Input(shape=(L,self.embedding_size))
        drop_out = Dropout(0.1, name='dropout')(encoding_input)
        lstm_fwd = LSTM(NEURONS, return_sequences=True, name='lstm_fwd')(drop_out)
        lstm_bwd = LSTM(NEURONS, return_sequences=True, go_backwards=True, name='lstm_bwd')(drop_out)
        bilstm = merge([lstm_fwd, lstm_bwd], name='bilstm', mode='concat')
        drop_out = Dropout(0.1)(bilstm)

        encoded_text2 = Flatten()(drop_out)

        out = Dense(self.Labels_one_hot_len, activation='softmax')(encoded_text2)
        output = out
        model = Model(input=[encoding_input], output=output)
        return model
    
    
    
    def structrue_11_simple_attention_topic(self, topic):
        L = self.CONTEXT_LENGTH
        NEURONS = self.NEURONS
        encoding_input = Input(shape=(L,self.embedding_size))
        drop_out = Dropout(0.1, name='dropout')(encoding_input)
        lstm_fwd = LSTM(NEURONS, return_sequences=True, name='lstm_fwd')(drop_out)
        lstm_bwd = LSTM(NEURONS, return_sequences=True, go_backwards=True, name='lstm_bwd')(drop_out)
        bilstm = merge([lstm_fwd, lstm_bwd], name='bilstm', mode='concat')
        drop_out = Dropout(0.1)(bilstm)

        topic_input = Input(shape=(topic,))
        attention_topic = Dense(1, activation='tanh')(topic_input)
        attention_topic =  RepeatVector(L)(attention_topic) #bidirectional attention model
        attention_topic = Flatten()(attention_topic)



        # compute importance for each step
        attention = Dense(1, activation='tanh')(drop_out)
        attention = Flatten()(attention)

        attention = merge([attention,attention_topic], mode="mul")


        attention = Activation('softmax')(attention)
        attention = RepeatVector(NEURONS*2)(attention) #bidirectional attention model
        attention = Permute([2, 1])(attention)#what is this step doing?
        sent_representation = merge([drop_out, attention], mode='mul')
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(NEURONS*2,))(sent_representation)

        out = Dense(self.Labels_one_hot_len, activation='softmax')(sent_representation)
        #out2 = Dense(expected_length, activation='softmax')(sent_representation)
        #output = [out,out2]
        model = Model(input=[encoding_input,topic_input], output=out)

        return model    


    def structrue_11_simple_attention_topic_2(self, topic):
        L = self.CONTEXT_LENGTH
        NEURONS = self.NEURONS
        encoding_input = Input(shape=(L,self.embedding_size))
        drop_out = Dropout(0.1, name='dropout')(encoding_input)
        lstm_fwd = LSTM(NEURONS, return_sequences=True, name='lstm_fwd')(drop_out)
        lstm_bwd = LSTM(NEURONS, return_sequences=True, go_backwards=True, name='lstm_bwd')(drop_out)
        bilstm = merge([lstm_fwd, lstm_bwd], name='bilstm', mode='concat')
        drop_out = Dropout(0.1)(bilstm)
        
        topic_input = Input(shape=(topic,))
        attention_topic = Dense(NEURONS, activation='linear')(topic_input)
        attention_topic =  RepeatVector(L)(attention_topic) #bidirectional attention model
        #attention_topic = Flatten()(attention_topic)
        
        
        # compute importance for each step
        attention = Dense(NEURONS, activation='linear')(drop_out)
        #attention = Flatten()(attention)
        
        attention = merge([attention,attention_topic], mode="sum")
        
        
        attention = Dense(1, activation='tanh')(attention) # sequence + topic
        attention = Flatten()(attention)
        
        attention = Activation('softmax')(attention)
        attention = RepeatVector(NEURONS*2)(attention) #bidirectional attention model
        attention = Permute([2, 1])(attention)#what is this step doing?
        sent_representation = merge([drop_out, attention], mode='mul')
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(NEURONS*2,))(sent_representation)
        out = Dense(self.Labels_one_hot_len, activation='softmax')(sent_representation)
        #out2 = Dense(expected_length, activation='softmax')(sent_representation)
        #output = [out,out2]
        model = Model(input=[encoding_input,topic_input], output=out)
        
        return model  
    
    #this outputs topics -> for the next publication.
    def structrue_11_simple_attention_double(self):
        L = self.CONTEXT_LENGTH
        NEURONS = self.NEURONS
        encoding_input = Input(shape=(L,self.embedding_size))
        drop_out = Dropout(0.1, name='dropout')(encoding_input)
        lstm_fwd = LSTM(NEURONS, return_sequences=True, name='lstm_fwd')(drop_out)
        lstm_bwd = LSTM(NEURONS, return_sequences=True, go_backwards=True, name='lstm_bwd')(drop_out)
        bilstm = merge([lstm_fwd, lstm_bwd], name='bilstm', mode='concat')
        drop_out = Dropout(0.1)(bilstm)
    
        # compute importance for each step
        attention = Dense(1, activation='tanh')(drop_out)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(NEURONS*2)(attention) #bidirectional attention model
        attention = Permute([2, 1])(attention)#what is this step doing?
        sent_representation = merge([drop_out, attention], mode='mul')
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(NEURONS*2,))(sent_representation)

        out = Dense(self.Labels_one_hot_len, activation='softmax')(sent_representation)
        out2 = Dense(self.ONE_HOT_LEN, activation='softmax')(sent_representation)
        output = [out,out2]
        model = Model(input=[encoding_input], output=output)
        return model
    
    
    
    def structure_11(self):
        k = 256
        L = self.CONTEXT_LENGTH
        encoding_input = Input(shape=(self.CONTEXT_LENGTH-1,self.embedding_size))
        drop_out = Dropout(0.1, name='dropout')(encoding_input)
        lstm_fwd = LSTM(128, return_sequences=True, name='lstm_fwd')(drop_out)
        lstm_bwd = LSTM(128, return_sequences=True, go_backwards=True, name='lstm_bwd')(drop_out)
        bilstm = merge([lstm_fwd, lstm_bwd], name='bilstm', mode='concat')
        drop_out = Dropout(0.1)(bilstm)
        h_n = Lambda(self.get_H_n, output_shape=(k,), name="h_n")(drop_out)
        Y = Lambda(self.get_Y, arguments={"xmaxlen": L}, name="Y", output_shape=(L, k))(drop_out)
        Whn = Dense(k, W_regularizer=l2(0.01), name="Wh_n")(h_n)
        Whn_x_e = RepeatVector(L, name="Wh_n_x_e")(Whn)
        WY = TimeDistributed(Dense(k, W_regularizer=l2(0.01)), name="WY")(Y)
        merged = merge([Whn_x_e, WY], name="merged", mode='sum')
        M = Activation('tanh', name="M")(merged)

        alpha_ = TimeDistributed(Dense(1, activation='linear'), name="alpha_")(M)
        flat_alpha = Flatten(name="flat_alpha")(alpha_)
        alpha = Dense(L, activation='softmax', name="alpha")(flat_alpha)

        Y_trans = Permute((2, 1), name="y_trans")(Y)  # of shape (None,300,20)

        r_ = merge([Y_trans, alpha], output_shape=(k, 1), name="r_", mode=self.get_R)

        r = Reshape((k,), name="r")(r_)

        Wr = Dense(k, W_regularizer=l2(0.01))(r)
        Wh = Dense(k, W_regularizer=l2(0.01))(h_n)
        merged = merge([Wr, Wh], mode='sum')
        h_star = Activation('tanh')(merged)
        #out = Dense(ONE_HOT_LEN, activation='softmax')(h_star)
        out = Dense(self.Labels_one_hot_len, activation='softmax')(h_star)

        output = out
        model = Model(input=[encoding_input], output=output)
        return model

    
    def structure_11_double(self, Neurons):
        k = Neurons*2
        L = self.CONTEXT_LENGTH-1
        encoding_input = Input(shape=(self.CONTEXT_LENGTH-1,self.embedding_size))
        drop_out = Dropout(0.1, name='dropout')(encoding_input)
        lstm_fwd = LSTM(Neurons, return_sequences=True, name='lstm_fwd')(drop_out)
        lstm_bwd = LSTM(Neurons, return_sequences=True, go_backwards=True, name='lstm_bwd')(drop_out)
        bilstm = merge([lstm_fwd, lstm_bwd], name='bilstm', mode='concat')
        drop_out = Dropout(0.1)(bilstm)
        h_n = Lambda(self.get_H_n, output_shape=(k,), name="h_n")(drop_out)
        Y = Lambda(self.get_Y, arguments={"xmaxlen": L}, name="Y", output_shape=(L, k))(drop_out)
        Whn = Dense(k, W_regularizer=l2(0.01), name="Wh_n")(h_n)
        Whn_x_e = RepeatVector(L, name="Wh_n_x_e")(Whn)
        WY = TimeDistributed(Dense(k, W_regularizer=l2(0.01)), name="WY")(Y)
        merged = merge([Whn_x_e, WY], name="merged", mode='sum')
        M = Activation('tanh', name="M")(merged)

        alpha_ = TimeDistributed(Dense(1, activation='linear'), name="alpha_")(M)
        flat_alpha = Flatten(name="flat_alpha")(alpha_)
        alpha = Dense(L, activation='softmax', name="alpha")(flat_alpha)

        Y_trans = Permute((2, 1), name="y_trans")(Y)  # of shape (None,300,20)

        r_ = merge([Y_trans, alpha], output_shape=(k, 1), name="r_", mode=self.get_R)

        r = Reshape((k,), name="r")(r_)

        Wr = Dense(k, W_regularizer=l2(0.01))(r)
        Wh = Dense(k, W_regularizer=l2(0.01))(h_n)
        merged = merge([Wr, Wh], mode='sum')
        h_star = Activation('tanh')(merged)
        #
        out = Dense(self.Labels_one_hot_len, activation='softmax')(h_star)
        out2 = Dense(self.ONE_HOT_LEN, activation='softmax')(h_star)
       
        output = [out,out2]
        model = Model(input=[encoding_input], output=output)
        return model


    
    
    def structure_13(self):

        context_model = Sequential()
        context_model.add(Convolution2D(32, 2, 2, activation='relu', border_mode='same', input_shape=(1, self.CONTEXT_LENGTH-1, self.CONTEXT_LENGTH-1)))
        context_model.add(MaxPooling2D((2, 2)))
        context_model.add(Convolution2D(32, 2, 2, activation='relu', border_mode='same'))
        context_model.add(Convolution2D(32, 2, 2, activation='relu'))
        context_model.add(MaxPooling2D((2, 2)))
        context_model.add(Flatten())

        # now let's get a tensor with the output of our vision model:
        image_input = Input(shape=(1, self.CONTEXT_LENGTH-1, self.CONTEXT_LENGTH-1))
        encoded_context = context_model(image_input)

        k = 256
        L = self.CONTEXT_LENGTH-1
        encoding_input = Input(shape=(self.CONTEXT_LENGTH-1,self.embedding_size))
        drop_out = Dropout(0.1, name='dropout')(encoding_input)
        lstm_fwd = LSTM(128, return_sequences=True, name='lstm_fwd')(drop_out)
        lstm_bwd = LSTM(128, return_sequences=True, go_backwards=True, name='lstm_bwd')(drop_out)
        bilstm = merge([lstm_fwd, lstm_bwd], name='bilstm', mode='concat')
        drop_out = Dropout(0.1)(bilstm)
        h_n = Lambda(self.get_H_n, output_shape=(k,), name="h_n")(drop_out)
        Y = Lambda(self.get_Y, arguments={"xmaxlen": L}, name="Y", output_shape=(L, k))(drop_out)
        Whn = Dense(k, W_regularizer=l2(0.01), name="Wh_n")(h_n)
        Whn_x_e = RepeatVector(L, name="Wh_n_x_e")(Whn)
        WY = TimeDistributed(Dense(k, W_regularizer=l2(0.01)), name="WY")(Y)
        merged = merge([Whn_x_e, WY], name="merged", mode='sum')
        M = Activation('tanh', name="M")(merged)

        alpha_ = TimeDistributed(Dense(1, activation='linear'), name="alpha_")(M)
        flat_alpha = Flatten(name="flat_alpha")(alpha_)

        flat_beta  = merge([flat_alpha, encoded_context], name='m2', mode='concat')

        alpha = Dense(L, activation='softmax', name="alpha")(flat_beta)

        Y_trans = Permute((2, 1), name="y_trans")(Y)  # of shape (None,300,20)

        r_ = merge([Y_trans, alpha], output_shape=(k, 1), name="r_", mode=self.get_R)

        r = Reshape((k,), name="r")(r_)

        Wr = Dense(k, W_regularizer=l2(0.01))(r)
        Wh = Dense(k, W_regularizer=l2(0.01))(h_n)
        merged = merge([Wr, Wh], mode='sum')
        h_star = Activation('tanh')(merged)


        #out = Dense(ONE_HOT_LEN, activation='softmax')(h_star)
        #variation below
        out = Dense(self.Labels_one_hot_len, activation='softmax')(h_star)

        output = out
        self.model = Model(input=[encoding_input,image_input], output=output)
        return self.model   
    
    
    
   
    def structure_14(self):

        context_model = Sequential()
        context_model.add(Convolution2D(32, 2, 2, activation='relu', border_mode='same', input_shape=(1, 13, 13)))
        context_model.add(MaxPooling2D((2, 2)))
        context_model.add(Convolution2D(32, 2, 2, activation='relu', border_mode='same'))
        context_model.add(Convolution2D(32, 2, 2, activation='relu'))
        context_model.add(MaxPooling2D((2, 2)))
        context_model.add(Flatten())

        # now let's get a tensor with the output of our vision model:
        image_input = Input(shape=(1, self.CONTEXT_LENGTH-1, self.CONTEXT_LENGTH-1))
        encoded_context = context_model(image_input)
        repeated_context = RepeatVector(self.CONTEXT_LENGTH-1)(encoded_context)


        k = 256
        L = self.CONTEXT_LENGTH-1
        encoding_input = Input(shape=(self.CONTEXT_LENGTH-1,self.embedding_size))
        drop_out = Dropout(0.1, name='dropout')(encoding_input)



        flat_beta  = merge([drop_out, repeated_context], name='m2', mode='concat')


        lstm_fwd = LSTM(128, return_sequences=True, name='lstm_fwd')(flat_beta)
        lstm_bwd = LSTM(128, return_sequences=True, go_backwards=True, name='lstm_bwd')(flat_beta)
        bilstm = merge([lstm_fwd, lstm_bwd], name='bilstm', mode='concat')
        drop_out = Dropout(0.1)(bilstm)
        h_n = Lambda(self.get_H_n, output_shape=(k,), name="h_n")(drop_out)
        Y = Lambda(self.get_Y, arguments={"xmaxlen": L}, name="Y", output_shape=(L, k))(drop_out)
        Whn = Dense(k, W_regularizer=l2(0.01), name="Wh_n")(h_n)
        Whn_x_e = RepeatVector(L, name="Wh_n_x_e")(Whn)
        WY = TimeDistributed(Dense(k, W_regularizer=l2(0.01)), name="WY")(Y)
        merged = merge([Whn_x_e, WY], name="merged", mode='sum')
        M = Activation('tanh', name="M")(merged)

        alpha_ = TimeDistributed(Dense(1, activation='linear'), name="alpha_")(M)
        flat_alpha = Flatten(name="flat_alpha")(alpha_)

        alpha = Dense(L, activation='softmax', name="alpha")(flat_alpha)

        Y_trans = Permute((2, 1), name="y_trans")(Y)  # of shape (None,300,20)

        r_ = merge([Y_trans, alpha], output_shape=(k, 1), name="r_", mode=self.get_R)

        r = Reshape((k,), name="r")(r_)

        Wr = Dense(k, W_regularizer=l2(0.01))(r)
        Wh = Dense(k, W_regularizer=l2(0.01))(h_n)
        merged = merge([Wr, Wh], mode='sum')
        h_star = Activation('tanh')(merged)

        #out = Dense(ONE_HOT_LEN, activation='softmax')(h_star)
        #variation below
        out = Dense(self.Labels_one_hot_len, activation='softmax')(h_star)

        output = out
        model = Model(input=[encoding_input,image_input], output=output)
        return model    
    

    
    
    
    
    def structure_15(self,batch_size):
        k = 256
        L = self.CONTEXT_LENGTH-1
        encoding_input = Input(batch_shape=(batch_size,self.CONTEXT_LENGTH-1,self.embedding_size))
        drop_out = Dropout(0.5, name='dropout')(encoding_input)
        lstm_fwd = LSTM(128, batch_input_shape=(batch_size,self.CONTEXT_LENGTH-1, self.embedding_size), return_sequences=True, name='lstm_fwd', stateful=True)(drop_out)
        lstm_bwd = LSTM(128, batch_input_shape=(batch_size,self.CONTEXT_LENGTH-1, self.embedding_size), return_sequences=True, go_backwards=True, name='lstm_bwd', stateful=True)(drop_out)
        bilstm = merge([lstm_fwd, lstm_bwd], name='bilstm', mode='concat')
        drop_out = Dropout(0.5)(bilstm)
        h_n = Lambda(self.get_H_n, output_shape=(k,), name="h_n")(drop_out)
        Y = Lambda(self.get_Y, arguments={"xmaxlen": L}, name="Y", output_shape=(L, k))(drop_out)
        Whn = Dense(k, W_regularizer=l2(0.01), name="Wh_n")(h_n)
        Whn_x_e = RepeatVector(L, name="Wh_n_x_e")(Whn)
        WY = TimeDistributed(Dense(k, W_regularizer=l2(0.01)), name="WY")(Y)
        merged = merge([Whn_x_e, WY], name="merged", mode='sum')
        M = Activation('tanh', name="M")(merged)

        alpha_ = TimeDistributed(Dense(1, activation='linear'), name="alpha_")(M)
        flat_alpha = Flatten(name="flat_alpha")(alpha_)

        alpha = Dense(L, activation='softmax', name="alpha")(flat_alpha)

        Y_trans = Permute((2, 1), name="y_trans")(Y)  # of shape (None,300,20)

        r_ = merge([Y_trans, alpha], output_shape=(k, 1), name="r_", mode=self.get_R)

        r = Reshape((k,), name="r")(r_)

        Wr = Dense(k, W_regularizer=l2(0.01))(r)
        Wh = Dense(k, W_regularizer=l2(0.01))(h_n)
        merged = merge([Wr, Wh], mode='sum')
        h_star = Activation('tanh')(merged)

        #out = Dense(ONE_HOT_LEN, activation='softmax')(h_star)
        #variation below
        out = Dense(self.Labels_one_hot_len, activation='softmax')(h_star)

        output = out
        model = Model(input=[encoding_input], output=output)
        return model    

    
    
def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale

class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(tf.keras.backend.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = tf.keras.backend.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

# DropConnect
# https://github.com/andry9454/KerasDropconnect

from keras.layers import Wrapper

class DropConnect(Wrapper):
    def __init__(self, layer, prob=1., **kwargs):
        self.prob = prob
        self.layer = layer
        super(DropConnect, self).__init__(layer, **kwargs)
        if 0. < self.prob < 1.:
            self.uses_learning_phase = True

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(DropConnect, self).build()

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def call(self, x):
        if 0. < self.prob < 1.:
            self.layer.kernel = K.in_train_phase(K.dropout(self.layer.kernel, self.prob), self.layer.kernel)
            self.layer.bias = K.in_train_phase(K.dropout(self.layer.bias, self.prob), self.layer.bias)
        return self.layer.call(x)

    
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
