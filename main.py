import os
from NumpyEncoder import NumpyEncoder
#import magic
import urllib.request
from app import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import keras as k
import audioop
import os
import threading
import time
import traceback
import wave
#from Queue import Queue
from array import array
from collections import deque
from struct import pack
from sys import byteorder
import time
#from StringIO import StringIO
import datetime
import numpy as np
import pandas as pd
import pyaudio
import threading;
import requests
from keras.preprocessing.sequence import pad_sequences
from FeatureExtractor import FeatureExtractor
import EmotionExtractor_tf
import tensorflow as tf
import json
import Structures as str2
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras.utils import to_categorical
import keras
from keras.layers.merge import Concatenate
from keras.layers import Bidirectional, Dropout, Flatten
from keras.layers.merge import Dot
import keras
from keras_self_attention import SeqSelfAttention
from pydub import AudioSegment




graph = tf.get_default_graph()

ft = FeatureExtractor('mean_std.csv', BIT_PRECISION=16,sampling_rate=8000)
#em = EmotionExtractor_tf.EmotionExtractor('baseline_context5_conv_simple2.weights', 'mean_std.csv')



sequences_length = 256
classes = 4
context_len = 5

structure=  str2.Structures(context_len,136,classes,256)
my_attention_network = structure.structure_11_LCA_attention_dot3()
my_attention_network.load_weights("vera_4_classes_lca_attention.npy")

intermediate_model = Model(inputs=my_attention_network.input, 
                                 outputs=my_attention_network.layers[-3].output)


from keras.models import load_model



def create_stress_model():
    model_nn = Sequential()
    model_nn.add(Bidirectional(LSTM(64,return_sequences=True),input_shape=(256,256),name="LSTM_Layer")) # you can add a label
    model_nn.add(Dropout(0.1))

    model_nn.add(SeqSelfAttention(
                       attention_type=SeqSelfAttention.ATTENTION_TYPE_ADD,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_activation = 'softmax',
                       attention_regularizer_weight=1e-4,
                       name='Attention'))
        
    model_nn.add(Flatten())
    model_nn.add((Dense(64, activation="relu")))

    model_nn.add(Dense(1, activation="sigmoid")) # let's predict the class
    model_nn.compile(loss='binary_crossentropy',  optimizer='adam', metrics=['accuracy'])
    return model_nn


def create_sat_model():
    model_nn = Sequential()
    model_nn.add(Bidirectional(LSTM(64,return_sequences=True),input_shape=(256,256),name="LSTM_Layer")) # you can add a label
    model_nn.add(Dropout(0.1))

    model_nn.add(SeqSelfAttention(
                       attention_type=SeqSelfAttention.ATTENTION_TYPE_ADD,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_activation = 'softmax',
                       attention_regularizer_weight=1e-4,
                       name='Attention'))
        
    model_nn.add(Flatten())
    model_nn.add((Dense(64, activation="relu")))

    model_nn.add(Dense(1, activation="linear")) # let's predict the class
    model_nn.compile(loss='mse',  optimizer='adam', metrics=['accuracy'])
    return model_nn



stress_model = create_stress_model()
stress_model.load_weights('Stress_Detector.h5')

#sat_model = create_sat_model()
#sat_model.load_weights('SAT_Detector.h5')




ALLOWED_EXTENSIONS = set(['mp3'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
	if request.method == 'POST':
        # check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No file selected for uploading')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			flash('File(s) successfully uploaded')
			
			return analyse(filename)


#given the name of the mp3 file, it will produce a json, with the emotions associated.

def analyse(name):
    
    song = AudioSegment.from_file(name)    
    list_feature_extractions=ft.split_song(song)
	#collect sliding windows of arrays
    print(np.shape(list_feature_extractions))

    list_of_sliding_windows = []
    for i in range(5,len(list_feature_extractions),1):
      	list_of_sliding_windows.append(list_feature_extractions[i-5:i])
    
    with graph.as_default():
        #remember the double output
        emotions_output = my_attention_network.predict(np.array(list_of_sliding_windows))[0]
        #emotions_output2 = em.my_attention_network.predict(np.array(list_of_sliding_windows))   
    dict_emotions = {}
    starting_second = 15
    #many emotions down here
    
    #print(emotions_output)
    
    for i in range(0,len(emotions_output)):
        prediction = emotions_output[i]
        #prediction2=   emotions_output2[i]

        mydict ={"Anger/Disgust": prediction[0], "Surprise or Fear": prediction[1], "Happiness": prediction[2],
                                   "Neutral": prediction[3]}
        #mydict2 ={"Anger": prediction2[0], "Disgust": prediction2[1], "Fear": prediction2[3],
        #                           "Happiness": prediction2[5], "Neutral": prediction2[6], "Sadness": prediction2[2],
        #                           "Surprise": prediction2[4]}

        dict_emotions[starting_second] = mydict #(mydict,mydict2)
        starting_second = starting_second+3

    with graph.as_default():
        sequences_features= intermediate_model.predict(np.array(list_of_sliding_windows))	

    print(np.shape(sequences_features))    
    sequences_for_stress=pad_sequences([sequences_features],sequences_length, truncating = 'post', padding='post', value = np.zeros(256))
    
    with graph.as_default():
        prob_stress_confidence = stress_model.predict(sequences_for_stress)
        #sat_prediction = np.abs((sat_model.predict(sequences_for_stress)/7)*10)

	#create a Keras model here
	#apply feature extraction
    #output ={'stress': prob_stress_confidence.tolist(), 'sat':sat_prediction.tolist(), 'emotions_series':dict_emotions}
    output ={'stress': prob_stress_confidence.tolist(), 'emotions_series':dict_emotions}

    os.remove(name)
    
    return json.dumps(output, cls=NumpyEncoder)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8882)