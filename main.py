import os
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
from Queue import Queue
from array import array
from collections import deque
from struct import pack
from sys import byteorder
import time
from StringIO import StringIO
import datetime
import numpy as np
import pandas as pd
import pyaudio
import threading;
import requests
from keras.preprocessing.sequence import pad_sequences
import FeatureExtractor
import EmotionExtractor_tf


em = EmotionExtractor_tf.EmotionExtractor('baseline_context5_conv_simple2.weights', 'mean_std.csv', Conv=False)
ft = FeatureExtractor('mean_std.csv')

graph = tf.get_default_graph()


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
			analyse(filename)
			return redirect('/')


#given the name of the mp3 file, it will produce a json, with the emotions associated.

def analyse(name):
	list_feature_extractions=ft.split_song(name)
	#collect sliding windows of arrays

	list_of_sliding_windows = []
	for i in range(5,len(list_feature_extractions),1):
		list_of_sliding_windows.append(list_feature_extractions[i-5,i])


	emotions_output = model.predict(np.array(list_of_sliding_windows))
    dict_emotions = {}
    
    starting_second = 15
    #many emotions down here
    for i in range(0,len(emotion_output)):
    	prediction = emotion_output[i]
    	mydict ={"Anger": prediction[0], "Disgust": prediction[1], "Fear": prediction[3],
                                   "Happiness": prediction[5], "Neutral": prediction[6], "Sadness": prediction[2],
                                   "Surprise": prediction[4]}

    	dict_emotions[starting_second] = mydict
    	starting_second = starting_second+3

	sequences_features= intermediate_model.predict(list_of_sliding_windows)	


	sequence_for_stress=pad_sequence([sequences_features],250, truncating = 'post', padding='post', value = np.zeroes(250))
    

	prob_stress_confidence = stress_model.predict(sequences_features)

	#create a Keras model here
	#apply feature extraction
	output = ['stress':prob_stress_confidence, 'emotions_series':dict_emotions]


	return output

if __name__ == "__main__":
    app.run()