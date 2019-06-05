from flask import Flask
from flask import Flask, request, url_for

import os
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import json
import sys
import pandas as pd
import tensorflow as tf


UPLOAD_FOLDER = './'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024