
from flask import Flask, render_template,request
#scientific computing library for saving, reading, and resizing images
#from scipy.misc import imsave, imread, imresize
#for matrix math
import numpy as np
#for importing our keras model
import keras.models
#for regular expressions, saves time dealing with string data
import re

#system level operations (like loading files)
import sys
#for reading operating system data
import os
#tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
from load import *


#initalize our flask app
app = Flask(__name__)



#global vars for easy reusability
global model, graph
#initialize these variables
#model, graph = init()
graph = tf.get_default_graph()
reponse="1"

from common import load_track, GENRES
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Dropout, Activation, \
        TimeDistributed, Convolution1D, MaxPooling1D, BatchNormalization
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from optparse import OptionParser
from sys import stderr, argv
import os
import keras
from keras.models import load_model

from create_data_pickle import get_default_shape

import time

import easygui



from flask import url_for
from werkzeug.utils import secure_filename

#from flask import send_from_directory

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'au'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




savedData=os.path.join(os.getcwd(),'data/data.pkl')
with open(savedData, 'rb') as f:
    data = pickle.load(f)

savedModel=os.path.join(os.getcwd(),'models/model.h5')
model = load_model(savedModel)
labels = {0:"blues",1:"classical",2:'country', 3:'disco',4:'hiphop',5:'jazz',6:'metal',
        7:'pop',8:'reggae',9:'rock'}


def ini():
	testset_path=easygui.fileopenbox()
	return 1


@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")

@app.route('/genres/')
def genre():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("genres.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    time.sleep(2)
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            print(filename,"",url_for('uploaded_file',
                                    filename=filename))
            global myFile
            myFile=os.path.join(app.config['UPLOAD_FOLDER'], filename)

            print(myFile)


    return render_template('index.html')



@app.route('/upload/',methods=['GET','POST'])
def upload():

	global testset_path
	testset_path=easygui.fileopenbox()
	return testset_path

@app.route('/predict/',methods=['GET','POST'])
def predict():



	print ('Model loaded')
	testset_path=myFile

	x = data['x']

	y = data['y']
	t=data['track_paths']
	print("classifying audio...")

	default_shape=(647, 128)


	TRACK_COUNT = 1000
	test_pos=TRACK_COUNT+100
	t1=np.zeros((TRACK_COUNT+200,) + default_shape, dtype=np.float32)
	t1=x

	print(testset_path)
	file_name='blues.00000.au'
	print('Processing', file_name)
    #path = os.path.join(testset_path, file_name)

	t1, _ = load_track(testset_path, default_shape)
	with graph.as_default():
		pred1=model.predict(np.array([t1]))[0]
		predict_class=np.argmax(np.round(pred1))
		index=predict_class
		time.sleep(3)
		print("Prediction for the selected song is ",labels[index])
		os.remove(myFile)
		print("debug3")
		#convert the response to a string
		#response = np.array_str(np.argmax(out,axis=1))
		return labels[index]


if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5001))

	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)

	#optional if we want to run in debugging mode
	#app.run(debug=True)
