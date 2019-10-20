from flask import Flask, render_template, request
from werkzeug import secure_filename
import os
import sys
from PIL import Image
import pytesseract
from pytesseract import Output
import argparse
import cv2
import pickle

import numpy as np
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from tensorflow import keras
import tensorflow as tf
#from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array

from img_model import load_model
import logging

__author__ = 'Ayush Kumar <kayush206@gmail.com>'
__source__ = ''

app = Flask(__name__)
UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

HEADERS = {'content-type': 'application/json'}
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

MODEL_1 = 'models/text_model.save'
MODEL_2 = 'models/image_model_saved.h5'


# Image resizing utils
def resize_image_array(img, img_size_dims):
  img = cv2.resize(img, dsize=img_size_dims, 
                     interpolation=cv2.INTER_CUBIC)
  img = np.array(img, dtype=np.float32)
  return img

# load the text spam model
text_model = pickle.load(open(MODEL_1, 'rb'))
app.logger.info(text_model)
image_model = load_model(MODEL_2)
app.logger.info(image_model)

def text_spam(img):
  text = pytesseract.image_to_string(img)
  app.logger.info(type(text))
  app.logger.info(len(text))
  #TODO need the dict to pass in below function,
  #f = feature_extraction.text.CountVectorizer(stop_words = 'english')
  #X = f.fit_transform(text)
  #app.logger.info(X) 
  #_score = text_model.predict(X)
  return 1

def image_spam(img):
  app.logger.info("in image spam")

  return 0.6

@app.route("/spam_buster/api/v1/liveness")
def liveness():
  return 'API live!'

@app.route("/spam_buster/api/v1/model")
def model():
  return render_template("index.html")  

@app.route("/spam_buster/api/v1/about")
def about():
  return render_template("about.html")

@app.route('/spam_buster/api/v1/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      
        # TODO for empty submit

      def allowed_file(filename):
          return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

      # create a secure filename
      filename = secure_filename(f.filename)

      # save file to /static/uploads
      filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)
      f.save(filepath)
      
      # save the processed image in the /static/uploads directory
      #ofilename = os.path.join(app.config['UPLOAD_FOLDER'],"{}.png".format(os.getpid()))
      #cv2.imwrite(ofilename, gray)
      
      img = keras.preprocessing.image.load_img(filepath, target_size=(150, 150))
      app.logger.info(type(img))
      # convert to numpy array
      img_array =  np.array([keras.preprocessing.image.img_to_array(img)/255.])
      app.logger.info(img_array.dtype)
      app.logger.info(img_array.shape)


      # image classification on processed image
      pred = image_model.predict(img_array)
      pred = pred[0]

      # load the example image and convert it to grayscale
      image = cv2.imread(filepath)

      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      
      # apply thresholding to preprocess the image
      gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

      # apply median blurring to remove any blurring
      img = cv2.medianBlur(gray, 3)

      # perform OCR on the processed image
      _score = text_spam(img)

      out_img = image.copy()
      #_score = 0.67
      d = pytesseract.image_to_data(img, output_type=Output.DICT)
      n_boxes = len(d['level'])
      for i in range(n_boxes):
         (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
         cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

      pfilename = os.path.join(app.config['UPLOAD_FOLDER'],"out_"+filename)
      cv2.imwrite(pfilename, out_img)

      # remove the processed image
      # os.remove(ofilename)

      return render_template("uploaded.html", fname=filename, fname2="out_" +filename, 
        result='SPAM', score=str(_score+pred))

if __name__ == '__main__': 
   app.run(host="0.0.0.0", port=5000, debug=True)
