from __future__ import division, print_function

import sys
import os
import glob
import re
import numpy as np

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import sqlite3

from warnings import filterwarnings
filterwarnings("ignore")
from flask import Flask
from flask import render_template, request, redirect, send_file
import os

from matplotlib import cm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

static_path = 'static'
app.config['IMAGE_UPLOADS'] = static_path

model_path2 = 'svm.h5'

CTS = load_model(model_path2)
from keras.preprocessing.image import load_img, img_to_array

def model_predict2(image_path,model):
    image = load_img(image_path,target_size=(224,224))
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image,axis=0)
    
    result = np.argmax(model.predict(image))
     
    if result == 0:
        return "BENIGN","result.html"        
    elif result == 1:
        return "MALIGNANT","result.html"
   
@app.route("/")
def index():
    return render_template("home.html")

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("index.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signup.html")

@app.route('/home')
def home():
	return render_template('index.html')

@app.route('/segment', methods = ['GET','POST'])
def segment():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            filename = "input_image.bmp"
            image.save(os.path.join(app.config['IMAGE_UPLOADS'], filename))
            print("Saved!")
            convert(filename)

            img = Image.open('static/'+filename)
            new_img = img.resize( (256, 256) )
            new_img.save( UPLOAD_FOLDER + '/'+'image.png', 'png')
            file_path = 'static/uploads/image.png'

            print("@@ Predicting class......")
            pred, output_page = model_predict2(file_path,CTS)
                
            return render_template(output_page, pred_output = pred)

@app.route('/predict2',methods=['GET','POST'])
def predict2():
   
    if request.files:
        file = request.files['files'] 
        #image = request.files["files"]
        filename = file.filename        
        
            
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        
        pred, output_page = model_predict2(file_path,CTS)

        filename = "input_image.bmp"
        image.save(os.path.join(app.config['IMAGE_UPLOADS'], filename))
        # print("Saved!")
        convert(file_path)
        # print("Converted!")
              
        return render_template(output_page, pred_output = pred, img_src=UPLOAD_FOLDER + file.filename)

def enhance(img):
    sub = (CTS.predict(img)).flatten()

    for i in range(len(sub)):
        if sub[i] > 0.5:
            sub[i] = 1
        else:
            sub[i] = 0
    return sub

def get_segment_crop(img,tol=0, mask=None):
    if mask is None:
        mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]

def convert(filename):
    filename = os.path.join(static_path,filename)
    inp_image = np.array(Image.open(filename).resize((224,224)))
    inp_image = np.expand_dims(inp_image,axis=0)

    img_pred = enhance(inp_image)
    #img_crop = get_segment_crop(img=inp_image, mask= img_pred)
   

    im_1 = Image.fromarray(np.uint8(cm.gist_earth(img_pred)*255))
    im_1 = im_1.convert("L")
    im_1.save(os.path.join(static_path,'segmented.bmp'))
   
@app.after_request
def add_header(r):
    
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/notebook')
def notebook():
	return render_template('Notebook.html')

if __name__ == '__main__':
    app.run(debug=False)