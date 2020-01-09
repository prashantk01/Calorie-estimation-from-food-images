import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import time
import flask
from waitress import serve
import werkzeug
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from werkzeug.serving import WSGIRequestHandler
from tensorflow.keras.preprocessing import image
app = flask.Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def handle_request():
  files_ids = list(flask.request.files)
  print("\nNumber of Received Images : ", len(files_ids))
  image_num = 1
  stri=""
  label=['burger','burrito','club_sandwich','donuts','drink','french_fries','ice_cream','pizza','samosa']
  ctable=[295,206,220,300,41,312,207,266,262]
  for file_id in files_ids:
    print("\nSaving Image ", str(image_num), "/", len(files_ids))
    imagefile = flask.request.files[file_id]
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("Image Filename : " + imagefile.filename)
    image_name=imagefile.filename
    timestr = time.strftime("%Y%m%d-%H%M%S")
    imagefile.save(timestr+'_'+filename)
    # path="C:/Users/user/Desktop/Calorie_Estimation/python_code/burger.jpg"
    path="C:/Users/user/Desktop/Calorie_Estimation/python_code/"+image_name
    images = []
    # img = cv2.imread(path,1)
    images.append(path)
    for img in images:
  	  img = image.load_img(img, target_size=(224, 224))
  	  img = image.img_to_array(img)
  	  img = np.expand_dims(img, axis=0)
  	  img /= 255.
  	  model_best = load_model('best_model_pretrain.hdf5',compile = False)
  	  pred = model_best.predict(img)
  	  index = np.argmax(pred)
  	  pred_value = label[index]
  	  stri = pred_value+ " Calories [" + str(ctable[index])+"]"
    image_num = image_num + 1
    print("\n")
  return stri

WSGIRequestHandler.protocol_version = "HTTP/1.1"
serve(app,host="0.0.0.0",port=5000, url_scheme='https',cleanup_interval=200,threads=5,asyncore_loop_timeout=10
)
# app.run(host="0.0.0.0", port=5000, debug=True)