import os
import numpy as np
from flask import Flask,request,render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model=load_model('ECG.h5')


@app.route('/')
def about():
    return render_template("about.html")

@app.route('/about')
def home():
    return render_template("about.html")

@app.route('/info')
def info():
    return render_template("info.html")

@app.route('/upload')
def test():
    return render_template("index6.html")


@app.route("/predict", methods=["GET", "POST"])  # route for our prediction
def upload():
    if request.method == 'POST':
        f = request.files['file']  # requesting the file
        basepath = os.path.dirname('__file__')  # storing the file directory
        filepath = os.path.join(basepath, "uploads", f.filename)  # storing the file in uploads folder
        f.save(filepath)  # saving the file

        img = image.load_img(filepath, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        pred = model.predict(x)
        y_pred = np.argmax(pred)
        print("prediction", y_pred)

        index = ['Left Bundle Branch Block', 'Normal', 'Premature Atrial Contraction',
                 'Premature Ventricular Contractions', 'Right Bundle Branch Block', 'Ventricular Fibrillation']
        result = str(index[y_pred])

        return result  # resturing the result
    return None

if __name__=="__main__":
    app.run(debug=False)