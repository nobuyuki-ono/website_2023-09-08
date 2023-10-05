import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

app = Flask(__name__)

#アップロードする画像を保存場所の作成
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#画像の保存先がない場合、作成
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

#保存ができる拡張子の設定
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# ファイルの拡張子をチェックする機能
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 画像を読み込んで前処理する機能
def load_and_preprocess_image(file_path, target_size=(150, 150)):
    if not os.path.isfile(file_path):
        raise FileNotFoundError("The specified file does not exist.")
    try:
        img = Image.open(file_path)
        img = img.convert("RGB")  
        img = img.resize(target_size)  
        return img_to_array(img)
    except Exception as e:
        print(f"Error details: {str(e)}")
        raise ValueError("An error occurred during image loading and preprocessing.") from e

#画像の読み込みの処理
@app.route('/', methods=['GET', 'POST'])
def upload_and_classify():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file selected.'
        file = request.files['file']
        if file.filename == '':
            return 'File name is empty.'
        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            try:
                model_path = os.path.abspath("C:/test_python/app.py/models/fashion_model.h5")
                model = load_model(model_path)
                image_data = load_and_preprocess_image(file_path)
                prediction = model.predict(np.expand_dims(image_data, axis=0))
                label = 'Yes' if prediction[0][0] > 0.5 else 'No'
                
                if label == 'Yes':
                    return redirect(url_for('oshare_html', uploaded_image_filename=file.filename))
                else:
                    return redirect(url_for('dasai_html', uploaded_image_filename=file.filename, label=label))

            except Exception as e:
                return str(e)
        else:
            return 'Unsupported file extension. Supported extensions are {}.'.format(', '.join(ALLOWED_EXTENSIONS))
    return render_template('index.html')

#oshare.htmlへのルート設定
@app.route('/oshare.html')
def oshare_html():
    uploaded_image_filename = request.args.get('uploaded_image_filename', default='', type=str)
    return render_template('oshare.html', uploaded_image_filename=uploaded_image_filename)

#dasai.htmlへのルート設定
@app.route('/dasai.html')
def dasai_html():
    uploaded_image_filename = request.args.get('uploaded_image_filename', default='', type=str)
    label = request.args.get('label', default='', type=str)
    return render_template('dasai.html', uploaded_image_filename=uploaded_image_filename, label=label)

if __name__ == '__main__':
    app.run(debug=True)
