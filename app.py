import sys
import os
sys.path.append(r'C:\Users\Jacopo\Desktop\photo2painting\photo2painting_webapp\cgan')
from flask import Flask, render_template, request, send_file
from commons import input_photo, load_photo, tensor_to_PIL
from flask_dropzone import Dropzone
from pathlib import Path
from PIL import Image

app = Flask(__name__)

if __name__ == '__main__':
    app.run(debug= False)   
    
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('index.html')

@app.route('/create/', methods=['GET', 'POST'])
def create():
    if request.method == 'GET':
        ModelName = request.args.get('model')      

        return render_template('create.html', ModelName= ModelName)

    if request.method == 'POST':
                #get model and photo querys
                PhotoName = request.form['upload']
                ModelName = request.args.get('model')  

                #get input photo and resize it
                photo_path = os.path.join(app.root_path, 'static\images', PhotoName + '.jpg')
                photo = load_photo(photo_path) #load photo and scale it if necessary
                photo = input_photo(photo) #get tensor of photo (input of the model)

                #run inference on the photo
                painting_tensor = get_painting_tensor(photo, ModelName).cpu() 
                painting_image = tensor_to_PIL(painting_tensor) #transform output tensor to PIL Image

                #save painting output
                save_path = os.path.join(app.root_path, 'static\images', 'result' + '.jpg')
                painting_image.save(save_path)

                return render_template('result-download.html')    

@app.route('/result_download/', methods=['GET', 'POST'])
def result_download():
        if request.method == 'GET':
                return render_template('result-download.html')

@app.route('/download_files/')
def download_files():
	try:
		return send_file(os.path.join(app.root_path, 'static\images', 'result.jpg'), attachment_filename='photo2painting.jpg')
	except Exception as e:
		return str(e)
