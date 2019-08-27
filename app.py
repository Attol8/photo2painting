import sys
import os
sys.path.append(r'C:\Users\Jacopo\Desktop\photo2painting\photo2painting_webapp\cgan')
from flask import Flask, render_template, request
from commons import get_photo, tensor_to_PIL
from inference import get_painting_tensor
from flask_dropzone import Dropzone

app = Flask(__name__)

if __name__ == '__main__':
    app.run(debug= False)   
    
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        if 'file' not in request.files:
            print('file not uploaded')
            return

        file = request.files['file']
        print(file)
        image_bytes = file.read() #read uploaded file
        photo = get_photo(image_bytes) #make all the passage from image_bytes to painting_image shorter and faster
        painting_tensor = get_painting_tensor(photo) #run inference on the photo
        painting_image = tensor_to_PIL(painting_tensor) #transform output tensor to PIL Image
        painting_path = 'static\images\gen_painting' #path where to save static image
        painting_image.save(painting_path, format= 'JPEG')
        photo_path = 'static\images\photo.jpg'
        return render_template('result.html', painting_path = painting_path, photo_path= photo_path)
