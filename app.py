from flask import Flask, render_template, request
import cv2
import numpy as np
import pickle
from PIL import Image

app = Flask(__name__)

# Function to predict the fingerprint reconstruction
def reconstruct_fingerprint(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(224,224))

    img = np.asarray(img)
    img = img.astype('float32')

    img = img / np.max(img)

    img = img.reshape(-1, 224,224, 1)

    model = pickle.load(open("finalized_model.sav", 'rb'))
    res = model.predict(img)

    res = np.reshape(res, (224, 224, 3))

    res = (res * 255).astype(np.uint8)

    res = Image.fromarray(res)

    res.save('static/reconstructed.jpg')

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    original_image_path = None
    reconstructed_image_path = None

    if request.method == 'POST':
        image = request.files['image']
        if image:
            img_path = "static/real_data/" + image.filename

            original_image_path = img_path
            reconstructed_img = reconstruct_fingerprint(img_path)
            
            reconstructed_image_path = "static/reconstructed.jpg"
            
    return render_template('index.html', original_image=original_image_path, reconstructed_image=reconstructed_image_path)


if __name__ == '__main__':
    app.run(debug=True)
