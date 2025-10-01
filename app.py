from flask import Flask, render_template, request, url_for
import tifffile as tiff
from preprocessing.preprocess_input import preprocess_input , normalize
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def head():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def input():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    img = tiff.imread(image_path)
    
    rgb = img[..., 1:4]
    rgb_norm = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        rgb_norm[..., c] = normalize(rgb[..., c])
    rgb_unit8 = (rgb_norm * 255).astype(np.uint8)
    rgb_img = Image.fromarray(rgb_unit8)
    rgb_filename = "rgb_" + os.path.splitext(imagefile.filename)[0] + ".png"
    rgb_path = os.path.join("static/uploads", rgb_filename)
    rgb_img.save(rgb_path, format="PNG")

    preprocessed_image = preprocess_input(image=img)
    print("Preprocessed shape:", preprocessed_image.shape)

    model = load_model(r'models/satalite_model_73_iou.keras')
    
    y_pred = model.predict(np.expand_dims(preprocessed_image, axis=0))[0]
    y_pred = np.squeeze(y_pred > 0.5)

    mask = y_pred.astype(np.uint8) * 255
    
    print("Final mask unique values:", np.unique(mask))
    print("Final white pixels:", np.sum(mask == 255))
    print("Final black pixels:", np.sum(mask == 0))

    mask_img = Image.fromarray(mask)
    mask_filename = "mask_" + os.path.splitext(imagefile.filename)[0] + ".png"
    mask_path = os.path.join("static/uploads", mask_filename)
    mask_img.save(mask_path, format="PNG")

    return render_template("index.html",
                          uploaded_image=url_for("static", filename=f"uploads/{rgb_filename}"),
                          mask_image=url_for("static", filename=f"uploads/{mask_filename}"))

if __name__ == "__main__":
    app.run(port=3000, debug=True)