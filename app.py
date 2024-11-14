from flask import Flask, render_template, request, redirect, url_for, send_file
import cv2
import pytesseract
import re
from scipy import ndimage
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Your existing code here (functions for rotate, preprocessing, aadhar_mask_and_ocr)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename != '':
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            # Process the image
            image = cv2.imread(image_path)
            if image is None:
                return "Error: Unable to load image."

            thres_image, resized_image = preprocessing(image)
            masked_image, UID = aadhar_mask_and_ocr(thres_image, resized_image)

            # Save and serve the masked image
            masked_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'masked_' + file.filename)
            cv2.imwrite(masked_image_path, masked_image)

            return send_file(masked_image_path, as_attachment=True)

    return '''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Upload Aadhaar Image</title>
    </head>
    <body>
      <h2>Upload Aadhaar Image for Masking</h2>
      <form action="/" method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload">
      </form>
    </body>
    </html>
    '''

if __name__ == "__main__":
    app.run(debug=True)
