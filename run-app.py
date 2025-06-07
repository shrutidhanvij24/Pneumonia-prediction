from flask import Flask, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('pneumonia_predictor.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Preprocess image function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Main Page with Upload Form
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if file is uploaded
        if 'file' not in request.files:
            return "No file uploaded. Go back and try again."

        img_file = request.files['file']
        if img_file.filename == '':
            return "No selected file. Go back and try again."

        # Save and process image
        img_path = "uploaded_image.jpg"
        img_file.save(img_path)

        # Predict
        img_array = preprocess_image(img_path)
        prediction = model.predict(img_array)[0][0]
        diagnosis = 'Pneumonia' if prediction > 0.5 else 'Normal'
        confidence = round(float(prediction) * 100, 2)

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body {{
                    background: url('/static/lungs.webp') no-repeat center center fixed;
                    background-size: cover;
                    background-color: #ADD8E6;
                    color: #fff;
                    font-family: Arial, sans-serif;
                    text-align: center;
                    padding-top: 10%;
                }}
                .container {{
                    background: rgba(0, 0, 0, 0.7);
                    padding: 20px;
                    border-radius: 20px;
                    display: inline-block;
                }}
                h1 {{ color: #FFD700; }}
                p {{ font-size: 20px; }}
                a {{
                    display: inline-block;
                    padding: 10px 20px;
                    background-color: #4CAF50;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                    margin-top: 20px;
                }}
                a:hover {{ background-color: #45a049; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Prediction Result:</h1>
                <p><b>Diagnosis:</b> {diagnosis}</p>
                <p><b>Confidence:</b> {confidence}%</p>
                <a href="/">Upload Another Image</a>
            </div>
        </body>
        </html>
        """

    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pneumonia Detection</title>
        <style>
            body {
                background: url('/static/lungs.webp') no-repeat center center fixed;
                background-size: cover;
                background-color: #ADD8E6;
                color: #fff;
                font-family: Arial, sans-serif;
                text-align: center;
                padding-top: 10%;
            }
            .container {
                background: rgba(0, 0, 0, 0.7);
                padding: 40px;
                border-radius: 20px;
                display: inline-block;
            }
            h1 { color: #FFD700; }
            label { font-size: 18px; }
            input, button {
                padding: 10px;
                margin-top: 20px;
                border-radius: 10px;
                border: none;
            }
            button {
                background-color: #4CAF50;
                color: white;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Pneumonia Detection</h1>
            <form method="post" enctype="multipart/form-data">
                <label for="file">Upload Chest X-ray Image:</label>
                <input type="file" name="file" required>
                <button type="submit">Predict</button>
            </form>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logs
    app.run(debug=True, port=5000)
