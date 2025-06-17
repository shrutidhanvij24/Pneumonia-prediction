# 🧬Pneumonia-prediction
This repository contains code for training a pneumonia detection model from chest X-ray images and a Flask-based web app to make predictions using the trained model.
## 📁Repository Contents
| File                     | Description                                                                 |
| ------------------------ | --------------------------------------------------------------------------- |
| model_training.py      | Script to train the pneumonia detection CNN model on chest X-ray dataset.   |
| rup-aap.py             | Flask web application to upload X-ray images and get pneumonia predictions. |
| pneumonia_predictor.h5 | Pre-trained model file (download separately, see below).                    |
| requirements.txt       | Python packages required to run both scripts.                               |
| README.md              | This file.                                                                  |
## ⚙️Setup Instructions
### 1. Clone the repository
git clone <https://github.com/shrutidhanvij24/Pneumonia-prediction>
cd <Pneumonia-prediction>
### 2. Install dependencies
Make sure you have Python 3.7+ installed. Then install required packages:
pip install -r requirements.txt
### 3. Train the Model (Optional)
If you want to train the model yourself:
Make sure you have the chest X-ray dataset downloaded and extracted.
Edit model-training.py to update dataset paths if needed.
Run the training script:
python model-training.py
The training process will save the model as pneumonia_predictor.h5 when done.
Note: Training requires a good GPU and enough disk space for the dataset.
### 🧠 Model Training
You can retrain the model using model_training.py if needed. The script loads a dataset from the directory structure, preprocesses the images, builds a CNN model, trains it, and saves the model as pneumonia_predictor.h5.
⚠️Dataset used: Kaggle Chest X-ray Pneumonia Dataset
### 5. Download Pre-trained Model (Recommended)
👉If you don’t want to train yourself, download the pre-trained model:
  https://drive.google.com/file/d/1jAaZ7-4NIPJyYBRCsxpGbFVf26sORBdH/view?usp=drive_link  
Download pneumonia_predictor.h5 (Google Drive)
Place the downloaded pneumonia_predictor.h5 file in the root folder of the repository (same folder as rup-aap.py).
### 6. Run the Flask Web Application
Start the Flask app by running:
python rup-app.py
You will see output indicating the server is running.
### 7. Access the App in Browser
Open your web browser
You will see the pneumonia detection web interface.
### 8. Use the App
Upload a chest X-ray image using the upload form.
Click Predict.
See the prediction result: whether pneumonia is detected or normal, with confidence score.
🌐Additional Notes
The app background image (lungs.webp) is optional. You can add your own image in a folder named static or remove the background code in rup-aap.py.
Make sure your Python virtual environment is activated when running the scripts (optional but recommended).
The prediction model expects images resized to 150x150 pixels (handled automatically).
Troubleshooting
If you get errors loading the model, confirm pneumonia_predictor.h5 is in the same directory as rup-aap.py.
For dataset issues during training, verify dataset paths and image files.

### ✍️ Author
Shruti Dhanvij  
Student of Compute Science Engineering & Data Science
## Thankyou!
