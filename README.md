Coffee disease detection Application Computer Science Project Portfolio for the study program MSc. Computer Science at IU University of Applied Sciences <br>
Student Name: Thomas Luke <br>
Matriculation Number: 4243144 <br>
Course Code: CSEMCSPCSP01 <br>

<h2> Project Description </h2> 

CoffeeGuard is a hybrid mobile-cloud AI application designed to help farmers identify coffee plant diseases quickly using smartphone images. The system uses a deep learning convolutional neural network (CNN) trained on coffee leaf images to classify diseases such as Leaf Rust, Leaf Miner, Phoma, and Healthy leaves.

The mobile application allows users to capture or upload a leaf image, which is then analyzed either:

Online using a backend API server for higher computational power, or

Offline using an embedded TensorFlow Lite model for real-time diagnosis in low-connectivity environments.

This hybrid architecture ensures the application remains functional even in rural farming regions with unreliable internet access. By providing instant disease detection and confidence scores, the system helps farmers take timely action to reduce crop loss and improve coffee yield.

<br><br>

<h2>  Project Overview </h2> <br>

This project is a hybrid mobile–cloud system for detecting coffee leaf diseases using deep learning.

It consists of:

📱 Mobile Application (Android – Flutter)

🖥️ Backend API (FastAPI)

🤖 Deep Learning Model (MobileNet-based CNN)

📦 TensorFlow Lite model for offline inference

The system supports:

Online cloud-based prediction

Offline on-device prediction (TensorFlow Lite)

<br> <br> <br>

<h2> Installation & Setup Guide </h2> <br>

<h3> 1. Clone the Respository </h3> <br>

      git clone https://github.com/your-username/coffee-disease-app.git 
      cd coffee-disease-app

<br>

<h2> 🖥️ Backend Setup (FastAPI) </h2> <br>

<b> Requirements </b> <br>

  Python 3.9+
  
  pip
  
  virtualenv (recommended) <br> <br>

<h3> Step 1: Create Virtual Environment </h3>

      cd backend <br>
      python -m venv venv <br>

Activate environment: <br>

<b> Windows </b> 

    venv\Scripts\activate  <br>

<b> Mac/Linux </b>

    source venv/bin/activate <br>

<H3> Step 2: Install Dependencies </H3>

    pip install -r requirements.txt <br>

<h3> Step 3: Run Backend Server </h3>

    uvicorn main:app --reload <br>

Server will run at http://127.0.0.1:8000 <br>
API documentation available at: http://127.0.0.1:8000/docs <br> <br>

<h2> 📱 Mobile App Setup (Flutter) </h2>

<h3> Requirements </h3> <br>

-Flutter SDK (latest stable) <br>
-Android Studio or VS Code <br>
-Android Studio or VS Code <br>

Check Flutter installation: 

      flutter doctor 

<br>

<h3> Step 1: Navigate to Mobile App Folder </h3> <br>


      cd mobile_app
<br>

<h3> Step 2: Install Dependencies </h3> <br>

      flutter pub get
<br>

<h3> Step 3: Run App </h3> <br>

      flutter run

<br> Make sure: Emulator is running OR USB debugging enabled on physical device <br> <br>

<h2> Model Setup </h2> <br>

If retraining the model: <br>

      <>BASH
      cd model
<br> Open: <br>

      training.ipynb
<br> Train model and export: 

      Python
      model.save("saved_model/") 

<br>

Convert to TensorFlow Lite: 

            Python
            converter = tf.lite.TFLiteConverter.from_saved_model("saved_model/")
            tflite_model = converter.convert()
            
            with open("model.tflite", "wb") as f:
                f.write(tflite_model)

<br> Place model.tflite inside: 

      mobile_app/assets/
<br> Update pubspec.yaml:

      assets:
        - assets/model.tflite


<br> <br>

<h3> Google Sign-In Setup </h3> <br>

1. Go to Google Cloud Console <br>

2. Create OAuth 2.0 credentials <br>

3. Add SHA-1 fingerprint <br>

4. Download google-services.json <br>

5. Place inside: mobile_app/android/app/ <br>

Add required dependencies in pubspec.yaml: 

      google_sign_in: ^6.0.0
      firebase_auth: ^4.0.0

Run: 

      Bash
      flutter pub get
<br>

<h3> Running in Offline Mode </h3> <br>

The app automatically switches to:

-Backend API when internet is available

-TensorFlow Lite when offline

Ensure:

-model.tflite exists in assets

-Proper asset path declared 

<br> <br>

<h2> (Optional) Docker Setup for Backend </h2> <br>

Build Docker image:

      docker build -t coffee-api .

Run container:

      docker run -p 8000:8000 coffee-api

<br>

<h2> API Endpoint Example </h2> <br>

POST /predict

Request:

      multipart/form-data
      image: file

<br>

multipart/form-data
image: file

      JSON
      {
        "prediction": "Coffee Leaf Rust",
        "confidence": 0.94
      }

<br>

<h3> Troubleshooting </h3>

<b> ❌ ModuleNotFoundError </b>

Ensure virtual environment is activated.

<b> ❌ Flutter not detecting device </b>

RUN: 

      flutter doctor

<b> Model not loading </b>

Verify:

Correct asset path

Correct TensorFlow Lite version


<br> <br>

<h2> Production Deployment </h2> <br>

Backend:

Deploy via Docker

Use Nginx + Gunicorn

Mobile:

Generate APK:

      Bash
      flutter build apk --release



<h2> System Architecuture </h2>





<img width="4436" height="396" alt="mermaid-diagram" src="https://github.com/user-attachments/assets/463c0e2d-2973-4d3d-adca-3fe8955ba195" />


<h3> Architecture Explanation </h3>

The system consists of three main components:

1️⃣ Mobile Application

Built using Flutter / React Native

Allows users to:

Capture coffee leaf images

Upload existing images

Receive disease diagnosis

Handles both online and offline inference <br>

2️⃣ Backend Server

Built with FastAPI / Flask

Hosts the trained deep learning model (.h5)

Receives image requests from the mobile app

Returns:

Disease classification

Prediction confidence score <br>

3️⃣ AI Model Layer

Two models are used:

Cloud Model

Full TensorFlow/Keras CNN

Higher accuracy

Runs on backend server

Offline Model

Converted TensorFlow Lite (.tflite)

Optimized for mobile devices

Enables offline disease detection <br>

<b> Workflow </b>

1. User captures a coffee leaf image.

2. Mobile app checks internet connectivity.

3. If online → image sent to backend server.

4. If offline → TensorFlow Lite model processes image locally.

5. Model predicts disease category.

6. App displays disease name and confidence score.

  

