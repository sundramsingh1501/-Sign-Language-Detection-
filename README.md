Sign Language Detection

An interactive Sign Language Detection web app that recognizes human actions/sign gestures using machine learning and computer vision techniques. This project is deployed live on Hugging Face Spaces (Gradio).

🔗 Live Demo: https://huggingface.co/spaces/sundram1501/Sign_Language_Detection

🧠 Overview

This project uses hand pose extraction + deep learning to detect sign language gestures from video or webcam input. It’s designed to help bridge communication gaps by translating hand gestures into text (or later audio). 
GitHub

🚀 Features

📹 Real-time sign gesture detection

🔍 Trained ML model for recognizing specific action/sign patterns

🧪 Easy-to-use Gradio web interface (Hugging Face Space)

🧰 Includes Jupyter Notebook and app.py for local testing

📌 Demo

👇 Try the live app here:
🔗 https://huggingface.co/spaces/sundram1501/Sign_Language_Detection

📁 Project Structure
📦 -Sign-Language-Detection-
├── app.py                       # Main application code
├── sign_language_detection.ipynb  # Notebook with model & processing
├── action.h5                    # Trained model file
└── README.md                   # This documentation

🛠️ Installation (Local)

To run locally:

# Clone repository
git clone https://github.com/sundramsingh1501/-Sign-Language-Detection-.git
cd -Sign-Language-Detection-

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate.bat     # Windows

# Install dependencies
pip install -r requirements.txt

▶️ Run

To launch locally:

python app.py


Then open in browser:

http://localhost:7860

📌 Model Details

✔ Uses a trained neural network model (action.h5) to classify sign gestures.
✔ The notebook contains all steps from preprocessing to training and evaluation.

📦 Dependencies

Make sure you have the following installed:

Python 3.8+

OpenCV

TensorFlow / Keras

Gradio (for Spaces deployment)

Mediapipe (optional for pose keypoint extraction)

(Exact versions should be in requirements.txt)

🧠 How It Works

Video frames are captured from webcam or video file

Hand landmarks are detected and passed to the ML model

The model returns a predicted sign/action

Output is shown in real-time on the UI

This general pipeline is common in sign gesture detection projects using CV + ML. 
GitHub

📈 Future Improvements

✨ Add support for more sign vocabularies (A–Z, numbers, sentences)
✨ Integrate text-to-speech conversion for accessibility
✨ Improve accuracy with larger datasets

❤️ Contributing

Contributions are welcome! You can:

Report bugs

Add new gestures or datasets

Improve UI/UX
Extend to more languages

Feel free to fork and submit a pull request.
