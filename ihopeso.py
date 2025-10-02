import streamlit as st
import soundfile as sf
from pydub import AudioSegment
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os
import time
import base64
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
""", unsafe_allow_html=True)

import requests

def download_model_from_drive(file_id, destination):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    with open(destination, "wb") as f:
        f.write(response.content)

# Usage
download_model_from_drive("1rkcX2WkO8zMFGK5mfvSWP5JHLeHy8mXn", "model.h5")

# Constants
MAX_PAD_LEN = 174
DATA_DIR = "DATA"
TEST_DIR = os.path.join(DATA_DIR, "TEST")
FFMPEG_PATH = r"C:/ffmpeg/bin/ffmpeg.exe"
FFPROBE_PATH = r"C:/ffmpeg/bin/ffprobe.exe"

# Ensure TEST_DIR exists
os.makedirs(TEST_DIR, exist_ok=True)

# Set FFmpeg paths
AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffprobe = FFPROBE_PATH

# Load model and encoder
model = tf.keras.models.load_model("model.h5")
encoder = LabelEncoder()
encoder.classes_ = np.load("classes.npy", allow_pickle=True)

# Disease image mapping
disease_images = {
    "asthma": "Asthma.png",
    "bronchitis": "Bronchitis.png",
    "COPD" : "COPD.png",
    "Bronchiectasis": "Bronchiectasis.png",
    "Covid": "Covid.png",
    "Bronchiolitis" : "Bronchiolitis.png",
    "Healthy" : "Healthy.png",
    "pertussis" : "Pertussis.png",
    "Pneumonia" : 'Pneumonia.png',
    "URTI" : "URTI.png"
    
}

st.markdown("""
<style>
    body, .stApp {
        background: linear-gradient(to right, #1a2733, #2c3e50, #375a6c);
        font-family: 'Segoe UI', sans-serif;
        color: #e0f7fa;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        text-shadow: 0 0 10px #00ffff;
    }

    .stMarkdown, .stText, .stRadio label, .stSelectbox label, .stSlider label {
        color: #e0f7fa !important;
    }

    .stButton>button {
        background-color: #00ffff;
        color: #003153;
        font-weight: bold;
        border: none;
        padding: 0.6em 1.4em;
        border-radius: 10px;
        box-shadow: 0 0 12px #00ffff;
        transition: background-color 0.3s ease, box-shadow 0.3s ease;
        animation: pulse 2s infinite;
    }

    .stButton>button:hover {
        background-color: #00e6e6;
        box-shadow: 0 0 16px #00ffff;
        color: #001f2e;
    }

    .stSidebar {
        background-color: #001f2e;
        border-right: 2px solid #00ffff;
    }

    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        color: #ffffff !important;
    }

    .stSidebar p {
        color: #e0f7fa !important;
    }

    .stImage img {
        border-radius: 12px;
        box-shadow: 0 0 16px #00ffff;
    }

    .stAlert {
        border-radius: 10px;
        padding: 1em;
        font-weight: bold;
    }

    .stAlert[data-testid="stAlert-error"] {
        background-color: #2c0f0f;
        border-left: 5px solid #ff4e50;
    }

    .stAlert[data-testid="stAlert-success"] {
        background-color: #0f2c1f;
        border-left: 5px solid #00ffff;
    }

    .stAlert[data-testid="stAlert-info"] {
        background-color: #0f1f2c;
        border-left: 5px solid #00ffff;
    }

    .footer {
        text-align: center;
        padding: 20px;
        font-size: 0.9rem;
        color: #e0f7fa;
        margin-top: 40px;
        border-top: 2px solid #00ffff;
    }

    .footer a {
        color: #00ffff;
        text-decoration: none;
    }

    .footer a:hover {
        text-decoration: underline;
    }

    h1::after {
        content: ' ✨';
        color: #ffcc00;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 10px #00ffff; }
        50% { box-shadow: 0 0 20px #00e6e6; }
        100% { box-shadow: 0 0 10px #00ffff; }
    }
</style>
""", unsafe_allow_html=True)

# 🧠 Sidebar Instructions
with st.sidebar:
    st.header("🧠 How It Works")
    st.markdown("""
    - 🎙️ Record or 📁 upload a cough sample  
    - 🔍 MFCC features are extracted  
    - 🧠 CNN model predicts disease  
    - 🖼️ Visual + textual feedback provided  
    """)
    st.markdown("✅ Supported: Asthma, Bronchitis, Broncheictasis, Brochiolitis, Pertussis, URTI, COPD, Pneumonia, Covid, ")

# 🩺 Header
st.markdown("""
    <div style='text-align: center; padding: 20px; border-bottom: 2px solid #00ffd5;'>
        <h1 style='color: #00ffd5;'>🩺 Cough Disease Analyzer</h1>
        <p style='font-size: 18px; color: #ccc;'>From waveform to wellness—every breath tells a story.</p>
    </div>
""", unsafe_allow_html=True)



import streamlit.components.v1 as components


def show_disease_card(disease_name):
    # Normalize disease name
    normalized_name = disease_name.strip().lower().replace("_", " ")
    # Disease descriptions and links
    disease_info = {
        "asthma": {
            "desc": "A chronic condition causing inflammation and narrowing of the airways, leading to wheezing and shortness of breath.",
            "link": "https://www.who.int/news-room/fact-sheets/detail/asthma"
        },
         "bronchitis": {
            "desc": "Inflammation of the bronchial tubes, often caused by infection or irritants, resulting in coughing and mucus.",
            "link": "https://www.nhs.uk/conditions/bronchitis/"
        },
        "copd": {
            "desc": "Chronic Obstructive Pulmonary Disease—a progressive lung disease that makes breathing difficult.",
            "link": "https://www.who.int/news-room/fact-sheets/detail/chronic-obstructive-pulmonary-disease-(copd)"
        },
        "pneumonia": {
            "desc": "An infection that inflames the air sacs in one or both lungs, which may fill with fluid.",
        "   link": "https://www.who.int/news-room/fact-sheets/detail/pneumonia"
        },
        "urti": {
            "desc": "Upper Respiratory Tract Infection—includes common cold, sinusitis, and laryngitis.",
            "link": "https://www.ncbi.nlm.nih.gov/books/NBK532961/"
        },
        "covid": {
            "desc": "A viral respiratory illness caused by SARS-CoV-2, with symptoms ranging from mild to severe.",
            "link": "https://www.who.int/health-topics/coronavirus"
        },
        "healthy": {
            "desc": "No signs of respiratory disease detected.",
            "link": "https://www.cdc.gov/healthyweight/index.html"
        },
        "bronchiolitis": {
            "desc": "Common in children, this involves inflammation of the small airways in the lungs.",
            "link": "https://www.nhs.uk/conditions/bronchiolitis/"
        },
        "bronchiectasis": {
            "desc": "A condition where the airways become widened and scarred, leading to mucus buildup.",
            "link": "https://www.nhlbi.nih.gov/health/bronchiectasis"
        },
        "pertussis": {
            "desc": "Also known as whooping cough, a highly contagious bacterial infection causing severe coughing fits.",
            "link": "https://www.cdc.gov/pertussis/index.html"
        },
        "non cough": {
            "desc": "The input sound does not resemble a cough. Please try again with a clearer sample.",
            "link": "https://en.wikipedia.org/wiki/Cough"
        }
    }

    # Get description and link
    info = disease_info.get(normalized_name)
    if not info:
        st.warning(f"⚠️ No description found for '{disease_name}'. Please check your labels or add it to the dictionary.")
        return

    # Load image from local folder (relative path for Streamlit Cloud)
    image_filename = f"{normalized_name.replace(' ', '_')}.png"
    image_path = os.path.join("images", image_filename)  # assumes 'images/' folder is in repo

    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode()
        image_tag = f'data:image/png;base64,{encoded_image}'
    else:
        image_tag = ""  # fallback: no image

    # HTML card with embedded image
    html_content = f"""
    <html>
    <head>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{
                background-color: #0b1f3f;
                font-family: 'Segoe UI', sans-serif;
                padding: 40px;
                display: flex;
                justify-content: center;
            }}
            .diagnostic-card {{
                display: flex;
                flex-direction: row;
                background: linear-gradient(to right, #0b0f1a, #1a1f2e, #003153);
                color: white;
                border-radius: 16px;
                box-shadow: 0 10px 30px rgba(0,255,213,0.2);
                max-width: 900px;
                width: 100%;
                overflow: hidden;
                transition: transform 0.3s ease;
            }}
            .diagnostic-card:hover {{
                transform: scale(1.01);
            }}
            .card-img {{
                width: 40%;
                object-fit: contain;
                background-color: #0b1f3f;
                padding: 20px;
                border-right: 1px solid #003153;
            }}
            .card-body {{
                padding: 30px;
                width: 60%;
            }}
            .card-title {{
                font-size: 2rem;
                color: #00ffd5;
                margin-bottom: 10px;
            }}
            .card-text {{
                font-size: 1.1rem;
                line-height: 1.6;
                color: #cccccc;
            }}
            .btn-info {{
                margin-top: 20px;
                font-weight: bold;
                background-color: #1e90ff;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
               text-decoration: none;
                color: white;
                display: inline-block;
            }}
            .btn-info:hover {{
                background-color: #63b3ed;
            }}
        </style>
    </head>
    <body>
        <div class="diagnostic-card">
            {"<img src='" + image_tag + "' class='card-img' alt='Disease Image'>" if image_tag else ""}
            <div class="card-body">
                <h2 class="card-title">🩺 Prediction: {disease_name}</h2>
                <p class="card-text">{info['desc']}</p>
                <a href="{info['link']}" target="_blank" class="btn-info">🔎 Learn More</a>
            </div>
        </div>
    </body>
    </html>
    """


    # Render it
    components.html(html_content, height=400)


from gtts import gTTS
import tempfile

def speak_diagnosis(disease_name):
    messages = {
        "asthma": "Your cough pattern suggests signs of asthma. Please consult a respiratory specialist for further evaluation.",
        "bronchiectasis": "This may be indicative of bronchiectasis. A medical checkup is strongly recommended.",
        "bronchiolitis": "The sound resembles bronchiolitis. It's best to seek pediatric or pulmonary advice.",
        "bronchitis": "Your cough may be consistent with bronchitis. Consider seeing a doctor if symptoms persist.",
        "copd": "This could be a sign of chronic obstructive pulmonary disease. Please consult a pulmonologist.",
        "covid": "The cough characteristics match COVID-19 patterns. Please get tested and follow health guidelines.",
        "healthy": "Your cough does not show signs of disease. You appear to be on the safer side.",
        "pertussis": "This may resemble pertussis, also known as whooping cough. Medical attention is advised.",
        "pneumonia": "Your cough may be consistent with pneumonia. Please consult a healthcare provider promptly.",
        "urti": "This could be an upper respiratory tract infection. Rest and medical advice are recommended.",
        "non_cough": "The input sound does not resemble a cough. Please try again with a clearer sample."
    }

    message = messages.get(disease_name.lower(), "Unable to determine the condition. Please try again.")
    tts = gTTS(text=message, lang='en')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        with open(fp.name, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            st.markdown(f"""
                <audio autoplay>
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
            """, unsafe_allow_html=True)
def record_audio():
    st.subheader("🎙️ Record Your Cough")
    audio_data = st.audio_input("Tap to record")
    if audio_data:
        st.success("✅ Audio recorded!")
        st.audio(audio_data)
        return audio_data
    return None


# 📁 UPLOADING FUNCTION
def upload_audio_file():
    uploaded_file = st.file_uploader("📁 Upload audio file", type=["wav", "mp3", "m4a", "mp4"])
    if uploaded_file:
        st.success("✅ File uploaded!")

        file_ext = uploaded_file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as f:
            f.write(uploaded_file.read())
            raw_path = f.name

        if file_ext == "wav":
            wav_path = raw_path
        else:
            wav_path = raw_path.replace(f".{file_ext}", ".wav")
            try:
                if file_ext == "mp3":
                    audio = AudioSegment.from_mp3(raw_path)
                elif file_ext == "m4a":
                    audio = AudioSegment.from_file(raw_path, format="m4a")
                elif file_ext == "mp4":
                    audio = AudioSegment.from_file(raw_path, format="mp4")
                else:
                    st.error("❌ Unsupported format.")
                    return None
                audio.export(wav_path, format="wav")
            except Exception as e:
                st.error(f"❌ Conversion failed: {e}")
                return None

        st.audio(wav_path)
        st.info(f"📁 Saved as: {os.path.basename(wav_path)}")
        return wav_path
    else:
        st.info("📂 Waiting for file upload...")
        return None

def extract_mfcc(file_path):
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

        # Extract 40 MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        # Pad or truncate to MAX_PAD_LEN
        if mfccs.shape[1] < MAX_PAD_LEN:
            pad_width = MAX_PAD_LEN - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :MAX_PAD_LEN]

        # Reshape to (1, 40, MAX_PAD_LEN, 1)
        mfccs = mfccs.reshape(1, 40, MAX_PAD_LEN, 1)

        return mfccs

    except Exception as e:
        st.error(f"❌ MFCC extraction failed: {e}")
        return None
        
def classify_audio(mfcc):
    try:
        # Ensure correct shape for model input
        mfcc = mfcc.reshape(1, 40, MAX_PAD_LEN, 1)

        # Make prediction
        prediction = model.predict(mfcc)
        predicted_label = encoder.inverse_transform([np.argmax(prediction)])[0]

        # Normalize label for comparison
        label_clean = str(predicted_label).lower().replace("-", "_").replace(" ", "_").strip()

        # Handle non-cough case
        if "non" in label_clean and "cough" in label_clean:
            st.error("🔴 Please upload a valid cough file.")
        else:
            st.markdown(
                f"<h3 style='color: #00ff99;'>🟢 Predicted Disease: <strong>{predicted_label}</strong></h3>",
                unsafe_allow_html=True
            )
            show_disease_card(label_clean)
            speak_diagnosis(label_clean)

        return label_clean

    except Exception as e:
        st.error(f"❌ Classification failed: {e}")
        return None



st.set_page_config(page_title="Breathe", layout="centered")
audio_path = None  # ✅ Always initialize

option = st.radio("Choose input method", ["Record", "Upload"])

if option == "Record":
    audio_path = record_audio()
elif option == "Upload":
    audio_path = upload_audio_file()

if audio_path:
    st.audio(audio_path)
    mfcc_features = extract_mfcc(audio_path)
    if mfcc_features is not None:
        prediction = classify_audio(mfcc_features)

st.markdown("""
    <hr style="border: none; height: 2px; background: linear-gradient(to right, #00ffd5, #00bfff); margin-top: 40px;">

    <div style="text-align: center; padding: 20px; font-size: 0.9rem; color: #cccccc;">
        <p style='font-size: 18px; color: #ccc;'>This is an early diagnosis that endicates that you might have this disease, for a proper diagnosis and treatment it is recommended to visit a medical practitioners</p>
        Built with ❤️ by Anu · Powered by Streamlit · <a href="https://github.com/your-repo" target="_blank" style="color:#00ffd5; text-decoration: none;">GitHub</a>
    </div>
""", unsafe_allow_html=True)










