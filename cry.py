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

# COLOR
st.markdown("""
 <style>
    body, .stApp {
        background: linear-gradient(to right, #0b0f1a, #1a1f2e, #003153);
        color: #f0f0f0;
        font-family: 'Segoe UI', sans-serif;
    }

    h1, h2, h3, h4, h5, h6, p, div, span {
        color: #f0f0f0 !important;
    }

    .stButton>button {
        background-color: #003153;  /* Prussian blue */
        color: white;
        border: none;
        padding: 0.6em 1.4em;
        border-radius: 10px;
        font-weight: bold;
        transition: background-color 0.3s ease;
        box-shadow: 0 0 10px #00bfff;
    }

    .stButton>button:hover {
        background-color: #005f8f;
        color: #ffffff;
        box-shadow: 0 0 12px #00ffd5;
    }

    .stRadio label {
        color: #00ffd5 !important;
        font-weight: bold;
    }

    .stSlider .css-1y4p8pa {
        color: #ff6b6b !important;
    }

    .stMarkdown {
        color: #f0f0f0 !important;
    }

    .stSidebar {
        background-color: #111827;
        border-right: 2px solid #003153;
    }

    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4 {
        color: #00bfff !important;
    }

    .stSidebar p {
        color: #cccccc !important;
    }

    .stImage img {
        border-radius: 12px;
        box-shadow: 0 0 12px #00bfff;
    }

    .stAlert {
        border-radius: 10px;
        padding: 1em;
    }

    .stAlert[data-testid="stAlert-error"] {
        background-color: #2c0f0f;
        border-left: 5px solid #ff4e50;
    }

    .stAlert[data-testid="stAlert-success"] {
        background-color: #0f2c1f;
        border-left: 5px solid #00ffd5;
    }

    .stAlert[data-testid="stAlert-info"] {
        background-color: #0f1f2c;
        border-left: 5px solid #00bfff;
    }

    /* Add a fun accent to headers */
    h1::after {
        content: ' ✨';
        color: #ffcc00;
    }

    /* Add subtle animation to buttons */
    .stButton>button {
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 10px #00bfff; }
        50% { box-shadow: 0 0 20px #00ffd5; }
        100% { box-shadow: 0 0 10px #00bfff; }
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
        <p style='font-size: 18px; color: #ccc;'>this app is made using AI which cannot diagonis you, Please refer to a doctor for a proper diagonsis.</p>
    </div>
""", unsafe_allow_html=True)

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
            "link": "https://www.cdc.gov/pneumonia/index.html"
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
        "flu": {
            "desc": "Influenza—a viral infection causing fever, chills, and respiratory symptoms.",
            "link": "https://www.cdc.gov/flu/index.htm"
        },
        "common cold": {
            "desc": "A mild viral infection of the nose and throat, often causing sneezing and coughing.",
            "link": "https://www.nhs.uk/conditions/common-cold/"
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

    # Load image from local folder
    image_filename = f"{normalized_name.replace(' ', '_')}.png"
    image_path = os.path.join(os.getcwd(), image_filename)

    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode()
        image_html = f'<img src="data:image/png;base64,{encoded_image}" class="card-img-top" alt="{disease_name}" style="width:100%; height:200px; object-fit:cover; border-radius:12px 12px 0 0;">'
    else:
        image_html = ""

    import streamlit.components.v1 as components 
    html_content = f"""
    <html>
        <head>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{
                    background-color: #0b1f3f;
                    padding-top: 40px;
                    display: flex;
                    justify-content: center;
                    font-family: 'Segoe UI', sans-serif;
                }}
                .card {{
                    background-color: #112d5c;
                    color: white;
                    border-radius: 12px;
                    box-shadow: 0 8px 20px rgba(0,0,0,0.4);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                    max-width: 24rem;
                }}
                .card:hover {{
                    transform: scale(1.03);
                    box-shadow: 0 12px 24px rgba(0,0,0,0.5);
                }}
                .card-img-top {{
                    height: 180px;
                    object-fit: contain;
                    background-color: transparent;
                    padding: 10px;
                    border-bottom: 1px solid #ccc;
                }}
                .btn-info {{
                    font-weight: bold;
                    background-color: #1e90ff;
                    border: none;
                    transition: background-color 0.3s ease;
                }}
                .btn-info:hover {{
                    background-color: #63b3ed;
                }}
            </style>
        </head>
        <body>
            <div class="card mb-3">
                <img src="{image_html}" class="card-img-top" alt="Disease Icon">
                <div class="card-body">
                    <h5 class="card-title">🩺 Prediction: {disease_name}</h5>
                    <p class="card-text">{info['desc']}</p>
                    <a href="{info['link']}" target="_blank" class="btn btn-info mt-3">🔎 Learn More</a>
                </div>
            </div>
        </body>
    </html>
    """


    # Render it with components.html
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

from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import wave


def record_audio():
    st.subheader("🎙️ Record Your Cough")

    class AudioProcessor:
        def __init__(self):
            self.frames = []

        def recv(self, frame):
            audio = frame.to_ndarray()
            self.frames.append(audio)
            return frame

    # Start WebRTC stream
    ctx = webrtc_streamer(
        key="cough-recorder",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
        processor_factory=AudioProcessor,
    )

    # Simulated progress bar during recording
    if ctx.state.playing:
        st.info("🔴 Recording in progress...")
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.05)  # ~5 seconds total
            progress.progress(i + 1)


# Convert any audio to WAV
def convert_to_wav(file_path, file_ext):
    try:
        audio = AudioSegment.from_file(file_path, format=file_ext)
        wav_path = os.path.join(TEST_DIR, "converted.wav")
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        st.error(f"Conversion failed: {e}")
        return None

# Extract MFCC
def extract_mfcc(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        if mfccs.shape[1] < MAX_PAD_LEN:
            pad_width = MAX_PAD_LEN - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :MAX_PAD_LEN]
        return mfccs
    except Exception as e:
        st.error(f"MFCC extraction failed: {e}")
        return None



def classify_audio(mfcc):
    # Reshape input for model
    mfcc = mfcc.reshape(1, 40, MAX_PAD_LEN, 1)

    # Make prediction
    prediction = model.predict(mfcc)
    predicted_label = encoder.inverse_transform([np.argmax(prediction)])[0]

    # Normalize label for comparison
    label_clean = str(predicted_label).lower().replace("-", "_").replace(" ", "_").strip()



    # Handle non-cough case
    if "non" in label_clean and "cough" in label_clean:
        st.error("🔴 Please upload a valid cough file.")
        show_disease_card(label_clean)



    else:
        st.markdown(
            f"<h3 style='color: #00ff99;'>🟢 Predicted Disease: <strong>{predicted_label}</strong></h3>",
            unsafe_allow_html=True
        )
        show_disease_card(label_clean)
        speak_diagnosis(label_clean)

        
        

    return label_clean

# Input method
option = st.radio("Choose input method:", ["🎙️ Record Audio", "📁 Upload File"])

if option == "🎙️ Record Audio":
    duration = st.slider("Recording Duration (seconds)", 2, 10, 5)
    if st.button("Start Recording"):
        mp3_path = record_audio(duration=duration)
        if mp3_path:
            wav_path = convert_to_wav(mp3_path, "mp3")
            if wav_path:
                mfcc = extract_mfcc(wav_path)
                if mfcc is not None:
                    classify_audio(mfcc)

elif option == "📁 Upload File":
    uploaded_file = st.file_uploader("Upload .wav, .mp3, or .m4a", type=["wav", "mp3", "m4a"])
    if uploaded_file and st.button("Upload & Analyze"):
        file_ext = uploaded_file.name.split(".")[-1].lower()
        raw_path = os.path.join(TEST_DIR, f"uploaded.{file_ext}")
        with open(raw_path, "wb") as f:
            f.write(uploaded_file.read())

        wav_path = raw_path if file_ext == "wav" else convert_to_wav(raw_path, file_ext)
        if wav_path:
            mfcc = extract_mfcc(wav_path)
            if mfcc is not None:
                classify_audio(mfcc)