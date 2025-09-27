import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Constants
DATA_DIR = "DATA"
MAX_PAD_LEN = 174
N_MFCC = 40

# Your full disease class list
DISEASE_CLASSES = [
    "Asthma",
    "Bronchiectasis",
    "Bronchiolitis",
    "Bronchitis",
    "COPD",
    "Covid",
    "Healthy",
    "pertussis",
    "Pneumonia",
    "URTI",
    "non_cough"
]

# Extract MFCC features
def extract_mfcc(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
        if mfccs.shape[1] < MAX_PAD_LEN:
            pad_width = MAX_PAD_LEN - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :MAX_PAD_LEN]
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load dataset
def load_dataset(data_dir):
    features = []
    labels = []
    for label in DISEASE_CLASSES:
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            print(f"Skipping missing class folder: {label}")
            continue
        for file in os.listdir(class_dir):
            if file.endswith(".wav"):
                file_path = os.path.join(class_dir, file)
                mfcc = extract_mfcc(file_path)
                if mfcc is not None:
                    features.append(mfcc)
                    labels.append(label)
    return np.array(features), np.array(labels)

# Load and preprocess
X, y = load_dataset(DATA_DIR)
X = X.reshape(X.shape[0], N_MFCC, MAX_PAD_LEN, 1)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_onehot = tf.keras.utils.to_categorical(y_encoded)

# Save encoder classes
np.save("classes.npy", encoder.classes_)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(N_MFCC, MAX_PAD_LEN, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("model.h5")
print("✅ Model and encoder saved.")

