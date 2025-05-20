import librosa
import numpy as np
import joblib

class DeepFakeModel:
    def __init__(self, model_path="svm_model.pkl", scaler_path="scaler.pkl"):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def extract_features(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        return np.mean(mfcc.T, axis=0).reshape(1, -1)

    def predict(self, file_path):
        features = self.extract_features(file_path)
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)
        return "Spoof" if prediction[0] == 1 else "Real"