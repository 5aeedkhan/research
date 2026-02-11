import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
import scipy.signal as signal

class AudioPreprocessor:
    def __init__(self, sample_rate: int = 16000, n_mels: int = 128, 
                 n_fft: int = 1024, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """Load audio file and resample if necessary."""
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram from audio."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """Extract MFCC features from audio."""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return mfcc
    
    def extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """Extract chroma features from audio."""
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return chroma
    
    def extract_spectral_contrast(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral contrast features."""
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return spectral_contrast
    
    def extract_tonnetz(self, audio: np.ndarray) -> np.ndarray:
        """Extract tonnetz features."""
        tonnetz = librosa.feature.tonnetz(
            y=audio,
            sr=self.sample_rate
        )
        return tonnetz
    
    def extract_all_features(self, audio: np.ndarray) -> dict:
        """Extract all audio features."""
        features = {
            'mel_spectrogram': self.extract_mel_spectrogram(audio),
            'mfcc': self.extract_mfcc(audio),
            'chroma': self.extract_chroma(audio),
            'spectral_contrast': self.extract_spectral_contrast(audio),
            'tonnetz': self.extract_tonnetz(audio)
        }
        return features
    
    def normalize_features(self, features: dict) -> dict:
        """Normalize features to zero mean and unit variance."""
        normalized = {}
        for key, feature in features.items():
            mean = np.mean(feature)
            std = np.std(feature)
            normalized[key] = (feature - mean) / (std + 1e-8)
        return normalized

class SpeechDataset(Dataset):
    def __init__(self, audio_files: list, labels: list, preprocessor: AudioPreprocessor):
        self.audio_files = audio_files
        self.labels = labels
        self.preprocessor = preprocessor
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        # Load and preprocess audio
        audio = self.preprocessor.load_audio(audio_path)
        features = self.preprocessor.extract_all_features(audio)
        features = self.preprocessor.normalize_features(features)
        
        # Convert to tensors
        mel_spec = torch.FloatTensor(features['mel_spectrogram']).unsqueeze(0)
        mfcc = torch.FloatTensor(features['mfcc']).unsqueeze(0)
        
        return {
            'mel_spectrogram': mel_spec,
            'mfcc': mfcc,
            'features': features,
            'label': torch.tensor(label, dtype=torch.long)
        }
