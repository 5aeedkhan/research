import numpy as np
import torch
import librosa
import soundfile as sf
import os
from typing import List, Tuple, Dict
import random
from scipy import signal
import matplotlib.pyplot as plt

class SpeechDisorderDataGenerator:
    """
    Generate synthetic speech disorder dataset for testing the classification pipeline.
    """
    
    def __init__(self, sample_rate: int = 16000, duration: float = 2.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        
    def generate_healthy_speech(self) -> np.ndarray:
        """Generate healthy speech-like audio."""
        t = np.linspace(0, self.duration, self.n_samples)
        
        # Fundamental frequency (typical male/female speech: 100-200 Hz)
        f0 = np.random.uniform(100, 200)
        
        # Generate harmonics
        audio = np.zeros(self.n_samples)
        for harmonic in range(1, 6):
            freq = f0 * harmonic
            amplitude = 1.0 / harmonic
            phase = np.random.uniform(0, 2*np.pi)
            audio += amplitude * np.sin(2 * np.pi * freq * t + phase)
        
        # Add formant-like resonances (typical speech formants)
        formants = [500, 1500, 2500, 3500]  # Hz
        for formant_freq in formants:
            bandwidth = formant_freq / 10  # Q factor of 10
            b, a = signal.butter(2, [formant_freq - bandwidth, formant_freq + bandwidth], 
                                btype='band', fs=self.sample_rate)
            audio = signal.lfilter(b, a, audio)
        
        # Add natural variation
        audio += 0.1 * np.random.randn(self.n_samples)
        
        # Apply envelope for speech-like dynamics
        envelope = np.exp(-3 * (t - self.duration/2)**2 / (self.duration/2)**2)
        audio *= envelope
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio
    
    def generate_dysarthria_speech(self) -> np.ndarray:
        """Generate dysarthric speech (impaired articulation, reduced prosody)."""
        t = np.linspace(0, self.duration, self.n_samples)
        
        # Dysarthria characteristics: reduced fundamental frequency variation
        f0 = np.random.uniform(80, 120)  # Lower and more stable F0
        
        # Generate harmonics with reduced variation
        audio = np.zeros(self.n_samples)
        for harmonic in range(1, 4):  # Fewer harmonics
            freq = f0 * harmonic
            amplitude = 1.0 / (harmonic ** 1.5)  # Faster amplitude decay
            phase = np.random.uniform(0, np.pi)  # Reduced phase variation
            audio += amplitude * np.sin(2 * np.pi * freq * t + phase)
        
        # Add tremor (common in dysarthria)
        tremor_freq = np.random.uniform(4, 8)  # Hz
        tremor = 0.2 * np.sin(2 * np.pi * tremor_freq * t)
        audio *= (1 + tremor)
        
        # Reduced formant structure
        formants = [400, 1200, 2000]  # Fewer and lower formants
        for formant_freq in formants:
            bandwidth = formant_freq / 5  # Wider bandwidth (less defined)
            b, a = signal.butter(2, [formant_freq - bandwidth, formant_freq + bandwidth], 
                                btype='band', fs=self.sample_rate)
            audio = signal.lfilter(b, a, audio)
        
        # Add noise (imprecise articulation)
        audio += 0.2 * np.random.randn(self.n_samples)
        
        # Slower speech rate
        slow_factor = np.random.uniform(0.7, 0.9)
        audio = signal.resample(audio, int(self.n_samples * slow_factor))
        if len(audio) < self.n_samples:
            audio = np.pad(audio, (0, self.n_samples - len(audio)))
        else:
            audio = audio[:self.n_samples]
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.6
        
        return audio
    
    def generate_apraxia_speech(self) -> np.ndarray:
        """Generate apraxic speech (inconsistent articulation errors)."""
        t = np.linspace(0, self.duration, self.n_samples)
        
        # Apraxia characteristics: inconsistent articulation
        f0 = np.random.uniform(120, 250)
        
        # Generate inconsistent harmonics
        audio = np.zeros(self.n_samples)
        for harmonic in range(1, 7):
            if np.random.random() > 0.3:  # Randomly drop harmonics
                freq = f0 * harmonic * np.random.uniform(0.9, 1.1)  # Frequency instability
                amplitude = np.random.uniform(0.5, 1.5) / harmonic
                phase = np.random.uniform(0, 2*np.pi)
                audio += amplitude * np.sin(2 * np.pi * freq * t + phase)
        
        # Add articulation breaks (gaps in speech)
        n_breaks = np.random.randint(2, 5)
        for _ in range(n_breaks):
            break_start = np.random.randint(0, self.n_samples - 1000)
            break_duration = np.random.randint(50, 200)
            audio[break_start:break_start + break_duration] = 0
        
        # Distorted formants
        formants = [600, 1800, 3000, 4000]
        for i, formant_freq in enumerate(formants):
            if np.random.random() > 0.3:  # Randomly distort formants
                formant_freq *= np.random.uniform(0.8, 1.2)
            bandwidth = formant_freq / 8
            b, a = signal.butter(2, [formant_freq - bandwidth, formant_freq + bandwidth], 
                                btype='band', fs=self.sample_rate)
            audio = signal.lfilter(b, a, audio)
        
        # Add effortful speech characteristics
        effort_noise = 0.15 * np.random.randn(self.n_samples)
        audio += effort_noise
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.7
        
        return audio
    
    def generate_dysphonia_speech(self) -> np.ndarray:
        """Generate dysphonic speech (voice quality disorders)."""
        t = np.linspace(0, self.duration, self.n_samples)
        
        # Dysphonia characteristics: voice quality issues
        f0 = np.random.uniform(150, 300)
        
        # Generate harmonics with voice breaks
        audio = np.zeros(self.n_samples)
        for harmonic in range(1, 5):
            freq = f0 * harmonic
            amplitude = 1.0 / harmonic
            
            # Add voice breaks (vocal fry, creaky voice)
            if np.random.random() > 0.7:
                amplitude *= np.random.uniform(0.1, 0.5)
            
            phase = np.random.uniform(0, 2*np.pi)
            audio += amplitude * np.sin(2 * np.pi * freq * t + phase)
        
        # Add breathiness (noise component)
        breathiness = 0.3 * np.random.randn(self.n_samples)
        # Filter to high frequencies for breathy quality
        b, a = signal.butter(4, 2000, btype='high', fs=self.sample_rate)
        breathiness = signal.lfilter(b, a, breathiness)
        audio += breathiness
        
        # Add roughness (amplitude modulation)
        roughness_freq = np.random.uniform(20, 50)  # Hz
        roughness = 0.2 * np.sin(2 * np.pi * roughness_freq * t)
        audio *= (1 + roughness)
        
        # Strained voice (add subharmonics)
        subharmonic_freq = f0 / 2
        audio += 0.3 * np.sin(2 * np.pi * subharmonic_freq * t)
        
        # Apply formant filtering
        formants = [400, 1400, 2400, 3400]
        for formant_freq in formants:
            bandwidth = formant_freq / 12  # Narrower bandwidth (strained)
            b, a = signal.butter(2, [formant_freq - bandwidth, formant_freq + bandwidth], 
                                btype='band', fs=self.sample_rate)
            audio = signal.lfilter(b, a, audio)
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.5
        
        return audio
    
    def generate_dataset(self, n_samples_per_class: int = 100, 
                       output_dir: str = "synthetic_data") -> Tuple[List[str], List[int]]:
        """
        Generate complete synthetic dataset.
        
        Args:
            n_samples_per_class: Number of samples per class
            output_dir: Directory to save audio files
            
        Returns:
            Tuple of (file_paths, labels)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        file_paths = []
        labels = []
        
        # Class definitions
        classes = [
            ("healthy", 0, self.generate_healthy_speech),
            ("dysarthria", 1, self.generate_dysarthria_speech),
            ("apraxia", 2, self.generate_apraxia_speech),
            ("dysphonia", 3, self.generate_dysphonia_speech)
        ]
        
        print(f"Generating {n_samples_per_class} samples per class...")
        
        for class_name, label, generator in classes:
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for i in range(n_samples_per_class):
                # Generate audio
                audio = generator()
                
                # Save audio file
                filename = f"{class_name}_{i:04d}.wav"
                filepath = os.path.join(class_dir, filename)
                sf.write(filepath, audio, self.sample_rate)
                
                file_paths.append(filepath)
                labels.append(label)
                
                if (i + 1) % 20 == 0:
                    print(f"Generated {i + 1}/{n_samples_per_class} {class_name} samples")
        
        print(f"Dataset generation complete! Total samples: {len(file_paths)}")
        
        return file_paths, labels
    
    def visualize_samples(self, n_samples: int = 2, output_dir: str = "visualizations"):
        """Visualize different types of generated speech samples."""
        os.makedirs(output_dir, exist_ok=True)
        
        classes = [
            ("healthy", self.generate_healthy_speech),
            ("dysarthria", self.generate_dysarthria_speech),
            ("apraxia", self.generate_apraxia_speech),
            ("dysphonia", self.generate_dysphonia_speech)
        ]
        
        fig, axes = plt.subplots(len(classes), n_samples, figsize=(15, 10))
        if len(classes) == 1:
            axes = axes.reshape(1, -1)
        if n_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (class_name, generator) in enumerate(classes):
            for j in range(n_samples):
                audio = generator()
                t = np.linspace(0, self.duration, len(audio))
                
                axes[i, j].plot(t, audio)
                axes[i, j].set_title(f"{class_name.capitalize()} - Sample {j+1}")
                axes[i, j].set_xlabel("Time (s)")
                axes[i, j].set_ylabel("Amplitude")
                axes[i, j].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "speech_samples.png"), dpi=300)
        plt.show()
        
        # Plot spectrograms
        fig, axes = plt.subplots(len(classes), 1, figsize=(12, 10))
        
        for i, (class_name, generator) in enumerate(classes):
            audio = generator()
            
            # Compute spectrogram
            f, t_spec, Sxx = signal.spectrogram(audio, fs=self.sample_rate, nperseg=256)
            
            im = axes[i].pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud')
            axes[i].set_title(f"{class_name.capitalize()} Spectrogram")
            axes[i].set_ylabel("Frequency (Hz)")
            axes[i].set_xlabel("Time (s)")
            plt.colorbar(im, ax=axes[i], label="Power (dB)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "spectrograms.png"), dpi=300)
        plt.show()

# Test the data generator
if __name__ == "__main__":
    generator = SpeechDisorderDataGenerator()
    
    # Visualize samples
    print("Visualizing speech samples...")
    generator.visualize_samples(n_samples=2)
    
    # Generate dataset
    print("\nGenerating synthetic dataset...")
    file_paths, labels = generator.generate_dataset(n_samples_per_class=50)
    
    print(f"\nGenerated {len(file_paths)} audio files")
    print("Class distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        class_names = ["healthy", "dysarthria", "apraxia", "dysphonia"]
        print(f"  {class_names[label]}: {count} samples")
    
    print("\nData generator test complete!")
