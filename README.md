# Speech Disorder Classification Pipeline

**Multi-Feature Fusion-Based Speech Disorder Classification Using MobileNetV3-EfficientNetB7-Linformer-Performer and SHAP-Aware XGBoost**

## 🎯 Overview

This implementation reproduces and tests the research paper on speech disorder classification using advanced deep learning techniques. The system combines:

- **Dual CNN Pathway**: MobileNetV3 + EfficientNetB7 for visual feature extraction
- **Dual Transformer Pathway**: Linformer + Performer for sequential modeling  
- **Multi-Feature Fusion**: Combines CNN, Transformer, and acoustic features
- **SHAP-Aware XGBoost**: Explainable final classifier with feature importance analysis

## 🏗️ Architecture

```
Audio Input
    ↓
Audio Preprocessing
    ↓
┌─────────────────┬─────────────────┬─────────────────┐
│   CNN Pathway   │ Transformer     │  Acoustic       │
│                 │ Pathway         │  Features       │
│ MobileNetV3     │ Linformer       │ MFCC            │
│ EfficientNetB7  │ Performer       │ Chroma          │
│                 │                 │ Spectral        │
│                 │                 │ Tonnetz         │
└─────────────────┴─────────────────┴─────────────────┘
    ↓                 ↓                 ↓
    └─────────────┬─────────────────────┘
                  ↓
        Multi-Feature Fusion
                  ↓
        SHAP-Aware XGBoost
                  ↓
        Classification + Explanation
```

## 📁 Project Structure

```
research/
├── src/
│   ├── __init__.py
│   ├── audio_preprocessor.py      # Audio feature extraction
│   ├── cnn_features.py            # MobileNetV3 + EfficientNetB7
│   ├── transformer_attention.py   # Linformer + Performer
│   ├── fusion_model.py           # Multi-feature fusion architecture
│   ├── shap_xgboost.py           # SHAP-aware XGBoost classifier
│   ├── training_pipeline.py      # Training and evaluation
│   └── data_generator.py         # Synthetic data generation
├── main.py                       # Main testing script
├── test_components.py           # Component testing
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 🚀 Installation

1. **Clone/Download the project**
2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Required Packages:
- `torch>=2.0.0` - Deep learning framework
- `tensorflow>=2.13.0` - For pre-trained models
- `librosa>=0.10.0` - Audio processing
- `xgboost>=1.7.0` - Gradient boosting
- `shap>=0.42.0` - Explainable AI
- `timm>=0.9.0` - Pre-trained vision models
- `scikit-learn>=1.3.0` - Machine learning utilities
- `numpy>=1.24.0`, `pandas>=2.0.0` - Data processing
- `matplotlib>=3.7.0`, `seaborn>=0.12.0` - Visualization

## 🧪 Testing

### Quick Component Test
```bash
python test_components.py
```

### Full Pipeline Test
```bash
python main.py
```

## 📊 Key Features

### 1. **Advanced CNN Architecture**
- **MobileNetV3**: Lightweight, efficient feature extraction
- **EfficientNetB7**: High-performance compound scaling
- **Dual pathway**: Complementary feature learning

### 2. **Efficient Transformer Attention**
- **Linformer**: O(N) complexity linear attention
- **Performer**: Kernel-based efficient attention
- **Dual processing**: Sequential and global context

### 3. **Multi-Modal Fusion**
- **CNN features**: Visual patterns from spectrograms
- **Transformer features**: Temporal dependencies
- **Acoustic features**: Traditional speech features (MFCC, chroma, etc.)
- **Attention-based fusion**: Learnable feature weighting

### 4. **Explainable AI**
- **SHAP values**: Feature contribution analysis
- **XGBoost**: Interpretable tree-based classifier
- **Visual explanations**: Waterfall plots, feature importance

## 🎛️ Usage

### 1. **Data Preparation**
```python
from src.data_generator import SpeechDisorderDataGenerator

# Generate synthetic dataset
generator = SpeechDisorderDataGenerator()
file_paths, labels = generator.generate_dataset(
    n_samples_per_class=100,
    output_dir="data"
)
```

### 2. **Model Training**
```python
from src.training_pipeline import CompletePipeline
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipeline = CompletePipeline(device=device)

# Train complete pipeline
results = pipeline.complete_training(dataset)
```

### 3. **Feature Explanation**
```python
from src.shap_xgboost import SHAPAwareXGBoost

# Explain predictions
shap_xgb = SHAPAwareXGBoost()
explanation = shap_xgb.explain_prediction(X, instance_idx=0)
print(f"Top features: {explanation['feature_importance'][:5]}")
```

## 📈 Performance Metrics

The system provides comprehensive evaluation:

- **Accuracy**: Overall classification performance
- **Cross-validation**: Robustness assessment
- **Classification Report**: Precision, Recall, F1-score per class
- **Confusion Matrix**: Detailed error analysis
- **SHAP Analysis**: Feature importance and contribution

## 🏥 Clinical Applications

### Speech Disorder Types:
1. **Healthy**: Normal speech patterns
2. **Dysarthria**: Motor speech impairment
3. **Apraxia**: Speech planning disorder
4. **Dysphonia**: Voice quality disorders

### Use Cases:
- **Clinical Diagnosis**: Assist speech-language pathologists
- **Telemedicine**: Remote speech assessment
- **Progress Monitoring**: Track treatment effectiveness
- **Research**: Large-scale speech disorder analysis

## 🔧 Configuration

### Model Parameters:
```python
model = MultiFeatureFusionModel(
    input_dim=128,           # Mel spectrogram height
    seq_len=256,             # Sequence length
    cnn_feature_dim=512,     # CNN feature dimension
    transformer_dim=512,      # Transformer dimension
    num_classes=4,           # Number of disorder types
    dropout=0.3              # Dropout rate
)
```

### Training Parameters:
```python
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    learning_rate=1e-4,
    patience=10,
    weight_decay=1e-5
)
```

## 📝 Research Paper Implementation

This implementation follows the research paper's methodology:

1. **Multi-Feature Extraction**: Combines deep and traditional features
2. **Efficient Attention**: Reduces computational complexity
3. **Fusion Architecture**: Integrates multiple feature types
4. **Explainable Classification**: Provides interpretable results

## 🚀 Future Extensions

1. **Real-time Processing**: Streaming audio classification
2. **Mobile Deployment**: On-device speech analysis
3. **Multi-lingual Support**: Language-agnostic features
4. **Extended Disorders**: Additional speech condition categories
5. **Web Interface**: User-friendly clinical tool

## 📚 References

- MobileNetV3: "Searching for MobileNetV3" (Howard et al., 2019)
- EfficientNet: "EfficientNet: Rethinking Model Scaling" (Tan & Le, 2019)
- Linformer: "Linformer: Self-Attention with Linear Complexity" (Wang et al., 2020)
- Performer: "Rethinking Attention with Performers" (Choromanski et al., 2020)
- SHAP: "A Unified Approach to Interpreting Model Predictions" (Lundberg & Lee, 2017)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for research and educational purposes. Please cite the original research paper if used in academic work.

---

**🎯 This code right here is Developed according to the research paper i have which i have to follow to make my own Research for Mphil CS**
