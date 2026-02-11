import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from .cnn_features import DualCNNFeatureExtractor
from .transformer_attention import DualTransformerEncoder

class MultiFeatureFusionModel(nn.Module):
    def __init__(self, 
                 input_dim: int = 128,
                 seq_len: int = 256,
                 cnn_feature_dim: int = 512,
                 transformer_dim: int = 512,
                 num_classes: int = 4,  # e.g., healthy, dysarthria, apraxia, dysphonia
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        
        # CNN feature extractor for mel spectrograms
        self.cnn_extractor = DualCNNFeatureExtractor(feature_dim=cnn_feature_dim)
        
        # Transformer encoder for sequential features
        self.transformer_encoder = DualTransformerEncoder(
            dim=transformer_dim, 
            seq_len=seq_len, 
            depth=4,
            heads=8,
            dim_head=64,
            mlp_dim=1024,
            dropout=dropout
        )
        
        # Feature projection layers
        self.cnn_projection = nn.Sequential(
            nn.Linear(cnn_feature_dim, transformer_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Traditional acoustic features processor
        self.acoustic_processor = nn.Sequential(
            nn.Linear(13 * 4, transformer_dim),  # MFCC(13) + chroma(12) + spectral_contrast(7) + tonnetz(6) ≈ 38
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim, transformer_dim)
        )
        
        # Multi-head attention for feature fusion
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=transformer_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-modal fusion layers
        self.cross_fusion = nn.Sequential(
            nn.Linear(transformer_dim * 3, transformer_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim * 2, transformer_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim // 2, num_classes)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(transformer_dim)
        
    def extract_acoustic_features(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract and process traditional acoustic features."""
        batch_size = features['mfcc'].shape[0]
        
        # Average pool temporal dimension
        mfcc_avg = F.adaptive_avg_pool1d(features['mfcc'], 1).squeeze(-1)
        chroma_avg = F.adaptive_avg_pool1d(features['chroma'], 1).squeeze(-1)
        spectral_avg = F.adaptive_avg_pool1d(features['spectral_contrast'], 1).squeeze(-1)
        tonnetz_avg = F.adaptive_avg_pool1d(features['tonnetz'], 1).squeeze(-1)
        
        # Concatenate features
        acoustic_features = torch.cat([mfcc_avg, chroma_avg, spectral_avg, tonnetz_avg], dim=1)
        
        # Process through acoustic processor
        processed_acoustic = self.acoustic_processor(acoustic_features)
        
        return processed_acoustic
    
    def forward(self, mel_spectrogram: torch.Tensor, 
                mfcc: torch.Tensor, 
                features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = mel_spectrogram.shape[0]
        
        # CNN features from mel spectrogram
        cnn_features = self.cnn_extractor(mel_spectrogram)  # (batch, cnn_feature_dim)
        cnn_features = self.cnn_projection(cnn_features)  # (batch, transformer_dim)
        
        # Prepare sequential features for transformer
        # Use MFCC as sequential input
        seq_features = mfcc.transpose(1, 2)  # (batch, seq_len, features)
        
        # Project to transformer dimension
        target_dim = 512  # Fixed dimension
        if seq_features.shape[-1] != target_dim:
            seq_projection = nn.Linear(seq_features.shape[-1], target_dim).to(seq_features.device)
            seq_features = seq_projection(seq_features)
        
        # Transformer features
        transformer_features = self.transformer_encoder(seq_features)  # (batch, seq_len, transformer_dim)
        
        # Global pooling of transformer features
        transformer_features = transformer_features.mean(dim=1)  # (batch, transformer_dim)
        
        # Acoustic features
        acoustic_features = self.extract_acoustic_features(features)  # (batch, transformer_dim)
        
        # Stack features for attention fusion
        feature_stack = torch.stack([cnn_features, transformer_features, acoustic_features], dim=1)
        # (batch, 3, transformer_dim)
        
        # Apply multi-head attention for feature fusion
        fused_features, attention_weights = self.fusion_attention(
            feature_stack, feature_stack, feature_stack
        )
        
        # Flatten and apply cross-modal fusion
        fused_flat = fused_features.view(batch_size, -1)  # (batch, transformer_dim * 3)
        cross_fused = self.cross_fusion(fused_flat)  # (batch, transformer_dim)
        
        # Apply layer normalization
        cross_fused = self.layer_norm(cross_fused)
        
        # Classification
        logits = self.classifier(cross_fused)  # (batch, num_classes)
        
        return {
            'logits': logits,
            'features': {
                'cnn': cnn_features,
                'transformer': transformer_features,
                'acoustic': acoustic_features,
                'fused': cross_fused
            },
            'attention_weights': attention_weights
        }

class FeatureImportanceAnalyzer(nn.Module):
    """Module to analyze feature importance using attention weights."""
    
    def __init__(self, feature_dim: int, num_feature_types: int = 3):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_feature_types = num_feature_types
        
    def compute_feature_importance(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        """Compute importance scores for each feature type."""
        # attention_weights: (batch, num_heads, seq_len, seq_len)
        # For our case, we have 3 feature types (CNN, Transformer, Acoustic)
        
        # Average across heads and sequence
        avg_attention = attention_weights.mean(dim=(1, 2, 3))  # (batch,)
        
        # Compute importance scores (simplified)
        importance_scores = {
            'cnn_importance': avg_attention.mean().item(),
            'transformer_importance': avg_attention.std().item(),
            'acoustic_importance': avg_attention.max().item()
        }
        
        return importance_scores

# Test the fusion model
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy inputs
    batch_size = 4
    mel_spec = torch.randn(batch_size, 1, 128, 256).to(device)
    mfcc = torch.randn(batch_size, 13, 256).to(device)
    
    # Create dummy acoustic features
    features = {
        'mfcc': torch.randn(batch_size, 13, 256).to(device),
        'chroma': torch.randn(batch_size, 12, 256).to(device),
        'spectral_contrast': torch.randn(batch_size, 7, 256).to(device),
        'tonnetz': torch.randn(batch_size, 6, 256).to(device)
    }
    
    # Initialize model
    model = MultiFeatureFusionModel(
        input_dim=128,
        seq_len=256,
        cnn_feature_dim=512,
        transformer_dim=512,
        num_classes=4
    ).to(device)
    
    # Forward pass
    output = model(mel_spec, mfcc, features)
    
    print(f"Logits shape: {output['logits'].shape}")
    print(f"CNN features shape: {output['features']['cnn'].shape}")
    print(f"Transformer features shape: {output['features']['transformer'].shape}")
    print(f"Acoustic features shape: {output['features']['acoustic'].shape}")
    print(f"Fused features shape: {output['features']['fused'].shape}")
    print(f"Attention weights shape: {output['attention_weights'].shape}")
    
    # Test feature importance analyzer
    analyzer = FeatureImportanceAnalyzer(512)
    importance = analyzer.compute_feature_importance(output['attention_weights'])
    print(f"Feature importance: {importance}")
