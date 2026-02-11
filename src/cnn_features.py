import torch
import torch.nn as nn
import torchvision.models as models
import timm
from typing import Tuple

class MobileNetV3FeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = True, feature_dim: int = 512):
        super().__init__()
        # Load MobileNetV3 from timm
        self.mobilenet = timm.create_model('mobilenetv3_large_100', 
                                          pretrained=pretrained, 
                                          num_classes=0)  # Remove classification head
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.mobilenet(dummy_input)
            self.in_features = dummy_output.shape[1]
        
        # Add projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.in_features, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (batch_size, 1, height, width) - mel spectrogram
        # Convert to 3 channels by repeating
        x = x.repeat(1, 3, 1, 1)
        
        # Resize to 224x224 if needed
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract features
        features = self.mobilenet(x)
        projected_features = self.projection(features)
        
        return projected_features

class EfficientNetB7FeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = True, feature_dim: int = 512):
        super().__init__()
        # Load EfficientNetB7 from timm
        self.efficientnet = timm.create_model('efficientnet_b7', 
                                             pretrained=False,  # Use pretrained=False to avoid download issues
                                             num_classes=0)  # Remove classification head
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.efficientnet(dummy_input)
            self.in_features = dummy_output.shape[1]
        
        # Add projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.in_features, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (batch_size, 1, height, width) - mel spectrogram
        # Convert to 3 channels by repeating
        x = x.repeat(1, 3, 1, 1)
        
        # Resize to 224x224 if needed
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract features
        features = self.efficientnet(x)
        projected_features = self.projection(features)
        
        return projected_features

class DualCNNFeatureExtractor(nn.Module):
    def __init__(self, feature_dim: int = 512, fusion_method: str = 'concat'):
        super().__init__()
        self.mobilenet = MobileNetV3FeatureExtractor(feature_dim=feature_dim)
        self.efficientnet = EfficientNetB7FeatureExtractor(feature_dim=feature_dim)
        self.fusion_method = fusion_method
        
        if fusion_method == 'concat':
            combined_dim = feature_dim * 2
        elif fusion_method == 'add':
            combined_dim = feature_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Final projection layer
        self.final_projection = nn.Sequential(
            nn.Linear(combined_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features from both networks
        mobilenet_features = self.mobilenet(x)
        efficientnet_features = self.efficientnet(x)
        
        # Fusion
        if self.fusion_method == 'concat':
            combined = torch.cat([mobilenet_features, efficientnet_features], dim=1)
        elif self.fusion_method == 'add':
            combined = mobilenet_features + efficientnet_features
        
        # Final projection
        output = self.final_projection(combined)
        
        return output

# Test the feature extractors
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy input (batch_size=4, channels=1, height=128, width=256)
    dummy_input = torch.randn(4, 1, 128, 256).to(device)
    
    # Test MobileNetV3
    mobilenet = MobileNetV3FeatureExtractor().to(device)
    mobilenet_output = mobilenet(dummy_input)
    print(f"MobileNetV3 output shape: {mobilenet_output.shape}")
    
    # Test EfficientNetB7
    efficientnet = EfficientNetB7FeatureExtractor().to(device)
    efficientnet_output = efficientnet(dummy_input)
    print(f"EfficientNetB7 output shape: {efficientnet_output.shape}")
    
    # Test Dual CNN
    dual_cnn = DualCNNFeatureExtractor().to(device)
    dual_output = dual_cnn(dummy_input)
    print(f"Dual CNN output shape: {dual_output.shape}")
