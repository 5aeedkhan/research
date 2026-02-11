import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns

class SHAPAwareXGBoost:
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 random_state: int = 42,
                 use_gpu: bool = True):
        """
        SHAP-aware XGBoost classifier for speech disorder classification.
        
        Args:
            n_estimators: Number of trees in the ensemble
            max_depth: Maximum depth of trees
            learning_rate: Learning rate for gradient boosting
            random_state: Random seed for reproducibility
            use_gpu: Whether to use GPU acceleration
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.use_gpu = use_gpu
        
        # Initialize XGBoost classifier
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            tree_method='gpu_hist' if use_gpu else 'hist',
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        
        self.scaler = StandardScaler()
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        self.class_names = None
        
    def extract_features_from_neural_model(self, 
                                         neural_model: nn.Module, 
                                         dataloader: torch.utils.data.DataLoader,
                                         device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from the trained neural model for XGBoost classification.
        
        Args:
            neural_model: Trained neural network model
            dataloader: DataLoader containing the data
            device: Device to run the model on
            
        Returns:
            Tuple of (features, labels)
        """
        neural_model.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                mel_spec = batch['mel_spectrogram'].to(device)
                mfcc = batch['mfcc'].to(device)
                features = {k: v.to(device) for k, v in batch['features'].items()}
                labels = batch['label'].cpu().numpy()
                
                # Get features from neural model
                output = neural_model(mel_spec, mfcc, features)
                
                # Extract fused features for XGBoost
                fused_features = output['features']['fused'].cpu().numpy()
                
                # Combine with other features
                cnn_features = output['features']['cnn'].cpu().numpy()
                transformer_features = output['features']['transformer'].cpu().numpy()
                acoustic_features = output['features']['acoustic'].cpu().numpy()
                
                # Concatenate all features
                combined_features = np.hstack([
                    fused_features,
                    cnn_features,
                    transformer_features,
                    acoustic_features
                ])
                
                all_features.append(combined_features)
                all_labels.append(labels)
        
        X = np.vstack(all_features)
        y = np.hstack(all_labels)
        
        return X, y
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None,
            class_names: Optional[List[str]] = None) -> None:
        """
        Train the XGBoost model and compute SHAP values.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Names of features for interpretation
            class_names: Names of classes for interpretation
        """
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        self.class_names = class_names or [f'class_{i}' for i in range(len(np.unique(y)))]
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train XGBoost model
        self.model.fit(
            X_train_scaled, y_train
        )
        
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        
        # Compute SHAP values on validation set
        self.shap_values = self.explainer.shap_values(X_val_scaled)
        
        print(f"Model trained with validation accuracy: {self.model.score(X_val_scaled, y_val):.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance."""
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, target_names=self.class_names, output_dict=True)
        cm = confusion_matrix(y, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def explain_prediction(self, X: np.ndarray, instance_idx: int = 0) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP values.
        
        Args:
            X: Feature matrix
            instance_idx: Index of the instance to explain
            
        Returns:
            Dictionary containing explanation details
        """
        X_scaled = self.scaler.transform(X)
        
        # Get SHAP values for the specific instance
        shap_values_instance = self.explainer.shap_values(X_scaled[instance_idx:instance_idx+1])
        
        # Get prediction
        prediction = self.model.predict(X_scaled[instance_idx:instance_idx+1])[0]
        prediction_proba = self.model.predict_proba(X_scaled[instance_idx:instance_idx+1])[0]
        
        # Get feature importance for this prediction
        if isinstance(shap_values_instance, list):
            # Multi-class case
            shap_values_class = shap_values_instance[prediction]
        else:
            # Binary case
            shap_values_class = shap_values_instance[0]
        
        # Get top contributing features
        feature_importance = list(zip(self.feature_names, shap_values_class))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return {
            'prediction': prediction,
            'predicted_class': self.class_names[prediction],
            'prediction_proba': prediction_proba,
            'feature_importance': feature_importance[:10],  # Top 10 features
            'shap_values': shap_values_class
        }
    
    def plot_feature_importance(self, plot_type: str = 'bar', max_features: int = 20) -> None:
        """Plot global feature importance."""
        if self.shap_values is None:
            print("SHAP values not computed. Please fit the model first.")
            return
        
        # Calculate mean absolute SHAP values
        if isinstance(self.shap_values, list):
            # Multi-class case - average across classes
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in self.shap_values], axis=0)
        else:
            mean_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Get top features
        top_indices = np.argsort(mean_shap)[-max_features:]
        top_features = [self.feature_names[i] for i in top_indices]
        top_values = mean_shap[top_indices]
        
        plt.figure(figsize=(10, 8))
        if plot_type == 'bar':
            plt.barh(top_features, top_values)
            plt.xlabel('Mean |SHAP value|')
            plt.title('Global Feature Importance')
        elif plot_type == 'summary':
            shap.summary_plot(self.shap_values, features=self.scaler.transform(np.random.randn(100, len(self.feature_names))), 
                             feature_names=self.feature_names, plot_type="bar")
        
        plt.tight_layout()
        plt.show()
    
    def plot_shap_waterfall(self, X: np.ndarray, instance_idx: int = 0) -> None:
        """Plot SHAP waterfall plot for a single instance."""
        if self.explainer is None:
            print("Model not fitted. Please fit the model first.")
            return
        
        X_scaled = self.scaler.transform(X)
        shap_values_instance = self.explainer.shap_values(X_scaled[instance_idx:instance_idx+1])
        
        if isinstance(shap_values_instance, list):
            # Multi-class case
            prediction = self.model.predict(X_scaled[instance_idx:instance_idx+1])[0]
            shap_values_class = shap_values_instance[prediction]
        else:
            shap_values_class = shap_values_instance[0]
        
        # Create waterfall plot
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(values=shap_values_class,
                           base_values=self.explainer.expected_value,
                           data=X_scaled[instance_idx],
                           feature_names=self.feature_names)
        )
        plt.title(f'SHAP Waterfall Plot - Predicted: {self.class_names[prediction] if isinstance(shap_values_instance, list) else ""}')
        plt.tight_layout()
        plt.show()
    
    def get_feature_contributions(self, X: np.ndarray, class_idx: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature contributions for all instances.
        
        Args:
            X: Feature matrix
            class_idx: Specific class to analyze (for multi-class)
            
        Returns:
            DataFrame with feature contributions
        """
        X_scaled = self.scaler.transform(X)
        shap_values_all = self.explainer.shap_values(X_scaled)
        
        if isinstance(shap_values_all, list):
            # Multi-class case
            if class_idx is not None:
                shap_values = shap_values_all[class_idx]
            else:
                # Average across classes
                shap_values = np.mean(shap_values_all, axis=0)
        else:
            shap_values = shap_values_all
        
        # Create DataFrame
        contributions_df = pd.DataFrame(shap_values, columns=self.feature_names)
        
        return contributions_df

# Test the SHAP-aware XGBoost
if __name__ == "__main__":
    # Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 4, n_samples)  # 4 classes
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    class_names = ['healthy', 'dysarthria', 'apraxia', 'dysphonia']
    
    # Initialize and train model
    shap_xgb = SHAPAwareXGBoost(n_estimators=50, use_gpu=False)
    shap_xgb.fit(X, y, feature_names=feature_names, class_names=class_names)
    
    # Evaluate model
    results = shap_xgb.evaluate(X, y)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"CV Score: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
    
    # Explain a prediction
    explanation = shap_xgb.explain_prediction(X, instance_idx=0)
    print(f"\nPrediction: {explanation['predicted_class']}")
    print("Top contributing features:")
    for feature, importance in explanation['feature_importance'][:5]:
        print(f"  {feature}: {importance:.4f}")
    
    print("SHAP-aware XGBoost classifier implemented successfully!")
