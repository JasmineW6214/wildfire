import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import math
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from v4.model.loss_v4  import improved_weighted_focal_loss_v1
from v4.model.utils import calculate_metrics
import joblib


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward layers"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual connection and layer norm
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + attended)
        
        # Feed-forward with residual connection and layer norm
        x = self.norm2(x + self.ff(x))
        return x

class FireTransformer(nn.Module):
    """Transformer-based model for fire prediction"""
    def __init__(self, feature_size, window_size=15, n_layers=6, d_model=128, n_heads=8, d_ff=256, dropout=0.1, use_feature_norm=True):
        super().__init__()
        
        # Initialize positional encoding
        position = torch.arange(window_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, window_size, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoder', pe)
        
        # Add BatchNorm1d 
        self.use_feature_norm = use_feature_norm
        if use_feature_norm:
            self.feature_norm = nn.BatchNorm1d(feature_size)
        
        # Input dropout
        self.input_dropout = nn.Dropout(dropout)
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(feature_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)  # Dropout after input projection
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Block dropout
        self.block_dropout = nn.Dropout(dropout)
        
        # Output layers with slightly higher dropout
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),  # Maintain dimensionality longer
            nn.GELU(),
            nn.Dropout(dropout * 0.5),    # Reduce dropout
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),   # Additional normalization
            nn.Linear(d_model // 2, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        # Modified initialization for better gradient flow
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_normal_(
                    module.weight, 
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_size)
        batch_size, seq_len, feature_size = x.shape
        
        # Apply feature normalization if enabled
        if self.use_feature_norm:
            # Reshape for BatchNorm1d: (batch_size * seq_len, feature_size)
            x_reshaped = x.view(-1, feature_size)
            # Apply BatchNorm1d
            x_normalized = self.feature_norm(x_reshaped)
            # Reshape back: (batch_size, seq_len, feature_size)
            x = x_normalized.view(batch_size, seq_len, feature_size)
        
        # Input dropout
        x = self.input_dropout(x)
        
        # Project input and add positional encoding
        x = self.input_proj(x) + self.pos_encoder
        
        # Apply transformer blocks with dropout
        for block in self.blocks:
            x = block(x)
            x = self.block_dropout(x)  # Dropout after each block
        
        # Take last token and apply output layers
        x = x[:, -1]
        logits = self.output(x).squeeze(-1)
        
        # Handle NaN/inf values gracefully
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            # Log the issue
            print("Warning: Model output contains NaN or infinite values. Applying correction.")
            
            # Replace NaN/inf with safe values
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Clamp to reasonable range
            logits = torch.clamp(logits, -100, 100)
            
        return logits

class BaselineModels:
    """Traditional baseline models (Logistic Regression, XGBoost)"""
    def __init__(self, pos_weight=2.0, fn_penalty=15.0, fp_penalty=8.0):
        self.pos_weight = pos_weight
        self.fn_penalty = fn_penalty
        self.fp_penalty = fp_penalty
        
        # Initialize models
        self.logistic = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.xgb = xgb.XGBClassifier(
            scale_pos_weight=pos_weight,
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False
        )
        
    def fit_logistic(self, X, y, is_prob_target=False):
        """Fit logistic regression model"""
        print("Training Logistic Regression...")
        
        # Convert probability targets to binary if needed
        if is_prob_target:
            y = (y > 0.5).astype(int)
        
        self.logistic.fit(X, y)
        
    def fit_xgb(self, X, y, is_prob_target=False):
        """Fit XGBoost model"""
        print("Training XGBoost...")
        
        # Convert probability targets to binary if needed
        if is_prob_target:
            y = (y > 0.5).astype(int)
        
        self.xgb.fit(X, y)
        
    def predict(self, model_name, X):
        """Make predictions using the specified model"""
        if model_name == 'logistic':
            return self.logistic.predict_proba(X)[:, 1]  # Return probabilities
        elif model_name == 'xgb':
            return self.xgb.predict_proba(X)[:, 1]  # Return probabilities
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
    def save_model(self, model_name, path):
        """Save the specified model to disk"""
        if model_name == 'logistic':
            joblib.dump(self.logistic, path)
        elif model_name == 'xgb':
            joblib.dump(self.xgb, path)
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
    def load_model(self, model_name, path):
        """Load the specified model from disk"""
        if model_name == 'logistic':
            self.logistic = joblib.load(path)
        elif model_name == 'xgb':
            self.xgb = joblib.load(path)
        else:
            raise ValueError(f"Unknown model: {model_name}")

class BaselineNN(nn.Module):
    """Neural network baseline model for fire prediction"""
    def __init__(self, input_size, hidden_sizes=[128, 64, 64], dropout=0.3):
        super().__init__()
        
        # Build MLP layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze(-1)


class BaselineCNN(nn.Module):
    """CNN baseline model for fire prediction"""
    def __init__(self, input_size, hidden_channels=[32, 64, 128], kernel_size=3, dropout=0.3):
        super().__init__()
        
        # Reshape input for 1D convolution (batch_size, 1, input_size)
        self.input_size = input_size
        
        # Build CNN layers
        cnn_layers = []
        in_channels = 1
        
        for out_channels in hidden_channels:
            cnn_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            cnn_layers.append(nn.BatchNorm1d(out_channels))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool1d(2))
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate output size after convolutions and pooling
        output_size = input_size
        for _ in hidden_channels:
            output_size = output_size // 2  # Effect of MaxPool1d(2)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels[-1] * output_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # Reshape for CNN (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.flatten(1)
        return self.fc(x).squeeze(-1)


class NNBaselineModels:
    """Neural network baseline models with GPU support"""
    def __init__(self, input_size, pos_weight=2.0, fn_penalty=15.0, fp_penalty=8.0):
        self.input_size = input_size
        self.pos_weight = pos_weight
        self.fn_penalty = fn_penalty
        self.fp_penalty = fp_penalty
        
        # Initialize models
        self.mlp = BaselineNN(input_size)
        self.cnn = BaselineCNN(input_size)
        
        self.models = {
            'mlp': self.mlp,
            'cnn': self.cnn
        }
        
        # Move models to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for model in self.models.values():
            model.to(self.device)
    
    def fit_model(self, model_name, train_loader, val_loader=None, epochs=10, lr=0.001):
        """Train a neural network model"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.models[model_name]
        model.train()
        
        # Setup optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
        
        # Use improved_weighted_focal_loss_v1 for better handling of imbalanced data
        def loss_fn(y_pred, y_true):           
            # Then use the improved loss function
            return improved_weighted_focal_loss_v1(
                y_pred, 
                y_true,
                base_pos_weight=self.pos_weight,
                fn_penalty=self.fn_penalty,
                fp_penalty=self.fp_penalty
            )
        
        # Training loop
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            all_preds = []
            all_labels = []
            
            for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                X, y = X.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                logits = model(X)
                loss = loss_fn(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                all_preds.extend(torch.sigmoid(logits).cpu().detach().numpy())
                all_labels.extend(y.cpu().numpy())
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Calculate training metrics
            train_metrics = calculate_metrics(np.array(all_preds), np.array(all_labels), is_prob_target=True)
            
            # Validation
            if val_loader:
                model.eval()
                val_loss = 0
                val_preds = []
                val_labels = []
                
                with torch.no_grad():
                    for X, y in val_loader:
                        X, y = X.to(self.device), y.to(self.device)
                        logits = model(X)
                        loss = loss_fn(logits, y)
                        val_loss += loss.item()
                        val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                        val_labels.extend(y.cpu().numpy())
                
                avg_val_loss = val_loss / len(val_loader)
                val_metrics = calculate_metrics(np.array(val_preds), np.array(val_labels), is_prob_target=True)
                
                # Print detailed metrics
                print(f"\nEpoch {epoch+1} Metrics:")
                print(f"Train Loss = {avg_train_loss:.4f}")
                print(f"Val Loss = {avg_val_loss:.4f}")
                print("\nTraining Metrics:")
                print(f"AUC-PR = {train_metrics['auc_pr']:.4f}")
                print(f"Accuracy = {train_metrics['accuracy']:.4f}")
                print(f"Recall = {train_metrics['recall']:.4f}")
                print(f"Precision = {train_metrics['precision']:.4f}")
                print(f"F1 = {train_metrics['f1']:.4f}")
                print("\nValidation Metrics:")
                print(f"AUC-PR = {val_metrics['auc_pr']:.4f}")
                print(f"Accuracy = {val_metrics['accuracy']:.4f}")
                print(f"Recall = {val_metrics['recall']:.4f}")
                print(f"Precision = {val_metrics['precision']:.4f}")
                print(f"F1 = {val_metrics['f1']:.4f}")
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"\nEpoch {epoch+1} Metrics:")
                print(f"Train Loss = {avg_train_loss:.4f}")
                print("\nTraining Metrics:")
                print(f"AUC-PR = {train_metrics['auc_pr']:.4f}")
                print(f"Accuracy = {train_metrics['accuracy']:.4f}")
                print(f"Recall = {train_metrics['recall']:.4f}")
                print(f"Precision = {train_metrics['precision']:.4f}")
                print(f"F1 = {train_metrics['f1']:.4f}")
    
    def predict(self, model_name, X):
        """Get probability predictions from a model"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.models[model_name]
        model.eval()
        
        # Convert to tensor if numpy array
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            logits = model(X)
            probs = torch.sigmoid(logits)
        
        return probs.cpu().numpy()
    
    def save_model(self, model_name, path):
        """Save model to disk"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        torch.save(self.models[model_name].state_dict(), path)
    
    def load_model(self, model_name, path):
        """Load model from disk"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.models[model_name].load_state_dict(torch.load(path)) 
