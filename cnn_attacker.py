import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class CNNAttacker(nn.Module):
    """
    1D CNN optimized for side-channel analysis on power traces.
    Designed for 6GB VRAM with CUDA enforcement and memory management.
    """
    
    def __init__(self, input_length=1000, num_classes=5):
        super(CNNAttacker, self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Enforce CUDA requirement - hard fail if not available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This model requires CUDA for execution.")
        
        print(f"Using device: {self.device}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Enable float16 for faster tensor core utilization
        self.dtype = torch.float16
        print(f"Using precision: {self.dtype} for tensor core optimization")
        
        # 1D CNN architecture for power traces
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        # Calculate the output size after conv layers
        self._calculate_conv_output_size(input_length)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Move model to CUDA
        self.to(self.device)
        
        # Enable automatic mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
        
        print("Using float16 precision for tensor core optimization")
        
    def _calculate_conv_output_size(self, input_length):
        """Calculate the output size after convolutional layers."""
        x = torch.randn(1, 1, input_length)
        with torch.no_grad():
            x = self.conv_layers(x)
        self.conv_output_size = x.numel()
        
    def forward(self, x):
        # Input shape: (batch_size, sequence_length)
        # Reshape for Conv1d: (batch_size, 1, sequence_length)
        x = x.unsqueeze(1)
        
        # Pass through convolutional layers
        x = self.conv_layers(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Pass through fully connected layers
        x = self.fc_layers(x)
        
        return x
    
    def train_model(self, traces, labels, epochs=50, learning_rate=0.001):
        """
        Train the CNN model with memory-optimized DataLoader for 6GB VRAM.
        
        Args:
            traces: Power traces (numpy array)
            labels: Secret coefficient labels (numpy array)
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            
        Returns:
            Training history
        """
        print(f"Training CNN on {len(traces)} traces...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            traces, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Convert to PyTorch tensors (keep on CPU for DataLoader)
        X_train = torch.FloatTensor(X_train)
        X_val = torch.FloatTensor(X_val)
        y_train = torch.LongTensor(y_train)
        y_val = torch.LongTensor(y_val)
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        # Memory-optimized DataLoader for 6GB VRAM
        train_loader = DataLoader(
            train_dataset, 
            batch_size=256,  # Optimized for 6GB VRAM
            shuffle=True, 
            num_workers=6,   # Optimized for 6GB VRAM
            pin_memory=True   # Optimized for 6GB VRAM
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=256, 
            shuffle=False, 
            num_workers=6, 
            pin_memory=True
        )
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        
        # Training history
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device, dtype=self.dtype), target.to(self.device)
                
                optimizer.zero_grad()
                
                # Use automatic mixed precision
                with torch.cuda.amp.autocast():
                    output = self(data)
                    loss = criterion(output, target)
                
                # Scale gradients and update
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                # Memory optimization: detach and move to CPU
                train_loss += loss.detach().cpu().item()
                _, predicted = torch.max(output.detach(), 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().detach().cpu().item()
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device, dtype=self.dtype), target.to(self.device)
                    
                    # Use automatic mixed precision
                    with torch.cuda.amp.autocast():
                        output = self(data)
                        loss = criterion(output, target)
                    
                    # Memory optimization: detach and move to CPU
                    val_loss += loss.detach().cpu().item()
                    _, predicted = torch.max(output.detach(), 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().detach().cpu().item()
            
            # Calculate metrics
            train_loss /= len(train_loader)
            train_acc = 100. * train_correct / train_total
            val_loss /= len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Store history (detached from GPU)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.state_dict(), 'best_cnn_model.pth')
        
        print(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')
        return history
    
    def evaluate(self, traces, labels):
        """
        Evaluate the trained model on test data.
        
        Args:
            traces: Test power traces (numpy array)
            labels: Test labels (numpy array)
            
        Returns:
            Accuracy score
        """
        print(f"Evaluating CNN on {len(traces)} traces...")
        
        self.eval()
        
        # Convert to PyTorch tensors (keep on CPU for DataLoader)
        X_test = torch.FloatTensor(traces)
        y_test = torch.LongTensor(labels)
        
        # Create dataset and dataloader
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=256, 
            shuffle=False, 
            num_workers=6, 
            pin_memory=True
        )
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device, dtype=self.dtype), target.to(self.device)
                
                # Use automatic mixed precision
                with torch.cuda.amp.autocast():
                    outputs = self(data)
                
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().detach().cpu().item()
        
        accuracy = 100. * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        
        return accuracy
    
    def load_best_model(self):
        """Load the best saved model."""
        try:
            self.load_state_dict(torch.load('best_cnn_model.pth'))
            print("Loaded best CNN model.")
        except FileNotFoundError:
            print("No saved model found. Using current model state.")

def main():
    """Test the CNN attacker model."""
    print("Testing CNN Attacker...")
    
    # Generate some dummy data for testing
    num_traces = 1000
    trace_length = 1000
    num_classes = 5
    
    # Create dummy traces and labels
    traces = np.random.randn(num_traces, trace_length).astype(np.float32)
    labels = np.random.randint(0, num_classes, num_traces)
    
    # Initialize and test the model
    try:
        model = CNNAttacker(input_length=trace_length, num_classes=num_classes)
        print("CNN Attacker initialized successfully!")
        
        # Test forward pass
        with torch.no_grad():
            sample_input = torch.randn(32, trace_length).to(model.device)
            output = model(sample_input)
            print(f"Forward pass successful. Output shape: {output.shape}")
            
    except RuntimeError as e:
        print(f"Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
