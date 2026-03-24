import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np

# Hard Hardware Guard
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. GPU is strictly required for this project.")

class SCA_CNN(nn.Module):
    def __init__(self, trace_length=1000, num_classes=5):
        super(SCA_CNN, self).__init__()
        # 1D Convolutional Architecture optimized for Power Traces
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, padding=5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Calculate flattened size
        reduced_length = trace_length // 8
        self.classifier = nn.Sequential(
            nn.Linear(64 * reduced_length, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1) # Add channel dimension
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.classifier(x)
        return x

def train_and_evaluate(traces_path, labels_path, epochs):
    device = torch.device('cuda')
    
    # Load Data
    traces = torch.tensor(np.load(traces_path), dtype=torch.float32)
    labels = torch.tensor(np.load(labels_path), dtype=torch.long)
    
    # Split 80/20
    dataset_size = len(traces)
    train_size = int(0.8 * dataset_size)
    
    train_dataset = TensorDataset(traces[:train_size], labels[:train_size])
    val_dataset = TensorDataset(traces[train_size:], labels[train_size:])
    
    # Optimized DataLoader for 6GB VRAM
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    model = SCA_CNN().to(device)
    # PyTorch 2.0 Optimization
    # model = torch.compile(model) 
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Add the Learning Rate Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"Training on {traces_path}...")
    for epoch in range(epochs):
        model.train()
        for batch_traces, batch_labels in train_loader:
            batch_traces = batch_traces.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Automatic Mixed Precision
            with torch.amp.autocast('cuda'):
                outputs = model(batch_traces)
                loss = criterion(outputs, batch_labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Step the scheduler at the end of each epoch
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} complete. Current LR: {scheduler.get_last_lr()[0]}")

    # Evaluation Phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_traces, batch_labels in val_loader:
            batch_traces = batch_traces.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            
            outputs = model(batch_traces)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().detach().cpu().item()
            
    accuracy = 100 * correct / total
    return accuracy