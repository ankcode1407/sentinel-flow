import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

# Import the architecture and dataset you just built
from ebpf_builder import eBPFDataset, KernelMamba

def train_model():
    # 1. Hardware Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Aegis-State Training on: {device}")

    # 2. Load the Dataset
    print("⏳ Loading eBPF Dataset into RAM...")
    dataset = eBPFDataset(seq_length=64)
    # We use a larger batch size (128) to speed up training on 760k+ samples
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # 3. Initialize Model and Optimizer
    model = KernelMamba(vocab_size=dataset.max_syscall_id + 1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # 4. The Training Loop (1 Epoch for rapid prototyping)
    epochs = 1
    print("\n🔥 Starting Training Loop...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (syscalls, labels) in enumerate(dataloader):
            # Move data to GPU/CPU
            syscalls, labels = syscalls.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits, hidden_states = model(syscalls)
            
            # Calculate error and backpropagate
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print progress every 500 batches 
            if batch_idx % 500 == 0 and batch_idx > 0:
                elapsed = time.time() - start_time
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(dataloader)}] | Loss: {running_loss/500:.4f} | Time: {elapsed:.2f}s")
                running_loss = 0.0
                start_time = time.time()

    # 5. Save the trained brain
    torch.save(model.state_dict(), "aegis_kernel_mamba.pth")
    print("\n🏆 Training Complete! Model weights saved as 'aegis_kernel_mamba.pth'")

if __name__ == "__main__":
    train_model()