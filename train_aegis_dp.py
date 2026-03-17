import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from opacus import PrivacyEngine # The cryptographic wrapper

from ebpf_builder import eBPFDataset, KernelMamba

def train_dp_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Privacy-Preserving Aegis-State on: {device}")

    print("⏳ Loading eBPF Dataset into RAM...")
    dataset = eBPFDataset(seq_length=64)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    model = KernelMamba(vocab_size=dataset.max_syscall_id + 1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # --- THE DIFFERENTIAL PRIVACY ENGINE ---
    print("🛡️ Attaching Opacus Privacy Engine...")
    privacy_engine = PrivacyEngine()
    
    # This automatically modifies the model to clip gradients and inject noise
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=1.2, # The amount of "blur" added to the memory
        max_grad_norm=1.0,    # The strictness of the gradient clipping
    )

    epochs = 1
    print("\n🔥 Starting Differentially Private Training Loop...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (syscalls, labels) in enumerate(dataloader):
            syscalls, labels = syscalls.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, hidden_states = model(syscalls)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0 and batch_idx > 0:
                elapsed = time.time() - start_time
                epsilon = privacy_engine.get_epsilon(delta=1e-5)
                print(f"DP-Epoch [1/1] | Batch [{batch_idx}/{len(dataloader)}] | Loss: {running_loss/100:.4f} | Epsilon (ε): {epsilon:.2f} | Time: {elapsed:.2f}s")
                
               
                torch.save(model.state_dict(), "aegis_kernel_mamba_DP.pth")
                print("Early DP weights saved as 'aegis_kernel_mamba_DP.pth'")
                return 
                # ---------------------------------------

    torch.save(model.state_dict(), "aegis_kernel_mamba_DP.pth")
    print("\n🏆 DP Training Complete! Model weights saved as 'aegis_kernel_mamba_DP.pth'")

if __name__ == "__main__":
    train_dp_model()