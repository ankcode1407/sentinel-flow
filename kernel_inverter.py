import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

from ebpf_builder import eBPFDataset, KernelMamba

# --- 1. The Kernel-Level Forensic Inverter ---
class SyscallInverter(nn.Module):
    def __init__(self, state_dim=2048, seq_length=64, vocab_size=1011):
        super().__init__()
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        # Expands the compressed 2048 state back into the sequence timeline
        self.decoder = nn.Sequential(
            nn.Linear(state_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.2),
            nn.Linear(4096, seq_length * vocab_size)
        )

    def forward(self, state):
        x = self.decoder(state)
        # Reshape to [Batch, Sequence_Length, Vocab_Size]
        return x.view(-1, self.seq_length, self.vocab_size)

# --- 2. The Inversion Attack Pipeline ---
def train_inverter():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Preparing Kernel Inverter on: {device}")

    # Load the frozen Target Model (The one currently training)
    print("Loading Frozen Aegis-State Gateway...")
    target_model = KernelMamba(vocab_size=1011).to(device)
    
    try:
        # 1. Load the raw dictionary saved by Opacus
        raw_state_dict = torch.load("aegis_kernel_mamba_DP.pth", map_location=device)
        
        # 2. Strip the '_module.' prefix from every key
        clean_state_dict = {}
        for key, value in raw_state_dict.items():
            clean_key = key.replace('_module.', '')
            clean_state_dict[clean_key] = value
            
        # 3. Load the clean dictionary into the target model
        target_model.load_state_dict(clean_state_dict)
        target_model.eval() 
        print("✅ DP Aegis-State Gateway Loaded Successfully!")
        
    except FileNotFoundError:
        print("Error: aegis_kernel_mamba_DP.pth not found. Wait for training to finish!")
        return

    # Initialize the attacker model
    inverter = SyscallInverter(vocab_size=1011).to(device)
    
    # We use CrossEntropyLoss because we are predicting discrete Syscall IDs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(inverter.parameters(), lr=0.001)

    # Load Data
    dataset = eBPFDataset(seq_length=64)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    print("\nStarting Inverter Attack Training...")
    epochs = 1
    
    for epoch in range(epochs):
        inverter.train()
        running_loss = 0.0
        
        for batch_idx, (syscalls, labels) in enumerate(dataloader):
            syscalls = syscalls.to(device)
            
            # Step A: Get the frozen hidden state from the Gateway
            with torch.no_grad():
                _, hidden_states = target_model(syscalls)
            
            # Step B: Attempt to reconstruct the raw syscalls from the state
            optimizer.zero_grad()
            reconstructed_logits = inverter(hidden_states)
            
            # Reshape for CrossEntropyLoss: [Batch * Seq_Len, Vocab_Size] vs [Batch * Seq_Len]
            loss = criterion(reconstructed_logits.view(-1, 1011), syscalls.view(-1))
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 500 == 0 and batch_idx > 0:
                print(f"Inversion Attack | Batch [{batch_idx}/{len(dataloader)}] | Reconstruction Loss: {running_loss/500:.4f}")
                running_loss = 0.0

    torch.save(inverter.state_dict(), "aegis_inverter.pth")
    print("Inverter Training Complete. Weights saved as aegis_inverter.pth")

if __name__ == "__main__":
    train_inverter()