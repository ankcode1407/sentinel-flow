import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

# --- 1. eBPF Data Ingestion Engine ---
class eBPFDataset(Dataset):
    def __init__(self, seq_length=64):
        self.seq_length = seq_length
        
        # Pointing to the BETH dataset file
        filename = "labelled_training_data.csv" 
        self.csv_path = rf"C:\Users\mishr\Downloads\BETH eBPF dataset\{filename}"
        
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Error: Could not find the dataset at: {self.csv_path}")

        print(f"Loading eBPF kernel traces from {self.csv_path}...")
        
        # BETH dataset uses 'eventId' for the syscall number and 'sus' for the malicious flag
        self.df = pd.read_csv(self.csv_path, usecols=['eventId', 'sus'])
        
        # Drop any missing values to prevent PyTorch errors
        self.df = self.df.dropna()
        
        self.syscalls = self.df['eventId'].astype(int).values
        self.labels = self.df['sus'].astype(int).values 
        
        # Dynamically determine the maximum syscall ID for the embedding layer
        self.max_syscall_id = int(self.syscalls.max())
        print(f"Maximum Syscall ID detected: {self.max_syscall_id}")
        
        self.num_sequences = len(self.syscalls) - self.seq_length
        print(f"Successfully extracted {self.num_sequences:,} continuous kernel sequences.")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Extract a window of consecutive syscalls
        seq = self.syscalls[idx : idx + self.seq_length]
        
        # Label is malicious if ANY syscall in this window is flagged 'sus' == 1
        label = 1 if 1 in self.labels[idx : idx + self.seq_length] else 0
        
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# --- 2. Kernel Mamba Architecture ---
class KernelMamba(torch.nn.Module):
    def __init__(self, vocab_size, d_model=2048):
        super().__init__()
        # Embedding layer turns discrete syscall IDs into dense vectors
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        
        # Mamba Core (Simplified placeholder for the Mamba SSM logic)
        self.mamba_compressor = torch.nn.Linear(d_model, d_model) 
        
        self.classifier = torch.nn.Linear(d_model, 2)

    def forward(self, syscall_sequence):
        # 1. Convert Syscalls to Vectors: [Batch, 64] -> [Batch, 64, 2048]
        x = self.embedding(syscall_sequence)
        
        # 2. Compress the sequence (Simulating Mamba's h_t update)
        state = torch.mean(torch.relu(self.mamba_compressor(x)), dim=1) 
        
        # 3. Classify
        logits = self.classifier(state)
        return logits, state


# --- 3. Execution & Testing ---
if __name__ == "__main__":
    print("Initializing eBPF Data Loader... (This may take a few seconds)")
    
    # Initialize the data loader
    dataset = eBPFDataset(seq_length=64)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize the Kernel Mamba model dynamically based on the dataset's vocabulary size
    # We add 1 to the max ID because index counting starts at 0
    model = KernelMamba(vocab_size=dataset.max_syscall_id + 1)
    
    # Fetch one batch of Kernel Data
    syscall_batch, label_batch = next(iter(dataloader))
    
    print("\n--- Aegis-State Pipeline Test ---")
    print(f"Input Syscall Tensor Shape: {syscall_batch.shape} (Batch Size, Sequence Length)")
    
    # Pass it through the model
    logits, hidden_state = model(syscall_batch)
    
    print(f"Hidden State Shape: {hidden_state.shape} (Ready for Forensic Inversion)")
    print(f"Prediction Logits Shape: {logits.shape}")
    print("---------------------\n")
    print("Success: eBPF tokenization and Mamba ingestion pipeline is operational.")