import time
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Sentinel-Flow Forensic Proxy")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. The ResNet Inverter Architecture ---
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(x + self.block(x))

class ResNetInverter(nn.Module):
    def __init__(self, state_dim=2048, out_dim=1312): # 32 * 41 = 1312
        super().__init__()
        hidden_dim = 4096 
        self.entry = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim)
        )
        self.exit = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.entry(x)
        x = self.res_blocks(x)
        return self.exit(x)

inverter_model = None

@app.on_event("startup")
def load_model():
    global inverter_model
    print(f"Loading Forensic Proxy on {device}...")
    inverter_model = ResNetInverter().to(device)
    inverter_model.eval()

# --- 2. The Latency-Optimized Inference Route ---
class StatePayload(BaseModel):
    state: list[float]

@app.post("/invert")
async def invert_state(payload: StatePayload):
    start_time = time.perf_counter()
    
    # Format input
    state_tensor = torch.tensor([payload.state], dtype=torch.float32).to(device)
    
    # Reconstruct
    with torch.no_grad():
        reconstruction = inverter_model(state_tensor)
        
    # Calculate latency (Crucial for our research claim)
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    
    print(f"🚨 ALERT: Forensic inversion completed in {latency_ms:.2f} ms")
    
    return {
        "status": "success", 
        "latency_ms": latency_ms
    }