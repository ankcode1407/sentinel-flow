import torch
import torch.nn as nn
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import httpx  # For sending state to forensic proxy asynchronously

app = FastAPI(title="Sentinel-Flow Gateway")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Mamba Model Stub (Cleaned up from the notebook) ---
class MambaIDS(nn.Module):
    def __init__(self):
        super().__init__()
        # Standardized dimensions based on our research phase
        self.encoder = nn.Linear(32 * 41, 2048) 
        self.classifier = nn.Linear(2048, 2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        state = torch.relu(self.encoder(x))
        logits = self.classifier(state)
        return logits, state

mamba_model = None

@app.on_event("startup")
def load_model():
    global mamba_model
    print(f"Loading Mamba Gateway on {device}...")
    mamba_model = MambaIDS().to(device)
    mamba_model.eval()

# --- 2. Inference Route ---
class TrafficPayload(BaseModel):
    packet_data: list[list[float]] # Expecting a 32x41 array

async def trigger_forensics(state_vector: list):
    # Sends the hidden state to the Glass-Box proxy without slowing down the Gateway
    async with httpx.AsyncClient() as client:
        try:
            await client.post("http://forensic_proxy:8000/invert", json={"state": state_vector})
        except Exception as e:
            print(f"Forensic proxy unreachable: {e}")

@app.post("/scan")
async def scan_traffic(payload: TrafficPayload, background_tasks: BackgroundTasks):
    # 1. Format input
    input_tensor = torch.tensor([payload.packet_data], dtype=torch.float32).to(device)
    
    # 2. Fast Inference
    with torch.no_grad():
        logits, hidden_state = mamba_model(input_tensor)
        prediction = torch.argmax(logits, dim=1).item()
        
    # 3. If malicious, trigger forensics asynchronously
    if prediction == 1:
        state_list = hidden_state[0].cpu().tolist()
        background_tasks.add_task(trigger_forensics, state_list)
        
    return {"status": "Malicious" if prediction == 1 else "Benign"}