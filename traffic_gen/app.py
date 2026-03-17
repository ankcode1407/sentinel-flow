import time
import requests
import random

# Generate a fake 32x41 packet payload (Normal Traffic)
def generate_normal_traffic():
    return [[random.random() for _ in range(41)] for _ in range(32)]

# Generate the "Staircase" OOD Attack (Malicious Traffic)
def generate_ood_staircase():
    payload = [[0.0 for _ in range(41)] for _ in range(32)]
    for t in range(5, 15):
        payload[t][2] = (t - 5) * 0.5
    return payload

print("🚀 Starting Traffic Simulator...")
time.sleep(5) # Wait for the gateway to boot up

while True:
    try:
        if random.random() > 0.1:
            # 90% Normal Traffic
            requests.post("http://sentinel_gateway:8000/scan", json={"packet_data": generate_normal_traffic()})
            print("Sent Normal Traffic")
        else:
            # 10% OOD Attack
            print("🕵️ Injecting OOD Staircase Attack...")
            requests.post("http://sentinel_gateway:8000/scan", json={"packet_data": generate_ood_staircase()})
            
    except Exception as e:
        print(f"Gateway unreachable: {e}")
        
    time.sleep(2) # Send a request every 2 seconds