# Aegis-State: State-Space Models for Intrusion Detection and Privacy Vulnerability Analysis

## Overview
Aegis-State is a cloud-native Intrusion Detection System (IDS) and AI security research project. It explores the deployment of Selective State-Space Models (Mamba) for real-time Linux kernel telemetry (eBPF) analysis. 

Beyond high-accuracy threat detection, this project conducts a rigorous empirical audit of AI data privacy. It demonstrates a critical structural vulnerability in Split-Learning architectures, proving that standard enterprise Differential Privacy (DP-SGD) fails to protect against Feature Space Hijacking Attacks (FSHA) during live inference.

## Key Features
* **eBPF Kernel Telemetry:** Ingests and tokenizes live Linux system calls using the BETH dataset, creating overlapping 64-step temporal sequences.
* **Mamba-SSM Gateway:** Utilizes a linear-time State-Space Model ($O(N)$) to achieve deep contextual sequence modeling of kernel behavior, bypassing Transformer latency bottlenecks.
* **Deep Residual Inversion:** Implements a ResNet-based Forensic Inverter to reverse-engineer compressed 2048-dimensional Mamba hidden states back into raw syscalls.
* **Privacy Vulnerability Proof:** Integrates PyTorch `Opacus` to apply DP-SGD. The project empirically proves that while DP-SGD protects the backward pass (weights), the deterministic forward pass (activations) leaks reconstructable data, negating the privacy guarantee in edge-to-cloud architectures.

## Repository Structure

```text
SENTINEL_FLOW/
├── forensic_proxy/          # Microservice: Asynchronous forensic analysis
│   ├── app.py
│   └── Dockerfile
├── gateway/                 # Microservice: Mamba inference engine
│   ├── app.py
│   └── Dockerfile
├── traffic_gen/             # Microservice: eBPF sequence simulation
│   ├── app.py
│   └── Dockerfile
├── venv/                    # Virtual Environment
├── .gitignore
├── docker-compose.yml       # Multi-container orchestration
├── ebpf_builder.py          # Data ingestion and tensor pipeline
├── train_aegis.py           # Standard Mamba training loop
├── train_aegis_dp.py        # Differentially Private (DP-SGD) training loop
├── kernel_inverter.py       # Feature Space Hijacking attack execution
├── aegis_kernel_mamba.pth   # Standard model weights (Generated)
├── aegis_kernel_mamba_DP.pth# Private model weights (Generated)
└── aegis_inverter.pth       # Attacker model weights (Generated)
