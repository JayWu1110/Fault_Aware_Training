# Fault Aware Training (FAT) for Spiking Neural Networks

This project implements a **Fault Aware Training (FAT)** framework designed to improve the robustness and reliability of Spiking Neural Networks (SNNs) deployed on fault-prone neuromorphic hardware. 

## Overview
While SNNs are highly energy-efficient and biologically inspired, neuromorphic hardware is vulnerable to physical faults (e.g., neuron failures, synaptic errors, and random noise). Traditional error correction mechanisms add unwanted computational overhead. Our FAT framework solves this by integrating fault resilience directly into the software-level training process, teaching the network to adapt to errors dynamically.

## Key Techniques

Our modified backpropagation approach relies on three core mechanisms:

* **Fault Simulation:** During training, we simulate real-world hardware failures (e.g., "Hard to Spike" faults). We inject faults into neurons with a 25% probability, updating their weights with randomly selected wrong values (between -10 and 10).
* **Targeted Weight Recovery:** To prevent the network from completely overfitting to the simulated distortions, 25% of neurons are randomly selected during each iteration. Their weights are restored to their original values after backpropagation, stabilizing the learning process.
* **Gradient Clipping:** To stabilize deep SNN training and prevent exploding gradients caused by discrete spike activations, the maximum gradient norm is capped at a threshold of 10.

## Experimental Results
Evaluated on the **MNIST dataset**, the FAT framework consistently outperforms normal backpropagation. Key findings include:
* Higher classification accuracy in **both** fault-free and faulty hardware scenarios.
* An average accuracy improvement of **~2.0%** specifically under faulty conditions, proving the effectiveness of the targeted weight recovery and fault simulation.

## Environmental Setup
Download the tool for constructing virtual environment
```bash
sudo apt install python3-venv
```

Construct virtual environment
```bash
python3 -m venv venv
```

Enter into virtual environment
```bash
source venv/bin/activate
```

Install all required tool
```bash
pip3 install -r requirement.txt
```

Compilation and Executing
```bash
python3 fat.py
```



