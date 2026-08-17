# Intelligent Network Performance Optimization Using Machine Learning-Driven Adaptive QoS

## Overview

This project presents an intelligent network performance optimization system that uses **Machine Learning-driven Adaptive Quality of Service (QoS)** to improve network performance under varying traffic and congestion conditions.

The system simulates three different network approaches and compares their performance:

1. **FIFO Queue (No QoS)**
2. **Static Priority QoS**
3. **ML-Adaptive QoS**

The system evaluates these approaches using **delay, throughput, packet loss, and queue length**.

## Key Features

* Poisson-based network traffic generation
* Simulation of **VoIP, HTTP, and FTP** traffic
* FIFO queue-based network simulation
* Static priority-based QoS
* ML-based congestion prediction
* Decision Tree classifier
* Random Forest classifier
* Hyperparameter tuning using GridSearchCV
* 5-fold cross-validation
* Dynamic priority adaptation based on predicted congestion
* Performance comparison and visualization

## System Architecture

```text
                Traffic Generation
                       │
          ┌────────────┼────────────┐
          │            │            │
         VoIP         HTTP         FTP
          │            │            │
          └────────────┼────────────┘
                       │
                Network Traffic
                       │
       ┌───────────────┼────────────────┐
       │               │                │
      FIFO        Static Priority   ML-Adaptive
     (No QoS)          QoS              QoS
       │               │                │
       │               │        ML Congestion
       │               │          Prediction
       │               │                │
       │               │        Dynamic Priority
       └───────────────┼────────────────┘
                       │
                Performance Analysis
                       │
        ┌──────────────┼──────────────┐
        │              │              │
      Delay        Throughput     Packet Loss
                       │
                 Queue Length
```

## Traffic Model

Network traffic is generated using a **Poisson arrival process**.

The simulated applications are:

| Application | Priority | Packet Size |
| ----------- | -------: | ----------: |
| VoIP        |  Highest |   160 bytes |
| HTTP        |   Medium |  1500 bytes |
| FTP         |   Lowest |  1500 bytes |

The simulation uses different arrival rates for each application to represent heterogeneous network traffic.

## QoS Approaches

### 1. FIFO

The FIFO model provides a baseline without QoS. Packets are processed in their order of arrival.

### 2. Static Priority QoS

Packets are placed into separate queues according to application type.

```text
VoIP  → Highest Priority
HTTP  → Medium Priority
FTP   → Lowest Priority
```

The scheduler always selects the highest-priority non-empty queue.

### 3. ML-Adaptive QoS

The ML-based approach predicts the current congestion level and dynamically modifies application priorities.

The congestion levels are:

```text
Low
Medium
High
```

Dynamic priorities are adjusted according to the predicted congestion level, giving stronger protection to VoIP traffic during high congestion.

## Machine Learning

Two classification models are trained and compared:

* **Decision Tree**
* **Random Forest**

### Input Features

The congestion prediction model uses:

* Arrival rate
* Current queue length
* Previous throughput

### Target Classes

```text
0 → Low Congestion
1 → Medium Congestion
2 → High Congestion
```

The dataset is divided into **80% training data and 20% testing data**.

GridSearchCV and 5-fold cross-validation are used for model selection and hyperparameter tuning.

## Performance Metrics

The project compares the three approaches using:

* **Average Delay**
* **Average Throughput**
* **Packet Loss Rate**
* **Queue Length**

The program also calculates the improvement of ML-Adaptive QoS over the FIFO baseline.

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Machine Learning
* Network Simulation
* Quality of Service (QoS)

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/intelligent-network-performance-optimization.git
cd intelligent-network-performance-optimization
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

```text
numpy
pandas
matplotlib
scikit-learn
```

## Running the Project

Run:

```bash
python network_performance.py
```

The program will:

1. Generate network traffic.
2. Simulate FIFO.
3. Simulate Static Priority QoS.
4. Prepare ML training data.
5. Train Decision Tree and Random Forest models.
6. Select the best-performing model.
7. Run ML-Adaptive QoS.
8. Compare all approaches.
9. Generate performance visualizations.

The comparison graph is saved in:

```text
outputs/qos_comparison.png
```

## Results

The program automatically reports:

```text
Average Delay
Average Throughput
Packet Loss Rate
Delay Reduction
Throughput Improvement
Packet Loss Reduction
```

The actual performance values are generated during execution and should be used when reporting experimental results.

## Project Objective

The primary objective is to demonstrate how **machine learning can be integrated with network QoS mechanisms to dynamically respond to congestion instead of relying only on fixed priority rules**.

## Future Improvements

* Real-time network traffic integration
* Deep learning-based congestion prediction
* Reinforcement learning for QoS optimization
* Additional traffic classes
* Real network testbed validation
* More advanced fairness and starvation analysis

## Author

**Pulla Siddardha**

B.Tech – Electronics and Communication Engineering
