# Final Multi-Phase Implementation Plan

## Phase 1: Foundation & Infrastructure

### 1. Project Setup
- [X] Use uv to configure dependencies: ultralytics, pandas, numpy, pulp, psutil, pytest
- [X] Create directory structure: src/, tests/, logs/, results/
- [X] Setup error-focused logging

### 2. Data Models & Classes
- [X] Battery class: 4000 mAh capacity, charge/discharge simulation
- [X] YOLOModel class: wrapper for YOLOv10 models with psutil power profiling
- [X] EnergyData class: load/process clean energy CSVs
- [X] Controller base class: abstract interface

## Phase 2: Power Benchmarking

### 1. Power Measurement System
- [X] Implement psutil-based CPU power monitoring (cpu_percent, battery sensors)
- [X] Create benchmark suite for each YOLOv10 model (N,S,M,B,L,X)
- [X] Run models on benchmark images, measure power consumption
- [X] Store power profiles for 4000 mAh battery simulation

### 2. Battery Simulation
- [X] Implement realistic battery behavior with 100W USB-C charging
- [X] Validate power consumption models

## Phase 3: Controller Implementation

### 1. Base Controllers
- [X] NaiveWeakController: always smallest model, charge only when battery ≤ 20%
- [X] NaiveStrongController: always largest model, charge only when battery ≤ 20%
- [X] OracleController: PuLP MILP solver using future knowledge, charges when clean energy percentage is maximal

### 2. Custom Controller Training
- [X] Create MIPS solver to generate training data with diverse scenarios:
  - Battery levels: 5-100% (step 5%)
  - Clean energy: 0-100% (step 10%)
  - Accuracy requirements: 70-95% (step 5%)
  - Latency requirements: 1000-3000ms (step 250ms)
  - Target: 10,000 training samples
- [X] Cache MIPS results to JSON file for reuse
- [X] Gradient descent training with dual outputs (model selection + charging decision):
  - Loss: α * (1 - accuracy) + β * latency + γ * non-clean-energy-usage
  - Initial weights: α=0.5, β=0.3, γ=0.2
- [X] Implement CustomController that utilizes trained weights

### 3. Desired Functionality
- [X] A python file exists that solves the MIPS problem and outputs training data to JSON
- [X] A python file exists that trains the CustomController from scratch and saves weights to JSON

## Phase 4: Simulation Engine

### 1. Core Simulation
- [ ] 7-day simulation with 5-second task intervals
- [ ] Clean energy data integration (5-minute updates)
- [ ] 4 weeks (seasonal), 4 locations support

### 2. Metrics Collection
- [ ] Small/Large miss rates
- [ ] Total/clean energy usage
- [ ] CSV output generation

## Phase 5: Testing & Validation

### 1. Unit Tests
- [ ] pytest-based individual component testing

### 2. Integration Tests
- [ ] pytest-based full simulation validation

### 3. Performance Tests
- [ ] 192 total simulations

## Unresolved Questions
- None