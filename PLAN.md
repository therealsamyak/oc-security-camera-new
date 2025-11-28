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
- [ ] Implement psutil-based CPU power monitoring (cpu_percent, battery sensors)
- [ ] Create benchmark suite for each YOLOv10 model (N,S,M,B,L,X)
- [ ] Run models on benchmark images, measure power consumption
- [ ] Store power profiles for 4000 mAh battery simulation

### 2. Battery Simulation
- [ ] Implement realistic battery behavior with 100W USB-C charging
- [ ] Validate power consumption models

## Phase 3: Controller Implementation

### 1. Base Controllers
- [ ] NaiveWeakController: always smallest model
- [ ] NaiveStrongController: always largest model
- [ ] OracleController: PuLP MILP solver using future knowledge

### 2. Custom Controller Training
- [ ] Gradient descent with loss: α * (1 - accuracy) + β * latency + γ * non-clean-energy-usage
- [ ] Optimize weights for accuracy, latency, clean energy factors
- [ ] Implement CustomController with trained weights

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