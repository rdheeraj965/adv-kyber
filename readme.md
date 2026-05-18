# Mitigating Deep Learning Side-Channel Attacks on CRYSTALS-Kyber via In-Band Noise Injection

This repository presents a simulation-driven study of side-channel leakage in CRYSTALS-Kyber and evaluates a lightweight hiding-based countermeasure called **In-Band Noise Injection**. The project combines synthetic power-trace generation, a 1D convolutional neural network attacker, and an interactive Streamlit dashboard to demonstrate how baseline leakage can enable secret recovery and how protected execution can push the attacker toward random guessing.

This README is written in a report-friendly format so it can be directly reused as source material for drafting a detailed project report.

## Table of Contents

1. [Introduction](#1-introduction)
   1. [Terminology](#11-terminology)
   2. [Purpose](#12-purpose)
   3. [Motivation](#13-motivation)
   4. [Problem Statement](#14-problem-statement)
   5. [Existing System](#15-existing-system)
   6. [Proposed System](#16-proposed-system)
   7. [Objectives](#17-objectives)
2. [Requirement Specification](#2-requirement-specification)
   1. [Hardware Requirements](#21-hardware-requirements)
   2. [Software Requirements](#22-software-requirements)
   3. [Functional Requirements](#23-functional-requirements)
   4. [Non-Functional Requirements](#24-non-functional-requirements)
3. [System Design](#3-system-design)
   1. [System Architecture](#31-system-architecture)
   2. [Data Flow](#32-data-flow)
   3. [1D-CNN Attack Design](#33-1d-cnn-attack-design)
   4. [Algorithmic Hiding Design](#34-algorithmic-hiding-design)
4. [Implementation](#4-implementation)
   1. [Implementation Overview](#41-implementation-overview)
   2. [Trace Generation](#42-trace-generation)
   3. [Dataset Preparation](#43-dataset-preparation)
   4. [Deep Learning Model](#44-deep-learning-model)
   5. [Protected Mode](#45-protected-mode)
   6. [Interactive Dashboard](#46-interactive-dashboard)
5. [Testing and Results](#5-testing-and-results)
   1. [Testing Strategy](#51-testing-strategy)
   2. [Baseline Attack Test](#52-baseline-attack-test)
   3. [Protected Mode Test](#53-protected-mode-test)
   4. [Data Stability Test](#54-data-stability-test)
   5. [Performance Test](#55-performance-test)
   6. [Dashboard Validation](#56-dashboard-validation)
6. [Conclusion](#6-conclusion)
7. [Future Scope](#7-future-scope)
8. [Project Files](#8-project-files)
9. [How to Run](#9-how-to-run)
10. [References](#10-references)

## 1. Introduction

CRYSTALS-Kyber, standardized as ML-KEM, is a post-quantum cryptographic scheme designed to resist attacks from both classical and quantum adversaries. While its mathematical hardness is strong, practical implementations can still leak secret-dependent information through physical side channels such as power consumption. This project studies that implementation-level risk and demonstrates a mitigation strategy tailored for deep-learning-based side-channel attacks.

### 1.1 Terminology

- **CRYSTALS-Kyber / ML-KEM**: A lattice-based post-quantum key encapsulation mechanism.
- **Side-Channel Attack (SCA)**: An attack that exploits physical leakage such as timing, power, or electromagnetic emissions.
- **Power Trace**: A sequence of sampled power values recorded during device execution.
- **Chosen Ciphertext Attack (CCA)**: An adversarial setting where crafted ciphertexts are used to influence targeted internal computation.
- **NTT**: Number Theoretic Transform used for efficient polynomial multiplication in lattice cryptography.
- **Hamming Weight Leakage**: A leakage model where power correlates with the number and position of active bits.
- **1D-CNN**: A one-dimensional convolutional neural network used to classify traces.
- **Algorithmic Hiding**: A defense method that obscures leakage by modifying how computation appears physically without changing the cryptographic result.

### 1.2 Purpose

The purpose of this project is to simulate a realistic side-channel setting for Kyber, train a deep learning attacker on the resulting traces, and evaluate whether a lightweight protection mechanism can reduce the attacker's effectiveness.

### 1.3 Motivation

Post-quantum algorithms are being deployed in security-sensitive systems, but practical deployment must account for physical leakage. Deep learning has made side-channel analysis more powerful because neural networks can automatically learn discriminative features from noisy traces. This creates a need for defenses that are effective, lightweight, and suitable for constrained hardware.

### 1.4 Problem Statement

Even when CRYSTALS-Kyber is mathematically secure, secret-dependent NTT operations can leak through power consumption. A profiled attacker using a deep neural network may recover secret key coefficients from single traces. The problem addressed here is how to reduce this leakage-driven attack accuracy without relying on expensive masking or excessive random noise.

### 1.5 Existing System

In a conventional unprotected implementation:

- the secret coefficient directly influences the intermediate arithmetic value,
- the arithmetic value maps to a predictable power pattern,
- a CNN can learn this pattern from training data,
- and single-trace classification can achieve very high accuracy.

Traditional defenses often rely on Gaussian noise injection or Boolean masking. Gaussian noise can be partially filtered out by strong models, while masking may introduce RNG cost and implementation complexity.

### 1.6 Proposed System

The proposed system introduces **In-Band Noise Injection**, a hiding-based mechanism that blends the real leakage with a mathematically valid dummy operation. Instead of adding external random noise alone, the defense injects meaningful internal activity so the observed power spike becomes ambiguous. This reduces the separability of classes seen by the CNN.

### 1.7 Objectives

- Simulate Kyber-like side-channel traces for secret coefficients in the range `[-2, -1, 0, 1, 2]`.
- Model leakage using a weighted Hamming weight approach.
- Train a 1D-CNN to recover secret classes from baseline traces.
- Generate protected traces using algorithmic hiding.
- Compare baseline and protected attack accuracy.
- Provide an interactive dashboard for visualization and explanation.
- Demonstrate that protected traces can drive the attacker near the random-guessing limit of `20%` for five classes.

## 2. Requirement Specification

### 2.1 Hardware Requirements

- NVIDIA GPU with CUDA support for CNN training
- Minimum 6 GB VRAM recommended for the configured data loaders and batch size
- Standard CPU for data generation and dashboard usage
- Sufficient disk space for `.npy` datasets
  - `baseline_traces.npy` and `protected_traces.npy` are each approximately 400 MB

### 2.2 Software Requirements

- Python 3.x
- PyTorch with CUDA
- NumPy
- SciPy
- Matplotlib
- Scikit-learn
- Streamlit
- Pandas

Current `requirements.txt` includes:

```txt
numpy
scipy
matplotlib
scikit-learn
--index-url https://download.pytorch.org/whl/cu128

torch
torchvision
torchaudio
```

Note: `streamlit` and `pandas` are used in `app.py` and should be installed if the dashboard is to be executed.

### 2.3 Functional Requirements

- Generate synthetic baseline and protected power traces.
- Label traces according to secret key coefficient classes.
- Save datasets to disk as `.npy` files.
- Train and evaluate a 1D-CNN on generated traces.
- Display comparative results between vulnerable and protected settings.
- Allow interactive simulation of trace parameters such as noise, jitter, key coefficient, and execution mode.
- Visualize trace waveforms, predicted probabilities, leakage profiles, and defense comparison.

### 2.4 Non-Functional Requirements

- High baseline attack success on unprotected traces.
- Protected mode should reduce attack accuracy close to random guessing.
- Reproducible trace generation through configurable random seeds.
- Clear modularity across generation, training, orchestration, and visualization.
- Dashboard should remain intuitive for demonstration and academic presentation.
- Defense should remain conceptually lightweight and edge-device friendly.

## 3. System Design

### 3.1 System Architecture

The project is organized into four main components:

1. **Trace Generator**: Produces baseline and protected traces using a weighted leakage model.
2. **CNN Attacker**: Learns to classify secret-dependent leakage from power traces.
3. **Pipeline Runner**: Executes dataset generation, training, and result reporting.
4. **Streamlit Dashboard**: Presents simulated traces, inference behavior, and defense insights interactively.

### 3.2 Data Flow

The logical flow of the system is:

1. Sample secret coefficient classes.
2. Map each class to a Kyber-relevant modular value.
3. Simulate NTT-related leakage using weighted Hamming weight.
4. Create a baseline trace or a protected trace with in-band blending.
5. Normalize and save traces.
6. Train the CNN on baseline or protected data.
7. Measure classification accuracy.
8. Visualize the attack and defense behavior in the dashboard.

### 3.3 1D-CNN Attack Design

The attacker is implemented in `cnn_attacker.py` as `SCA_CNN`. Its structure is:

- Input trace length: `1000`
- Conv1D block 1: `1 -> 16`, kernel size `11`, batch normalization, ReLU, max pooling
- Conv1D block 2: `16 -> 32`, kernel size `7`, batch normalization, ReLU, max pooling
- Conv1D block 3: `32 -> 64`, kernel size `5`, batch normalization, ReLU, max pooling
- Fully connected layers: flattened features -> `128` -> `5` output classes

This design is appropriate for local waveform pattern extraction around the leakage point and for robust classification under moderate noise and temporal jitter.

### 3.4 Algorithmic Hiding Design

The hiding mechanism is implemented by generating a dummy but mathematically valid operation and blending its weighted leakage with the real leakage. In the dashboard simulation, the target leakage is blended equally between the real and fake activity. In the dataset generator, protected traces are produced with full replacement of the original target leakage by dummy-derived leakage using `blend_weight=1.0`.

The key idea is that the observed power signal still looks structurally meaningful, but no longer maps cleanly to the actual secret coefficient.

## 4. Implementation

### 4.1 Implementation Overview

The repository consists of the following main files:

- `trace_generator.py`: Creates baseline and protected datasets.
- `cnn_attacker.py`: Defines and trains the CNN attack model.
- `main.py`: Runs the full data-generation and evaluation pipeline.
- `app.py`: Provides the interactive dashboard and explanatory visualizations.
- `references/`: Contains supporting papers and background material.

### 4.2 Trace Generation

The `TraceGenerator` class in `trace_generator.py` creates traces with the following logic:

- Secret classes are sampled from five values representing CBD coefficients.
- These classes are mapped to `Z_q` values under modulus `3329`.
- A chosen ciphertext condition is modeled using `u = 1`.
- The multiplication result is converted to a weighted Hamming weight leakage value.
- A power spike is injected around operation index `500`.
- Gaussian noise with standard deviation `0.4` is added.
- Temporal jitter in the range `[-2, 2]` is introduced.
- Final traces are Z-score normalized.

This produces traces that are simple enough for controlled experimentation yet expressive enough to support attack-versus-defense comparison.

### 4.3 Dataset Preparation

The dataset generator creates:

- `baseline_traces.npy`
- `baseline_labels.npy`
- `protected_traces.npy`
- `protected_labels.npy`

The pipeline uses `100000` traces. Labels correspond to the original secret classes, and the training script performs an `80/20` train-validation split.

### 4.4 Deep Learning Model

The CNN training pipeline in `cnn_attacker.py` includes:

- GPU-only execution with CUDA enforcement
- Batch size `128`
- Adam optimizer with learning rate `0.001`
- StepLR scheduler with `step_size=10` and `gamma=0.5`
- Mixed precision training through `torch.amp`
- Validation accuracy computation after training

The evaluation metric is classification accuracy on the held-out validation split.

### 4.5 Protected Mode

The protected mode uses in-band noise as an algorithmic hiding strategy. Rather than simply increasing external noise, it changes the leakage source itself by mixing in a dummy operation. The goal is to:

- preserve computational plausibility,
- break direct correspondence between leakage and the true key,
- and lower the confidence and correctness of the attacker.

The dashboard describes the intended protected-mode target as approximately `20%`, which corresponds to random guessing over five classes.

### 4.6 Interactive Dashboard

The Streamlit dashboard in `app.py` acts as a presentation and analysis layer. It provides:

- a sidebar for secret coefficient, execution mode, noise, jitter, and seed selection,
- KPI cards for baseline accuracy, protected accuracy, active key, and weighted leakage,
- an oscilloscope-style trace view,
- simulated CNN inference probability plots,
- leakage analysis charts,
- a threat and defense comparison section,
- a requirements checklist for validation reporting.

This makes the project useful not only as an experiment but also as a demonstration artifact for presentations and reports.

## 5. Testing and Results

### 5.1 Testing Strategy

The project evaluates two scenarios:

1. **Baseline attack test** on unprotected traces to confirm vulnerability.
2. **Protected-mode test** on defended traces to measure mitigation effectiveness.

Additional checks are made for data normalization, class separability, training stability, and dashboard consistency.

### 5.2 Baseline Attack Test

Expected baseline outcome:

- CNN is trained on unprotected traces.
- Leakage remains directly correlated with the secret coefficient.
- The attacker should achieve very high single-trace classification accuracy.

The project's stated acceptance target is:

- **Baseline accuracy > 95%**

The dashboard default shows a representative baseline accuracy of **97.3%**.

### 5.3 Protected Mode Test

Expected protected outcome:

- CNN is trained on protected traces.
- Leakage becomes ambiguous because the observed spike no longer corresponds cleanly to the real secret-dependent operation.
- Classification accuracy should collapse toward the random-guessing floor.

The project's stated acceptance target is:

- **Protected accuracy <= 25%**

The dashboard default shows a representative protected accuracy of **20.1%**.

### 5.4 Data Stability Test

Data stability is supported by:

- fixed random seeds,
- consistent trace length,
- controlled Gaussian noise,
- bounded temporal jitter,
- and Z-score normalization.

These design choices reduce training instability and improve repeatability of experiments.

### 5.5 Performance Test

Performance considerations visible in the implementation include:

- GPU-only training for faster deep learning execution,
- pinned memory and multi-worker data loading,
- mixed precision for better throughput,
- and learning-rate scheduling for convergence control.

This setup is suitable for running experiments on moderately capable CUDA hardware.

### 5.6 Dashboard Validation

The dashboard validates the conceptual behavior of the system by showing:

- clear trace differences between baseline and protected modes,
- correct mapping between coefficients and class labels,
- entropy-based interpretation of prediction uncertainty,
- side-by-side comparison of baseline and protected accuracy,
- and alignment of displayed metrics with the intended security claims.

## 6. Conclusion

This project demonstrates that deep-learning-based side-channel analysis can be highly effective against unprotected Kyber-style leakage, especially when the leakage source is localized and class-dependent. At the same time, it shows that **In-Band Noise Injection** can serve as a practical hiding mechanism by making the leakage semantically ambiguous rather than merely noisy.

Within the project's simulation framework, the baseline system remains strongly vulnerable, while the protected system reduces CNN performance to nearly random guessing. This suggests that algorithm-aware hiding can be a promising direction for securing post-quantum implementations against modern profiling attacks.

## 7. Future Scope

- Extend the work from synthetic traces to hardware-collected traces from real Kyber implementations.
- Evaluate the defense against stronger models such as residual CNNs, transformers, or multi-trace attacks.
- Measure overhead in terms of latency, energy, and area on embedded platforms.
- Explore adaptive or tunable blend ratios instead of fixed hiding behavior.
- Compare the proposed defense with masking, shuffling, dual-rail, and hybrid countermeasures using a common benchmark.
- Add confusion matrices, ROC-style metrics, and repeated-run statistics for more rigorous evaluation.
- Integrate automated experiment logging and result export for reproducible report generation.

## 8. Project Files

- [app.py](/c:/Users/deepa/Desktop/adv-kyber/app.py)
- [main.py](/c:/Users/deepa/Desktop/adv-kyber/main.py)
- [trace_generator.py](/c:/Users/deepa/Desktop/adv-kyber/trace_generator.py)
- [cnn_attacker.py](/c:/Users/deepa/Desktop/adv-kyber/cnn_attacker.py)
- [requirements.txt](/c:/Users/deepa/Desktop/adv-kyber/requirements.txt)
- [references](/c:/Users/deepa/Desktop/adv-kyber/references)

## 9. How to Run

### Generate data and train the CNN

```powershell
python main.py
```

### Launch the interactive dashboard

```powershell
streamlit run app.py
```

### Install dependencies

```powershell
pip install -r requirements.txt
pip install streamlit pandas
```

## 10. References

Reference material currently included in the repository:

- `references/Investigating_CRYSTALS-Kyber_Vulnerabilities_Attac.pdf`
- `references/cryptography-09-00064.pdf`
- `references/2024021606.pdf`

## Notes for Report Drafting

The project currently combines two complementary layers:

- a **real training pipeline** in `main.py`, `trace_generator.py`, and `cnn_attacker.py`,
- and a **presentation-oriented interactive simulator** in `app.py`.

For report writing, it is best to describe the dashboard as a visualization and explanatory interface, while treating the pipeline scripts as the primary experimental core.
