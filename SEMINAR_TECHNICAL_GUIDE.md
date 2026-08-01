# 🎓 Digital Twin Heart: Technical & Seminar Presentation Guide

This guide contains everything you need to deliver a seminar, present your final year project, or answer technical viva questions with confidence.

---

## 1. Executive Summary & Elevator Pitch (30 Seconds)

> **"Our project is an integrated Digital Twin Heart platform that simulates how drugs, dose levels, and patient physiology impact cardiovascular health in real time.**
> 
> **It combines a 7-layer biophysical simulation engine (pharmacokinetics, pharmacodynamics, and arterial Windkessel pressure modeling) with a multi-model machine learning stack (XGBoost, LightGBM, Random Forest, and Keras LSTM) and a local RAG medical assistant (MediRAG) powered by LLaMA 3.2."**

---

## 2. Problem Statement & Motivation

- **Problem**: Clinical drug trials are expensive, time-consuming, and carry inherent risks of adverse cardiac events (such as QTc prolongation and Torsades de Pointes arrhythmia).
- **Solution**: A **Digital Twin** creates a patient-specific in-silico model that predicts drug distribution, target receptor binding, arterial blood pressure, and ECG changes before administering medication in real life.

---

## 3. Core Architecture: The 7-Layer Simulation Engine

When presenting the simulation engine, walk through the 7 layers sequentially:

```
Drug Name & Dose ➔ Layer 1 (SMILES/FP) ➔ Layer 2 (Receptor Binding) ➔ Layer 3 (PK ADME) 
➔ Layer 4 (PD Emax) ➔ Layer 5 (CV Windkessel/ECG) ➔ Layer 6 (ML Risk Scoring) ➔ Layer 7 (Visual Twin)
```

| Layer | Technical Component | Mathematical / Algorithmic Concept |
| :--- | :--- | :--- |
| **Layer 1: Representation** | Molecular Structure | Converts drug name to canonical SMILES string and 1024-bit Morgan Fingerprint vector. |
| **Layer 2: Target Binding** | Ion Channels & Receptors | Computes dynamic binding occupancy for $\text{hERG}$, $\text{Na}_{\text{v}}1.5$, $\text{Ca}_{\text{v}}1.2$, $\beta_1$, $\text{ACE}$, and $\text{AT}_1$ using Hill-equation affinity scaling. |
| **Layer 3: Pharmacokinetics** | ADME Clearance | Solves 1-compartment and 2-compartment ODE systems using `scipy.integrate.solve_ivp` to compute $C_{\text{max}}$, $T_{\text{max}}$, and $\text{AUC}$. |
| **Layer 4: Pharmacodynamics** | Organ-level Response | Applies sigmoid $E_{\text{max}}$ saturating effect models to determine heart rate, blood pressure, and QTc interval changes. |
| **Layer 5: Cardiovascular** | Hemodynamics & Electrophysiology | Computes arterial blood pressure via 2-element Windkessel model ($R \cdot C$) and synthesizes 12-lead ECG waveforms. |
| **Layer 6: ML Assessment** | Risk Classification | Calculates Arrhythmia Risk, Cardiac Event Probability, and Therapeutic Index scores. |
| **Layer 7: Output & Canvas** | 3D Visualization | Renders real-time anatomical heart contractions on an HTML5 canvas with dynamic QTc risk heatmaps. |

---

## 4. Machine Learning & Deep Learning Stack

### Activity Heart Rate Predictor (Phase 2)
Translates natural-language activity prompts (e.g. *"running 5km in morning"*) into heart rate predictions.

1. **Feature Engineering**:
   - Classifies activity into MET (Metabolic Equivalent of Task) categories.
   - Calculates baseline HR using the **Karvonen Formula**: 
     $$\text{Target HR} = \text{Resting HR} + \text{Intensity} \times (\text{HR}_{\text{max}} - \text{Resting HR})$$
   - Generates a 22-dimensional feature vector combining patient age, weight, height, gender, and activity intensity.

2. **Model Architecture**:
   - **Ensemble Model**: Weighted average of **XGBoost**, **LightGBM**, and **Random Forest Regressors**.
   - **LSTM Recurrent Neural Network**: Sequential 10-step model trained with a custom **Heteroscedastic Loss**:
     $$\mathcal{L} = \frac{1}{2} \left( \frac{(y - \hat{y})^2}{\sigma^2} + \ln(\sigma^2) \right)$$
     *(Accounts for inherent variance and uncertainty in real-world activity data).*

---

## 5. Generative AI & MediRAG (Phase 3)

- **Architecture**: Retrieval-Augmented Generation (RAG) pipeline for answering medical queries over uploaded documents.
- **Components**:
  1. **Document Loader & Text Splitter**: `PyPDFLoader` / `TextLoader` + `RecursiveCharacterTextSplitter` (chunk size: 1000, overlap: 200).
  2. **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (`HuggingFaceEmbeddings`).
  3. **Vector Database**: Disk-persisted **ChromaDB** (`./chroma_db`).
  4. **LLM**: Local **Ollama LLaMA 3.2** executing private inference without cloud data leaks.

---

## 6. Advanced Technical Features

- **Polypharmacy (Combination Therapy)**:
  - Models competitive and additive ion channel blockade when administering two drugs concurrently (e.g. Metoprolol + Amlodipine):
    $$\text{Occupancy}_{\text{combined}} = 1 - (1 - p_1)(1 - p_2)$$
- **Patient Physiology Factors**:
  - **eGFR (Glomerular Filtration Rate)**: Scales renal elimination rate $k_e$.
  - **Serum Potassium ($\text{K}^+$)**: Hypokalemia ($<3.5\text{ mM}$) increases $\text{hERG}$ sensitivity and QTc prolongation risk.
  - **Baseline QTc**: Dynamically shifts individual patient baseline repolarization time.

---

## 7. Top 10 Viva / Presentation Questions & Winning Answers

### Q1: What is a "Digital Twin" in healthcare?
> **Answer**: A Digital Twin is a dynamic, computational virtual model of a physical organ or biological system. It uses patient-specific physiological and pharmacological parameters to simulate real-world responses to therapies before administering them to the actual patient.

### Q2: How does your system model drug pharmacokinetics (PK)?
> **Answer**: PK is modeled using first-order absorption and elimination kinetics. We solve systems of differential equations ($\text{ODEs}$) for 1-compartment and 2-compartment clearance using `scipy.integrate.solve_ivp`, calculating concentration curves over time, $C_{\text{max}}$, $T_{\text{max}}$, and Area Under the Curve ($\text{AUC}$).

### Q3: Why did you choose a 2-element Windkessel model for blood pressure?
> **Answer**: The Windkessel model simulates arterial hemodynamics by treating arterial compliance ($C$) as a capacitor and total peripheral resistance ($R$) as a resistor. This provides mathematically sound systolic, diastolic, and Mean Arterial Pressure ($\text{MAP}$) waveforms.

### Q4: How does the app handle polypharmacy (multi-drug interaction)?
> **Answer**: When two drugs are selected, the engine computes independent receptor affinities for both compounds and applies a combined probabilistic occupancy formula on ion channels ($\text{hERG}$, $\text{Na}_{\text{v}}1.5$, $\text{Ca}_{\text{v}}1.2$) to detect synergistic QTc prolongation and arrhythmia risks.

### Q5: What is heteroscedastic loss in your LSTM model?
> **Answer**: Heteroscedastic loss models data uncertainty by simultaneously predicting both the mean heart rate ($\hat{y}$) and variance ($\sigma^2$). It penalizes prediction errors proportionally to data uncertainty, improving neural network robustness on noisy fitness data.

### Q6: How does MediRAG preserve patient data privacy?
> **Answer**: MediRAG runs entirely locally. Embeddings are generated on the machine using HuggingFace models, stored in a local ChromaDB instance, and answered using a locally hosted Ollama LLaMA 3.2 model — no data is sent to external cloud APIs.

### Q7: What happens if Ollama or RDKit is not installed on a user's computer?
> **Answer**: The application features a fault-tolerant fallback architecture. If RDKit is missing, it uses deterministic string hashing for fingerprints; if Ollama is absent, it generates structured rule-based medical summaries without crashing.

### Q8: How does the anatomical heart canvas update?
> **Answer**: The main Flask dashboard calculates cardiac parameters and passes them via HTML5 `postMessage` API to the embedded iframe canvas (`heart-simulation.html`), which adjusts contraction frequency, stroke volume pulsation, and canvas heatmap colors in real time.

### Q9: How is the drug safety score calculated?
> **Answer**: Safety score starts at 10.0 and applies weighted demerits for $\text{hERG}$ binding strength ($>0.3$), QTc prolongation ($>50\text{ ms}$), severe BP fluctuations, contractility reduction, non-selectivity, and overdose ratios.

### Q10: What is the future scope of this project?
> **Answer**: Future enhancements include WebSockets for real-time wearable watch streaming, 3D Mesh heart rendering via Three.js, and training target interaction weights against PubChem/ChEMBL bioassay datasets.
