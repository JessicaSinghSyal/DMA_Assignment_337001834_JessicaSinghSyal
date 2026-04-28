# 🔍 FraudLens: Isolating, Clustering, and Reconstructing Fraud Without a Single Label

> **Can unsupervised methods — trained with zero knowledge of fraud labels — learn representations of "normal" transaction behavior strong enough to flag the 0.17% of transactions that deviate from it?**

**Author:** Jessica Singh Syal | **UIN:** 337001834  
**Course:** Data Mining — CSCE 676 (Spring 2026)  
**Institution:** Texas A&M University  

---

## 🎥 Project Video

📺 **[Watch the 2-minute project video on YouTube →](https://youtu.be/nSqaTecwNxg?si=OFX2HMlFCv9m2iU4)**

---

## 👉 Start Here: [`main_notebook.ipynb`](main_notebook.ipynb)

This is the final curated notebook. It tells the complete story: motivation → EDA → methods → results → analysis. Run all cells top-to-bottom in Google Colab.

---

## 📋 Project Overview

Credit card fraud costs the global economy over **$33 billion annually**. Detection systems must identify 492 needles in a haystack of 284,315 transactions — *without* original transaction features (PCA-transformed for privacy), and often without reliable labels.

**FraudLens** investigates this challenge through three methodological pillars:

| Pillar | Technique | Type |
|--------|-----------|------|
| **Anomaly Detection** | Isolation Forest vs Local Outlier Factor | Course method |
| **Unsupervised Clustering** | DBSCAN vs K-Means | Course method |
| **Deep Reconstruction** | MLP Autoencoder | External (beyond course) |

All three methods operate **without fraud labels during training** — labels are used only for post-hoc evaluation.

---

## ❓ Research Questions

**RQ1 (Course — Anomaly Detection):** Do fraud transactions form geometrically isolable anomalies in feature space, and does tree-based isolation (Isolation Forest) outperform density-based proximity (LOF)?

**RQ2 (Course — Clustering):** Does unsupervised transaction clustering recover behaviorally coherent fraud concentrations without access to labels?

**RQ3 (External — Deep Anomaly Detection):** Does the normal-class transaction manifold carry a nonlinear reconstruction signal strong enough to outperform classical anomaly detectors?

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Name** | Credit Card Fraud Detection |
| **Source** | [ULB Machine Learning Group / Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) / [TensorFlow mirror](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv) |
| **Size** | 284,807 transactions × 31 features |
| **Features** | `Time`, `V1`–`V28` (PCA-anonymous), `Amount`, `Class` |
| **Class imbalance** | 492 fraud (0.17%) vs 284,315 normal — **577:1** |
| **Time window** | 48 hours, September 2013, European cardholders |
| **License** | Open Database License (ODbL) |
| **Missing values** | None |

**Data access:** The notebook downloads the dataset automatically from the TensorFlow public mirror. No manual download needed. If the download fails, download `creditcard.csv` from Kaggle and place it in a `data/` folder.

### Preprocessing

| Feature | Treatment | Rationale |
|---------|-----------|-----------|
| V1–V28 | No scaling | Already zero-mean, unit-variance (PCA-constructed) |
| Amount | RobustScaler (fit on normal class only) | Right-skewed; outliers must not dominate scale |
| Time | RobustScaler (fit on normal class only) | Raw range 0–172,792 would dominate distance calculations |

---

## 🏆 Key Results

| Method | PR-AUC | Notes |
|--------|--------|-------|
| Random baseline | 0.0017 | All-positive classifier |
| Isolation Forest (best) | *see notebook* | Top-ADI-8, optimized contamination |
| LOF (best) | *see notebook* | k-NN sweep |
| Linear PCA Proxy | *see notebook* | Normal-fit PCA, linear reconstruction |
| **MLP Autoencoder (best)** | **see notebook** | **Winner — Bottleneck & depth swept** |

**Big Takeaway:** Fraud in this dataset is geometrically *subtle*, not dramatic. Tree-based isolation methods underperform because fraud is not simply isolated in random-split space. The reconstruction-based autoencoder — trained only on normal transactions — dramatically outperforms classical anomaly detectors by learning the full joint structure of normal behavior and flagging deviations from it.

**Custom metrics introduced:**
- **ADI (Anomaly Discriminability Index):** `ADI(f) = KS(f) × |mean_fraud(f) − mean_normal(f)|` — combines distributional shape separation with mean shift
- **TFCS (Temporal Fraud Concentration Score):** KL divergence between fraud and normal temporal distributions — confirmed significant at p < 0.001 via 200-permutation test

---

## 🔁 How to Reproduce

### Recommended: Google Colab

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `main_notebook.ipynb`
3. Run all cells sequentially — dataset downloads automatically
4. Runtime: ~20–40 minutes depending on Colab GPU availability

### Local (Jupyter)

```bash
# Clone the repository
git clone https://github.com/JessicaSinghSyal/DMA_Assignment_337001834_JessicaSinghSyal.git
cd DMA_Assignment_337001834_JessicaSinghSyal

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook main_notebook.ipynb
```

**Python version:** 3.10+ (developed on Colab — Python 3.10.12)

---

## 📦 Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `python` | 3.10+ | Runtime |
| `numpy` | 1.26+ | Numerical computation |
| `pandas` | 2.2+ | Data handling |
| `scikit-learn` | 1.4+ | Isolation Forest, LOF, DBSCAN, K-Means, metrics |
| `torch` | 2.2+ | MLP Autoencoder |
| `matplotlib` | 3.8+ | Visualizations |
| `seaborn` | 0.13+ | Statistical plots |
| `scipy` | 1.12+ | KS test, KL divergence |

Full dependency list: [`requirements.txt`](requirements.txt)

---

## 🗂 Repository Structure

```
DMA_Assignment_337001834_JessicaSinghSyal/
│
├── main_notebook.ipynb          # 👉 THE MAIN DELIVERABLE — start here
├── README.md                    # This file
├── requirements.txt             # Full dependency list
│
├── checkpoints/
│   ├── checkpoint_1.ipynb       # Dataset selection & initial EDA
│   └── checkpoint_2.ipynb       # Research question formation
│
├── assets/                      # Auto-generated by main_notebook
│   ├── eda1_imbalance.png
│   ├── eda2_adi.png
│   ├── eda3_amount_paradox.png
│   ├── eda4_tfcs.png
│   ├── eda5_pca.png
│   ├── eda6_proxy.png
│   ├── rq1_results.png
│   ├── rq1_best_if.png
│   ├── rq2_clustering.png
│   ├── rq3_autoencoder.png
│   └── final_comparison.png
│
└── data/                        # Downloaded automatically by notebook
    └── creditcard.csv           # (not tracked — auto-downloaded)
```

---

## 📚 Citations

**Dataset:**
> Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification*. IEEE Symposium on Computational Intelligence and Data Mining.

**Methods:**
> Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation Forest*. ICDM 2008.  
> Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). *LOF: Identifying Density-Based Local Outliers*. SIGMOD 2000.

**License:** ODbL v1.0 (dataset) | This project is for educational purposes as part of a graduate-level Data Mining course at Texas A&M University.

---

*Jessica Singh Syal | jessicasinghsyalinternship@gmail.com | Texas A&M University*



