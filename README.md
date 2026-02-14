# Data Mining Project: Credit Card Fraud Detection

**Author:** Jessica  
**Course:** Data Mining (Spring 2026)  
**Institution:** Texas A&M University

---

## Project Overview

This project explores three candidate datasets for data mining analysis, ultimately selecting **Credit Card Fraud Detection** for in-depth study. The project demonstrates:

- Comparative analysis of multiple datasets
- Exploratory Data Analysis (EDA) techniques
- Handling extreme class imbalance (577:1 ratio)
- Feature importance analysis on PCA-transformed features
- Preparation for anomaly detection algorithms

---

## Repository Structure
```
├── checkpoint1_dataset_selection_eda.ipynb    # Main Jupyter notebook
├── README.md                                   # This file
├── data/                                       # Downloaded datasets (not tracked)
│   ├── creditcard.csv
│   ├── ml-25m/
│   └── smsspam/
└── visualizations/                             # Generated plots
    ├── fraud_eda_visualizations.png
    ├── fraud_deep_dive_visualizations.png
    ├── movielens_eda_visualizations.png
    └── sms_spam_eda_visualizations.png
```

---

## Datasets Analyzed

### 1. Credit Card Fraud Detection ✅ **SELECTED**
- **Size:** 284,807 transactions, 67 MB
- **Source:** Kaggle / TensorFlow Datasets
- **Features:** 31 (Time, V1-V28 PCA components, Amount, Class)
- **Challenge:** Extreme class imbalance (577:1)
- **Use Case:** Anomaly detection, classification

### 2. MovieLens 25M
- **Size:** 25M ratings, 763 MB
- **Source:** GroupLens Research
- **Features:** User-movie ratings with timestamps
- **Challenge:** Extreme sparsity (99.74%)
- **Use Case:** Graph mining, collaborative filtering

### 3. SMS Spam Collection
- **Size:** 5,572 messages, 1 MB
- **Source:** UCI Machine Learning Repository
- **Features:** Text messages labeled spam/ham
- **Challenge:** Small dataset size, temporal drift
- **Use Case:** Text mining, classification

---

## Key Findings

### Credit Card Fraud Dataset

**Most Discriminative Features:**
1. **V3** - Mean difference: 7.05 between fraud/normal
2. **V14** - Mean difference: 6.98
3. **V17** - Mean difference: 6.68

**Counterintuitive Discovery:**
- Fraud median amount (€9.25) is **lower** than normal (€22.00)
- Fraudsters use many small transactions to avoid detection

**Temporal Patterns:**
- Fraud rates vary 30× across hours (0.04% to 1.17%)
- Hour 13 has highest fraud rate (1.17%)

**Class Imbalance:**
- 492 frauds (0.17%) vs. 284,315 normal (99.83%)
- Requires specialized handling (SMOTE, class weights, anomaly detection)

---

## Technologies Used

**Languages & Libraries:**
- Python 3.10+
- pandas 2.2.2
- NumPy
- Matplotlib
- Seaborn

**Development Environment:**
- Google Colab
- Jupyter Notebook

**Version Control:**
- Git
- GitHub

---

## Planned Techniques

### Course-Aligned Techniques
- Anomaly detection (fraud as rare events)
- Clustering (K-means, DBSCAN)
- Classification (with class imbalance handling)

### Beyond-Course Techniques
1. **Autoencoders** - Deep learning for anomaly detection via reconstruction error
2. **Isolation Forest** - Tree-based anomaly detection algorithm
3. **SMOTE** - Synthetic minority oversampling for imbalance handling
4. **One-Class SVM** - Semi-supervised anomaly detection

---

## How to Run

### Prerequisites
```bash
# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Notebook

**Option 1: Google Colab (Recommended)**
1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `checkpoint1_dataset_selection_eda.ipynb`
3. Run all cells sequentially
4. Datasets will download automatically

**Option 2: Local Jupyter**
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Launch Jupyter
jupyter notebook checkpoint1_dataset_selection_eda.ipynb
```

---

## Results & Visualizations

### Comparison Analysis
- Comprehensive comparison table across 3 datasets
- Evaluation on: data quality, algorithmic feasibility, bias, ethics

### EDA Highlights
- Class distribution showing 577:1 imbalance
- Feature importance ranking (V3, V14, V17 top discriminators)
- Temporal fraud rate patterns
- Amount distribution analysis

### Visual Gallery
See `/visualizations/` folder for:
- Fraud vs. normal distribution comparisons
- Feature importance bar charts
- Temporal pattern analysis
- Correlation heatmaps

---

## Future Work (Checkpoint 2 & 3)

**Next Steps:**
1. Implement baseline models (Logistic Regression, Random Forest)
2. Apply Isolation Forest for anomaly detection
3. Train autoencoders for reconstruction-based detection
4. Evaluate with SMOTE oversampling
5. Compare precision-recall across all methods
6. Business-focused recommendations (cost-benefit analysis)

---

## Citations

**Datasets:**
- Dal Pozzolo, A., et al. (2015). "Calibrating Probability with Undersampling for Unbalanced Classification." *IEEE Symposium on Computational Intelligence*.
- Harper, F.M., Konstan, J.A. (2015). "The MovieLens Datasets: History and Context." *ACM TiiS*.
- Almeida, T.A., Hidalgo, J.M.G. (2011). "Contributions to the Study of SMS Spam Filtering." *ACM Document Engineering*.

---

## License

This project is for educational purposes as part of a graduate-level Data Mining course.

Datasets used are publicly available and cited appropriately:
- Credit Card Fraud: Open Database License (ODbL)
- MovieLens: Available for research and education
- SMS Spam: Public domain (UCI repository)

---

## Contact

**Jessica Singh Syal**  
UIN: 337001834
Graduate Student, Computer Science  
Texas A&M University  
Email: jessicasinghsyalinternship@gmail.com

---

## Acknowledgments

- **Anthropic Claude** for code generation assistance and documentation
- **GroupLens Research** for MovieLens dataset
- **UCI Machine Learning Repository** for SMS Spam dataset
- **Kaggle Community** for fraud detection dataset and kernels

---

*Last Updated: February 13, 2026*
