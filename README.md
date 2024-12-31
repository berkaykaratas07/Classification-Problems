# Classification Algorithms on Diverse Datasets

 This repository contains Jupyter notebooks that explore various classification algorithms applied to three distinct datasets. Each notebook is located in the notebooks/ directory and focuses on a specific classification problem, leveraging machine learning techniques to analyze, preprocess, and evaluate the data.
 
## Datasets

1. **Banking Dataset - Marketing Targets**  
   - **Description**: Telephonic marketing campaign data from a Portuguese bank. The goal is to predict whether a client will subscribe to a term deposit.  
   - **Source**: [Banking Dataset - Marketing Targets](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets) 

2. **Molecular Biology (Promoter Gene Sequences)**  
   - **Description**: Classify DNA sequences as promoter or non-promoter regions for genetic analysis.  
   - **Source**: [Molecular Biology](https://archive.ics.uci.edu/dataset/67/molecular+biology+promoter+gene+sequences)

3. **Non-Linear Classification Dataset (Moons)**  
   - **Description**: A synthetic dataset designed to demonstrate the challenges of non-linear separability in classification problems.  
   - **Source**: [Moons](https://www.kaggle.com/datasets/emadmakhlouf/linearly-inseperable-dataset)

  ## Notebooks Overview

1. **Banking Dataset Notebook**  
   - **Path**: `notebooks/Bank_Deposit_Classification.ipynb`  
   - **Objective**: Predict customer subscription to a term deposit.  
   - **Key Steps**:  
     - Data preprocessing with one-hot encoding.  
     - Feature scaling using StandardScaler.  
   - **Algorithms**:  
     - K-Nearest Neighbors (KNN)  
     - Logistic Regression  
     - Decision Tree  
     - Random Forest  
   - **Evaluation**: Cross-validation for performance evaluation.

2. **DNA Sequence Classification Notebook**  
   - **Path**: `notebooks/DNA_Sequence_Classification.ipynb`  
   - **Objective**: Classify DNA sequences into promoter or non-promoter regions.  
   - **Key Steps**:  
     - Preprocessing: Sequence cleaning and invalid character replacement.  
     - Feature Engineering:  
       - k-mer frequencies  
       - GC content  
       - Dinucleotide frequencies  
     - Feature scaling and encoding.  
   - **Algorithms**:  
     - K-Nearest Neighbors (KNN)  
     - Support Vector Machine (SVM)  
     - Random Forest  
     - XGBoost  
     - Logistic Regression  
   - **Evaluation**: 5-fold cross-validation for evaluation.

3. **Moon Dataset Notebook**  
   - **Path**: `notebooks/Moon_Dataset_Classification .ipynbb`  
   - **Objective**: Demonstrate non-linear classification using synthetic moon data.  
   - **Key Steps**:  
     - Data preprocessing: Scaling and splitting.  
   - **Algorithms**:  
     - K-Nearest Neighbors (KNN) with hyperparameter tuning (GridSearchCV).  
     - Support Vector Machine (SVM) with RBF and linear kernels.  
   - **Evaluation**: Test set evaluation and accuracy reporting.

  ## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/berkaykaratas07/Classification-Problems.git
   ```

2. **Install dependencies:** Install the required Python packages by running:
   ```bash
   pip install -r requirements.txt
   ```

## Contributing

If you would like to contribute to this project, please feel free to submit pull requests or report any issues.

## License

[License Information - For example, MIT License, Apache 2.0, etc.]

## Contact

You can reach me at: berkaykaratas054@gmail.com or [GitHub](https://github.com/berkaykaratas07)
