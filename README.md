# Credit Card Fraud Detection

This project applies machine learning, specifically Logistic Regression, to identify fraudulent credit card transactions. The dataset used for training and evaluation is highly imbalanced, reflecting the rarity of fraudulent transactions in real-world data. The model is trained to differentiate between legitimate and fraudulent transactions based on anonymized transaction features.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Data Exploration and Preparation](#data-exploration-and-preparation)
  - [Class Imbalance](#class-imbalance)
  - [Data Sampling](#data-sampling)
- [Model Development and Training](#model-development-and-training)
- [Evaluation and Results](#evaluation-and-results)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

---

## Project Overview

The goal of this project is to build a classification model that detects fraudulent credit card transactions using logistic regression. By using a balanced subset of the original data, the model is optimized to handle the significant class imbalance between legitimate and fraudulent transactions. Fraud detection models are crucial for minimizing unauthorized transactions and protecting users.

## Technologies Used

- **Python** for data processing and model training
- **Pandas** and **NumPy** for data handling and manipulation
- **Scikit-Learn** for machine learning model implementation

## Dataset

The dataset is available on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and contains transaction data from European credit cardholders over two days in September 2013. The dataset includes 284,807 transactions, of which 492 (0.172%) are fraudulent.

### Dataset Features
- **Time**: Seconds elapsed between each transaction and the first transaction in the dataset.
- **Amount**: Transaction amount, useful for cost-sensitive learning.
- **Class**: Target variable indicating fraud (1) or legitimate transaction (0).
- **V1 to V28**: Principal components obtained through PCA (to anonymize the data).

This dataset is highly imbalanced, and measuring accuracy with the Area Under the Precision-Recall Curve (AUPRC) is recommended. For more background, refer to the [practical handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html) on machine learning for credit card fraud detection.

## Data Exploration and Preparation

### Class Imbalance

The dataset is highly imbalanced, with fraudulent transactions making up only 0.172% of all transactions. This imbalance requires careful handling to avoid model bias towards the majority class.

```python
credit_card_data['Class'].value_counts()
```

### Data Sampling

To address the imbalance, the dataset is undersampled by taking a subset of legitimate transactions equal in number to the fraudulent ones. This balancing step ensures that the model is exposed to an equal number of legitimate and fraudulent samples during training.

```python
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)
```

## Model Development and Training

### Data Splitting

The data is split into features (`X`) and target (`Y`) and then further divided into training and test sets using an 80-20 split, stratified by class to maintain balance in each set.

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
```

### Logistic Regression Model

A Logistic Regression model is trained on the balanced dataset to classify each transaction as fraudulent or legitimate.

```python
model = LogisticRegression()
model.fit(X_train, Y_train)
```

## Evaluation and Results

The model's performance is evaluated based on accuracy scores on both the training and test datasets. Due to class imbalance, other metrics like the Precision-Recall AUC are also useful for future evaluations.

```python
# Training data accuracy
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data:', training_data_accuracy)

# Test data accuracy
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data:', test_data_accuracy)
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   ```
2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn
   ```

## Usage

1. Run the script to load, train, and evaluate the model:
   ```bash
   python main.py
   ```

2. **Sample Prediction**: To perform individual predictions, provide sample transaction data in the code for testing fraud detection.

## References

- Dataset Source: [Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Citations**: 
  - Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. "Calibrating Probability with Undersampling for Unbalanced Classification." *IEEE Symposium on Computational Intelligence and Data Mining*, 2015.
  - Dal Pozzolo, Andrea et al. "Learned lessons in credit card fraud detection from a practitioner perspective." *Expert Systems with Applications*, 41(10), 2014, Pergamon.
  - Dal Pozzolo, Andrea et al. "Credit card fraud detection: a realistic modeling and a novel learning strategy." *IEEE Transactions on Neural Networks and Learning Systems*, 29(8), 2018, IEEE.
