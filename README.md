# 🎯 Rock vs Mine Prediction using Logistic Regression

> A binary classification project that predicts whether a sonar signal bounced off a **Rock** or a **Mine** (metal cylinder) using **Logistic Regression**. This project demonstrates supervised learning for sonar data classification.

---

## 📁 Project Overview

This project uses sonar signal data to classify underwater objects as either rocks or mines. The dataset contains 60 features representing sonar signal frequencies, with labels 'R' (Rock) and 'M' (Mine).

**Files:**
* **`Rock-vs-Mine-Prediction.ipynb`**: Complete implementation with data analysis, preprocessing, training, and evaluation
* **`Sonar data.csv`**: Dataset containing 208 samples with 60 frequency features and 1 target column

---

## 📊 Dataset Information

### Dataset Details
- **Total Samples**: 208
- **Features**: 60 numerical columns (sonar signal frequencies at different angles)
- **Target Column**: Column 60 with values 'R' (Rock) or 'M' (Mine)
- **Class Distribution**:
  - Mines (M): 111 samples
  - Rocks (R): 97 samples

### Feature Characteristics
- All features are continuous numerical values between 0 and 1
- Features represent energy within specific frequency bands
- Higher values indicate stronger signal reflections at that frequency

---

## 🔧 Implementation Pipeline

### 1. **Data Collection & Loading**
```python
sonar_data = pd.read_csv("Sonar data.csv", header=None)
```
**Note**: The dataset has no header row, so `header=None` is specified to use index values as column names.

### 2. **Data Preprocessing & Exploration**

#### Exploratory Data Analysis
- **Shape Analysis**: 208 rows × 61 columns (60 features + 1 target)
- **Statistical Summary**: Using `describe()` to understand feature distributions
- **Class Distribution**: Checking balance between Rock and Mine samples
- **Group Analysis**: Comparing mean feature values for each class
```python
# Check target distribution
sonar_data[60].value_counts()

# Compare feature means by class
sonar_data.groupby(60).mean()
```

### 3. **Data Splitting**

#### Feature-Target Separation
```python
X = sonar_data.drop(columns=60, axis=1)  # Features (60 columns)
Y = sonar_data[60]                        # Target (R or M)
```

#### Train-Test Split
```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, 
    test_size=0.1,      # 10% for testing, 90% for training
    stratify=Y,         # Maintains class distribution in both sets
    random_state=1      # For reproducibility
)
```

**Split Results**:
- Training Set: 187 samples (90%)
- Test Set: 21 samples (10%)

**Why Stratify?**: Ensures both train and test sets have proportional representation of Rocks and Mines, crucial for small datasets.

### 4. **Model Training**

Using scikit-learn's Logistic Regression:
```python
model = LogisticRegression()
model.fit(X_train, Y_train)
```

The model learns the decision boundary that best separates rocks from mines based on the 60 sonar frequency features.

### 5. **Model Evaluation**

#### Training Accuracy
```python
train_prediction = model.predict(X_train)
accuracy_score(train_prediction, Y_train)
# Result: 83.42%
```

#### Testing Accuracy
```python
test_prediction = model.predict(X_test)
accuracy_score(test_prediction, Y_test)
# Result: 76.19%
```

**Performance Analysis**:
- Good training accuracy (83.42%) shows the model learned patterns effectively
- Reasonable test accuracy (76.19%) indicates decent generalization
- Small gap between train and test suggests minimal overfitting

---

## 🚀 Making Predictions

### Single Prediction Example
```python
# Sample sonar reading (60 frequency values)
input_data = (0.0181, 0.0146, 0.0026, ..., 0.0089, 0.0085)

# Reshape for prediction
input_as_numpy = np.asarray(input_data)
input_reshaped = input_as_numpy.reshape(1, -1)

# Predict
prediction = model.predict(input_reshaped)

if prediction[0] == "R":
    print("IT is ROCK")
else:
    print("IT IS MINE")
```

**Why Reshape?**: The model expects 2D input (samples × features), so we reshape from (60,) to (1, 60).

---

## 🛠️ Setup and Run

### Prerequisites
```bash
pip install numpy pandas scikit-learn jupyter
```

### Running the Project

1. **Clone the repository**:
```bash
   git clone https://github.com/PranavbalajiGit/SONAR-Rock-vs-Mine-Prediction.git
   cd SONAR-Rock-vs-Mine-Prediction
```

2. **Ensure dataset is present**:
   - Place `Sonar data.csv` in the same directory as the notebook

3. **Launch Jupyter**:
```bash
   jupyter notebook Rock-vs-Mine-Prediction.ipynb
```

4. **Run all cells** to train and evaluate the model

---

## 📈 Key Concepts Demonstrated

### Binary Classification
Distinguishing between two classes (Rock vs Mine) based on multiple features.

### Supervised Learning
The model learns from labeled training data (known Rocks and Mines) to predict unlabeled test data.

### Stratified Splitting
Maintaining class distribution during train-test split to ensure representative evaluation, especially important for small or imbalanced datasets.

### Model Evaluation
Using accuracy metrics to assess model performance on both training and unseen test data.

---

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Loading and exploring datasets without headers
- ✅ Performing exploratory data analysis (EDA)
- ✅ Proper train-test splitting with stratification
- ✅ Training a Logistic Regression classifier
- ✅ Evaluating model performance
- ✅ Making predictions on new data
- ✅ Understanding classification metrics

---

## 🔍 Potential Improvements

- **Feature Engineering**: Create new features from frequency combinations
- **Cross-Validation**: Use k-fold cross-validation for more robust evaluation
- **Hyperparameter Tuning**: Optimize regularization parameters
- **Feature Selection**: Identify most important frequency bands
- **Other Models**: Compare with SVM, Random Forest, or Neural Networks
- **Confusion Matrix**: Analyze specific misclassification patterns

---

## 📝 Dataset Source

The Sonar dataset is a classic machine learning dataset used for binary classification tasks in signal processing applications.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/PranavbalajiGit/SONAR-Rock-vs-Mine-Prediction/issues).

---

## 👤 Author

**PRANAV BALAJI P MA**
- GitHub: [@PranavbalajiGit](https://github.com/PranavbalajiGit)

---

## 🌟 Acknowledgments

- Dataset: Sonar Mines vs Rocks dataset
- Built with: Python, NumPy, Pandas, scikit-learn
- Task: Binary Classification (Supervised Learning)

---

**⭐ Star this repo if you find it helpful for learning ML classification!**