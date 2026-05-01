import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from imblearn.over_sampling import SMOTE

# ==============================
# STEP 1: LOAD DATA (NO DOWNLOAD NEEDED)
# ==============================
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

print("Dataset Loaded ✅")
print(df.shape)

# ==============================
# STEP 2: BASIC INFO
# ==============================
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df['Class'].value_counts())

# ==============================
# STEP 3: VISUALIZATION
# ==============================

# Fraud vs Normal
plt.figure()
df['Class'].value_counts().plot(kind='bar')
plt.title("Fraud vs Normal Transactions")
plt.savefig("images/fraud_vs_normal.png")
plt.close()

# Amount Distribution
plt.figure()
sns.histplot(df['Amount'], bins=50)
plt.title("Transaction Amount Distribution")
plt.savefig("images/amount_distribution.png")
plt.close()

# Fraud vs Amount
plt.figure()
sns.boxplot(x='Class', y='Amount', data=df)
plt.title("Fraud vs Amount")
plt.savefig("images/fraud_vs_amount.png")
plt.close()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Feature Correlation")
plt.savefig("images/correlation_heatmap.png")
plt.close()

# ==============================
# STEP 4: PREPARE DATA
# ==============================
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nFraud cases in training:", sum(y_train))
print("Fraud cases in test:", sum(y_test))

# ==============================
# STEP 5: HANDLE IMBALANCE (SMOTE)
# ==============================
smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_resampled).value_counts())

# ==============================
# STEP 6: TRAIN MODEL
# ==============================
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)

# ==============================
# STEP 7: EVALUATE MODEL
# ==============================
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==============================
# STEP 8: ROC-AUC SCORE
# ==============================
y_probs = model.predict_proba(X_test)[:, 1]
print("\nROC-AUC Score:", roc_auc_score(y_test, y_probs))
import joblib

joblib.dump(model, "model.pkl")
print("Model saved ✅")