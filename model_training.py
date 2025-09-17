import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle

# Load features from Hour 2
df = pd.read_csv("features_output.csv")

# Select features and labels
X = df[["url_length", "num_dots", "num_hyphens", "has_ip", "suspicious_keywords"]]
y = df["label"]

# Split dataset (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train RandomForest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Benign", "Phishing"]))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Feature Importance
importances = clf.feature_importances_
plt.bar(X.columns, importances)
plt.title("Feature Importance (RandomForest)")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save trained model for Hour 4 API use
with open("rf_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("\nModel saved successfully as 'rf_model.pkl'")


#Note -> Negative means Benign and Positive means phish , as our goal is to predict fake website
'''Precision -> How many predicted phishing are truly phishing
Recall -> How many real phishing sites were caught
F1-score -> Balance between precision and recall
Support -> How many samples you had of each class in the test set '''


#Our confusion matrix shows the model detected 57 out of 60 phishing sites
#and didn’t falsely block any safe sites. Only 3 phishing sites slipped through,
#giving us 97% accuracy.”

'''[Benign    WrongPhish
    WrongBenign  Phish]'''
