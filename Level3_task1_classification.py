# Codveda Internship - Level 3, Task 1: Predictive Modeling (Classification)
# Dataset: Iris Dataset
# Goal: Predict flower species using multiple classification models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


#- 1. LOAD DATASET-
df = pd.read_csv('1) iris.csv')
print("Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# -2. preprocess
#Encode species (text- numbers)
le = LabelEncoder()
df['species_encoded'] =le.fit_transform(df['species'])
print("\nClasses:", list(le.classes_))

X = df.drop(['species', 'species_encoded'], axis=1)
Y= df['species_encoded']

#-3. train/test/split

X_train, X_test, Y_train, Y_test= train_test_split(
X, Y,test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")

# ── 4. FEATURE SCALING ────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("Scaling done!")

# ── 5. TRAIN 3 MODELS ─────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Decision Tree'      : DecisionTreeClassifier(random_state=42),
    'Random Forest'      : RandomForestClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, Y_train)
    Y_pred = model.predict(X_test_scaled)
    results[name] = {
        'Accuracy' : accuracy_score(Y_test, Y_pred),
        'Precision': precision_score(Y_test, Y_pred, average='weighted'),
        'Recall'   : recall_score(Y_test, Y_pred, average='weighted'),
        'F1-Score' : f1_score(Y_test, Y_pred, average='weighted')
    }
    print(f"\n{name}:")
    for metric, value in results[name].items():
        print(f"  {metric}: {value:.4f}")

# ── 6. RESULTS COMPARISON TABLE ───────────────────────────
results_df = pd.DataFrame(results).T
print("\nModel Comparison:\n", results_df.round(4))

# ── 7. PLOT: Model Comparison Bar Chart ───────────────────
results_df.plot(kind='bar', figsize=(10, 6), colormap='Set2')
plt.title('Model Comparison - Classification Metrics')
plt.xlabel('Model')
plt.ylabel('Score')
plt.xticks(rotation=15)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()

# ── 8. CONFUSION MATRIX (Best Model - Random Forest) ──────
rf_model = models['Random Forest']
Y_pred_rf = rf_model.predict(X_test_scaled)
cm = confusion_matrix(Y_test, Y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

plt.figure(figsize=(7, 5))
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# ── 9. HYPERPARAMETER TUNING (Random Forest) 
print("\nRunning Grid Search... (this may take a moment)")
param_grid = {
    'n_estimators' : [50, 100, 200],
    'max_depth'    : [None, 5, 10],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='accuracy'
)
grid_search.fit(X_train_scaled, Y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
Y_pred_best = best_model.predict(X_test_scaled)
print(f"Tuned Model Test Accuracy: {accuracy_score(Y_test, Y_pred_best):.4f}")
                                              