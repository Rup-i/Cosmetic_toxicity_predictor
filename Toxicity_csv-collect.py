import pandas as pd

# 1. Load the benchmark data
url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
raw_df = pd.read_csv(url)

# 2. Select only the structural string and your target class
# We use CT_TOX (1 if toxic/failed clinical trials, 0 if safe)
cleaned_df = raw_df[['smiles', 'CT_TOX']].copy()

# 3. Rename columns to match your project specifications
cleaned_df.columns = ['SMILES', 'Class']

# 4. Drop any rows with missing chemical structures or missing labels
cleaned_df = cleaned_df.dropna(subset=['SMILES', 'Class'])

# 5. Quick sanity check on your class balance
print(f"Dataset Shape: {cleaned_df.shape}")
print("\nClass Distribution:")
print(cleaned_df['Class'].value_counts())

# Save it locally as your clean starting point
cleaned_df.to_csv("clean_cosmetic_toxicology.csv", index=False)

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# 1. Define the RDKit featurization function
def smiles_to_fp(smiles, radius=2, n_bits=2048):
    """Converts a SMILES string into a binary Morgan Fingerprint array."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((1,))
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
    except:
        pass
    return None

# 2. Convert SMILES to features
print("Generating molecular fingerprints... this might take a few seconds.")
cleaned_df['fingerprint'] = cleaned_df['SMILES'].apply(smiles_to_fp)

# Drop any rows where RDKit failed to parse the chemical structure
final_df = cleaned_df.dropna(subset=['fingerprint']).reset_index(drop=True)

# 3. Create X (Features) and y (Labels)
X = np.stack(final_df['fingerprint'].values)
y = final_df['Class'].values

# 4. Stratified Train/Test Split (ensures both sets get a fair share of the 112 toxic molecules)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}\n")

# 5. Train Random Forest with class balancing
# 'class_weight="balanced"' forces the model to penalize missing a toxic molecule heavily
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# 6. Predict and Evaluate
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]

print("=== MODEL PERFORMANCE REPORT ===")
print(classification_report(y_test, y_pred, target_names=['Safe (0)', 'Toxic (1)']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Calculate the exact imbalance ratio to feed into XGBoost
# 1372 (safe) / 112 (toxic) ≈ 12.25
scale_pos_weight_value = 1372 / 112

print(f"Calculated scale_pos_weight: {scale_pos_weight_value:.2f}")

# 2. Initialize XGBoost with the class weight correction
xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight_value, # This is the magic parameter for imbalance
    random_state=42,
    eval_metric='logloss'
)

# 3. Fit the model
xgb_model.fit(X_train, y_train)

# 4. Predict
y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# 5. Print the new results
print("\n=== XGBOOST MODEL PERFORMANCE REPORT ===")
print(classification_report(y_test, y_pred_xgb, target_names=['Safe (0)', 'Toxic (1)']))
print(f"New ROC-AUC Score: {roc_auc_score(y_test, y_proba_xgb):.4f}")

# 1. Instead of using .predict(), we use the raw probabilities we already calculated
# Let's set a conservative safety threshold of 0.25
custom_threshold = 0.25
y_pred_custom = (y_proba_xgb >= custom_threshold).astype(int)

# 2. Print the threshold-adjusted results
print(f"=== XGBOOST WITH CUSTOM THRESHOLD ({custom_threshold}) ===")
print(classification_report(y_test, y_pred_custom, target_names=['Safe (0)', 'Toxic (1)']))
