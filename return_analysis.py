from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -------------------------
# Load Dataset
# -------------------------

base_path = Path(__file__).resolve().parent.parent
file_path = base_path / "Dataset" / "Ecommerce_Return_Analysis.csv"

df = pd.read_csv(file_path)

# -------------------------
# Create Return Flag
# -------------------------

df["Return_Flag"] = df["returned"].map({
    "Yes": 1,
    "No": 0
})

# Show Return Flag
print("Return Flag Created:")
print(df[["returned", "Return_Flag"]].head())

# -------------------------
# Calculate Overall Return Rate
# -------------------------

return_rate = df["Return_Flag"].mean() * 100

print("\nOverall Return Rate:")
print(round(return_rate, 2), "%")

# -------------------------
# Return Rate by Category
# -------------------------

category_return = (
    df.groupby("category")["Return_Flag"]
      .mean() * 100
)

print("\nReturn Rate by Category:")
print(category_return.round(2))

# Save category analysis
category_return.round(2).to_csv(
    base_path / "Outputs" / "category_return_rate.csv",
    header=["Return Rate (%)"]
)

print("\ncategory_return_rate.csv saved successfully!")

# -------------------------
# Return Rate by Region
# -------------------------

region_return = (
    df.groupby("region")["Return_Flag"]
      .mean() * 100
)

print("\nReturn Rate by Region:")
print(region_return.round(2))

# Save Region Analysis
region_return.round(2).to_csv(
    base_path / "Outputs" / "region_return_rate.csv",
    header=["Return Rate (%)"]
)

print("\nregion_return_rate.csv saved successfully!")

# -------------------------
# Return Rate by Payment Method
# -------------------------

payment_return = (
    df.groupby("payment_method")["Return_Flag"]
      .mean() * 100
)

print("\nReturn Rate by Payment Method:")
print(payment_return.round(2))

# Save Payment Analysis
payment_return.round(2).to_csv(
    base_path / "Outputs" / "payment_return_rate.csv",
    header=["Return Rate (%)"]
)

print("\npayment_return_rate.csv saved successfully!")

# -------------------------
# Return Reasons Analysis
# -------------------------

return_reason_count = df[df["returned"] == "Yes"]["return_reason"].value_counts()

print("\nMost Common Return Reasons:")
print(return_reason_count)

# Save Return Reasons
return_reason_count.to_csv(
    base_path / "Outputs" / "return_reasons.csv",
    header=["Count"]
)

print("\nreturn_reasons.csv saved successfully!")

# -------------------------
# Machine Learning - Logistic Regression
# -------------------------

# Select required columns
ml_df = df[[
    "category",
    "payment_method",
    "region",
    "customer_gender",
    "customer_age",
    "price",
    "discount",
    "quantity",
    "Return_Flag"
]].copy()
# Convert text columns into numbers
encoder = LabelEncoder()

ml_df["category"] = encoder.fit_transform(ml_df["category"])
ml_df["payment_method"] = encoder.fit_transform(ml_df["payment_method"])
ml_df["region"] = encoder.fit_transform(ml_df["region"])
ml_df["customer_gender"] = encoder.fit_transform(ml_df["customer_gender"])

# Features (X)
X = ml_df.drop("Return_Flag", axis=1)

# Target (y)
y = ml_df["Return_Flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Predict return probability
ml_df["Return_Probability"] = model.predict_proba(X)[:, 1]

# High-risk products (Probability >= 70%)
high_risk = df.copy()

high_risk["Return_Probability"] = ml_df["Return_Probability"]

high_risk = high_risk[
    high_risk["Return_Probability"] >= 0.20
]

high_risk.to_csv(
    base_path / "Outputs" / "high_risk_products.csv",
    index=False
)

print("\nhigh_risk_products.csv saved successfully!")
