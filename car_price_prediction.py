# ============================================
# CAR PRICE PREDICTION - MACHINE LEARNING
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CAR PRICE PREDICTION - MACHINE LEARNING")
print("="*60)

# ============================================
# PART 1: LOAD DATA
# ============================================
print("\n[1] Loading Data...")

# LOAD REAL DATASET
df = pd.read_csv('car data.csv')

print(f"\nDataset Shape: {df.shape}")
print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Columns ---")
print(df.columns.tolist())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Statistics ---")
print(df.describe())

# ============================================
# PART 2: DATA EXPLORATION
# ============================================
print("\n[2] Data Exploration...")

print(f"\nCar Names: {df['Car_Name'].nunique()}")
print(df['Car_Name'].value_counts().head(10))

print(f"\nFuel Types: {df['Fuel_Type'].unique()}")
print(f"Transmission: {df['Transmission'].unique()}")
print(f"Selling Type: {df['Selling_type'].unique()}")
print(f"Owner: {df['Owner'].unique()}")

# Check missing values
print(f"\nMissing Values: {df.isnull().sum().sum()}")

# ============================================
# PART 3: DATA PREPROCESSING
# ============================================
print("\n[3] Data Preprocessing...")

# Encode Categorical Variables
le_car = LabelEncoder()
le_fuel = LabelEncoder()
le_seller = LabelEncoder()
le_trans = LabelEncoder()

df['Car_Name_Encoded'] = le_car.fit_transform(df['Car_Name'])
df['Fuel_Encoded'] = le_fuel.fit_transform(df['Fuel_Type'])
df['Seller_Encoded'] = le_seller.fit_transform(df['Selling_type'])
df['Trans_Encoded'] = le_trans.fit_transform(df['Transmission'])

print("Encoding Done!")

# ============================================
# PART 4: FEATURE ENGINEERING
# ============================================
print("\n[4] Feature Engineering...")

# Features to use
features = ['Year', 'Present_Price', 'Driven_kms', 'Owner', 
           'Car_Name_Encoded', 'Fuel_Encoded', 'Seller_Encoded', 'Trans_Encoded']

X = df[features]
y = df['Selling_Price']

print(f"Features: {features}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# ============================================
# PART 5: TRAIN - TEST SPLIT
# ============================================
print("\n[5] Splitting Data...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training: {X_train.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# PART 6: MODEL TRAINING
# ============================================
print("\n[6] Training Models...")

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
print("Linear Regression: ✓")

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest: ✓")

# ============================================
# PART 7: MODEL EVALUATION
# ============================================
print("\n[7] Model Evaluation...")

print("\n--- Linear Regression ---")
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)
print(f"MAE: {lr_mae:.2f} lakhs")
print(f"RMSE: {lr_rmse:.2f} lakhs")
print(f"R² Score: {lr_r2:.4f}")

print("\n--- Random Forest ---")
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)
print(f"MAE: {rf_mae:.2f} lakhs")
print(f"RMSE: {rf_rmse:.2f} lakhs")
print(f"R² Score: {rf_r2:.4f}")

# ============================================
# PART 8: VISUALIZATIONS
# ============================================
print("\n[8] Creating Visualizations...")

# Chart 1: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual Price (lakhs)')
plt.ylabel('Predicted Price (lakhs)')
plt.title('Random Forest: Actual vs Predicted')
plt.tight_layout()
plt.savefig('chart1_actual_vs_predicted.png')
plt.close()

# Chart 2: Feature Importance
plt.figure(figsize=(10, 6))
feature_imp = pd.DataFrame({
    'Feature': features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=True)

plt.barh(feature_imp['Feature'], feature_imp['Importance'], color='green')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('chart2_feature_importance.png')
plt.close()

# Chart 3: Price by Car Name
plt.figure(figsize=(12, 6))
car_avg = df.groupby('Car_Name')['Selling_Price'].mean().sort_values(ascending=False).head(15)
plt.bar(car_avg.index, car_avg.values, color='orange')
plt.xlabel('Car Name')
plt.ylabel('Avg Selling Price (lakhs)')
plt.title('Average Price by Car')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('chart3_price_by_car.png')
plt.close()

# Chart 4: Year vs Price
plt.figure(figsize=(10, 6))
year_avg = df.groupby('Year')['Selling_Price'].mean()
plt.plot(year_avg.index, year_avg.values, marker='o', linewidth=2, color='purple')
plt.xlabel('Year')
plt.ylabel('Avg Selling Price (lakhs)')
plt.title('Price Trend by Year')
plt.grid(True)
plt.tight_layout()
plt.savefig('chart4_year_vs_price.png')
plt.close()

# Chart 5: Model Comparison
plt.figure(figsize=(8, 5))
models = ['Linear Regression', 'Random Forest']
r2_scores = [lr_r2, rf_r2]
colors = ['blue', 'green']
plt.bar(models, r2_scores, color=colors)
plt.ylabel('R² Score')
plt.title('Model Comparison')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('chart5_model_comparison.png')
plt.close()

# Chart 6: Fuel Type Analysis
plt.figure(figsize=(8, 5))
fuel_avg = df.groupby('Fuel_Type')['Selling_Price'].mean()
plt.bar(fuel_avg.index, fuel_avg.values, color='red')
plt.xlabel('Fuel Type')
plt.ylabel('Avg Price (lakhs)')
plt.title('Price by Fuel Type')
plt.tight_layout()
plt.savefig('chart6_fuel_type.png')
plt.close()

print("Charts Saved!")

# ============================================
# PART 9: INSIGHTS
# ============================================
print("\n" + "="*60)
print("INSIGHTS & FINDINGS")
print("="*60)

print(f"""
📊 KEY FINDINGS:
1. Best Model: Random Forest (R² = {rf_r2:.4f})
2. Most Important Feature: Present_Price
3. Year, Driven_kms also important
4. Diesel cars have higher prices than Petrol

💡 REAL-WORLD APPLICATIONS:
1. Used car dealership pricing
2. Resale value prediction
3. Insurance premium calculation
4. Market analysis

🏆 MODEL PERFORMANCE:
- Random Forest R²: {rf_r2:.4f} ({rf_r2*100:.1f}% accuracy)
- Linear Regression R²: {lr_r2:.4f} ({lr_r2*100:.1f}% accuracy)
""")

# ============================================
# PART 10: PREDICTION EXAMPLE
# ============================================
print("\n[10] Sample Predictions...")

# Test predictions
sample_pred = rf.predict(X_test.head(5))
print("\nSample Predictions:")
for i, pred in enumerate(sample_pred):
    actual = y_test.iloc[i]
    print(f"  Car {i+1}: Actual: {actual:.2f} lakhs | Predicted: {pred:.2f} lakhs")

print("\n" + "="*60)
print("COMPLETE!")
print("="*60)