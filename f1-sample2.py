# importing libraries
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Paths to all original datasets
base_path = './data'




file_paths = {
    "status": "status.csv",
    "seasons": "seasons.csv",
    "results": "results.csv",
    "races": "races.csv",
    "qualifying": "qualifying.csv",
    "pit_stops": "pit_stops.csv",
    "drivers": "drivers.csv",
    "driver_standings": "driver_standings.csv",
    "constructors": "constructors.csv",
    "lap_timing": "lap_timing.csv",
    "constructor_standings": "constructor_standings.csv",
    "circuits": "circuits.csv",
    "constructor_results": "constructor_results.csv"
}

# Clean and save all datasets
def clean_dataframe(df):
    return df.dropna().drop_duplicates()

cleaned_dfs = {
    name: clean_dataframe(pd.read_csv(os.path.join(base_path, filename)))
    for name, filename in file_paths.items()
}

# Save cleaned CSVs
for name, df in cleaned_dfs.items():
    df.to_csv(os.path.join(base_path, f"{name}_cleaned.csv"), index=False)

print("Cleaned files:")
for name in cleaned_dfs:
    print(f"- {name}_cleaned.csv")

# Helper to merge safely
def safe_merge(left_df, right_df, on_cols, suffix=''):
    common_cols = [col for col in right_df.columns if col in left_df.columns and col not in on_cols]
    right_df = right_df.drop(columns=common_cols)
    return left_df.merge(right_df, how='left', on=on_cols, suffixes=('', suffix))

# Load cleaned datasets
status = cleaned_dfs["status"]
seasons = cleaned_dfs["seasons"]
results = cleaned_dfs["results"]
races = cleaned_dfs["races"]
qualifying = cleaned_dfs["qualifying"]
pit_stops = cleaned_dfs["pit_stops"]
drivers = cleaned_dfs["drivers"]
driver_standings = cleaned_dfs["driver_standings"]
constructors = cleaned_dfs["constructors"]
lap_timing = cleaned_dfs["lap_timing"]
constructor_standings = cleaned_dfs["constructor_standings"]
circuits = cleaned_dfs["circuits"]
constructor_results = cleaned_dfs["constructor_results"]

# Merge step-by-step, resolving column conflicts when necessary

races = safe_merge(races, circuits, ['circuitId'], '_circuits')
results = safe_merge(results, races, ['raceId'], '_races')
results = safe_merge(results, drivers, ['driverId'], '_drivers')
results = safe_merge(results, constructors, ['constructorId'], '_constructors')
results = safe_merge(results, driver_standings, ['raceId', 'driverId'], '_driverStandings')
results = safe_merge(results, constructor_standings, ['raceId', 'constructorId'], '_constructorStandings')
results = safe_merge(results, qualifying, ['raceId', 'driverId'], '_qualifying')
results = safe_merge(results, lap_timing, ['raceId', 'driverId'], '_lapTimes')
results = safe_merge(results, pit_stops, ['raceId', 'driverId'], '_pitStops')
results = safe_merge(results, constructor_results, ['raceId', 'constructorId'], '_constructorResults')

# Final output
print("\nMerged dataset shape:", results.shape)
print("Columns in final dataset:", results.columns.tolist())

#####################################
#####################################

#Graph visualisation

#Driver Performance(Points vs Wins)
# Top drivers by total points
top_drivers = results.groupby(['forename', 'surname'])['points'].sum().sort_values(ascending=False).head(10).reset_index()
top_drivers['name'] = top_drivers['forename'] + ' ' + top_drivers['surname']

plt.figure(figsize=(10, 6))
sns.barplot(data=top_drivers, x='points', y='name', hue='name', palette='coolwarm', legend=False)
plt.title('Top 10 Drivers by Points')
plt.xlabel('Total Points')
plt.ylabel('Driver')
plt.tight_layout()
plt.show()

#Constructors Performance (Points Over Years)
# Total constructor points per year
constructor_points = results.groupby(['year', 'constructorRef'])['points'].sum().reset_index()
top5 = constructor_points.groupby('constructorRef')['points'].sum().sort_values(ascending=False).head(5).index
top5_data = constructor_points[constructor_points['constructorRef'].isin(top5)]
plt.figure(figsize=(10, 6))
for constructor in top5:
    data = top5_data[top5_data['constructorRef'] == constructor]
    plt.plot(data['year'], data['points'], marker='o', label=constructor)
plt.title("Top 5 Constructors' Points Over the Years")
plt.xlabel("Year")
plt.ylabel("Points")
plt.legend(title='Constructor')
plt.grid(True)
plt.tight_layout()
plt.show()


# Fastest Laps Distribution
results['fastestLapSpeed'] = pd.to_numeric(results['fastestLapSpeed'], errors='coerce')
lap_speeds = results['fastestLapSpeed'].dropna()
plt.figure(figsize=(10, 5))
sns.histplot(lap_speeds, bins=30, kde=True, color='blue')
plt.title('Distribution of Fastest Lap Speeds')
plt.xlabel('Fastest Lap Speed (km/h)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

#Save Plots (Optional)
plt.savefig("my_plot.png", dpi=300)


##################################################

#data preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 1: Choose relevant columns for prediction
# You can adjust this list as needed
columns_to_use = [
    'grid', 'positionOrder', 'points', 'laps', 'milliseconds', 'fastestLapSpeed',
    'year', 'round', 'constructorId', 'driverId'
]
df = results[columns_to_use].copy()

# Step 2: Replace '\\N' with NaN and drop missing values
df = df.replace('\\N', np.nan).dropna()

# Step 3: Convert all columns to numeric (skip if already numeric)
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except ValueError:
        pass

# Step 4: Label Encode constructorId and driverId
le_constructor = LabelEncoder()
le_driver = LabelEncoder()
df['constructorId'] = le_constructor.fit_transform(df['constructorId'])
df['driverId'] = le_driver.fit_transform(df['driverId'])

# Step 5: Feature Scaling (ensure all columns are numeric)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)

# Final processed DataFrame
processed_df = pd.DataFrame(scaled_features, columns=columns_to_use)
print("\n Data Preprocessing Completed!")
print(processed_df.head())


# ####################################################################


######################################################################
# Full Script (with classification integrated):

# --- REGRESSION SECTION: Predicting lap_time_seconds ---

# Drop missing lap times
regression_df = df.dropna(subset=['milliseconds'])

# Feature columns
reg_features = ['grid', 'constructorId', 'driverId', 'year', 'round', 'fastestLapSpeed']
X_reg = regression_df[reg_features]
X_reg = regression_df[reg_features]
y_reg = regression_df['milliseconds']

# Train/test split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train_reg, y_train_reg)

# Predict & evaluate
y_pred_reg = regressor.predict(X_test_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
print(f"Regression RMSE: {rmse:.2f} seconds")

# --- CLASSIFICATION SECTION: Predicting position class ---

# Convert position into classification class
def get_position_class(position):
    try:
        position = int(position)
        if position <= 5:
            return 0  # Top 5
        elif position <= 15:
            return 1  # Midfield
        else:
            return 2  # Backmarkers
    except:
        return np.nan

df['position_class'] = df['positionOrder'].apply(get_position_class)

# Drop missing lap time or class
clf_df = df.dropna(subset=['position_class', 'grid', 'constructorId', 'driverId', 'year', 'round', 'fastestLapSpeed'])

# Classification features
clf_features = ['grid', 'constructorId', 'driverId', 'year', 'round', 'fastestLapSpeed']
X_clf = clf_df[clf_features]
y_clf = clf_df['position_class']

# Train/test split
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_clf, y_train_clf)

# Predict & evaluate
y_pred_clf = classifier.predict(X_test_clf)

# Confusion Matrix
cm = confusion_matrix(y_test_clf, y_pred_clf)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Top 5', 'Midfield', 'Backmarkers'],
            yticklabels=['Top 5', 'Midfield', 'Backmarkers'])
plt.title("Confusion Matrix - Position Class Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test_clf, y_pred_clf, target_names=['Top 5', 'Midfield', 'Backmarkers']))

print("\n--- Regression Evaluation ---")
print(f"RMSE: {rmse:.2f} seconds")
mae = mean_absolute_error(y_test_reg, y_pred_reg)
print(f"MAE : {mae:.2f} seconds")



##############################################
#Data Processing Enhancements
# First, let's prepare qualifying-specific data
qualifying_data = results[['raceId', 'driverId', 'constructorId', 'q1', 'q2', 'q3']].copy()

# Convert timedelta strings to seconds
def time_to_seconds(time_str):
    if pd.isna(time_str) or time_str == '\\N':
        return np.nan
    try:
        # Handle formats like "1:23.456" or "1:23" or "23.456"
        parts = time_str.replace(',', '.').split(':')
        if len(parts) == 2:  # Format "1:23.456"
            return float(parts[0]) * 60 + float(parts[1])
        return float(time_str)  # Format "23.456"
    except:
        return np.nan

for col in ['q1', 'q2', 'q3']:
    qualifying_data[col] = qualifying_data[col].apply(time_to_seconds)

# Handle missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median', missing_values=np.nan)
qualifying_data[['q1', 'q2', 'q3']] = imputer.fit_transform(qualifying_data[['q1', 'q2', 'q3']])

# Merge with driver and constructor info
qualifying_data = qualifying_data.merge(drivers[['driverId', 'forename', 'surname']], on='driverId')
qualifying_data = qualifying_data.merge(constructors[['constructorId', 'constructorRef']], on='constructorId')
qualifying_data['driver_name'] = qualifying_data['forename'] + ' ' + qualifying_data['surname']


##################################################
#Model Development

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Features and target
X = qualifying_data[['q1', 'q2']].values
y = qualifying_data['q3'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline model
baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)

# Evaluate
y_pred = baseline_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Baseline Model Performance:")
print(f"MAE: {mae:.3f} seconds")
print(f"R²: {r2:.3f}")


######################################################
#Performance Factors Implementation

# Calculate team performance coefficients
team_coeff = qualifying_data.groupby('constructorRef')['q3'].mean() / qualifying_data['q3'].mean()
team_coeff = team_coeff.to_dict()

# Calculate driver performance adjustments
driver_adjustments = qualifying_data.groupby('driver_name')['q3'].mean() - qualifying_data.groupby('constructorRef')['q3'].mean()
driver_adjustments = driver_adjustments.groupby(level=0).mean().to_dict()  # Average across teams

# Base lap time (adjust based on your circuit)
BASE_LAP_TIME = 89.5  # seconds

# Function to apply performance factors
def apply_performance_factors(driver_name, constructor_name, predicted_time):
    # Get team coefficient and driver adjustment
    team_factor = team_coeff.get(constructor_name, 1.0)
    driver_adjustment = driver_adjustments.get(driver_name, 0.0)
    
    # Apply adjustments
    adjusted_time = (predicted_time * team_factor) + driver_adjustment
    
    # Add small random variation (0.1% of lap time)
    random_variation = np.random.normal(0, adjusted_time * 0.001)
    final_time = adjusted_time + random_variation
    
    return final_time

############################################
#Prediction System

def predict_qualifying_order(driver_team_pairs):
    """
    Predict qualifying order for given driver-team pairs
    
    Args:
        driver_team_pairs: List of tuples (driver_name, constructor_name)
    
    Returns:
        DataFrame with predicted times and positions
    """
    predictions = []
    
    # Calculate overall median times as fallback
    q1_median = qualifying_data['q1'].median()
    q2_median = qualifying_data['q2'].median()
    
    for driver, team in driver_team_pairs:
        # Initialize with median values
        q1_avg, q2_avg = q1_median, q2_median
        
        # Try to get team averages first
        team_mask = qualifying_data['constructorRef'] == team
        if team_mask.sum() > 0:
            q1_avg = qualifying_data[team_mask]['q1'].median()
            q2_avg = qualifying_data[team_mask]['q2'].median()
            
            # Then try to get driver-specific averages
            driver_mask = (qualifying_data['driver_name'] == driver) & team_mask
            if driver_mask.sum() > 0:
                q1_avg = qualifying_data[driver_mask]['q1'].median()
                q2_avg = qualifying_data[driver_mask]['q2'].median()
        
        # Get baseline prediction
        try:
            base_pred = baseline_model.predict([[q1_avg, q2_avg]])[0]
            final_pred = apply_performance_factors(driver, team, base_pred)
        except:
            final_pred = np.nan
        
        predictions.append({
            'Driver': driver,
            'Team': team,
            'Predicted Time': final_pred
        })
    
    # Create DataFrame and sort by predicted time
    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.sort_values('Predicted Time')
    pred_df['Position'] = range(1, len(pred_df)+1)
    
    return pred_df

# Example usage with 2023 teams/drivers
example_pairs = [
    ('Max Verstappen', 'red_bull'),
    ('Lewis Hamilton', 'mercedes'),
    ('Charles Leclerc', 'ferrari'),
    ('Lando Norris', 'mclaren'),
    ('Carlos Sainz', 'ferrari'),
    ('Sergio Perez', 'red_bull'),
    ('George Russell', 'mercedes'),
    ('Oscar Piastri', 'mclaren'),
    ('Fernando Alonso', 'aston_martin'),
    ('Lance Stroll', 'aston_martin')
]

predicted_qualifying = predict_qualifying_order(example_pairs)
print("\nPredicted Qualifying Order:")
print(predicted_qualifying[['Position', 'Driver', 'Team', 'Predicted Time']].to_string(index=False))

############################################
#Validation and Visualization

# Validation on historical data
import matplotlib.pyplot as plt
import seaborn as sns

def validate_model():
    predictions = []
    actuals = []
    
    for _, row in qualifying_data.iterrows():
        try:
            pred = apply_performance_factors(
                row['driver_name'],
                row['constructorRef'],
                baseline_model.predict([[row['q1'], row['q2']]])[0]
            )
            predictions.append(pred)
            actuals.append(row['q3'])
        except:
            continue  # Skip rows that cause errors
    
    # Only calculate metrics if we have predictions
    if predictions:
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        print(f"\n Validation Metrics (with performance factors):")
        print(f"MAE: {mae:.3f} seconds")
        print(f"R²: {r2:.3f}")
        
        # Visualization code remains the same
    else:
        print("\nWarning: No valid predictions could be made for validation")
        
    # Visualization
    plt.figure(figsize=(14, 6))

    # Scatter Plot: Actual vs Predicted
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=actuals, y=predictions, alpha=0.5, color='darkblue')
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')  # Diagonal line
    plt.xlabel("Actual Q3 Time (seconds)")
    plt.ylabel("Predicted Q3 Time (seconds)")
    plt.title("Actual vs Predicted Q3 Times")

    # Histogram of Errors
    plt.subplot(1, 2, 2)
    errors = np.array(predictions) - np.array(actuals)
    sns.histplot(errors, bins=30, kde=True, color='green')
    plt.title("Distribution of Prediction Errors")
    plt.xlabel("Error (Predicted - Actual) in seconds")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()