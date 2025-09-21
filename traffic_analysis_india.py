# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.set(style="whitegrid")
print("Libraries imported successfully!")

# Step 2 & 3: Generate Indian Traffic Dataset

# Set random seed
np.random.seed(42)

# Hourly timestamps for one week
dates = pd.date_range(start='2025-09-01', end='2025-09-07 23:00', freq='H')

# Popular Indian city locations
locations = ['Connaught Place', 'Bandra', 'MG Road', 'Electronic City', 'Jayanagar']

# Generate random vehicle counts
data = {
    'Timestamp': np.repeat(dates, len(locations)),
    'Location': locations * len(dates),
    'Vehicle_Count': np.random.randint(50, 800, size=len(dates)*len(locations))
}

# Create DataFrame
df = pd.DataFrame(data)

# Show first 10 rows
df.head(10)

# Step 4: Data Cleaning & Preprocessing
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.day_name()
df['Vehicle_Count'].fillna(df['Vehicle_Count'].median(), inplace=True)
df.info()

# Step 5: Exploratory Data Analysis (EDA)
hourly_traffic = df.groupby('Hour')['Vehicle_Count'].mean()
plt.figure(figsize=(10,5))
plt.plot(hourly_traffic.index, hourly_traffic.values, marker='o', color='blue')
plt.title("Average Traffic by Hour (India Cities)")
plt.xlabel("Hour of Day")
plt.ylabel("Average Vehicle Count")
plt.xticks(range(0,24))
plt.show()

daily_traffic = df.groupby('DayOfWeek')['Vehicle_Count'].mean()
daily_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
plt.figure(figsize=(10,5))
sns.barplot(x=daily_traffic.index, y=daily_traffic.values, order=daily_order, palette="viridis")
plt.title("Average Traffic by Day of Week (India Cities)")
plt.ylabel("Average Vehicle Count")
plt.show()

# Step 6: Location-Based Analysis
location_traffic = df.groupby('Location')['Vehicle_Count'].mean().sort_values(ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x=location_traffic.index, y=location_traffic.values, palette="coolwarm")
plt.title("Average Traffic by Location (Indian Cities)")
plt.xticks(rotation=45)
plt.ylabel("Average Vehicle Count")
plt.show()

# Step 7: Heatmap of Traffic (Hour vs Day)
traffic_pivot = df.pivot_table(index='DayOfWeek', columns='Hour', values='Vehicle_Count', aggfunc='mean')
traffic_pivot = traffic_pivot.reindex(daily_order)
plt.figure(figsize=(15,6))
sns.heatmap(traffic_pivot, cmap="YlOrRd", linewidths=.5, annot=True, fmt=".0f")
plt.title("Traffic Heatmap (Hour vs Day) - Indian Cities")
plt.show()

# Step 8: Simple Traffic Prediction using Hourly Average
hourly_avg = df.groupby('Hour')['Vehicle_Count'].mean()
predicted_traffic = hourly_avg.values
plt.figure(figsize=(10,5))
plt.plot(hourly_avg.index, hourly_avg.values, marker='o', label='Actual')
plt.plot(hourly_avg.index, predicted_traffic, marker='x', linestyle='--', label='Predicted')
plt.title("Traffic Prediction for Next Day (Indian Cities)")
plt.xlabel("Hour")
plt.ylabel("Vehicle Count")
plt.legend()
plt.show()

# Step 9: Save Cleaned Data & Plots
df.to_csv("cleaned_indian_traffic_data.csv", index=False)
plt.figure(figsize=(15,6))
sns.heatmap(traffic_pivot, cmap="YlOrRd", linewidths=.5, annot=True, fmt=".0f")
plt.title("Traffic Heatmap (Hour vs Day) - Indian Cities")
plt.savefig("traffic_heatmap_india.png", dpi=300)
print("Cleaned data and heatmap saved successfully!")
