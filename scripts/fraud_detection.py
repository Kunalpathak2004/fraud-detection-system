import random
import faker
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

num_of_data = 50000
fraud_ratio = 0.01
large_data = []
faker = Faker() ## creating an instance for not getting error

def random_date(start : datetime, end : datetime) -> datetime: ##creating a function for using in timestamp
    delta = end - start
    int_delta = delta.days * 24 * 60 * 60
    random_second = random.randint(0, int_delta)
    return start + timedelta(seconds=random_second)

# start and end date created for timestamp
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31)
merchants = ["Amazon", "Walmart", "Target", "Uber", "Starbucks", "Apple", "Best Buy", "Netflix", "Costco", "Nike"]
locations = [ "New York", "Los Angeles", "Chicago", "Houston", "Phoenix","San Francisco", "Seattle", "Atlanta", "Boston", "Miami"]
devices = ["Mobile", "Desktop", "Tablet", "POS Terminal", "Smartwatch"]


# list of all the sequences 

for i in range(num_of_data):
    is_fraud = 1 if random.random() < fraud_ratio else 0
    amount = round(np.random.exponential(scale=100),2)
    if is_fraud:
        amount *= random.randint(5,20)
    row = {
        "transactionID" : i+1,
        "timestamp" : random_date(start_date,end_date),
        "amount" : amount,
        "merchants" : random.choice(merchants),
        "locations" : random.choice(locations),
        "devices" : random.choice(devices),
        "card-holder" : faker.name(),
        "class" : is_fraud
    }
    # print(amount)
    # print(is_fraud)
    large_data.append(row)

df_large_realistic = pd.DataFrame(large_data) ## converting data into dataframe
df_large_realistic.to_csv("Synthetic-Fraud-Detection-Dataset" , index=False) ## converting into csv file
print("Dataset generated")
print(df_large_realistic["class"].value_counts())
print(df_large_realistic.head())
print("DATASET OVERVIEW: ")
print(f"dataset shape : {df_large_realistic.shape}")
print(f"dataset colums: {df_large_realistic.columns.to_list()}")
print(f"Dataset class distributions: ")
print(f"Fraud detection percentage: {df_large_realistic["class"].mean()*100:.4f}")
#  with this we have created and overviewd our dataset
print("Missing values per column: ")
print(df_large_realistic.isnull().sum()) ## we are checking for null values in the datset with this command
print("Duplocated values: ")
print(df_large_realistic.duplicated().sum()) ## we are checking for duplicate items in the dataset 
df_large_realistic.drop_duplicates(inplace=True) ## we are dropping duplicate data if any
print("Statistical summary of the data")
print(df_large_realistic.describe()) ## this command will create a statistical summary of the data

# Summary for the fraud and non-fraudulent transactions
fraudulent_transactions = df_large_realistic[df_large_realistic["class"]==1]
non_fraudulent_transaction = df_large_realistic[df_large_realistic["class"]==0]
fraud_stats = fraudulent_transactions['amount'].describe
non_fraud_stats = non_fraudulent_transaction['amount'].describe
print("Fraudulent Transactions: \n", fraud_stats)
print("Non-Fraudulebt Transaction: \n",non_fraud_stats)

# Now we will perform EDA: 
# 1. Correlational matrix
plt.figure(figsize=(15,15))
corr = df_large_realistic.select_dtypes(include="number").corr()
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Correlational Matrix (Numeic Values Only)")
plt.grid(True)
plt.show()
print("Correlational Matrix showcased\n")

# 2. Boxplot of Transactional Amount by Class
plt.figure(figsize=(10,6))
sns.barplot(x="class", y="amount",data=df_large_realistic)
plt.title("Transactional Amount by Class")
plt.xlabel("Class")
plt.ylabel("Transactional Amount")
plt.grid(True)
plt.show()
print("A boxplot of Transactional Amount by Class is showcased\n")

# 3 Pie Chart of Fraudulent vs Non-Fraudulent Transactions
class_count = df_large_realistic["class"].value_counts()
plt.figure(figsize=(10,8))
plt.pie(
    class_count.values,
    labels=["Non-Fraudulent","Fraudulent"],
    autopct="%1.1f%%",
    colors=["#9488d1", "#f8f5d1"],
    startangle=90,
    shadow=True
    )
plt.title("Non-Fraudulent vs Fraudulent Transactions")
plt.grid(True)
plt.show()
print("Showcasing a pie chart of Non-Fraudulent vs Fradulent Transactions")

# 4 FacetGrid of Fraudulent Transactions
fraudulent_transactions = df_large_realistic[df_large_realistic["class"]==1].copy()
fraudulent_transactions["day"] = pd.to_datetime(fraudulent_transactions["timestamp"]).dt.day_name()
fraudulent_transactions["amount-bin"] = pd.qcut(fraudulent_transactions["amount"], q=3,labels=["Low","Medium","High"] )
fraudulent_transactions["y_jitter"] = fraudulent_transactions["transactionID"] + np.random.uniform(-500, 500, size=len(fraudulent_transactions)) ##adding this line because locations and transactionID were getting overwritten 
graph = sns.FacetGrid(data=fraudulent_transactions,row="locations",col="amount-bin",hue="day",height=3,margin_titles=True)
graph.map(plt.scatter,"amount","transactionID",edgecolor = "w",alpha=0.6,s=40).add_legend()
plt.subplots_adjust(top=0.9,hspace=0.8,wspace=0.7)
graph.fig.suptitle("Scatterplot of Fraudulent Transactions by Loactions and Amount-Bin",fontsize = 16)
plt.grid(True)
plt.show()
print("Showcasing a scatterplot of Fraudulent Transactions\n")

# displaying data related to fraudulent transactions
fraudulent_transactions = df_large_realistic[df_large_realistic["class"]==1]
print(fraudulent_transactions)
# displaying the number of fraudulent transaction in the whole data
fraud_transaction = df_large_realistic["class"]==1
print("Number of fraud transaction in the data: ")
print(fraud_transaction.value_counts())

# 5 Correlation matrix for fraudulent transactions
plt.figure(figsize=(10,10))
fraud_corr = fraudulent_transactions.select_dtypes(include="number").corr()
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Correlational Matrix Of Fraudulent Transactions (Numeic Values Only)")
plt.grid(True)
plt.show()
print("Correlational Matrix of Fradulent Transaction showcased\n")

# 6 Line plot to show fraud trend over time
fraudulent_transactions = df_large_realistic[df_large_realistic["class"]==1].copy()
fraudulent_transactions["timestamp"] = pd.to_datetime(fraudulent_transactions["timestamp"])
fraudulent_transactions["hour"] = fraudulent_transactions["timestamp"].dt.hour
fraud_counts = fraudulent_transactions.groupby(fraudulent_transactions["hour"]).size()
plt.figure(figsize=(14,7))
plt.plot(fraud_counts.index,fraud_counts.values,linewidth = 2,marker = 'o')
plt.title("Fraud trends over time")
plt.xlabel("Time (Hours)")
plt.ylabel("Number of Fraud Transactions")
plt.grid(True)
plt.show()
print("Line chart of Fraud Trends over time\n")

# 7 Distributing fraud and non-fraud transactions based on amount
fig,ax = plt.subplots(1,2,figsize=(18,8)) ## we will be using the variables created as fraudulent_transactions and non_fraudulent_transaction
fig.suptitle("Fraudulent and Non-fraudulent transactions with respect to amount")
sns.kdeplot(non_fraudulent_transaction["amount"],label="Amount",ax=ax[0])
sns.kdeplot(fraudulent_transactions["amount"],label="Fraud-Amount",ax=ax[1])
plt.grid(True)
plt.show()
print("THe distribution of Fraud and Non-Fraud transactions over the amount\n")

# conveting time to date time
print(f"Original time column sample: {df_large_realistic["timestamp"].head()}")
df_large_realistic["datetime"] = pd.to_datetime(df_large_realistic["timestamp"],unit="s") ## here we convert timestamp into datetime and unit is kept as seconds
df_large_realistic = df_large_realistic.sort_values("datetime").reset_index(drop=True) ## here we sort the datetime values
df_large_realistic = df_large_realistic.drop("timestamp", axis=1) ## here we drop previous timestamp values

print(f"Timestamp range: {df_large_realistic["datetime"].min()} to {df_large_realistic["datetime"].max()}") ##here we are printing the range of time
duration = df_large_realistic["datetime"].max()- df_large_realistic["datetime"].min() ## here we calculate the total duration by doing max-datetime - min-datetime and conerting it into days
print(f"Total Duration: {duration.days} days\n") 

# IMPORTANT STEP : we need to convert individual transactions into time series
# since individiual transactions cannot directly go to ARIMA , because ARIMA needs a sequential time-series data
# so now we use feature engineering
# we will create following features :From Amount column: sum, mean, std, count, max per time window
                                     # From Class column: sum (fraud count), mean (fraud rate) per time window

print("FEATURE ENGINEERING: Creating Time Series")
print("="*50)
# set datetime as index for time series analysis
df_large_realistic_ts = df_large_realistic.set_index("datetime")
# CREATE HOURLY AGGREGATION - this is our time series frequency
print("Creating hourly time series features.........")
hourly_stats = df_large_realistic_ts.resample("1h").agg({
        "amount" : ["sum", "mean", "std", "count", "max", "min"],
        "class" : ["sum", "mean", "count"]
}).round(4)
# flatten the column names for making it easy
hourly_stats.columns = ["_".join(col).strip() if isinstance(col,tuple) else col for col in hourly_stats.columns] ## this line flatten multi-index colums like "amount","sum" into single strings like"amount_sum" and strip make sures no spaces are remained
hourly_stats = hourly_stats.fillna(0) ##filled all na/null values with 0
print(f"Time Series shape: {hourly_stats.shape}")
print("Time series colums created: ")
for col in hourly_stats.columns:
    print(f" -{col}") ##prints all the columns in hourly_stats
print(f"Time Series Date Range: {hourly_stats.index.min()} to {hourly_stats.index.max()}")
print(f"Total time points: {len(hourly_stats)}")
# sample of time series data (showing top columns of the time series data)
print("\nHourly Aggregated Data: ")
print(hourly_stats.head())
print("\n")

# From all possible time series, we select the most relevant ones: (FEATURE SELECTION)
# TRANSACTION VOLUME: amount_count (number of transactions per hour) - Helps detect unusual transaction patterns
# FRAUD COUNT: class_sum (number of frauds per hour) -  Direct measure of fraud occurrence
# FRAUD RATE: class_mean (percentage of frauds per hour) - Normalized fraud intensity
# TRANSACTION VALUE: amount_mean (average transaction amount per hour) -  Detects unusual spending patterns
# These 4 time series capture different aspects of fraudulent behavior!

print("SELECTING TIME SERIES FOR ANALYSIS: ")
print("="*50)
key_series = {
    "Transaction_Count" : hourly_stats["amount_count"],
    "Fraud_Count" : hourly_stats["class_sum"],
    "Fraud_Rate" : hourly_stats["class_mean"],
    "Avg_Amount" : hourly_stats["amount_mean"]
}
for name,series in key_series.items(): ## series here refers to items from hourly_stats
    print(f"\n {name}")
    print(f"Source: {"Amount Column" if "Amount" in name or "Transaction" in name  else "Class Column" }")
    print(f"Aggregation: {"Count" if "Count" in name else "Sum" if "Sum" in str(series.name) else "Mean"}")
    print(f"Length of Series: {len(series)}")
    print(f"Range: {series.min():.4f} to {series.max():.4f}")
    print(f"Non-zero periods: {(series > 0).sum()} out of {len(series)}\n")

# before doing analysis we will visualize our raw time series data to check
# Overall trends and patterns
# Seasonality (daily/weekly patterns)
# Outliers and anomalies
# Data quality issues
print("Visualization of Raw time series data")
print("="*50)
print("\n")

fig,axes = plt.subplots(2,2, figsize=(15,7)) ## axes is a 2D numpy array eg: {[a,b
                                                                            #    c,d  ]}
axes = axes.ravel() ## ravel turns this into [a,b,c,d]
colors = ["red","blue","green","brown"]
for idx,(name,series) in enumerate(key_series.items()):
    axes[idx].plot(series.index,series.values,color=colors[idx],alpha=0.7,marker="o")
    axes[idx].set_title(f"{name} Over Time (Raw Data)",fontweight="bold",fontsize=12)
    axes[idx].set_ylabel("Value")
    axes[idx].grid(True,alpha=0.3)
    axes[idx].tick_params(axis="x",rotation=45)
    stats_text = f"Mean:{series.mean():.2f}\n Std{series.std():.2f}\n Max{series.max():.2f}" ## this code converts basic statistics to text
    axes[idx].text(0.02,0.98,stats_text,transform=axes[idx].transAxes,
                                     verticalalignment='top', 
                                     bbox=dict(boxstyle='round', 
                                     facecolor='white', alpha=0.8))

plt.tight_layout()
plt.grid(True)
plt.show()

print("Initial Observations:")
print("- Look for obvious trends (increasing/decreasing over time)")
print("- Notice any cyclical patterns (daily/weekly cycles)")
print("- Identify periods with unusual activity")
print("- Check for missing data or zero periods")
print("\n")

# now we will do time series decomposition
# time series decomposition tells us about 
# ternds,seasonality,residual and for fraud_detection we will decompose fraud_rate
print("Decomposing Time Series Data")
print("="*50)
print("\n")

decomposition_results = {}
for name,series in key_series.items():
    print(f"Decomposing {name}")
# we are cleaning series from infinite and null items
    series_clean = series.replace([np.inf,-np.inf],np.nan).ffill().fillna(0)
    try:
        decomposition = seasonal_decompose(series_clean,model="additive",period=24)
        decomposition_results[name] = decomposition

        trend_var = np.var(decomposition.trend.dropna())
        seasonal_var = np.var(decomposition.seasonal)
        residual_var = np.var(decomposition.resid.dropna())
        total_var = trend_var + seasonal_var + residual_var

        print(f"Trend Variance: {trend_var:.6f} ({trend_var/total_var*100:.1f}%)")
        print(f"Seasonal Variance: {seasonal_var:.6f} ({seasonal_var/total_var*100:.1f}%)")
        print(f"Residual Variance: {residual_var:.6f} ({residual_var/total_var*100:.1f}%)")
    except Exception as e:
        print(f"Could not decompose {name} : {e}")

# now we take fraud_rate into decomposition it is very essential for fraud_detection
if "Fraud_Rate" in decomposition_results:
    print("\n Visualizing decomposition for Fraud Rate: ")

    decomp = decomposition_results["Fraud_Rate"]
    fig ,axes = plt.subplots(4,1,figsize=(18,14))
    # original series data
    axes[0].plot(decomp.observed.index,decomp.observed,color="red",alpha=0.8)
    axes[0].set_title("Original Fraud Rate time series",fontweight="bold")
    axes[0].grid(True,alpha=0.3)

    # trend series data
    axes[1].plot(decomp.trend.index,decomp.trend,color="blue",alpha=0.8)
    axes[1].set_title("Trend Of Fraud Rate Time Seies Data",fontweight="bold")
    axes[1].grid(True,alpha=0.3) 

    # seasonal series data
    axes[2].plot(decomp.seasonal.index,decomp.seasonal,color="green",alpha=0.8)
    axes[2].set_title("Seasonal Fraud Rate Time Series Data",fontweight="bold")
    axes[2].grid(True,alpha=0.3)

    # residual series data
    axes[3].plot(decomp.resid.index,decomp.resid,color="blue",alpha=0.8)
    axes[3].set_title("Residual Fraud Rate time series",fontweight="bold")
    axes[3].grid(True,alpha=0.3)

    plt.tight_layout()
    plt.grid(True)
    plt.show()






