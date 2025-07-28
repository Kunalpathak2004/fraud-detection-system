import random
import faker
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker
import matplotlib.pyplot as plt
import seaborn as sns

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
fraud_counts = fraudulent_transactions.groupby(fraudulent_transactions["timestamp"].astype(int)//3600).size()
plt.figure(figsize=(14,7))
plt.plot(fraud_counts.index,fraud_counts.values,linewidth = 2,marker = 'o')
plt.title("Fraud trends over time")
plt.xlabel("Time (Hours)")
plt.ylabel("Number of Fraud Transactions")
plt.grid(True)
plt.show()
print("Line chart of Fraud Trends over time\n")