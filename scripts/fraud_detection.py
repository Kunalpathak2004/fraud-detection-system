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
