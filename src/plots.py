import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "../data/external/energy_data_enschede.csv"
df = pd.read_csv(file_path, delimiter=";")

# # Show basic info
# print("Dataset preview:")
# print(df.head())
# print("\nMissing values:")
# print(df.isnull().sum())

print(df.columns.tolist())