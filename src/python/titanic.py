# Step 13: load packages and train.csv
import pandas as pd

train = pd.read_csv("data/train.csv")  # path depends on how the Dockerfile gets the data in
print("Loaded train.csv:", train.shape)

# Step 14: explore and adjust the data (print every change)

# Step 15: logistic regression on the training set

# Step 16: training accuracy

# Step 17: load test.csv and predict

# Step 18: test accuracy
