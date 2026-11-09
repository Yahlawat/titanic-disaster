# Step 23: load packages and train.csv
train <- read.csv("data/train.csv")  # path depends on how the Dockerfile gets the data in
cat("Loaded train.csv:", nrow(train), "rows,", ncol(train), "cols\n")

# Step 24: repeat 14-18 (explore/adjust, logistic regression, train accuracy, predict test, test accuracy)
