suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

options(width = 120)

section <- function(title) {
  cat("\n", strrep("=", 60), "\n", title, "\n", strrep("=", 60), "\n", sep = "")
}

# step 23: load train.csv
section("Step 23: load train.csv")
train <- read_csv("data/train.csv", show_col_types = FALSE)
cat("train.csv shape:", nrow(train), "rows,", ncol(train), "cols\n")
print(head(train))

# step 14 again: look at the data first
section("Step 14 (R): explore the training data")
cat("column types:\n")
print(sapply(train, class))
cat("\nmissing values per column:\n")
print(colSums(is.na(train)))
cat("\nsurvival rate:", round(mean(train$Survived), 3), "\n")
cat("\nsurvival rate by sex:\n")
print(train %>% group_by(Sex) %>% summarise(survival_rate = round(mean(Survived), 3)))
cat("\nsurvival rate by class:\n")
print(train %>% group_by(Pclass) %>% summarise(survival_rate = round(mean(Survived), 3)))

# test.csv has no Survived column, so hold out 20% of train.csv to measure accuracy on unseen rows
section("Step 14 (R): hold out 20% of train.csv for testing")
set.seed(42)
holdout_part <- train %>% group_by(Survived) %>% slice_sample(prop = 0.2) %>% ungroup()
train_part <- train %>% anti_join(holdout_part, by = "PassengerId")
cat("training rows:", nrow(train_part), "\n")
cat("held-out rows:", nrow(holdout_part), "\n")
cat("survival rate training / held-out:",
    round(mean(train_part$Survived), 3), "/", round(mean(holdout_part$Survived), 3), "\n")

# fill values come from the training rows only, so nothing leaks in from held-out or test
age_median <- median(train_part$Age, na.rm = TRUE)
fare_median <- median(train_part$Fare, na.rm = TRUE)
embarked_mode <- names(which.max(table(train_part$Embarked)))
cat("\nfill values from the training rows: Age median = ", age_median,
    ", Fare median = ", round(fare_median, 4), ", Embarked mode = ", embarked_mode, "\n", sep = "")

# same changes for every dataset
clean <- function(df, name) {
  cat("\ncleaning ", name, ", starting shape ", nrow(df), " x ", ncol(df), "\n", sep = "")

  df <- df %>% mutate(HasCabin = as.integer(!is.na(Cabin)))
  cat("  added HasCabin (1 if cabin known):", sum(df$HasCabin), "of", nrow(df), "rows\n")

  df <- df %>% select(-Name, -Ticket, -Cabin)
  cat("  dropped Name, Ticket, Cabin (free text, Cabin mostly missing)\n")

  cat("  filled", sum(is.na(df$Age)), "missing Age with", age_median, "\n")
  df <- df %>% mutate(Age = coalesce(Age, age_median))

  cat("  filled", sum(is.na(df$Fare)), "missing Fare with", fare_median, "\n")
  df <- df %>% mutate(Fare = coalesce(Fare, fare_median))

  cat("  filled", sum(is.na(df$Embarked)), "missing Embarked with", embarked_mode, "\n")
  df <- df %>% mutate(Embarked = coalesce(Embarked, embarked_mode))

  df <- df %>% mutate(Sex = ifelse(Sex == "female", 1, 0))
  cat("  coded Sex as male = 0, female = 1\n")

  # same levels everywhere so glm makes the same 0/1 columns for every dataset
  df <- df %>% mutate(Embarked = factor(Embarked, levels = c("C", "Q", "S")))
  cat("  made Embarked a factor with levels C, Q, S (glm turns it into 0/1 columns, C is the baseline)\n")

  cat("  missing values left:", sum(is.na(df)), "\n")
  cat("  final shape", nrow(df), "x", ncol(df), "\n")
  df
}

section("Step 14 (R): clean the training and held-out rows")
train_clean <- clean(train_part, "training rows")
holdout_clean <- clean(holdout_part, "held-out rows")

model_formula <- Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + HasCabin + Embarked
cat("\nmodel formula:", deparse(model_formula, width.cutoff = 500), "\n")

# step 15 again: logistic regression
section("Step 15 (R): fit a logistic regression")
model <- glm(model_formula, data = train_clean, family = binomial)
cat("model fit on", nrow(train_clean), "rows\n")
cat("\ncoefficients (log-odds, + means more likely to survive):\n")
print(round(sort(coef(model)), 3))

# predicted probability above 0.5 means survived
predict_class <- function(model, df) {
  as.integer(predict(model, newdata = df, type = "response") > 0.5)
}

# step 16 again: training accuracy
section("Step 16 (R): accuracy on the training set")
train_acc <- mean(predict_class(model, train_clean) == train_clean$Survived)
cat("training accuracy:", round(train_acc, 3), "\n")

# step 17 again: test.csv
section("Step 17 (R): load test.csv and predict")
test <- read_csv("data/test.csv", show_col_types = FALSE)
cat("test.csv shape:", nrow(test), "rows,", ncol(test), "cols\n")
cat("missing values per column:\n")
print(colSums(is.na(test)))
test_clean <- clean(test, "test.csv")
test_pred <- predict_class(model, test_clean)
cat("\npredicted survivors: ", sum(test_pred), " of ", length(test_pred),
    " (", round(mean(test_pred), 3), ")\n", sep = "")

write_csv(data.frame(PassengerId = test$PassengerId, Survived = test_pred), "data/predictions_r.csv")
cat("saved predictions to data/predictions_r.csv\n")

# step 18 again: test accuracy
section("Step 18 (R): accuracy on the test set")
cat("test.csv has no Survived column, so its accuracy can't be measured here.\n")
cat("the 20% of train.csv held out in step 14 is used as the test set instead.\n")
holdout_acc <- mean(predict_class(model, holdout_clean) == holdout_clean$Survived)
cat("\nheld-out test accuracy:", round(holdout_acc, 3), "\n")
baseline_acc <- mean(holdout_clean$Sex == holdout_clean$Survived)
cat("for comparison, guessing every woman survived and every man died:", round(baseline_acc, 3), "\n")
