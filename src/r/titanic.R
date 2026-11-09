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
# rule: every 5th passenger (by PassengerId) among survivors and among non-survivors is held out
# the Python script uses the same rule, so both hold out the exact same rows
section("Step 14 (R): hold out 20% of train.csv for testing")
train <- train %>%
  arrange(PassengerId) %>%
  group_by(Survived) %>%
  mutate(position = row_number() - 1) %>%
  ungroup()
holdout_part <- train %>% filter(position %% 5 == 4) %>% select(-position)
train_part <- train %>% filter(position %% 5 != 4) %>% select(-position)
cat("held out every 5th passenger by PassengerId, separately for survivors and non-survivors\n")
cat("training rows:", nrow(train_part), "\n")
cat("held-out rows:", nrow(holdout_part), "\n")
cat("survival rate training / held-out:",
    round(mean(train_part$Survived), 3), "/", round(mean(holdout_part$Survived), 3), "\n")

# title from the name, e.g. "Braund, Mr. Owen Harris" -> Mr
# rare titles (Dr, Rev, Col, Countess, ...) grouped as Other
get_title <- function(names) {
  title <- trimws(sub("^[^,]*,\\s*([^.]+)\\..*$", "\\1", names))
  title <- recode(title, Mlle = "Miss", Ms = "Miss", Mme = "Mrs")
  ifelse(title %in% c("Mr", "Mrs", "Miss", "Master"), title, "Other")
}

# fill values come from the training rows only, so nothing leaks in from held-out or test
# age is filled with the median for that title (a Master is a boy, a Mr is an adult man)
get_fill_values <- function(df) {
  list(
    age_by_title = df %>%
      mutate(Title = get_title(Name)) %>%
      group_by(Title) %>%
      summarise(title_age = median(Age, na.rm = TRUE)),
    fare_median = median(df$Fare, na.rm = TRUE),
    embarked_mode = names(which.max(table(df$Embarked)))
  )
}

fills <- get_fill_values(train_part)
cat("\nfill values from the training rows:\n")
cat("  median Age by title:",
    paste(fills$age_by_title$Title, fills$age_by_title$title_age, sep = " = ", collapse = ", "), "\n")
cat("  Fare median = ", round(fills$fare_median, 4), ", Embarked mode = ", fills$embarked_mode, "\n", sep = "")

# same changes for every dataset (verbose = FALSE keeps the cross-validation loop quiet)
clean <- function(df, name, fills, verbose = TRUE) {
  say <- function(...) if (verbose) cat(...)

  say("\ncleaning ", name, ", starting shape ", nrow(df), " x ", ncol(df), "\n", sep = "")

  df <- df %>% mutate(HasCabin = as.integer(!is.na(Cabin)))
  say("  added HasCabin (1 if cabin known):", sum(df$HasCabin), "of", nrow(df), "rows\n")

  df <- df %>% mutate(Title = get_title(Name))
  title_counts <- table(df$Title)
  say("  added Title from Name:", paste(names(title_counts), title_counts, sep = " = ", collapse = ", "), "\n")

  df <- df %>% mutate(FamilySize = SibSp + Parch + 1)
  say("  added FamilySize = SibSp + Parch + 1 (range ", min(df$FamilySize), " to ", max(df$FamilySize), ")\n", sep = "")

  df <- df %>% select(-Name, -Ticket, -Cabin)
  say("  dropped Name, Ticket, Cabin (free text, Cabin mostly missing)\n")

  say("  filled", sum(is.na(df$Age)), "missing Age with the median for their title\n")
  df <- df %>%
    left_join(fills$age_by_title, by = "Title") %>%
    mutate(Age = coalesce(Age, title_age)) %>%
    select(-title_age)

  # Title is only used for the age fill and this flag, since Mr/Mrs/Miss just repeat Sex
  df <- df %>% mutate(IsBoy = as.integer(Title == "Master")) %>% select(-Title)
  say("  added IsBoy (1 if title is Master):", sum(df$IsBoy), "rows, then dropped Title\n")

  say("  filled", sum(is.na(df$Fare)), "missing Fare with", fills$fare_median, "\n")
  df <- df %>% mutate(Fare = coalesce(Fare, fills$fare_median))

  say("  filled", sum(is.na(df$Embarked)), "missing Embarked with", fills$embarked_mode, "\n")
  df <- df %>% mutate(Embarked = coalesce(Embarked, fills$embarked_mode))

  df <- df %>% mutate(Sex = ifelse(Sex == "female", 1, 0))
  say("  coded Sex as male = 0, female = 1\n")

  # same levels everywhere so glm makes the same 0/1 columns for every dataset
  df <- df %>% mutate(Embarked = factor(Embarked, levels = c("C", "Q", "S")))
  say("  made Embarked a factor with levels C, Q, S (glm turns it into 0/1 columns, C is the baseline)\n")

  say("  missing values left:", sum(is.na(df)), "\n")
  say("  final shape", nrow(df), "x", ncol(df), "\n")
  df
}

section("Step 14 (R): clean the training and held-out rows")
train_clean <- clean(train_part, "training rows", fills)
holdout_clean <- clean(holdout_part, "held-out rows", fills)

# FamilySize replaces SibSp and Parch
model_formula <- Survived ~ Pclass + Sex + IsBoy + Age + FamilySize + Fare + HasCabin + Embarked
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

# auc: share of (survivor, non-survivor) pairs where the survivor gets the higher score, ties count half
auc_score <- function(actual, prob) {
  pos <- prob[actual == 1]
  neg <- prob[actual == 0]
  mean(outer(pos, neg, ">") + 0.5 * outer(pos, neg, "=="))
}

# accuracy alone hides what kind of mistakes the model makes, so print the other metrics too
print_metrics <- function(model, df, name) {
  actual <- df$Survived
  prob <- predict(model, newdata = df, type = "response")
  pred <- as.integer(prob > 0.5)
  tn <- sum(pred == 0 & actual == 0)
  fp <- sum(pred == 1 & actual == 0)
  fn <- sum(pred == 0 & actual == 1)
  tp <- sum(pred == 1 & actual == 1)
  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)
  f1 <- 2 * precision * recall / (precision + recall)
  auc <- auc_score(actual, prob)

  cat(sprintf("%s accuracy: %.3f\n", name, mean(pred == actual)))
  cat("\nconfusion matrix:\n")
  cat("                   predicted died  predicted survived\n")
  cat(sprintf("  actually died    %14d  %18d\n", tn, fp))
  cat(sprintf("  actually survived%14d  %18d\n", fn, tp))
  cat(sprintf("\nprecision: %.3f  (of those predicted to survive, share who did)\n", precision))
  cat(sprintf("recall:    %.3f  (of those who survived, share the model caught)\n", recall))
  cat(sprintf("f1:        %.3f  (balance of precision and recall)\n", f1))
  cat(sprintf("roc auc:   %.3f  (chance a survivor gets a higher score than a non-survivor)\n", auc))
}

# step 16 again: training accuracy
section("Step 16 (R): accuracy on the training set")
print_metrics(model, train_clean, "training")

# step 17 again: test.csv
section("Step 17 (R): load test.csv and predict")
test <- read_csv("data/test.csv", show_col_types = FALSE)
cat("test.csv shape:", nrow(test), "rows,", ncol(test), "cols\n")
cat("missing values per column:\n")
print(colSums(is.na(test)))
test_clean <- clean(test, "test.csv", fills)
test_pred <- predict_class(model, test_clean)
cat("\npredicted survivors: ", sum(test_pred), " of ", length(test_pred),
    " (", round(mean(test_pred), 3), ")\n", sep = "")

write_csv(data.frame(PassengerId = test$PassengerId, Survived = test_pred), "data/predictions_r.csv")
cat("saved predictions to data/predictions_r.csv\n")

# step 18 again: test accuracy
section("Step 18 (R): accuracy on the test set")
cat("test.csv has no Survived column, so its accuracy can't be measured here.\n")
cat("the 20% of train.csv held out in step 14 is used as the test set instead.\n")
cat("\n")
print_metrics(model, holdout_clean, "held-out test")
baseline_acc <- mean(holdout_clean$Sex == holdout_clean$Survived)
cat(sprintf("\nfor comparison, guessing every woman survived and every man died: %.3f accuracy\n", baseline_acc))

# extra: 5-fold cross-validation on all of train.csv, a steadier estimate than one 80/20 split
# fold k = every 5th passenger by PassengerId within survivors and non-survivors, starting at k
# (fold 5 is the hold-out set above), same folds as the Python script
section("Extra (R): 5-fold cross-validation on train.csv")
cv_acc <- c()
cv_auc <- c()
for (k in 0:4) {
  fit_rows <- train %>% filter(position %% 5 != k) %>% select(-position)
  check_rows <- train %>% filter(position %% 5 == k) %>% select(-position)
  fold_fills <- get_fill_values(fit_rows)
  fold_model <- glm(model_formula, data = clean(fit_rows, "", fold_fills, verbose = FALSE), family = binomial)
  check_clean <- clean(check_rows, "", fold_fills, verbose = FALSE)
  prob <- predict(fold_model, newdata = check_clean, type = "response")
  cv_acc <- c(cv_acc, mean(as.integer(prob > 0.5) == check_clean$Survived))
  cv_auc <- c(cv_auc, auc_score(check_clean$Survived, prob))
  cat(sprintf("fold %d: %d rows to fit, %d to check, accuracy %.3f, roc auc %.3f\n",
              k + 1, nrow(fit_rows), nrow(check_rows), tail(cv_acc, 1), tail(cv_auc, 1)))
}
cat(sprintf("\ncross-validated accuracy: %.3f (sd %.3f)\n", mean(cv_acc), sd(cv_acc)))
cat(sprintf("cross-validated roc auc:  %.3f (sd %.3f)\n", mean(cv_auc), sd(cv_auc)))
