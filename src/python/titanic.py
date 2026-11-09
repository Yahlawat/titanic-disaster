import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

pd.set_option("display.width", 120)


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# step 13: load train.csv
section("Step 13: load train.csv")
train = pd.read_csv("data/train.csv")
print("train.csv shape:", train.shape)
print(train.head())

# step 14: look at the data first
section("Step 14: explore the training data")
print("column types:")
print(train.dtypes)
print("\nmissing values per column:")
print(train.isna().sum())
print(f"\nsurvival rate: {train['Survived'].mean():.3f}")
print("\nsurvival rate by sex:")
print(train.groupby("Sex")["Survived"].mean().round(3))
print("\nsurvival rate by class:")
print(train.groupby("Pclass")["Survived"].mean().round(3))

# test.csv has no Survived column, so hold out 20% of train.csv to measure accuracy on unseen rows
# rule: every 5th passenger (by PassengerId) among survivors and among non-survivors is held out
# the R script uses the same rule, so both hold out the exact same rows
section("Step 14: hold out 20% of train.csv for testing")
train = train.sort_values("PassengerId").reset_index(drop=True)
position = train.groupby("Survived").cumcount()
is_holdout = position % 5 == 4
holdout_part = train[is_holdout]
train_part = train[~is_holdout]
print("held out every 5th passenger by PassengerId, separately for survivors and non-survivors")
print("training rows:", len(train_part))
print("held-out rows:", len(holdout_part))
print(
    f"survival rate training / held-out: "
    f"{train_part['Survived'].mean():.3f} / {holdout_part['Survived'].mean():.3f}"
)

# title from the name, e.g. "Braund, Mr. Owen Harris" -> Mr
# rare titles (Dr, Rev, Col, Countess, ...) grouped as Other
def get_title(names):
    title = names.str.extract(r",\s*([^.]+)\.")[0].str.strip()
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    return title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Other")


# fill values come from the training rows only, so nothing leaks in from held-out or test
# age is filled with the median for that title (a Master is a boy, a Mr is an adult man)
def get_fill_values(df):
    return {
        "age_by_title": df.assign(Title=get_title(df["Name"])).groupby("Title")["Age"].median(),
        "fare_median": df["Fare"].median(),
        "embarked_mode": df["Embarked"].mode()[0],
    }


fills = get_fill_values(train_part)
print("\nfill values from the training rows:")
print("  median Age by title:", fills["age_by_title"].to_dict())
print(f"  Fare median = {fills['fare_median']}, Embarked mode = {fills['embarked_mode']}")


# same changes for every dataset (verbose=False keeps the cross-validation loop quiet)
def clean(df, name, fills, verbose=True):
    def say(*args):
        if verbose:
            print(*args)

    say(f"\ncleaning {name}, starting shape {df.shape}")
    df = df.copy()

    df["HasCabin"] = df["Cabin"].notna().astype(int)
    say(f"  added HasCabin (1 if cabin known): {df['HasCabin'].sum()} of {len(df)} rows")

    df["Title"] = get_title(df["Name"])
    say("  added Title from Name:", df["Title"].value_counts().to_dict())

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    say(f"  added FamilySize = SibSp + Parch + 1 (range {df['FamilySize'].min()} to {df['FamilySize'].max()})")

    df = df.drop(columns=["Name", "Ticket", "Cabin"])
    say("  dropped Name, Ticket, Cabin (free text, Cabin mostly missing)")

    say(f"  filled {df['Age'].isna().sum()} missing Age with the median for their title")
    df["Age"] = df["Age"].fillna(df["Title"].map(fills["age_by_title"]))

    # Title is only used for the age fill and this flag, since Mr/Mrs/Miss just repeat Sex
    df["IsBoy"] = (df["Title"] == "Master").astype(int)
    df = df.drop(columns=["Title"])
    say(f"  added IsBoy (1 if title is Master): {df['IsBoy'].sum()} rows, then dropped Title")

    say(f"  filled {df['Fare'].isna().sum()} missing Fare with {fills['fare_median']}")
    df["Fare"] = df["Fare"].fillna(fills["fare_median"])

    say(f"  filled {df['Embarked'].isna().sum()} missing Embarked with {fills['embarked_mode']}")
    df["Embarked"] = df["Embarked"].fillna(fills["embarked_mode"])

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    say("  coded Sex as male = 0, female = 1")

    df = pd.get_dummies(df, columns=["Embarked"], dtype=int)
    say("  turned Embarked into 0/1 columns:", [c for c in df.columns if c.startswith("Embarked_")])

    say(f"  missing values left: {df.isna().sum().sum()}")
    say(f"  final shape {df.shape}")
    return df


section("Step 14: clean the training and held-out rows")
train_clean = clean(train_part, "training rows", fills)
holdout_clean = clean(holdout_part, "held-out rows", fills)

# Embarked_C is left out as the baseline (same as R), otherwise the 3 port columns always add up to 1
# FamilySize replaces SibSp and Parch
features = [
    "Pclass", "Sex", "IsBoy", "Age", "FamilySize", "Fare", "HasCabin",
    "Embarked_Q", "Embarked_S",
]
print("\nfeatures used in the model:", features)
print("Embarked_C left out as the baseline port")

# reindex so every dataset has the same columns in the same order
X_train = train_clean.reindex(columns=features, fill_value=0)
y_train = train_clean["Survived"]
X_holdout = holdout_clean.reindex(columns=features, fill_value=0)
y_holdout = holdout_clean["Survived"]

# step 15: logistic regression
section("Step 15: fit a logistic regression")
# no penalty and a tight tolerance, so it fits the same model as R's glm
model = LogisticRegression(penalty=None, tol=1e-8, max_iter=5000)
model.fit(X_train, y_train)
print("model fit on", X_train.shape[0], "rows and", X_train.shape[1], "features")
coefs = pd.Series(model.coef_[0], index=features).sort_values()
print("\ncoefficients (log-odds, + means more likely to survive):")
print(coefs.round(3))
print(f"intercept: {model.intercept_[0]:.3f}")



# accuracy alone hides what kind of mistakes the model makes, so print the other metrics too
def print_metrics(X, y, name):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(f"{name} accuracy: {accuracy_score(y, pred):.3f}")
    print("\nconfusion matrix:")
    print("                   predicted died  predicted survived")
    print(f"  actually died    {tn:14d}  {fp:18d}")
    print(f"  actually survived{fn:14d}  {tp:18d}")
    print(f"\nprecision: {precision_score(y, pred):.3f}  (of those predicted to survive, share who did)")
    print(f"recall:    {recall_score(y, pred):.3f}  (of those who survived, share the model caught)")
    print(f"f1:        {f1_score(y, pred):.3f}  (balance of precision and recall)")
    print(f"roc auc:   {roc_auc_score(y, prob):.3f}  (chance a survivor gets a higher score than a non-survivor)")


# step 16: training accuracy
section("Step 16: accuracy on the training set")
print_metrics(X_train, y_train, "training")

# step 17: test.csv
section("Step 17: load test.csv and predict")
test = pd.read_csv("data/test.csv")
print("test.csv shape:", test.shape)
print("missing values per column:")
print(test.isna().sum())
test_clean = clean(test, "test.csv", fills)
X_test = test_clean.reindex(columns=features, fill_value=0)
test_pred = model.predict(X_test)
print(f"\npredicted survivors: {test_pred.sum()} of {len(test_pred)} ({test_pred.mean():.3f})")

predictions = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred})
predictions.to_csv("data/predictions_python.csv", index=False)
print("saved predictions to data/predictions_python.csv")

# step 18: test accuracy
section("Step 18: accuracy on the test set")
print("test.csv has no Survived column, so its accuracy can't be measured here.")
print("the 20% of train.csv held out in step 14 is used as the test set instead.")
print()
print_metrics(X_holdout, y_holdout, "held-out test")
baseline_acc = accuracy_score(y_holdout, X_holdout["Sex"])
print(f"\nfor comparison, guessing every woman survived and every man died: {baseline_acc:.3f} accuracy")


# extra: 5-fold cross-validation on all of train.csv, a steadier estimate than one 80/20 split
# fold k = every 5th passenger by PassengerId within survivors and non-survivors, starting at k
# (fold 5 is the hold-out set above), same folds as the R script
section("Extra: 5-fold cross-validation on train.csv")
fold = train.groupby("Survived").cumcount() % 5
cv_acc, cv_auc = [], []
for k in range(5):
    fit_rows, check_rows = train[fold != k], train[fold == k]
    fold_fills = get_fill_values(fit_rows)
    X_fit = clean(fit_rows, "", fold_fills, verbose=False).reindex(columns=features, fill_value=0)
    X_check = clean(check_rows, "", fold_fills, verbose=False).reindex(columns=features, fill_value=0)
    fold_model = LogisticRegression(penalty=None, tol=1e-8, max_iter=5000)
    fold_model.fit(X_fit, fit_rows["Survived"])
    cv_acc.append(accuracy_score(check_rows["Survived"], fold_model.predict(X_check)))
    cv_auc.append(roc_auc_score(check_rows["Survived"], fold_model.predict_proba(X_check)[:, 1]))
    print(f"fold {k + 1}: {len(fit_rows)} rows to fit, {len(check_rows)} to check, "
          f"accuracy {cv_acc[-1]:.3f}, roc auc {cv_auc[-1]:.3f}")
print(f"\ncross-validated accuracy: {pd.Series(cv_acc).mean():.3f} (sd {pd.Series(cv_acc).std():.3f})")
print(f"cross-validated roc auc:  {pd.Series(cv_auc).mean():.3f} (sd {pd.Series(cv_auc).std():.3f})")
