import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

# fill values come from the training rows only, so nothing leaks in from held-out or test
age_median = train_part["Age"].median()
fare_median = train_part["Fare"].median()
embarked_mode = train_part["Embarked"].mode()[0]
print(
    f"\nfill values from the training rows: Age median = {age_median}, "
    f"Fare median = {fare_median}, Embarked mode = {embarked_mode}"
)


# same changes for every dataset
def clean(df, name):
    print(f"\ncleaning {name}, starting shape {df.shape}")
    df = df.copy()

    df["HasCabin"] = df["Cabin"].notna().astype(int)
    print(f"  added HasCabin (1 if cabin known): {df['HasCabin'].sum()} of {len(df)} rows")

    df = df.drop(columns=["Name", "Ticket", "Cabin"])
    print("  dropped Name, Ticket, Cabin (free text, Cabin mostly missing)")

    print(f"  filled {df['Age'].isna().sum()} missing Age with {age_median}")
    df["Age"] = df["Age"].fillna(age_median)

    print(f"  filled {df['Fare'].isna().sum()} missing Fare with {fare_median}")
    df["Fare"] = df["Fare"].fillna(fare_median)

    print(f"  filled {df['Embarked'].isna().sum()} missing Embarked with {embarked_mode}")
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    print("  coded Sex as male = 0, female = 1")

    df = pd.get_dummies(df, columns=["Embarked"], prefix="Embarked", dtype=int)
    print("  turned Embarked into 0/1 columns:", [c for c in df.columns if c.startswith("Embarked_")])

    print(f"  missing values left: {df.isna().sum().sum()}")
    print(f"  final shape {df.shape}")
    return df


section("Step 14: clean the training and held-out rows")
train_clean = clean(train_part, "training rows")
holdout_clean = clean(holdout_part, "held-out rows")

# Embarked_C is left out as the baseline (same as R), otherwise the 3 port columns always add up to 1
features = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "HasCabin",
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
# no penalty, so it's the same model as R's glm
model = LogisticRegression(penalty=None, max_iter=5000)
model.fit(X_train, y_train)
print("model fit on", X_train.shape[0], "rows and", X_train.shape[1], "features")
coefs = pd.Series(model.coef_[0], index=features).sort_values()
print("\ncoefficients (log-odds, + means more likely to survive):")
print(coefs.round(3))
print(f"intercept: {model.intercept_[0]:.3f}")

# step 16: training accuracy
section("Step 16: accuracy on the training set")
train_acc = accuracy_score(y_train, model.predict(X_train))
print(f"training accuracy: {train_acc:.3f}")

# step 17: test.csv
section("Step 17: load test.csv and predict")
test = pd.read_csv("data/test.csv")
print("test.csv shape:", test.shape)
print("missing values per column:")
print(test.isna().sum())
test_clean = clean(test, "test.csv")
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
holdout_acc = accuracy_score(y_holdout, model.predict(X_holdout))
print(f"\nheld-out test accuracy: {holdout_acc:.3f}")
baseline_acc = accuracy_score(y_holdout, X_holdout["Sex"])
print(f"for comparison, guessing every woman survived and every man died: {baseline_acc:.3f}")
