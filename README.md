<h1 align="center">Titanic Survival in Docker</h1>

<p align="center">
  Predicting who survived the Titanic with logistic regression, built twice, once in Python and once in R, each in its own Docker container.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white" alt="Python 3.13">
  <img src="https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn 1.7.2">
  <img src="https://img.shields.io/badge/R-4.6.1-276DC3?logo=r&logoColor=white" alt="R 4.6.1">
  <img src="https://img.shields.io/badge/docker-required-2496ED?logo=docker&logoColor=white" alt="Docker">
</p>

## Overview

Each container loads the [Kaggle Titanic](https://www.kaggle.com/competitions/titanic) data, cleans it, fits a logistic regression, scores it, and predicts `test.csv`. Every step is printed to the terminal, so you can follow what changed without opening the code.

The Python and R pipelines use the same split, cleaning and model, and they give the same coefficients, metrics and predictions. That's a built-in check that both are doing what they say.

## Quickstart

You need [Git](https://git-scm.com/downloads), [Docker Desktop](https://www.docker.com/products/docker-desktop/) (open and running) and a free [Kaggle](https://www.kaggle.com/) account. No local Python or R needed.

```bash
git clone https://github.com/Yahlawat/titanic-disaster.git
cd titanic-disaster
unzip ~/Downloads/titanic.zip -d src/data        # after downloading the data, see step 2

docker build -t titanic-python src/python
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-python

docker build -t titanic-r src/r
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r
```

## Step by step

Run every command from the repo root (the `titanic-disaster` folder).

### 1. Clone the repo

```bash
git clone https://github.com/Yahlawat/titanic-disaster.git
cd titanic-disaster
```

### 2. Download the data

The dataset is not in this repo.

1. Go to the [Titanic competition data page](https://www.kaggle.com/competitions/titanic/data) and sign in.
2. Join the competition and accept its rules. Kaggle won't let you download the data until you do.
3. Click **Download All** to get `titanic.zip`.
4. Unzip it into `src/data`:

```bash
unzip ~/Downloads/titanic.zip -d src/data
ls src/data
```

You should see `train.csv`, `test.csv` and `gender_submission.csv`. Only the first two are used.

On Windows, right-click `titanic.zip` > **Extract All** and choose the `src/data` folder.

### 3. Run the Python container

```bash
docker build -t titanic-python src/python
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-python
```

### 4. Run the R container

```bash
docker build -t titanic-r src/r
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r
```

The first R build takes a minute or two while it installs packages. Later builds are cached.

**Windows PowerShell:** use `${PWD}` instead of `$(pwd)`:

```powershell
docker run --rm -v "${PWD}/src/data:/app/data" titanic-python
docker run --rm -v "${PWD}/src/data:/app/data" titanic-r
```

### What the commands do

- `docker build -t <name> src/<language>` builds an image from that folder's Dockerfile: base image, pinned packages, then the script.
- `docker run --rm -v "$(pwd)/src/data:/app/data" <name>` runs the script. `-v` shares your `src/data` folder with the container, so the script reads the CSVs from it and saves its predictions back to it. `--rm` removes the container when it finishes.

## Approach

The steps below are the same in both scripts, and each one is printed as it runs.

1. **Explore.** Column types, missing values, survival rate overall, by sex and by class.
2. **Hold out a test set.** `test.csv` has no `Survived` column, so 20% of `train.csv` (177 rows) is held out as the test set. The rule is every 5th passenger by `PassengerId`, taken separately among survivors and non-survivors, so both parts keep the same survival rate and both languages hold out the same rows.
3. **Clean and add features.** Fill values are worked out from the training rows only.
   - `Title` pulled from `Name` (Mr, Mrs, Miss, Master, Other), used to fill missing `Age` with the median for that title
   - `IsBoy` = 1 if the title is Master (boys were saved ahead of men)
   - `FamilySize` = `SibSp` + `Parch` + 1
   - `HasCabin` = 1 if the cabin is known
   - missing `Fare` filled with the median, missing `Embarked` with the most common port
   - `Sex` coded 0/1, `Embarked` turned into 0/1 columns (port C is the baseline)
   - `Name`, `Ticket`, `Cabin` dropped
4. **Model.** Logistic regression on `Pclass`, `Sex`, `IsBoy`, `Age`, `FamilySize`, `Fare`, `HasCabin` and `Embarked`. No penalty, so scikit-learn fits the same model as R's `glm`.
5. **Evaluate.** Accuracy, confusion matrix, precision, recall, F1 and ROC AUC on the training rows and the held-out rows, plus a sex-only baseline.
6. **Predict `test.csv`** and save the predictions.
7. **Cross-validate.** 5-fold cross-validation on all 891 rows, with fill values recomputed inside each fold, as a steadier estimate than one split.

## Results

Same numbers in Python and R.

| Metric | Training (714 rows) | Held-out test (177 rows) | 5-fold CV (891 rows) |
| --- | ---: | ---: | ---: |
| Accuracy | 0.840 | 0.808 | 0.824 (sd 0.034) |
| Precision | 0.817 | 0.766 | |
| Recall | 0.752 | 0.721 | |
| F1 | 0.783 | 0.742 | |
| ROC AUC | 0.875 | 0.883 | 0.870 (sd 0.031) |

- The model beats guessing "women survive, men don't" (0.751 on the held-out rows).
- Training and held-out scores are close, so it isn't badly overfitting. The CV spread (0.78 to 0.87 by fold) shows how much one 177-row split can move the number.
- **Sex** (+3.36 log-odds) and **IsBoy** (+3.50) are the biggest effects. Each step down in class cuts the odds of surviving by about half (−0.79), and larger families did worse (−0.42 per person).
- 164 of the 418 passengers in `test.csv` are predicted to survive.

Predictions are saved to `src/data/predictions_python.csv` and `src/data/predictions_r.csv` (`PassengerId`, `Survived`).

## Design choices

- **Data is mounted, not copied into the image.** The dataset stays out of the repo and the image, and you only need to drop the CSVs in `src/data`.
- **Pinned versions.** Python packages are pinned in `requirements.txt`. The R image (`rocker/r-ver:4.6.1`) installs packages from a fixed, dated CRAN snapshot, so the versions don't change between builds.
- **Deterministic split, no random seed.** The hold-out and CV folds come from a rule on `PassengerId`, so Python and R use the exact same rows.
- **Fill values from training rows only.** Nothing from the held-out rows, the CV check fold or `test.csv` leaks into cleaning.

## Project structure

```
titanic-disaster/
├── README.md
├── CODEOWNERS
├── .gitignore
└── src/
    ├── data/                  # Kaggle CSVs go here (not tracked by git)
    ├── python/
    │   ├── Dockerfile         # python:3.13.13-slim
    │   ├── requirements.txt   # numpy, pandas, scikit-learn (pinned)
    │   └── titanic.py
    └── r/
        ├── Dockerfile         # rocker/r-ver:4.6.1
        ├── install_packages.R # readr, dplyr
        └── titanic.R
```

## Clean up

```bash
docker rmi titanic-python titanic-r
```

## Troubleshooting

- **`Cannot connect to the Docker daemon`**: open Docker Desktop and wait until it says it's running.
- **The script can't find `data/train.csv`**: the CSVs aren't in `src/data`, or `docker run` was run from a different folder. `cd` into the repo root and check `ls src/data`.
- **Kaggle won't let you download**: join the competition and accept its rules first (step 2).
