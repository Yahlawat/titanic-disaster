# titanic-disaster

Predicting who survived the Titanic with a logistic regression, run twice: once in a Python Docker container and once in an R Docker container. Each script loads the Kaggle data, cleans it, fits the model and prints every step to the terminal.

## What you need

- [Git](https://git-scm.com/downloads)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/), open and running (`docker info` should not show an error)
- A free [Kaggle](https://www.kaggle.com/) account, to download the data

You don't need Python or R installed. Everything runs inside the containers.

## Repo layout

```
titanic-disaster/
├── README.md
├── CODEOWNERS
├── .gitignore
└── src/
    ├── data/                  # put the Kaggle CSVs here (not tracked by git)
    ├── python/
    │   ├── Dockerfile
    │   ├── requirements.txt   # pinned package versions
    │   └── titanic.py
    └── r/
        ├── Dockerfile
        ├── install_packages.R
        └── titanic.R
```

## Steps to run

Run every command from the repo root (the `titanic-disaster` folder).

### 1. Clone the repo

```bash
git clone https://github.com/Yahlawat/titanic-disaster.git
cd titanic-disaster
```

### 2. Download the data

The dataset is not in this repo, so download it from Kaggle:

1. Go to the [Titanic competition data page](https://www.kaggle.com/competitions/titanic/data) and sign in.
2. Join the competition and accept its rules. Kaggle won't let you download the data until you do.
3. Click **Download All**. This gives you `titanic.zip`.
4. Unzip it into `src/data`:

```bash
unzip ~/Downloads/titanic.zip -d src/data
```

On Windows, right-click `titanic.zip` > **Extract All** and pick the `src/data` folder.

Check that the files are in the right place:

```bash
ls src/data
```

You should see `train.csv`, `test.csv` and `gender_submission.csv`. Only `train.csv` and `test.csv` are used.

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

The first R build takes a minute or two while it installs packages. Builds after that are cached.

**Windows PowerShell:** use `${PWD}` instead of `$(pwd)` in the `docker run` lines, e.g.

```powershell
docker run --rm -v "${PWD}/src/data:/app/data" titanic-python
```

### What the commands do

- `docker build -t <name> src/<language>` builds an image from the Dockerfile in that folder. It installs the pinned packages and copies in the script.
- `docker run --rm -v "$(pwd)/src/data:/app/data" <name>` runs the script. The `-v` part shares your `src/data` folder with the container, so the script can read the CSVs and save its predictions back to your machine. `--rm` deletes the container when it's done.

## What you'll see

Both scripts print the same steps:

1. Load `train.csv` (891 rows, 12 columns) and show the first rows.
2. Explore it: column types, missing values per column, survival rate overall, by sex and by class.
3. Hold out 20% of `train.csv` as a test set (split so both parts have the same survival rate).
4. Clean the data and print every change:
   - add `HasCabin` (1 if the cabin is known)
   - drop `Name`, `Ticket` and `Cabin`
   - fill missing `Age` and `Fare` with the median and missing `Embarked` with the most common port, all worked out from the training rows only
   - code `Sex` as 0/1 and turn `Embarked` into 0/1 columns
5. Fit a logistic regression on `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `HasCabin` and `Embarked`, and print the coefficients.
6. Print the accuracy on the training rows.
7. Load `test.csv`, clean it the same way and predict who survived.
8. Print the accuracy on the held-out rows.

Predictions for `test.csv` are saved to `src/data/predictions_python.csv` and `src/data/predictions_r.csv`.

### Results

| | Python | R |
| --- | --- | --- |
| Training accuracy | 0.805 | 0.794 |
| Held-out (test) accuracy | 0.804 | 0.836 |
| Guessing women survive, men don't (held-out) | 0.777 | 0.814 |
| Predicted survivors in `test.csv` | 158 of 418 | 159 of 418 |

### Why test accuracy uses a hold-out

Kaggle's `test.csv` has no `Survived` column, so accuracy can't be measured on it directly. Both scripts hold out 20% of `train.csv` before any cleaning, fit on the other 80%, and report accuracy on the held-out rows as the test accuracy. They still predict `test.csv` and save the predictions.

Python and R pick a different random 20%, so their numbers differ a bit. The R held-out rows happen to be easier: even the women-survive guess scores higher on them.

## Clean up

To remove the images when you're done:

```bash
docker rmi titanic-python titanic-r
```

## Troubleshooting

- **`Cannot connect to the Docker daemon`**: open Docker Desktop and wait until it says it's running.
- **The script can't find `data/train.csv`**: the CSVs aren't in `src/data`, or you ran `docker run` from a different folder. `cd` into the repo root and check `ls src/data`.
- **Kaggle won't let you download**: join the competition and accept its rules first (step 2).
