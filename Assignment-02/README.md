# Assignment 2: Experiment Tracking

**Course**: Applied Machine Learning
**Institute**: Chennai Mathematical Institute
**Student**: Jugaad Singh Sohal (MDS202421)

## Problem Statement

Extend the SMS spam classification workflow from Assignment 1 with two MLOps tools: DVC for data version control and MLflow for model experiment tracking. The goal is to demonstrate reproducible data management and systematic experiment tracking on the same classification task.

## Dataset

**UCI SMS Spam Collection**
- Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- Total samples: 5,572 messages
- Class distribution: ~87% ham, ~13% spam (imbalanced)
- Split: 70/15/15 stratified (train/validation/test)

## Approach

### Data Version Control with DVC (`prepare.ipynb`)

DVC tracks large data files separately from git, storing only lightweight pointer files (`.dvc`) in the repository while the actual data lives in local or remote storage.

1. Initialize DVC in the project subdirectory (`dvc init --subdir`)
2. Download and load the raw SMS dataset, save as `raw_data.csv`
3. Track raw data with `dvc add data/raw_data.csv`
4. **Version 1** (random_state=42): Stratified 70/15/15 split, track splits with DVC, tag as `v1`
5. **Version 2** (random_state=123): Re-split with different seed, update DVC tracking, tag as `v2`
6. Verify version switching with `git checkout <tag> -- data/*.dvc` followed by `dvc checkout`
7. Compare distributions across versions to confirm stratification is preserved but samples differ

### Model Experiment Tracking with MLflow (`train.ipynb`)

MLflow logs hyperparameters, metrics, and trained models for each experiment run. Using a SQLite backend enables the model registry for versioned model storage. Models are trained on both data versions to demonstrate how data versioning integrates with experiment tracking.

1. **Feature Extraction**: TF-IDF vectorization (5000 features, English stop words removed)
2. **Benchmark Models** with GridSearchCV (5-fold CV, scoring=average_precision):
   - Logistic Regression (C, penalty, class_weight)
   - Multinomial Naive Bayes (alpha, fit_prior)
   - Linear SVM (C, loss, class_weight)
3. **Version 1 Training**: Switch to v1 data via DVC, train all 3 models, log to MLflow and register as model version 1
4. **Version 2 Training**: Switch to v2 data via DVC, train all 3 models, log to MLflow and register as model version 2
5. **Cross-version Comparison**: Compare test AUCPR across data versions for each model
6. **Verification**: Query the MLflow REST API and SQLite database to confirm all runs, metrics, and registered models are stored correctly

## Results

### Model Comparison (Test AUCPR)

| Model | V1 (seed=42) | V2 (seed=123) | Difference |
|-------|-------------|---------------|------------|
| Logistic Regression | 0.9642 | 0.9737 | +0.0095 |
| Naive Bayes | 0.9708 | 0.9706 | -0.0002 |
| Linear SVM | 0.9672 | 0.9748 | +0.0076 |

All models achieve test AUCPR above 0.96 on both data versions. Performance differences between versions are small (< 1%), indicating robustness to the train/test split. Naive Bayes is the most stable across versions.

### Best Hyperparameters

| Model | V1 Parameters | V2 Parameters |
|-------|--------------|--------------|
| Logistic Regression | C=100, penalty=l2, solver=liblinear | C=100, penalty=l2, solver=liblinear |
| Naive Bayes | alpha=0.1, fit_prior=True | alpha=0.1, fit_prior=True |
| Linear SVM | C=1, loss=hinge | C=1, loss=squared_hinge |

Hyperparameters are largely consistent across data versions, with only Linear SVM selecting a different loss function.

### DVC Versions

| Version | Tag | Random State | Train | Validation | Test |
|---------|-----|-------------|-------|------------|------|
| v1 | `v1` | 42 | 3900 | 836 | 836 |
| v2 | `v2` | 123 | 3900 | 836 | 836 |

Both versions maintain the same stratified class distribution (~13% spam) but contain different samples.

### MLflow Registry

Three models registered in MLflow model registry, each with 2 versions:
- `logistic_regression` (version 1: v1 data, version 2: v2 data)
- `naive_bayes` (version 1: v1 data, version 2: v2 data)
- `linear_svc` (version 1: v1 data, version 2: v2 data)

6 total runs logged with parameters, metrics, and artifacts stored in `mlflow.db` (SQLite) and `mlruns/`.

## Files

```
Assignment-02/
├── README.md
├── prepare.ipynb            # Data download, splitting, DVC versioning
├── train.ipynb              # Model training on both versions, MLflow tracking
├── .dvc/
│   └── config               # DVC configuration
├── .dvcignore
└── data/
    ├── raw_data.csv.dvc     # DVC pointer to raw dataset
    ├── train.csv.dvc        # DVC pointer to training set
    ├── validation.csv.dvc   # DVC pointer to validation set
    └── test.csv.dvc         # DVC pointer to test set
```

## How to Run

1. Install dependencies: `pip install pandas numpy scikit-learn matplotlib dvc mlflow`
2. Run `prepare.ipynb` to download data, create splits, and set up DVC tracking
3. Run `train.ipynb` to train models on both data versions and log experiments to MLflow
4. View MLflow dashboard: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
5. Switch data versions: `git checkout v1 -- data/*.dvc && dvc checkout`

## Requirements

- pandas
- numpy
- scikit-learn
- matplotlib
- dvc
- mlflow
