# SDC485 2.5 Performance Assessment: Data Preprocessing

Coursework for SDC485 at ECPI University. The notebook cleans the Wisconsin breast cancer dataset (569 tumor samples, 30 features measured from fine needle aspirate images) and compares two classifiers for predicting a malignant or benign diagnosis, before and after hyperparameter tuning.

## What the notebook does

1. **Loads and explores** `breastCancerDataset.csv`.
2. **Preprocesses**: fills any missing numeric values with the column median, drops duplicate rows, encodes the diagnosis (M = 1, B = 0), standardizes `concavity1` and `concavity3`, and makes a 70/30 train/test split (398 / 171 samples).
3. **Trains an SVM** (`SVC`), then tunes `C` and `kernel` with 5-fold `GridSearchCV`.
4. **Trains XGBoost** (`XGBClassifier`), then tunes `n_estimators`, `learning_rate` and `max_depth` the same way.
5. **Evaluates** each model with accuracy and a classification report, and closes with a written analysis of bias in medical datasets.

## Results (test set)

| Model | Accuracy |
|---|---|
| SVM, default | 0.9357 |
| SVM, tuned (`C=10`, linear kernel) | **0.9708** |
| XGBoost, default | **0.9708** |
| XGBoost, tuned (`learning_rate=0.1`, `max_depth=3`, `n_estimators=100`) | 0.9591 |

Tuning helped the SVM a lot. It did not help XGBoost: the tuned model had the best cross-validation score (0.9673) but scored lower on the held-out test set than the defaults.

## Files

- `Breast_Cancer_Preprocessing_addsmi1720.ipynb`: the notebook
- `breastCancerDataset.csv`: the dataset
- `Lab_Report_2.5_addsmi1720.docx`: the lab report

## Run it

```
pip install pandas scikit-learn xgboost jupyter
jupyter notebook Breast_Cancer_Preprocessing_addsmi1720.ipynb
```

Run the cells top to bottom from the repository folder so the notebook can find the CSV.
