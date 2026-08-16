# CS601T - Deep Learning - Assignment 1
### Group 19

Perceptron implemented from scratch (only NumPy for matrix operations — 
no ML/DL libraries like sklearn, PyTorch, TensorFlow used anywhere) with 
manual gradient descent, applied to classification and regression tasks.

## Project Structure

utils/
perceptron.py - Core Perceptron class (sigmoid, tanh, linear activations + gradient descent)
metrics.py - Confusion matrix, precision, recall, F1-score (manual implementation)
data_utils.py - Train/test split (70-30) and standardization

classification/
classify_ls.py - Linearly Separable dataset, one-against-one, both activations
classify_nls.py - Non-Linearly Separable dataset, one-against-one, both activations

regression/
regress_univariate.py - 1D input regression using linear perceptron
regress_bivariate.py - 2D input regression using linear perceptron

plots/ - All generated plots (error curves, decision regions, RMSE scatter, etc.)

## How to Run

Data folder (`Group19/`) should be placed in the root directory, matching 
the structure of the dataset provided by the course.

```bash
cd classification
py classify_ls.py
py classify_nls.py

cd ../regression
py regress_univariate.py
py regress_bivariate.py
```

## Key Results

| Task | Activation | Test Accuracy / %RMSE |
|---|---|---|
| LS Classification | Sigmoid | 100% |
| LS Classification | Tanh | 100% |
| NLS Classification | Sigmoid | 55.6% |
| NLS Classification | Tanh | 55.6% |
| Univariate Regression | Linear | 4.00% RMSE |
| Bivariate Regression | Linear | 94.47% RMSE |

## Key Observations

- **LS dataset** is perfectly linearly separable, so both activation 
  functions achieve 100% accuracy using one-against-one perceptrons.
- **NLS dataset** forms concentric circular clusters, which cannot be 
  separated by straight-line decision boundaries. This demonstrates the 
  fundamental limitation of a single-layer perceptron on non-linear data.
- **Univariate regression** data is near-linear, so the linear perceptron 
  fits it well (low RMSE).
- **Bivariate regression** data has a non-linear underlying relationship, 
  so a linear perceptron cannot fit it well (high RMSE).

Random seed = 1 is used throughout for reproducibility of train/test splits.