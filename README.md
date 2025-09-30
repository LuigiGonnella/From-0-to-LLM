# From-0-to-LLM

This repository collects the "From 0 to LLM" learning path: a set of notebooks, scripts and experiments designed to teach, step-by-step, the practical techniques used to build, train, and understand language models and related deep learning workflows.

Materials are written with a hands-on approach: you will find Jupyter notebooks with reproducible examples, preprocessing scripts, theoretical notes and sample datasets.

## Mission

The mission of this repository is to guide learners from the basics of machine learning and data preprocessing up to practical techniques for training and fine-tuning deep learning models on real datasets. Key learning goals include:

- Understanding preprocessing fundamentals (imputation, encoding, scaling).
- Performing exploratory data analysis (EDA), detecting outliers and selecting proper transformations.
- Applying dimensionality reduction (PCA) correctly and interpreting results.
- Training baseline classifiers (LogisticRegression, DecisionTree, RandomForest) and evaluating them.
- Experimenting with transfer learning on image datasets (e.g. CIFAR-10, Flowers).
- Managing large datasets and model artifacts, with guidance for Git LFS.

## Repository structure

Top-level folders (partial view):

- `ML/` - machine learning experiments.
- `DL/` - deep learning experiments (CIFAR-10, Flowers, etc.).

Important files and notebooks:

- `DL/1) Training CIFAR10/CIFAR10_with_TransferLearning.ipynb` - transfer learning notebook for CIFAR-10.
- `ML/1) Regression/Lab8_Scikit-Learn_Regression.ipynb` - regression lab with scikit-learn examples.

The exact structure may evolve; inspect the repository tree for the most recent layout.

## Launching notebooks

1. Activate the virtual environment.
2. Start JupyterLab or Jupyter Notebook in the repo root:

```powershell
jupyter lab
# or
jupyter notebook
```

3. Open the notebook you want to run and execute cells from the top to reproduce results.

Tip: For reproducibility, run the setup cells first (those that define paths, seeds and dependencies). Avoid overwriting original DataFrames; use new variable names for transformed data (for example, `X_train_pca`).

## Contact

Author: Luigi Gonnella (or repository owner). See repository metadata for contact details.
