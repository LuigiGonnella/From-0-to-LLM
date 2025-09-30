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

- `1) Regression/` - notebooks and data for regression exercises.
- `DL/` - deep learning and transfer learning experiments (CIFAR-10, Flowers, etc.).
- `Python_Basics/` - example scripts and notes for Python fundamentals.

Important files and notebooks:

- `DL/1) Training CIFAR10/CIFAR10_with_TransferLearning.ipynb` - transfer learning notebook for CIFAR-10.
- `1) Regression/Lab8_Scikit-Learn_Regression.ipynb` - regression lab with scikit-learn examples.
- `Python_Basics/` - scripts demonstrating core Python concepts used throughout the course.

The exact structure may evolve; inspect the repository tree for the most recent layout.

## Requirements and environment

We recommend using Python 3.10+ inside a virtual environment. Example quick setup (PowerShell on Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

If a `requirements.txt` is not present, install core dependencies:

```powershell
pip install numpy pandas matplotlib seaborn scikit-learn jupyterlab notebook torch torchvision
```

Suggested versions (examples):

- Python 3.10 or 3.11
- pandas >= 1.5
- scikit-learn >= 1.1

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

## Handling large files and artifacts (Git LFS)

Some datasets or model files in this repository exceed GitHub's 100 MB file size limit. To avoid push errors, either use Git LFS (Large File Storage) or keep large artifacts outside the repository (cloud storage) and provide scripts or instructions to download them.

Recommended Git LFS flow (run locally):

1. Create a backup branch (optional):

```powershell
git checkout -b backup-main
```

2. Install Git LFS (if not already installed):

```powershell
git lfs install
```

3. Tell Git LFS which file patterns to track (examples: `*.tar.gz`, `*.pth`):

```powershell
git lfs track "DL/**/data/*.tar.gz" "DL/**/model*.pth" "*.pth"
git add .gitattributes
git commit -m "Track large files with Git LFS"
```

4. Migrate existing commits to LFS (this rewrites history!):

```powershell
git lfs migrate import --include="DL/**/data/*.tar.gz,DL/**/model*.pth" --include-ref=refs/heads/main
```

5. Force-push the rewritten branch to the remote (coordinate with collaborators):

```powershell
git push origin main --force
```

Alternatives:

- Remove the large files from history with tools like BFG or `git filter-repo` then recommit and push.
- Keep large data and models externally and provide `download_data.ps1` or notebook cells to fetch them on-demand.

Warnings:

- History rewrite affects all clones. After force-push, collaborators must re-clone or reset their local copies.

## How to use the main notebooks

- Example: `DL/1) Training CIFAR10/CIFAR10_with_TransferLearning.ipynb` demonstrates a preprocessing pipeline and a transfer learning experiment. Typical workflow:
  1. Download CIFAR-10 (if not included) using torchvision helpers or a provided script.
  2. Run preprocessing cells (imputation, encoding, scaling). Important: fit preprocessors only on the training set and reuse them to transform validation/test sets.
  3. Run training cells (configure `device`, define `dataloaders`, choose `model`, set hyperparameters).

## Best practices used throughout the notebooks

- Do not overwrite original DataFrames. Use suffixes such as `_transformed` or `_pca`.
- For PCA, inspect `pca.explained_variance_ratio_` and use `pca.n_components_` after fitting. To find the minimum number of components for a variance threshold, compute the cumulative sum and select the first index exceeding the threshold.
- When imputing or encoding, reconstruct DataFrames using `index=X_train.index` to preserve alignment with targets.
- For one-hot encoding with scikit-learn, use `OneHotEncoder(sparse=False)` (or `sparse_output=False` in newer versions) and `get_feature_names_out()` to rebuild column names.
- Prefer `RobustScaler` when data contains many outliers.

## Useful commands

- Create environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- Start Jupyter Lab:

```powershell
jupyter lab
```

- Quick Git LFS commands (remember the risks of history rewrite):

```powershell
git lfs install
git lfs track "*.tar.gz" "*.pth"
git add .gitattributes
git commit -m "Track large files with Git LFS"
git lfs migrate import --include="*.tar.gz,*.pth" --include-ref=refs/heads/main
git push origin main --force
```

## Contributing

Contributions are welcome. Suggested workflow:

1. Open an issue to discuss proposed changes or experiments.
2. Create a feature branch from `main`.
3. Submit a pull request describing what changed and how to test it.

Please avoid committing datasets or model files larger than 100 MB directly to the repository. Use Git LFS or external hosting.

## FAQ

- Why did I get a "pre-receive hook declined" error when pushing?  
  Because at least one file you tried to push is larger than GitHub's 100 MB limit. Use Git LFS or remove large files from history.

- How do I reproduce results?  
  Use the provided virtual environment, install dependencies, open the notebook and run all cells from the beginning. Set random seeds where applicable for deterministic behavior.

## Contact

Author: Luigi Gonnella (or repository owner). See repository metadata for contact details.
