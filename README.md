# MOVMUS-MODELS

This repository contains models and data analysis tools for muscle movement and grasp recognition, primarily focused on binary classification (cylindrical grasp against all others) and contextual model evaluation. The models are designed to handle large datasets, and the repository provides utilities for data preparation, testing, and training.

The data used for this project is obtained from: https://www.nature.com/articles/s41597-023-02723-w

**DISCLAIMER**: To obtain the prepared data, please contact the owner of this repository.

## Repository Structure

- **Folders**:
  - `MODELS_1000`: models only processing sEMG data.
  - `MODELS_SC`: models based on subject contextual information (both with only context and hybrid models).
  - `MODELS_TC`:  models based on task contextual information (both with only context andhybrid models).

- **Jupyter Notebooks**:
  - `0_id_HISTOGRAM_FIRSTCUT.ipynb` & `1_id_HISTOGRAM.ipynb`: Initial exploratory analysis, generating histograms and performing first-cut data analysis.
  - `2_SECOND_CUT.ipynb`: Data cleaning (deleting all of the null data).
  -  `3_DATA_ANALYSIS.ipynb` & `4_DATA_PREPARATION.ipynb`: Exploratory data analysis and splits for training-validation-test.
  - `5_BINARY_MODEL_1000.ipynb`, `6_BINARY_MODEL_2000.ipynb`, `7_BINARY_MODEL_3000.ipynb`, etc.: Training binary classification models on datasets with 1000, 2000, 3000, and 6337 samples, respectively.
  - `9_context_no_emg.ipynb`: Training contextual models.
  - `10_context_task_model.ipynb` & `11_context_subject_model.ipynb`: Training hybrid models based on task or subject context.
  - `12_TESTS.ipynb`, `13_TESTS_contextual.ipynb` & `14_TESTS_contextual(noshape).ipynb`: Various testing frameworks to evaluate model performance under different contextual scenarios. (13 and 14 include misclassification study at the end of the notebook)

- **Parquet Files**:
  - `TFM_METADATA.parquet`, `TFM_METADATA_FINAL.parquet`, `TFM_METADATA_FIRSTCUT.parquet`: Metadata for each of the different analysis in notebooks 0 to 4.

- **Scripts**:
  - `movmus.py` & `movmus2.py`: Scripts providing utility functions to manage data files, including saving/loading `.npz` data, label preparation, and visualization.

## Key Features

- **Data Processing**: Tools for loading, saving, and processing muscle movement data, including functions to handle metadata in `.npz` and `.parquet` formats.
  
- **Modeling**: Several models designed for binary classification tasks with a focus on identifying specific muscle grasp types.

- **Contextual Evaluation**: Tools for running contextual tests that consider different variables (e.g., task, subject) when evaluating model performance.

## Getting Started

1. **Dependencies**: Ensure you have the following Python packages installed:
    - `pandas`
    - `numpy`
    - `tensorflow`
    - `scikit-learn`
    
   You can install them using:
   ```bash
   pip install pandas numpy tensorflow scikit-learn
   ```
   Alternatively, you can install all dependencies at once using the `requirements.txt` file:

   ```bash
    pip install -r requirements.txt
   ```

2. **Runing the models**: Open the relevant Jupyter notebooks (e.g., `5_BINARY_MODEL_1000.ipynb`) and follow the instructions for running the model training and evaluation pipelines.

3. **Metadata**: Use the provided parquet files for metadata and labels to ensure the models work correctly with the appropriate data formats.
