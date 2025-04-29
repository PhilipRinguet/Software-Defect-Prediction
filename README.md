# Software Defect Prediction Project

This project implements Cross-Project Software Defect Prediction (CPDP) using machine learning techniques. It uses data from NASA's Metrics Data Program (MDP) to predict software defects in target projects by leveraging data from multiple source projects.

## Project Structure

```
├── data
│   ├── raw            # Raw ARFF files from NASA MDP
│   └── processed      # Processed and transformed data
├── models             # Trained and serialized models
├── notebooks          # Jupyter notebooks for exploration
├── reports
│   └── figures        # Generated plots and visualizations
├── src               
│   ├── data          # Data loading and preprocessing
│   ├── features      # Feature engineering
│   ├── models        # Model training and prediction
│   └── visualization # Visualization utilities
└── tests             # Unit tests
```

## Installation

1. Clone this repository
2. Create a virtual environment (recommended)
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The main script provides a command-line interface to run the complete pipeline:

```bash
python main.py --model-type svm --optimize --visualize
```

Key Arguments:
- `--model-type`: Choose from 'logistic', 'svm', 'balanced_rf', 'xgboost'
- `--optimize`: Enable hyperparameter optimization
- `--visualize`: Generate visualizations
- `--resampling-method`: Choose from 'smote' or 'adasyn' for handling class imbalance
- `--n-features`: Number of features to select

For full list of options:
```bash
python main.py --help
```

## Features

- Data preprocessing with Yeo-Johnson transformation
- Feature selection using mutual information
- Class imbalance handling with SMOTE/ADASYN
- Multiple model options (SVM, Balanced Random Forest, XGBoost)
- Hyperparameter optimization using Optuna
- Experiment tracking with MLflow
- Comprehensive visualizations

## Model Performance

Models are evaluated using:
- Precision
- Recall
- F-beta Score (β=2)
- PR-AUC

Performance metrics are logged using MLflow and can be viewed in the MLflow UI:
```bash
mlflow ui
```

## Testing

Run the test suite:
```bash
python -m unittest discover tests
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Model Comparison

### Running Model Comparisons

1. Run the comparison script to evaluate all models:
   ```bash
   python src/models/run_model_comparison.py
   ```
   This will:
   - Train all model configurations (Logistic Regression, SVM variants, Random Forest, XGBoost)
   - Evaluate models using multiple metrics (Precision, Recall, F-beta Score, PR-AUC)
   - Save results to the reports directory
   - Log runs in MLflow for experiment tracking

2. View results and visualizations:
   ```bash
   jupyter notebook notebooks/model_comparison_results.ipynb
   ```
   This notebook provides:
   - Comparative analysis of model performance
   - Visualization of metrics across models
   - Model ranking and recommendations

### Model Configurations Tested:
- Logistic Regression with feature selection
- SVM with balanced class weights
- SVM with SMOTE resampling
- SVM with ADASYN resampling
- Balanced Random Forest
- XGBoost

### Evaluation Metrics:
- Precision
- Recall
- F-beta Score (β=2)
- PR-AUC (Precision-Recall Area Under Curve)