# Cross-Project Software Defect Prediction

## Overview
This project aims to predict software defects across different projects using machine learning techniques. The goal is to assess the transferability of defect prediction models trained on one dataset and applied to another. The project involves data preprocessing, feature engineering, model training, and evaluation using performance metrics relevant to software defect prediction.

## Dataset
We use the JM1 and KC1 datasets, which is are small and imbalanced datasets for defect prediction. Data preprocessing includes:
- Handling missing values
- Normalization
- Feature selection
- Addressing class imbalance using techniques such as SMOTE and ADASYN

## Methodology
1. **Data Preprocessing**
   - Data cleaning and transformation
   - Feature engineering
   - Class balancing
2. **Model Selection & Training**
   - Logistic Regression, Support Vector Machine (SVM) with RBF Kernel, RandomForrest, XGBoost
   - Hyperparameter tuning using Optuna
   - Joint optimization of SMOTE and ADASYN
3. **Evaluation Metrics**
   - Precision, Recall, F1-Score
   - ROC-AUC
   - Confusion Matrix

## Repository Structure
```
├── data
│   ├── raw
│   ├── processed
├── notebooks
│   ├── exploratory_data_analysis.ipynb
│   ├── model_training.ipynb
├── src
│   ├── data_processing.py
│   ├── train_model.py
│   ├── evaluate.py
├── results
│   ├── model_performance_metrics.csv
├── README.md
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/PhilipRinguet/Software-Defect-Prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd cross-project-defect-prediction
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

## Usage
Run data preprocessing:
```bash
python src/data_processing.py
```
Train the model:
```bash
python src/train_model.py
```
Evaluate the model:
```bash
python src/evaluate.py
```

## Results & Analysis
- Model performance is analyzed using precision, recall, and F1-score.
- Performance comparison across different datasets is provided.
- The impact of hyperparameter tuning is visualized.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository
2. Create a new branch (`feature-new`)
3. Commit your changes
4. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any questions, feel free to reach out:
- **Email:** philipringuet@gmail.com
- **GitHub:** [your-username](https://github.com/PhilipRinguet)

