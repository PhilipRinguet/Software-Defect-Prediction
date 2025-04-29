# **Cross Project Software Defect Prediction with Machine Learning**

This project aims to predict software defects using machine learning models. We use historical defect data to train various classification models to predict defective modules in software systems, improving software quality and reducing maintenance costs.

## **Introduction**

For any software company its extremely important to manage the development phases of its projects as effectively as possible, which includes minimizing the potential bugs that could be costly and sometimes lead to recalling launched updates. Within each company, code review and testing procedures are set in place. However, manual reviews are costly and cannot cover the entirety of a project effectively. This is where Software Defect Prediction becomes crucial, assisting project managers as well as software professionnals in prioritising the most error prone sections of software to be reviewed and providing some likelyhood metrics for future potential defects.

## **Authors and Acknowledgment**

Project was created by Christian-Philippe Ringuet.

## **Instructions**

cpsdp.yml file in source folder can be ran to reproduce exact environment used.

The whole project can be ran as a single uninterrupted Jupyter Notebook file, containing all models, optimisations, results and analysis. 
MLFlow saves test runs and results with every run locally in the mlruns folder in the same directory as the Software_Defect_Prediction.ipynb file. 
The best pipeline with a model is also saved in the same directory: pipeline_svm_rbf_adasyn.pkl. 
Datasets are part of the datasets folder also in the source directory.