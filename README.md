RetailMind AI
RetailMind AI is an incremental end-to-end AI project developed by adding a new machine learning or deep learning capability each day.

The project currently covers customer churn prediction using classical machine learning and deep learning, followed by a computer vision module for fashion image classification.

Project Overview
The project is developed incrementally:

Day 1: Customer Churn Baseline
Day 2: Model Comparison
Day 3: Hyperparameter Tuning and Cross-Validation
Day 4: Feature Engineering and Ensemble Learning
Day 5: Artificial Neural Network
Day 6: CNN Fashion Image Classification
Day 1: Customer Churn Baseline
Objective
Build the first machine learning pipeline for predicting whether a telecom customer is likely to churn.

Dataset
Telco Customer Churn dataset.

The target variable is:

No = Customer does not churn
Yes = Customer churns
Workflow
Load the dataset using Pandas.
Inspect the data.
Handle missing or invalid values.
Remove unnecessary identifiers.
Separate features and target.
Preprocess numerical and categorical features.
Split the data into training and testing sets.
Train the baseline classifier.
Evaluate its performance.
Save the trained pipeline using Joblib.
This forms the baseline for later RetailMind experiments.

Day 2: Multi-Model Comparison
Objective
Compare multiple classification algorithms and determine which models are promising for churn prediction.

Models
Three models were evaluated:

Decision Tree
Random Forest
Support Vector Machine (SVM)
Results
Model	Accuracy	Precision	Recall	ROC-AUC
Decision Tree	0.7289	0.4896	0.5053	0.6573
SVM	0.7913	0.6418	0.4840	0.7905
Random Forest	0.7779	0.6034	0.4759	0.8162
The models show different trade-offs between accuracy, precision, recall and ROC-AUC.

Day 3: Hyperparameter Tuning and Validation
Objective
Improve the competing models using hyperparameter tuning and cross-validation.

Techniques
Day 3 introduces:

GridSearchCV
Cross-validation
Hyperparameter optimisation
Bias and variance analysis
Learning curves
Tuned Models
SVM
Best configuration included:

C = 0.1
Linear kernel
Balanced class weights
Gamma = scale
Results:

CV Training Recall: 0.8355
CV Validation Recall: 0.8355
Test Accuracy: 0.6941
Test Precision: 0.4575
Test Recall: 0.8209
Test F1: 0.5876
Test ROC-AUC: 0.8093
Random Forest
Results:

CV Training Recall: 0.8237
CV Validation Recall: 0.8054
Test Accuracy: 0.7395
Test Recall: 0.7861
Test F1: 0.6157
Test ROC-AUC: 0.8391
Decision Tree
Results:

CV Training Recall: 0.8249
CV Validation Recall: 0.7933
Test Accuracy: 0.7544
Test Recall: 0.7513
Test F1: 0.6189
Test ROC-AUC: 0.8375
Day 3 Champion
SVM was selected as the recall-focused champion.

Its validation recall and training recall were both 0.8355, indicating a small train-validation gap.

The learning curve also stabilised around a validation recall of approximately 0.8355.

Day 4: Feature Engineering and Ensemble Learning
Objective
Improve churn prediction using engineered features, dimensionality reduction and ensemble learning.

Feature Engineering
Additional domain-based features include:

Tenure groups
Average monthly charges
Charge differences
Month-to-month contract flag
Electronic payment flag
Fibre optic service flag
Automatic payment flag
Number of subscribed services
Security/support flags
Senior customer relationship indicators
Techniques
Day 4 explores:

Feature engineering
PCA / feature reduction
Random Forest
XGBoost
Comparison against the Day 3 champion
Results
Model	CV Validation Recall	Test Accuracy	Test Recall	Test F1	Test ROC-AUC
Day 3 Champion	0.8355	0.6941	0.8209	0.5876	0.8093
Random Forest + Features + PCA	0.7385	0.7601	0.7353	0.6194	0.8343
XGBoost + Features + PCA	0.7244	0.7502	0.7166	0.6036	0.8330
Conclusion
The feature-engineered ensemble models achieved higher ROC-AUC and accuracy in some comparisons, but the Day 3 champion retained the strongest recall.

Because detecting churn customers is an important objective, the Day 3 champion remained a strong candidate rather than automatically replacing it with the more complex models.

Day 5: Artificial Neural Network
Objective
Introduce deep learning and determine whether an Artificial Neural Network provides an improvement over the classical machine learning champion.

Architecture
The ANN uses:

Normalised inputs
Dense hidden layers
ReLU activation
Dropout
Sigmoid output
Binary cross-entropy
Adam optimiser
Early stopping
The Sigmoid output provides a probability of customer churn.

Early Stopping
Validation loss is monitored during training.

Training stops when validation performance stops improving, helping reduce overfitting.

ANN vs Classical Machine Learning
After running the Day 5 training script, record the results below.

Model	Accuracy	Precision	Recall	F1	ROC-AUC
Classical Champion	0.6941	0.4575	0.8209	0.5876	0.8093
ANN	ADD_RESULT	ADD_RESULT	ADD_RESULT	ADD_RESULT	ADD_RESULT
Conclusion
The ANN is not automatically considered better because it uses deep learning.

The ANN should only replace the classical champion if its evaluation demonstrates a meaningful benefit.

Day 6: CNN Fashion Image Classification
Objective
Add the first computer vision capability to RetailMind AI using a Convolutional Neural Network.

The CNN classifies fashion images automatically.

Dataset
Fashion-MNIST is used as the Day 6 prototype because an external fashion-product image dataset was not provided.

Fashion-MNIST contains 10 classes:

T-shirt / top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot
CNN Pipeline
The image classification process is:

Fashion-MNIST Image → Normalisation → Conv2D → Batch Normalisation → Max Pooling → Conv2D → Batch Normalisation → Max Pooling → Flatten → Dense Layer → Softmax Output → Fashion Category

Evaluation
The CNN is evaluated using:

Training accuracy
Validation accuracy
Test accuracy
Training loss
Validation loss
Confusion matrix
Classification report
The confusion matrix helps identify fashion categories that the CNN commonly confuses.

Day 6 Results
Test Accuracy: ADD_DAY6_ACCURACY

Replace ADD_DAY6_ACCURACY with the actual value generated after running the Day 6 CNN.

Generated Files
The Day 6 script generates:

models/fashion_mnist_cnn.keras
models/day6_fashion_mnist_results.csv
models/day6_fashion_mnist_confusion_matrix.csv
models/day6_fashion_mnist_history.csv
Project Structure
RetailMind/
│
├── data/
│
├── models/
│   ├── churn models
│   ├── model comparison results
│   ├── ANN model and results
│   ├── fashion_mnist_cnn.keras
│   └── Day 6 result files
│
├── src/
│   ├── train.py
│   ├── compare_models.py
│   ├── tune_models.py
│   ├── features.py
│   ├── finalize_model.py
│   ├── predict.py
│   ├── train_ann.py
│   └── train_fashion_mnist_cnn.py
│
├── README.md
├── requirements.txt
└── .gitignore
Technologies Used
Python
Pandas
Scikit-learn
Joblib
XGBoost
TensorFlow
Keras
Machine Learning Techniques
The project currently demonstrates:

Data preprocessing
Train-test splitting
Categorical encoding
Feature scaling
Logistic Regression
Decision Trees
Random Forest
Support Vector Machines
GridSearchCV
Cross-validation
Learning curves
Hyperparameter tuning
Feature engineering
PCA
Bagging
Boosting
XGBoost
Artificial Neural Networks
Convolutional Neural Networks
Early stopping
Image classification
Confusion matrices
Running the Project
Install dependencies:

python -m pip install -r requirements.txt
Example classical ML training:

python src/train.py
Day 3 tuning:

python src/tune_models.py
Day 5 ANN:

python src/train_ann.py
Day 6 CNN:

python src/train_fashion_mnist_cnn.py
Current RetailMind AI Progress
Day	Module	Status
Day 1	Churn Baseline	Completed
Day 2	Model Comparison	Completed
Day 3	Hyperparameter Tuning	Completed
Day 4	Feature Engineering + Ensembles	Completed
Day 5	Artificial Neural Network	Completed / Update Results
Day 6	CNN Fashion Classification	Completed / Update Accuracy
Key Learning
RetailMind AI demonstrates an incremental machine learning development workflow:

Baseline → Compare → Tune → Validate → Engineer Features → Ensemble → Deep Learning → Computer Vision

Each project increment is integrated into the same codebase so that new techniques can be evaluated against previous approaches rather than being treated as isolated exercises.