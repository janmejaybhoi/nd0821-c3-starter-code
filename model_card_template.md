# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A logistic regression model was used as a binary classification model for 
salary in the Census data
- Model type: LogisticRegression
- Model date: 10.04.2022
- Parameters: max_iter=1000
- OneHot encoding : For Categorical features
- Standard scaler : For Continuous features
- LabelBinarizer :  Binarize labels in a one-vs-all fashion for target variable

## Intended Use
Intended use: The primary intended use is to determine somone's salary range based on limited information about them. Hobbyist, not intended to be used in any official capacity. The project cover DVC, FastAPI, Regression task and Heroku.

## Training Data
The Census data provided by UCI : https://archive.ics.uci.edu/ml/datasets/census+income
The model train on 80 % of the clean data.(clean_cencus.csv)

## Evaluation Data
Only 20% of clean data is used for Evaluation and testing.

## Metrics
precision=0.7486832204665161
recall=0.6191661481020535
fbeta=0.6777929155313351

For slice performances two additional metrics were added:
- TNR: True Negative Rate
- NPR: Negative Predictive Value

## Ethical Considerations
This model was trained on Census data provided by UCI. This must kept in consideration 
when working with the uploaded model. This project was done for educational uses 
and meant to provide a proof of concept for deploying machine learning models.

## Caveats and Recommendations
Alternative models could be tested to further improve the performance.
Additionally, hyper-parameter tuning was not considered here as the model which
was used is a logistic regression model. However, hyper-parameter tuning should
be considered when other models are used, such as random forests. Then, the data
should be split into training, evaluation, and test. If the used model is not 
computationally expensive, K-fold cross-validation could be used on the training 
data. 

