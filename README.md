<h1 align = center>Credit Card Fraud Detection uisng Logistic Regression</h1>

## Requirements:
- pandas
- numpy
- sklearn
- matplotlib
- seaborn
- intertool

In this project we tries to detect credit card fraud using Logistic Regression also we preprocessing the data.

Database used is [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle

## Data Visualzation
We start by loading the data into the jupyter notebook. After loading the data, we convert the data into a data frame using the pandas to make it more easier to handel.
After loading the data, we visualize the data. First we need to know how our data looks so we use `dataframe.head()` to visualize the first 5 rows of the data also we need to know how our data is distributed so we plot our data.

<p align='center'><img src = 'https://user-images.githubusercontent.com/54438860/119931359-19632800-bf36-11eb-949e-3318c7e9fe54.png'></p>
<h5 align = 'center'> Fig 1: Frauds hppened with respect to the time frame and their respective amounts.</h5>

### Correlation of features
Using `dataframe.corr()`, we find the Pearson, Standard Correlation Coefficient matrix.
<p align = 'center'><img src = 'https://user-images.githubusercontent.com/54438860/119931225-d7d27d00-bf35-11eb-81e4-6bad164137ab.png'></p>
<h5 align = 'center'>Fig 2: Correlation of the futures</h5>

## Data Selection
Since the data is `highly Unbalanced` We need to undersample the data.

**Why are we undersampling instead of oversampling?**

We are undersampling the data because our data is highly unbalanced. The number of transactions which are not fradulent are labeled as 0 and the trancactions whoch are fradulent are labeled as 1.

The number of non fraudulent transactions are **284315** and the number of fradulent transactions are **492**.

If we oversample our data so inclusion of almost **284000** dummy elements will surely affect our outcome by a huge margin and outcoem will be hugely biased as non fradulant so undersampling is a much better approach to get an optimal and desired outcome.

## Confusion Matrix
We create a user defined function for the confusion matrix or we can use `confusion_matrix` from `sklearn.matrics` library.

# Applying Logistic Regression
We train our model using `LogisticRegression` from `sklearn.linear_model`.
The syntax is as follows:
```
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
pred = classifier.predict(X_train)
print(classifier.score(X_train,y_train))
```
We get accuracy of our training model more than 95% most of the time with random samples.
The confusion matrix is as follows: 
<p align = center><img src = https://user-images.githubusercontent.com/54438860/119933900-ce4b1400-bf39-11eb-964d-1d853a450eee.png></p>
<h5 align = center>Fig 3: Confusion matrix of training model</h5>
