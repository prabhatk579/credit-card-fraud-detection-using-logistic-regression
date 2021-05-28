<h1 align = center>Credit Card Fraud Detection using Logistic Regression</h1>

## Requirements:
- pandas
- numpy
- sklearn
- matplotlib
- seaborn

In this project we try to detect credit card fraud using Logistic Regression also we preprocessing the data.

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

If we oversample our data so inclusion of almost **284000** dummy elements will surely affect our outcome by a huge margin and it will be hugely biased as non-fradulant so undersampling is a much better approach to get an optimal and desired outcome.

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

### Precision, Recall, F1-Score, Mean Absolute Error, Mean Percentage Error and Mean Squared Error
We find the Precision, Recall, F1-Score, Mean Absolute Error, Mean Percentage Error and Mean Squared Error using the following syntax - 
```
from sklearn.metrics import classification_report,mean_absolute_error,mean_squared_error,r2_score
report= classification_report(y_train,pred)
print(report)
mean_abs_error = mean_absolute_error(y_train,pred)
mean_abs_percentage_error = np.mean(np.abs((y_train - pred) // y_train))
mse= mean_squared_error(y_train,pred)
r_squared_error = r2_score(y_train,pred)
print("Mean absolute error : {} \nMean Absolute Percentage error : {}\nMean Squared Error : {}\nR Squared Error  {}".format(mean_abs_error,mean_abs_percentage_error,mse,r_squared_error))
```

## Undersampling and Synthetic Minority Oversampling Technique (SMOTE) approach
To improve our performance, we use combination of undersampling and SMOTE on our dataset.
Syntax is as follows:
```
from imblearn.over_sampling import SMOTE
oversample=SMOTE()
X_train,y_train= oversample.fit_resample(X_train,y_train)
```
### Applying Logistic regression on training model with Undersampling and SMOTE.
We apply logistic regression on our dataset as usual. After applying logistic regression in most of the cases we observe that in most of the cases our accuracy is improved. Confusion matrix is as follows - 
<p align = center><img src = https://user-images.githubusercontent.com/54438860/119935609-9d201300-bf3c-11eb-98d3-21d4320ab67b.png></p>
<h5 align = center>Fig 4: Confusion matrix after Undersampling and SMOTE</h5>

## Hyperparameter Tuning
To improve our accuracy further, we tune the hyper parameter.
Syntax is as follows - 
```
classifier_b = LogisticRegression(class_weight={0:0.6,1:0.4})
classifier_b.fit(X_train,y_train)
pred_b = classifier_b.predict(X_test_all)
print(classifier_b.score(X_test_all,y_test_all))
```
The confusion matrix of the **Testing** model is as follows:
<p align = center><img src = https://user-images.githubusercontent.com/54438860/119936048-61397d80-bf3d-11eb-9946-0b44571e8418.png></p>
<h5 align = center>Fig 5: Confusion Matrix after Hyperparameter tuning on the testing model</h5>
