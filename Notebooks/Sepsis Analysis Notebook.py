#!/usr/bin/env python
# coding: utf-8

# # Analyzing Patterns and Predictors for Sepsis Incidence.

# ### Introduction 
# Sepsis, a life-threatening condition arising from infection, poses a significant global healthcare challenge. To better comprehend sepsis occurrence, researchers are turning to patient data analysis for uncovering hidden patterns and predictors. By harnessing advanced data analytics techniques and exploring diverse parameters such as vital signs, medical history, and demographic information, we aim to identify early warning signs and risk factors for sepsis development. This knowledge holds immense potential for developing risk stratification models, early detection systems, and targeted interventions, ultimately leading to improved patient outcomes and optimized sepsis management protocols.

# ### Hypothesis

# Null Hypothesis: There is no significant association between body mass index (BMI) values (M11) and the risk of sepsis.
# 
# Alternate Hypothesis: There is a significant association between body mass index (BMI) values (M11) and the risk of sepsis.

# In[62]:


#importing necessary packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import scipy.stats as stats
from scipy.stats import skew
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, roc_curve, auc, roc_auc_score
import pickle
from joblib import dump
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.linear_model import SGDClassifier
sns.set()
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score



# In[2]:


#loading of Dataset
train=pd.read_csv('Paitients_Files_Train.csv')
test=pd.read_csv('Paitients_Files_Train.csv')


# ### checking the information of the dataset

# In[3]:


train.head()
test.head()


# In[4]:


test.head()



# In[5]:


#printing the shape of the dataset
print("Shape of train dataset:", train)
print("Shape of test dataset:", test)


# In[6]:


# Print the information of the train dataset
print("Train Dataset Information:")
print(train.info())

# Print the information of the test dataset
print("\nTest Dataset Information:")
print(test.info())


# In[7]:


# Print the information of the train dataset
print("Train Dataset datatypes:")
print(train.dtypes)

# Print the information of the test dataset
print("\nTest Dataset dataypes:")
print(test.dtypes)


# In[8]:


# Check for null values in the train dataset
print("Null values in train dataset:")
print(train.isnull().sum())

# Check for null values in the test dataset
print("\nNull values in test dataset:")
print(test.isnull().sum())


# In[9]:


# Check for duplicated values in the train dataset
print("Duplicated values in train dataset:")
print(train.duplicated().sum())

# Check for duplicated values in the test dataset
print("\nDuplicated values in test dataset:")
print(test.duplicated().sum())


# In[10]:


train.describe()


# In[11]:


test.describe()


# ### Univariant Analysis 

# In[12]:


#taking a look at the frequency distribution 
print("\nBD2:")
print(train['BD2'].describe())
plt.hist(train['BD2'], bins=10)
plt.xlabel('BD2')
plt.ylabel('Frequency')
plt.show()


# Distribution: The data appears to be positively skewed (right-skewed) as the mean (0.481187) is greater than the median (0.383000). This suggests that there may be a few higher values that are pulling the mean towards the right.

# In[13]:


#taking a look at the distribution of the body mass index
print("\nM11:")
print(train['M11'].describe())
plt.hist(train['M11'], bins=10)
plt.xlabel('M11')
plt.ylabel('Frequency')
plt.show()


# Distribution: The data appears to have a relatively symmetrical distribution as the mean (31.920033) is close to the median (32.000000). There is no strong evidence of skewness in the distribution.

# In[14]:


print("\nPL (Blood Pressure):")
print(train['PL'].describe())
plt.boxplot(train['PL'])
plt.xlabel('Blood Pressure')
plt.ylabel('Values')
plt.show()


# It indicates that the majority of the data falls within the interquartile range, with some potential outliers present at both the lower and upper ends of the distribution.

# ### Bivariant Analysis

# In[15]:


# Calculate correlation coefficient between PL and TS
correlation = train['PL'].corr(train['TS'])
print("Correlation coefficient between PL and TS:", correlation)


# In summary, based on the correlation coefficient, there is a weak positive relationship between PL and TS. It suggests that as PL increases, there tends to be a slight tendency for TS to increase as well, but other factors may play a significant role in determining the cholesterol levels.

# In[16]:


# Create scatter plot of PL vs TS
plt.scatter(train['PL'], train['TS'])
plt.xlabel('PL (Blood Pressure)')
plt.ylabel('TS (Total Cholesterol)')
plt.title('Scatter Plot: PL vs TS')
plt.show()


# In[17]:


#sepsis Distribution across insurance
# Count plot for "Insurance"
sns.countplot(data=train, x='Insurance')

# Set labels
plt.xlabel('Insurance')
plt.ylabel('Count')

# Set title
plt.title('Distribution of Insurance')

# Calculate percentage distribution
total = len(train['Insurance'])
percentages = train['Insurance'].value_counts(normalize=True) * 100

# Add data labels and percentage annotations
for p, percentage in zip(plt.gca().patches, percentages):
    count = p.get_height()
    percentage_label = f'{percentage:.1f}%'
    plt.gca().annotate(f'{count}\n{percentage_label}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')


plt.show()


# In[18]:


# Numerical Variables - Histograms
numerical_vars = ['PRG', 'PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age']
for var in numerical_vars:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=train, x=var, hue='Sepssis', multiple='stack', kde=True)
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.title(f'{var} Distribution by Sepssis')
    
    plt.tight_layout()
    plt.show()


# In[19]:


# Perform ANOVA for M11 across different groups of Insurance
model = ols('M11 ~ Insurance', data=train).fit()
anova_table = sm.stats.anova_lm(model)
print("ANOVA results:\n", anova_table)


# Based on these results, the p-value (0.512276) is greater than the typical significance level (e.g., 0.05), indicating that there is no significant effect of "Insurance" on the variable being analyzed. The null hypothesis, which states that there is no difference in means between the insurance groups, cannot be rejected.

# In[20]:


# Create box plot of TS across different categories of Sepssis
sns.boxplot(x='Sepssis', y='TS', data=train)
plt.xlabel('Sepssis')
plt.ylabel('TS (Total Cholesterol)')
plt.title('Box Plot: TS across Sepssis')
plt.show()


# In[21]:


# Categorical Variables - Bar plots
categorical_vars = ['Insurance']
for var in categorical_vars:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=train, x=var, hue='Sepssis')
    plt.xlabel(var)
    plt.ylabel('Count')
    plt.title(f'{var} Distribution by Sepssis')

    # Calculate percentage distribution
    total = len(train['Sepssis'])
    percentages = train['Sepssis'].value_counts(normalize=True) * 100

    # Add data labels and percentage annotations
    for p, percentage in zip(plt.gca().patches, percentages):
        count = p.get_height()
        percentage_label = f'{percentage:.1f}%'
        plt.gca().annotate(f'{count}\n{percentage_label}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


# ### Multivariant Analysis

# In[22]:


# sepsis count by age range
age_ranges = pd.cut(train['Age'], bins=[20, 30, 40, 50, 60, 70, 80, 90, 100])
grouped_data = train.groupby(age_ranges)

# Calculate the count of 'Sepssis' for each age range
count_sepsis_by_age = grouped_data['Sepssis'].count()

# Plotting the count of 'Sepssis' for each age range
ax = count_sepsis_by_age.plot(kind='bar', xlabel='Age Range', ylabel='Sepssis Count', title='Sepssis Count by Age Range')
plt.xticks(rotation=45)

# Add data labels
for p in ax.patches:
    ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')

plt.show()


# In[23]:


numerical_vars = ['PRG', 'PL', 'PR']
sns.pairplot(data=train, vars=numerical_vars, hue='Sepssis', kind='scatter')
plt.show()


# In[24]:


numerical_vars = ['SK', 'TS', 'M11']
sns.pairplot(data=train, vars=numerical_vars, hue='Sepssis')


# ### Hypothesis Testing
# Null Hypothesis: There is no significant association between body mass index (BMI) values (M11) and the risk of sepsis.
# 
# Alternate Hypothesis: There is a significant association between body mass index (BMI) values (M11) and the risk of sepsis.

# In[25]:


from scipy.stats import ttest_ind

# Separate the samples based on sepsis occurrence
positive_sepsis = train[train['Sepssis'] == 'Positive']
negative_sepsis = train[train['Sepssis'] == 'Negative']

# Perform the two-sample t-test
t_statistic, p_value = ttest_ind(positive_sepsis['M11'], negative_sepsis['M11'])

# Print the t-test statistic and p-value
print('T-test statistic:', t_statistic)
print('P-value:', p_value)


# The p-value is significantly smaller than the commonly used significance level of 0.05, indicating strong evidence against the null hypothesis. Therefore, we can reject the null hypothesis and conclude that there is a significant association between body mass index (BMI) values (M11) and the risk of sepsis.
# 
# Furthermore, since the T-test statistic is positive, it suggests that patients with higher body mass index (BMI) values have a lower risk of sepsis. This provides support for the alternative hypothesis that patients with higher BMI values are less likely to develop sepsis.
# 
# In summary, the analysis suggests that there is a significant association between body mass index (BMI) values and the risk of sepsis, with higher BMI values being associated with a lower risk of sepsis.

# ## Feature Processing & Engineering 
# we would clean process and do feature creation

# ### checking for data imbalnce and outliers 

# In[26]:


# Check data imbalance
class_counts = train['Sepssis'].value_counts()
print(class_counts)

# Visualize data imbalance
plt.figure(figsize=(8, 6))
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('Sepssis')
plt.ylabel('Count')
plt.title('Distribution of Sepssis')
plt.show()



# we can infer that the dataset is imbalance and we would need to do balancing or choose models that can deal with imbalance dataset

# In[27]:


# Select numerical columns
numerical_cols = ['PRG', 'PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age', 'Insurance']

# Iterate over each numerical column
for col in numerical_cols:
    # Create a box plot
    plt.boxplot(train[col])
    plt.title(f'Boxplot of {col}')
    plt.ylabel(col)

    # Get the outliers
    outliers = train[train[col] > train[col].quantile(0.75) + 1.5 * (train[col].quantile(0.75) - train[col].quantile(0.25))]  
    # Print the number of outliers
    num_outliers = len(outliers)
    print(f"Number of outliers in {col}: {num_outliers}")

    # Show the plot
    plt.show()


# ### Dropping duplicated rows

# In[28]:


def check_duplicate_rows(data):
    duplicate_rows = data.duplicated()
    num_duplicates = duplicate_rows.sum()
    print("Number of duplicate rows:", num_duplicates)


# In[29]:


# Check duplicate rows in train data
check_duplicate_rows(train)

# Check duplicate rows in test data
check_duplicate_rows(test)


# ### Checking Missing values 

# In[30]:


def check_missing_values(data):
    missing_values = data.isna().sum()
    print("Missing values:\n", missing_values)


# In[31]:


# Check missing values in train data
check_missing_values(train)

# Check missing values in test data
check_missing_values(test)


# ### Feature Encoding

# In[32]:


def encode_target_variable(data, target_variable):
    # Encode the target variable using LabelEncoder
    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(data[target_variable])
    target_encoded = pd.DataFrame(encoded_target, columns=[target_variable])

    # Combine the features and the encoded target variable
    data_encoded = pd.concat([data.iloc[:, :-1], target_encoded], axis=1)
    data_encoded.drop('ID', axis=1, inplace=True)

    return data_encoded


# In[33]:


# Encode target variable in train data
train_encoded = encode_target_variable(train, 'Sepssis')

# Print the encoded train data
print(train_encoded.head())


# In[34]:


# Dropping the 'ID' column from train and test dataframes
test.drop('ID', axis=1, inplace=True)


# ### Splitting of dataset to train anf evaluation

# In[35]:


def split_data(X, y, test_size, random_state=42, stratify=None):
    # Split the data into train and validation sets
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

    return X_train, X_eval, y_train, y_eval


# In[36]:


# Split the data into train and validation sets for both X and y
X_train, X_eval, y_train, y_eval = split_data(train_encoded.iloc[:, :-1], train_encoded.iloc[:, -1:], test_size=0.2, random_state=42, stratify=train_encoded.iloc[:, -1:])

# Print the shapes of the train and validation sets
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_eval shape:", X_eval.shape)
print("y_eval shape:", y_eval.shape)


# ### Handling missing Data

# In[37]:


# Handle missing values
numerical_imputer = SimpleImputer(strategy='mean')
X_train_imputed = numerical_imputer.fit_transform(X_train)
X_eval_imputed = numerical_imputer.transform(X_eval)

# Make sure test data has the same columns as X_train
test = test[X_train.columns]

# Transform the test data using the imputer
X_test_imputed = numerical_imputer.transform(test)


# ### Feature Scaling

# In[38]:


scaler = StandardScaler()
scaler.fit(X_train_imputed)

columns = ['PRG','PL','PR','SK','TS','M11','BD2','Age','Insurance']

def scale_data(data, scaler, columns):
    scaled_data = scaler.transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=columns)
    return scaled_df


# In[39]:


# Scale the data
X_train_df = scale_data(X_train_imputed, scaler, columns)
X_eval_df = scale_data(X_eval_imputed, scaler, columns)
X_test = scale_data(X_test_imputed, scaler, columns)


# ## Machine Learning Model

# ### Linear Regression 

# In[40]:


def logistic_regression_model(X_train, y_train, X_eval, y_eval):
    # Fit logistic regression model
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)

    # Make predictions on the evaluation set
    lr_preds = lr_model.predict(X_eval)

    # Calculate F1 score
    lr_f1_score = f1_score(y_eval, lr_preds)

    # Calculate false positive rate, true positive rate, and thresholds using roc_curve
    fpr, tpr, thresholds = roc_curve(y_eval, lr_preds)

    # Calculate AUC score
    lr_auc_score = roc_auc_score(y_eval, lr_preds)

    return lr_model, lr_preds, lr_f1_score, fpr, tpr, thresholds, lr_auc_score


# In[41]:


# Call the function and get the outputs
lr_model, lr_preds, lr_f1_score, fpr, tpr, thresholds, lr_auc_score = logistic_regression_model(X_train_df, y_train, X_eval_df, y_eval)

print("F1 Score:", lr_f1_score)
print("AUC Score:", lr_auc_score)


# In[42]:


# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % lr_auc_score)
plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('(ROC) Curve for Logistic Regression')
plt.legend(loc='lower right')
plt.show()


# In[53]:


def logistic_regression_model(X_train, y_train, X_eval, y_eval):
    # Fit logistic regression model
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)

    # Make predictions on the training set
    train_preds = lr_model.predict(X_train)

    # Make predictions on the evaluation set
    eval_preds = lr_model.predict(X_eval)

    # Calculate F1 score on training and evaluation sets
    train_f1_score = f1_score(y_train, train_preds)
    eval_f1_score = f1_score(y_eval, eval_preds)

    # Calculate false positive rate, true positive rate, and thresholds using roc_curve on evaluation set
    fpr, tpr, thresholds = roc_curve(y_eval, eval_preds)

    # Calculate AUC score on evaluation set
    auc_score = roc_auc_score(y_eval, eval_preds)

    return lr_model, train_f1_score, eval_f1_score, fpr, tpr, thresholds, auc_score


# In[50]:


# Call the function and print F1 score and AUC score
lr_model, train_f1, eval_f1, _, _, _, auc = logistic_regression_model(X_train, y_train, X_eval, y_eval)
print("Training F1 Score:", train_f1)
print("Evaluation F1 Score:", eval_f1)
print("AUC Score:", auc)


# In[54]:


def perform_cross_validation(model, X, y, cv=5, scoring='f1'):
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    # Calculate the average score
    avg_score = np.mean(cv_scores)

    return cv_scores, avg_score


# In[57]:


def perform_cross_validation(model, X, y, cv=5, scoring='f1'):
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    # Calculate the average score
    avg_score = np.mean(cv_scores)

    return cv_scores, avg_score

# Call the function with your logistic regression model and train data
cv_scores, avg_f1_score = perform_cross_validation(lr_model, X_train_df, y_train, cv=5, scoring='f1')

# Print the cross-validation scores and average F1 score
print("Cross-Validation Scores:", cv_scores)
print("Average F1 Score:", avg_f1_score)


# ### Random Forest Classifier

# In[59]:


def random_forest_model(X_train, y_train, X_eval, y_eval):
    # Fit Random Forest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions on the evaluation set
    rf_preds = rf_model.predict(X_eval)

    # Calculate F1 score
    rf_f1_score = f1_score(y_eval, rf_preds)

    # Calculate false positive rate, true positive rate, and thresholds using roc_curve
    fpr, tpr, thresholds = roc_curve(y_eval, rf_preds)

    # Calculate AUC score
    rf_auc_score = roc_auc_score(y_eval, rf_preds)

    return rf_model, rf_preds, rf_f1_score, fpr, tpr, thresholds, rf_auc_score

rf_model, rf_preds, rf_f1_score, fpr, tpr, thresholds, rf_auc_score = random_forest_model(X_train, y_train, X_eval, y_eval)

print("F1 Score:", rf_f1_score)
print("AUC Score:", rf_auc_score)


# In[60]:


# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % rf_auc_score)
plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('(ROC) Curve for Randon Forest Classifier')
plt.legend(loc='lower right')
plt.show()


# In[64]:


# Calculate F1 scores for training and evaluation sets
rf_train_f1_score = f1_score(y_train, rf_model.predict(X_train_df))
rf_eval_f1_score = f1_score(y_eval, rf_model.predict(X_eval_df))

# Print the F1 scores
print("F1 Score on Training Set based on Random Forest:", rf_train_f1_score)
print("F1 Score on Evaluation Set based on Random Forest:", rf_eval_f1_score)


# In[65]:


# Call the function with your Random Forest model and train data
cv_scores, avg_f1_score = perform_cross_validation(rf_model, X_train_df, y_train, cv=5, scoring='f1')
# Print the cross-validation scores and average F1 score
print("Cross-Validation Scores:", cv_scores)
print("Average F1 Score:", avg_f1_score)


# ### Naives Bayes Model

# In[66]:


def naive_bayes_model(X_train, y_train, X_eval, y_eval):
    # Fit Naive Bayes model
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    # Make predictions on the evaluation set
    nb_preds = nb_model.predict(X_eval)

    # Calculate F1 score
    nb_f1_score = f1_score(y_eval, nb_preds)

    # Calculate false positive rate, true positive rate, and thresholds using roc_curve
    fpr, tpr, thresholds = roc_curve(y_eval, nb_preds)

    # Calculate AUC score
    nb_auc_score = roc_auc_score(y_eval, nb_preds)

    return nb_model, nb_preds, nb_f1_score, fpr, tpr, thresholds, nb_auc_score


# In[67]:


nb_model, nb_preds, nb_f1_score, fpr, tpr, thresholds, nb_auc_score = naive_bayes_model(X_train_df, y_train, X_eval_df, y_eval)

# Print the F1 score and AUC score
print("F1 Score on Evaluation Set based on Naive Bayes:", nb_f1_score)
print("AUC Score on Evaluation Set based on Naive Bayes:", nb_auc_score)


# In[68]:


# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % nb_auc_score)
plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('(ROC) Curve for Naive Bayes Model')
plt.legend(loc='lower right')
plt.show()


# In[70]:


# Calculate F1 scores for training and evaluation sets
nb_train_f1_score = f1_score(y_train, nb_model.predict(X_train_df))
nb_eval_f1_score = f1_score(y_eval, nb_model.predict(X_eval_df))

# Print the F1 scores
print("F1 Score on Training Set based on Naive Bayes:", nb_train_f1_score)
print("F1 Score on Evaluation Set based on Naive Bayes:", nb_eval_f1_score)


# In[71]:


# Call the function with your Naive Bayes model and train data
cv_scores, avg_f1_score = perform_cross_validation(nb_model, X_train_df, y_train, cv=5, scoring='f1')
# Print the cross-validation scores and average F1 score
print("Cross-Validation Scores based on Naive Bayes model:", cv_scores)
print("Average F1 Score based on Naive Bayes model:", avg_f1_score)


# ### Comparing the scores

# In[72]:


# Define the scores dictionary
scores_dict = {
    'Model': ['Linear Regression', 'Random Forest Classifier', 'Naive Bayes'],
    'F1 Score': [0.6265060240963854, 0.5783132530120483, 0.574712643678161],
    'AUC Score': [0.7133699633699634, 0.6767399267399268, 0.6694139194139194]
}

# Create a DataFrame from the scores dictionary
scores_df = pd.DataFrame(scores_dict)

# Print the DataFrame
print(scores_df)


# In[75]:


# Define colors for each model
colors = ['red', 'green', 'black']

# Plot the F1 scores
plt.figure(figsize=(8, 6))
plt.bar(scores_df['Model'], scores_df['F1 Score'], color=colors)
plt.xlabel('Model')
plt.ylabel('F1 Score')
plt.title('F1 Score for Different Models')
plt.xticks(rotation=45)
plt.show()

# Plot the AUC scores
plt.figure(figsize=(8, 6))
plt.bar(scores_df['Model'], scores_df['AUC Score'], color=colors)
plt.xlabel('Model')
plt.ylabel('AUC Score')
plt.title('AUC Score for Different Models')
plt.xticks(rotation=45)
plt.show()


# In[ ]:




