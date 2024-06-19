#!/usr/bin/env python
# coding: utf-8

# # Closed Form Linear Regression
# 
# This project demonstrates the application of closed form linear regression on the insurance dataset. The goal is to predict insurance charges based on various features such as age, sex, BMI, number of children, smoking status, and region. The notebook includes preprocessing steps, training and testing the model, and performing cross-validation.
# 
# # Setup
# 
# We start by importing the necessary libraries and setting a fixed random seed for reproducibility

# In[1]:


# Imports
import numpy as np
import pandas as pd 
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
pd.set_option('future.no_silent_downcasting', True)

# Set random seed
np.random.seed(0)


# # Data Preprocessing
# 
# In this section, we will analyze the dataset, preprocess the variables, and prepare the data for model training. We will convert categorical variables to numerical values, perform one-hot encoding where necessary, and ensure the data is shuffled and split into training and testing sets. First, let's load the dataset and take a look at its structure to understand the data we're working with.
# 

# In[2]:


# Read the dataset
df = pd.read_csv('insurance.csv')

# Display the first few rows of the dataset
df.head()


# We have the following columns in our dataset:
# 
# - **age**: Age of the individual
# - **sex**: Gender of the individual (male/female)
# - **bmi**: Body Mass Index of the individual
# - **children**: Number of children the individual has
# - **smoker**: Smoking status of the individual (yes/no)
# - **region**: Region where the individual resides (northeast/northwest/southeast/southwest)
# - **charges**: Medical insurance charges billed to the individual
# 
# 
# Next, we preprocess the variables and the dataframe, making it suitable for our regression model. We do the following preprocessing steps:
# 1. Convert `sex` and `smoker` columns to binary values.
# 2. Perform one-hot encoding on the `region` column.
# 3. Ensure all columns are of type float for processing.
# 4. Shuffle the dataset to ensure randomness.
# 5. Reset the index to maintain a clean index order.
# 

# In[3]:


# Convert sex, smoker to binary
df['sex'] = df['sex'].replace({'male': 1, 'female': 0})
df['smoker'] = df['smoker'].replace({'yes': 1, 'no': 0})

# One-hot encode the region column and concatenate to original df
df = pd.get_dummies(df, columns=['region'], prefix='', prefix_sep='')

# Convert all columns to float for processing
df = df.astype(float)

# Shuffle rows
df = df.sample(frac=1)

# Reset the index
df.reset_index(drop=True, inplace=True)


# We now separate the target variable (`charges`) from the features and add a dummy variable to the features to account for the bias term in the linear regression model.

# In[4]:


# Separate charges column as y value to predict
y = df['charges']

# Drop 'charges', add dummy variable to df first column and get feature matrix X
x = df.drop('charges', axis=1).assign(dummy=np.ones(df.shape[0]))[['dummy'] + df.drop('charges', axis=1).columns.tolist()]


# We now split the data into training and testing sets. This will allow us to train the model on one subset of the data and validate it on another to ensure that it generalizes well to unseen data.

# In[5]:


# Split into train and test set
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=(1/3))


# 
# Once we have our training and test/validation sets ready, we can move onto training and evaluating our model
# 
# # Model Training and Evaluation
# 
# We begin by calculating the weight vector using the closed form solution of linear regression on the training set. The training set is used for this computation to fit the model parameters without bias from the unseen test data. This approach ensures that the model learns only from the training data, which is crucial for evaluating its performance objectively on the validation or test set later. The closed form solution for the weight vector $ \mathbf{w} $ in linear regression is given by:
# 
# $$
# [
# \mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
# ]
# $$
# where:
# - $ \mathbf{X} $ is the matrix of input features (with a column of ones for the intercept term),
# - $ \mathbf{y} $  is the vector of target values (charges in this case),
# - $ \mathbf{X}^\top $ is the transpose of $ \mathbf{X} $, 
# - $(\mathbf{X}^\top \mathbf{X})^{-1} $ is the inverse of the matrix $ \mathbf{X}^\top \mathbf{X} $.
# 
# By using this equation, we can directly compute the optimal weight vector that minimizes the sum of squared errors.
# 

# In[6]:


# Calculate weight vector using closed form solution
w = np.dot(np.linalg.pinv(np.dot(x_train.T, x_train)), np.dot(x_train.T, y_train))


# The calculated weight vector $ \mathbf{w} $ can now be used to make predictions on both the training and validation datasets. This step involves taking the dot product of the input features $ \mathbf{X} $ with the weight vector to estimate the insurance charges.

# In[7]:


# Predictions on training set
y_train_pred = np.dot(x_train, w)

# Predictions on validation set
y_val_pred = np.dot(x_val, w)


# We then use the Root-Mean Squared Error (RMSE) as one of our evaluation metrics. RMSE is a standard way to measure the error of a model in predicting quantitative data. It represents the square root of the average of the squared differences between actual and predicted values. The formula for RMSE is:
# 
# $$
# \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}
# $$
# 
# where $ y_i $ are the actual values and $ \hat{y}_i $ are the predicted values, and $ n $ is the number of samples.

# In[8]:


# RMSE for training set
rmse_train = (sum((y_train - y_train_pred)**2)/len(y_train))**0.5;

# RMSE for validation set
rmse_val = (sum((y_val - y_val_pred)**2)/len(y_val))**0.5;


# 
# SMAPE is another robust measure of forecasting accuracy that we use. It scales the absolute percentage error by the sum of the absolute values of the actual and predicted values, which mitigates issues when the actual values are close to zero. The formula for SMAPE is:
# 
# $$
# \text{SMAPE} = \frac{100\%}{n} \sum_{i=1}^n \frac{|y_i - \hat{y}_i|}{\frac{(|y_i| + |\hat{y}_i|)}{2}}
# $$
# 

# In[9]:


# SMAPE for training set
smape_train = sum((abs(y_train - y_train_pred))/(abs(y_train) + abs(y_train_pred)))/len(y_train);

# SMAPE for validation set
smape_val = sum((abs(y_val - y_val_pred))/(abs(y_val) + abs(y_val_pred)))/len(y_val);


# Finally, we print the calculated RMSE and SMAPE values for both training and validation sets to assess the performance of our linear regression model.

# In[10]:


# Print the results
print(f"Training RMSE: {rmse_train:.2f}, Validation RMSE: {rmse_val:.2f}")
print(f"Training SMAPE: {smape_train:.2%}, Validation SMAPE: {smape_val:.2%}")


# 
# The RMSE and SMAPE values indicate our model performs consistently between training and validation sets, with a slightly better accuracy on the validation set. This suggests good generalization. However, to further confirm these results and ensure the model's robustness, we'll next employ cross-validation across different data subsets.
# 
# # Cross-Validation
# 
# Cross-validation is a robust statistical method used to estimate the accuracy of predictive models. It helps to mitigate overfitting by using different subsets of the data for training and validating the model multiple times. This process enhances the generalizability of the model as it must perform well on multiple different data splits to achieve a low average error.
# 
# In our implementation, we use a function to automate the cross-validation process with variable numbers of folds, $ S $. This function:
# - Randomly shuffles the dataset.
# - Splits the dataset into $ S $ distinct folds.
# - Iteratively uses one fold for validation and the remainder for training.
# - Computes the root mean square error (RMSE) for each fold.
# - Calculates and prints the mean and standard deviation of the RMSE across all folds to provide a comprehensive view of the model's performance stability across different subsets of data.
# 
# This method ensures each data point gets used for both training and validation, and the mean RMSE across folds gives us a robust measure of our model's predictive accuracy.
# 

# In[18]:


def cross_validation(df, S, num_seeds=20):
    """
    Perform cross-validation on the dataset to evaluate the model performance.

    Args:
        df (DataFrame): The dataset containing features and the target variable.
        S (int): The number of folds for cross-validation.
        num_seeds (int): The number of random seeds to use for averaging the results.

    Prints:
        The mean and standard deviation of the RMSE across all seeds.
    """
    rmse_vals = []  # List to store RMSE values for each seed

    for seed in range(num_seeds):
        np.random.seed(seed)  # Set the seed for reproducibility
        currdf = df.sample(frac=1)

        # Reset the index
        currdf.reset_index(drop=True, inplace=True)
        
        currdf.insert(0, 'dummy', np.ones(len(currdf)))  # Add a dummy variable for intercept

        # Declare variable to hold total sum of se values over different folds
        se_foldSum = 0;

        for i in range(S):
            # Get rows for training and validation
            row_train = currdf.index[currdf.index % S != i].tolist()
            row_val = currdf.index[currdf.index % S == i].tolist()
            
            
            # Split into train and test set
            training_data = currdf.loc[row_train]
            validation_data = currdf.loc[row_val]
            
            # Set x and y separate
            y_train = training_data['charges'];
            y_val = validation_data['charges'];
            x_train = training_data.drop('charges', axis=1);
            x_val = validation_data.drop('charges', axis=1);
            
            # Calculate w
            w = np.dot(np.linalg.pinv(np.dot(x_train.T,x_train)),np.dot(x_train.T,y_train));
            
            # Train the validation set
            y_val_pred = np.dot(x_val, w)
            
            # Get SE values
            se_val = (y_val - y_val_pred)**2;
            
            # Sum all se values in vector
            se_sum = np.sum(se_val);
            
            # Add to total sum over folds
            se_foldSum = se_foldSum + se_sum
                
        # Calculate RMSE for current seed
        currRmse = ((se_foldSum)/(currdf.shape[0]))**0.5
        
        # Append to all RMSE values     
        rmse_vals.append(currRmse);

    # Output the mean and standard deviation of the RMSE across all seeds
    print(f"Mean RMSE for S = {S}: {np.mean(rmse_vals)}")
    print(f"Standard Deviation for S = {S}: {np.std(rmse_vals)}")



# We then test our model with a smaller number of folds (S = 3) allows for a quick evaluation and provides a basic understanding of the model's performance across different subsets of data. This setup will likely give us a preliminary indication of the model's stability.

# In[19]:


# Perform cross-validation with 3 folds
cross_validation(df, S=3)


# The mean RMSE of 6105.73 and the standard deviation of 24.22 indicate a reasonably consistent performance across different folds, though the prediction error is relatively high. To gain a more detailed understanding of the model's performance, we will now increase the number of folds to 223.

# In[20]:


cross_validation(df, S=223)


# The mean RMSE of 6087.05 and the significantly lower standard deviation of 1.23 demonstrate that the model benefits from finer segmentation of the dataset. This setup leads to reduced prediction errors and more consistent performance. To further refine our assessment, we will now conduct leave-one-out cross-validation (LOOCV) with S equal to the number of data points, N = 1338.

# In[21]:


cross_validation(df, S=1338)


# The mean RMSE of 6087.39 and the near-zero standard deviation indicate that the model performs consistently across all individual data points. This consistency demonstrates the model's robustness and reliability in predicting insurance charges, making it a strong candidate for practical applications.
# 
# # Contact
# 
# **Prepared by:** Farzan Mirza
# 
# **Email:** [farzanmrz@gmail.com](mailto:farzanmrz@gmail.com), [fm474@drexel.edu](mailto:fm474@drexel.edu)
# 
# **GitHub:** [https://github.com/Farzanmrz](https://github.com/Farzanmrz)
# 
# **LinkedIn:** [https://www.linkedin.com/in/farzan-mirza13/](https://www.linkedin.com/in/farzan-mirza13/)
# 
