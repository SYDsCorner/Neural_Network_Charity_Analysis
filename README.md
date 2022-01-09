# Neural_Network_Charity_Analysis

![what-is-deep-learning-large](https://user-images.githubusercontent.com/89308251/148670046-54ee764e-52a5-4cfa-99f6-05497c0e4002.jpg)





## Challenge Overview

### Purpose:

   The purpose of this analysis is to create a deep-learning neural network by using the features in the provided dataset to help create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.
 
## Resources
- Software:
   - Jupyter Notebook 6.4.6
   - Machine Learning
      - Python 
         - scikit-learn library
         - tensorflow library
   
- Data source: 
   - [charity_data.csv](https://github.com/SYDsCorner/Neural_Network_Charity_Analysis/blob/main/Resources/charity_data.csv)

      - This dataset containes more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that
        capture metadata about each organization, such as the following:

        - `EIN` and `NAME` — Identification columns
        - `APPLICATION_TYPE` — Alphabet Soup application type
        - `AFFILIATION` — Affiliated sector of industry
        - `CLASSIFICATION` — Government organization classification
        - `USE_CASE` — Use case for funding
        - `ORGANIZATION` — Organization type
        - `STATUS` — Active status
        - `INCOME_AMT` — Income classification
        - `SPECIAL_CONSIDERATIONS` — Special consideration for application
        - `ASK_AMT` — Funding amount requested
        - `IS_SUCCESSFUL` — Was the money used effectively

   
   
## Results 

### *** Data Preprocessing ***
   - The model's **target** is the predicted outcome, or dependent variable, defined by:
      - `IS_SUCCESSFUL` column: this column is a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.
      
   - The model's **features** are the variables used to make a prediction, or independent variables, defined by:
      - `APPLICATION_TYPE` 
      - `AFFILIATION`
      - `CLASSIFICATION` 
      - `USE_CASE` 
      - `ORGANIZATION` 
      - `STATUS`
      - `INCOME_AMT` 
      - `SPECIAL_CONSIDERATIONS` 
      - `ASK_AMT` 
      
   - There are 2 features that **should be removed** from the input data, namely:
      - `EIN` and `NAME` columns: These two identification columns are not useful in the prediction. 
      
        ![Drop_columns](https://user-images.githubusercontent.com/89308251/148670603-121b5456-f25e-4564-9536-72a7fe787d2e.png)

	
### *** Compiling, Training, and Evaluating the Model ***
   - For the first model, I chose **2 hidden layers** with the **80** and **30 neurons** respectively.
      - **2 hidden layers**, this is because the additional layer was redundant or may increase the change of overfitting the train data.
      - **The number of neurons** was considered from a good rule of thumb to have *2 to 3 times* the amout of neurons in the hidden layers as the number of inputs.
      - **The activation functions**:
         - I selected the **relu** activation function for those **2 hidden layers** which is ideal when looking at positive nonlinear input data.
         - and selected the **sigmoid** activation function for the **output layer** because it is ideal for binary classification which will help us predict the probability of
           whether applicants will be successful.
           
          ![define_model](https://user-images.githubusercontent.com/89308251/148670697-983abeb0-93fc-4924-9252-2dd6f1bdcb8f.png)

      
### ** Optimizing the model: in order to achieve a target predictive accuracy higher than 75% from the **original 73%**
   - Besides **bucketing or binning** the `APPLICATION_TYPE` and `CLASSIFICATION` columns, I also binned the `ASK_AMT` column because the number of unique values in this
     column is pretty high and too different from the others. 
      
      ![nunique](https://user-images.githubusercontent.com/89308251/148671557-54b2addd-6740-4de5-ba11-35093a7581a6.png)

      
      ![bin_ask_amt](https://user-images.githubusercontent.com/89308251/148670530-a3ef4fee-d9d1-4ede-a300-165efa3dc6b8.png)

      
      - Then, I made multiple attempts at optimizing the model and each attempt was **slightly less optimal than the original state**.
      
         - **Original result**: approximately **73%**
            
            ![original_result](https://user-images.githubusercontent.com/89308251/148671119-e6b3c12c-9f1d-44bc-a6f3-2ff9e1e2b78f.png)

         - The **1st attempt**: 
            - I tried to **add neurons** to hidden layers and the target predictive accuracy became **72%**.
               
               ![first_attempt](https://user-images.githubusercontent.com/89308251/148671015-0760e5a2-2ca6-41f2-afae-b780b27c3d3d.png)

               
         - The **2nd attempt**:
            - I tried to **add the third hidden layers** and the target predictive accuracy is still **72%**.

               ![second_attempt](https://user-images.githubusercontent.com/89308251/148671290-1327c2df-40b9-4bcc-9889-c9ea4a56fc2a.png)


         - The **3rd attempt**: 
            - I tried to **change the activation function** of hidden layers **from "relu" to "tanh"** and the target predictive accuracy is also **not that different** from
              before.
               
               ![third_attempt](https://user-images.githubusercontent.com/89308251/148671355-c6337758-83e0-4578-9b13-04ec9346ca23.png)

   

## Summary: 

The overall results of the deep learning model was that I **was not able to achieve** the target model performance of **75%** after I tried multiple attempts. Adjusting the input data by dropping more columns, creating more bins for rare occurrences in columns, adding more neurons to a hidden layer, adding more hidden layers, and using different activation functions for the hidden layers all **did not result in a better model performance**. 

My recommendation on using a different model to solve the classification problem would be to try other types of supervised learning such as *Logistic Regression*, *Support Vector Machine (SVM)*, and *Random Forest*. In my opinion, in order to achieve a target predictive accuracy I would consider a ***Random Forest*** model more than the others. The random forest models are robust against overfitting as all of those weak learners are trained on different pieces of the data and robust to outliers and nonlinear data as well.    
   
--------------------------------------------

![types_machine_learning](https://user-images.githubusercontent.com/89308251/148670300-79c23207-49c7-4a6d-b3c5-bbf2b0fcc04a.png)

--------------------------------------------
 
   
