# Continuous Glucose Monitoring

## Built With
* Matlab

## Files
- Prediction.m
   - Extracted features for Meal and No Meal Data.
   - Trains each and every model using the mealData.csv and Nomeal.csv files.
- testScript.m
   - Allows users to test using test data where each models' predictions will be saved to its own .csv file. The different .csv files containing the predictions can then be used to calculate the accuracy, precision, recall, and F1 score. 
- 'modelName'.mat
   - There are four of these files, one for each model. 
   - 'modelName' is a placeholder for the actual model name (e.g. linearSVM.mat).
   - These files help run each model.

## How to Run the Project
* Open Matlab
* Open the project folder in Matlab
* In command window, call testScript() function (testScript.m) with the CSV filepath (test data filename) as argument.
* The different models should be used to predict the class labels (0 = no meal, 1 = meal) which will be stored in their respective .csv files (e.g. Gaussian SVM will be stored in gaussianSVM.csv) generated from testScript function.

## Information About the Project
* Each model will be stored in its corresponding .mat file (e.g. Linear SVM will be stored in linearSVM.mat file).
* Our features remain the same -- zero-crossing, RMS, coefficient variation, and FFT.

## Models Used
* Linear SVM 
* Polynomial SVM 
* Gaussian SVM 
* Naive Bayes 
