# Forecast the Car Insurance Risk Rating
This project attempts to forecast the vehicle risk rating for an insurance company. Different data mining applications will be used and compared to see which one is best fit for this dataset. Data validation, cleaning, and exploratory analysis was performed. Finally, three different classification methods were used: SVM with different kernels, Decision Tree, and Random Forest.

Code/Report: Forecast the Car Insurance Risk Rating

Skills: classification, decision tree, support vector machines, random forest, hyperparameter tuning, data validation, data visualization

Results:

The SVM with RBF kernel achieved the best accuracy of 67.68%, demonstrating that non-linear relationships between features and risk ratings are important. The Random Forest model, which aggregates the predictions of multiple decision trees, achieved a close accuracy of 67.78%. Despite this, further hyperparameter tuning and cross-validation could still enhance its performance. Moreover, based on the analysis, the categorical features have shown little importance in the feature importance plots and haven't significantly contributed to improving the accuracy of the models, it could be worth considering their removal to simplify the model, and potentially improve generalization by reducing noise.
