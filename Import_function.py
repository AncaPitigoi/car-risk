import seaborn as sns
import matplotlib.pyplot as plt
import openml

def import_openml(id):
    """
    Import a dataset from OpenML into a pandas DataFrame.

    This function will make the code cleaner and easier to manage.
    
    Parameters
    ----------
    dataset_id : int
        The OpenML dataset ID to import.
    
    Returns
    -------
    DataFrame
        A DataFrame containing the data from the OpenML dataset.
    """
    # Fetch the dataset from OpenML
    dataset = openml.datasets.get_dataset(id)
    
    # Get the data and its corresponding feature names
    X, *_ = dataset.get_data()
    
    return X


def get_outliers(df, column):
    """
    Identifies and retrieves the outliers for a specified column.

    This function shows the outliers in the given DataFrame using the Interquartile Range (IQR) method. Moreover, the output will only select the specified 
    columns needed to conduct the analysis.

    An outlier is defined as any value below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR, where:
    - Q1 is the 25th percentile (first quartile)
    - Q3 is the 75th percentile (third quartile)
    - IQR is the Interquartile Range (Q3 - Q1)

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame from which to identify outliers.
    
    column : str
        The name of the column in the DataFrame for which outliers should be identified.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the rows where the specified column contains outlier values, filtered to show specific columns.
    
    Example:
    --------
    >>> outliers_price = get_outliers(s_auto, 'price')
    >>> print(outliers_price)
    """

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    
    IQR = Q3 - Q1
    
    # Define the outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Retrieve rows where column values are outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    # Select only specific columns to output
    selected_columns = ['normalized_losses', 'engine_size', 'compression_ratio', 'horsepower',
                        'peak_rpm', 'price', 'make', 'num_of_cylinders', 'class']
    
    return outliers[selected_columns]



def plot_feature_importances(model, color, title, feature_columns):
    """
    Plots the feature importances of a given model.

    Parameters:
    -----------
    model : estimator object
        A trained model that contains the `feature_importances_` attribute, such as 
        a RandomForestClassifier or DecisionTreeClassifier.
    
    color : str
        The color of the bars in the plot. Use any valid color format, such as hex 
        ('#72BF45') or named colors ('blue').
    
    title : str
        The title of the plot. This string will be displayed as the plot's title.

    feature_columns : list or pandas.Index
        A list or pandas Index object containing the feature names (column names) to display
        on the y-axis of the plot.
    
    Returns:
    --------
    None
        The function generates and displays a feature importance bar plot.
    
    Example:
    --------
    plot_feature_importances(forest_model, '#72BF45', 'Feature Importance - Random Forest', X_final.columns)
    """
    # Get feature importances from the model
    importance = model.feature_importances_
    
    # Create the bar plot
    plt.figure(figsize=(3, 15))
    sns.barplot(x=importance, y=feature_columns, color=color)
    
    # Remove the box-like feature (spines)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Add the title
    plt.title(title)
    
    # Display the plot
    plt.show()