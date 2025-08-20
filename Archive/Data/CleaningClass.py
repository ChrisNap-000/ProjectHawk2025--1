import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE

class DataCleaning:
    """This class is meant for general data cleaning and preprocessing tasks.

    Specifically, it includes methods for:
    1. Load the data from a csv into Pandas DataFrame.
    2. Expore dataset for missing values
    3. Visualize data
    4. Detect outliers using Z-score or IQR method.
    5. Imputes missing values using SimpleImputer.
    6. Uses SMOTE to oversample the minority class in the dataset.
    7. Standardizes/normalizes the dataset using StandardScaler and MinMaxScaler.
    """

    def __init__(self, df: pd.DataFrame = None):
        
        """Initialize the DataCleaning class with a DataFrame."""

        self.df = df if df is not None else pd.DataFrame()
        self.imputer = None  # Will be defined in the imputeMissingValues method
        self.smote = None  # Will be defined in the oversampleData method
        self.scaler = None  # Will be defined in the standardizeData method
        self.minmax_scaler = None  # Will be defined in the normalizeData method
        self.data = None # Will be defined in the loadData method
        self.target_col = None

    def loadData(self, file_path: str):
        """Load the data from a csv into Pandas DataFrame.
        Must define the variable self as dc, self = dc
        Arguments should be (self, file_path)
        
        Inputs:
        1. self                 : Instance of the DataCleaning class.
        2. file_path (str)      : Path to the CSV file.
        """
        try:
            df = pd.read_csv(file_path)
            self.data = df
            print(f"Data loaded successfully from {file_path}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
        

    def exploreData(self, df: pd.DataFrame):
        """Explore dataset for missing values
        Arguments should be (df, self)
        
        Inputs:
        1. df                   : DataFrame to be explored.
        2. self                 : Instance of the DataCleaning class.
        """
        print("-------------------- Data Shape --------------------")
        print(self.shape)
        print("\n-------------------- Data Head --------------------")
        print(self.head())
        print("\n-------------------- Missing Data --------------------")
        print(self.isnull().sum())
        msno.matrix(self, figsize=(12, 6))
        plt.title("Missing Data Matrix")
        plt.show()
        print("\n-------------------- Data Info --------------------")
        print(self.info())
        print("\n-------------------- Data Description --------------------")
        print(self.describe())


    def visualizeData(self, df: pd.DataFrame, target_col: str):
        """Visualize data using pairplot and heatmap.
        Arguments should be (df, self)
        
        Inputs:
        1. df                   : DataFrame to be explored.
        2. self                 : Instance of the DataCleaning class.
        """
        sns.pairplot(df, diag_kind='kde', hue= target_col, corner=True)
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')


    def detectOutliers(self, columnName: str, method: str, threshold: float = 3):
        """Detect outliers in a numerical column using a chosen method.
        Filters the data to only return records with outliers.        
        Arguments should be (self, columnName, method, threshold)
        
        Inputs:
        1. self                 : Instance of the DataCleaning class.
        2. columnName (str)     : Name of the numerical column.
        3. method (str)         : Method to detect outliers. Options are 'zscore' or 'iqr'.
        4. threshold (float)    : Threshold for outlier detection.
        """
        if method == 'zscore':
            meanValue = self.data[columnName].mean()
            stdValue = self.data[columnName].std()
            z_scores = (self.data[columnName] - meanValue) / stdValue
            outliers = self.data[np.abs(z_scores) > threshold]
            return outliers
        elif method == 'iqr':
            Q1 = self.data[columnName].quantile(0.25)
            Q3 = self.data[columnName].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (1.5 * IQR)
            upper_bound = Q3 + (1.5 * IQR)
            outliers = self.data[(self.data[columnName] < lower_bound) | (self.data[columnName] > upper_bound)]
            return outliers
        else:
            raise ValueError("method must be either 'zscore' or 'iqr'.")
    