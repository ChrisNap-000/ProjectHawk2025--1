#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import missingno as mno
import seaborn as sns
from sklearn.preprocessing import StandardScaler

class FraminghamDataCleaning:
    def __init__(self, df: pd.DataFrame = None):
        """Initialize with an optional DataFrame."""
        self.df = df

    def explore_data(self, df: pd.DataFrame = None):
        """ Explore the DataFrame by printing its shape, head, missing values, info, and description.
        Args:
            df (pd.DataFrame)
        """
        if df is None:
            if self.df is None:
                raise ValueError("No DataFrame provided.")
            df = self.df

        print("-------------------- Data Shape --------------------")
        print(df.shape)

        print("\n-------------------- Data Head --------------------")
        print(df.head())

        print("\n-------------------- Missing Data --------------------")
        print(df.isnull().sum())
        mno.matrix(df, figsize=(12, 6))
        plt.title("Missing Data Matrix")
        plt.show()

        print("\n-------------------- Data Info --------------------")
        print(df.info())

        print("\n-------------------- Data Description --------------------")
        print(df.describe())

    def plot_all_columns(self, df: pd.DataFrame = None):
        """
        Args:
            df (pd.DataFrame, optional)
        """
        if df is None:
            if self.df is None:
                raise ValueError("No DataFrame provided.")
            df = self.df

        for column in df.columns:
            plt.figure(figsize=(8, 5))

            if df[column].dtype == 'object' or df[column].nunique() < 20:
                df[column].value_counts().plot(kind='bar')
                plt.title(f'Count of Categories in "{column}"')
                plt.xlabel(column)
                plt.ylabel("Count")
                plt.xticks(rotation=45)
            else:
                plt.hist(df[column].dropna(), bins=30, edgecolor='black')
                plt.title(f'Distribution of "{column}"')
                plt.xlabel(column)
                plt.ylabel("Frequency")

            plt.tight_layout()
            plt.show()

    def visualize_data(self, df: pd.DataFrame, target_col: str):
        """ Visualize the DataFrame using pairplot and scatterplot.
        Args:
            df (pd.DataFrame)
            target_col (str): Target column for visualization.

        Our Imputation function needs to be unique than just 'fill categorical null values with mode' 
        and 'fill numerical nulls with median/mean'
        """
        if df is None:
            if self.df is None:
                raise ValueError("No DataFrame provided.")
            df = self.df

        sns.pairplot(df, diag_kind='kde', hue=target_col, corner=True)
        plt.suptitle("Pairplot", y=1.02)
        plt.show()

    def impute_nulls(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Args:
            df (pd.DataFrame): _description_
        """
        if df is None:
            if self.df is None:
                raise ValueError("No DataFrame provided.")
            df = self.df

        for col in df.columns:
            if df[col].isnull().any():
                if col == 'education':
                    mode_value = df[col].mode(dropna=True)[0]
                    df[col].fillna(mode_value, inplace=True)

                elif col == 'cigsPerDay':
                    nonsmoker_mask = (df['currentSmoker'] == 0) & (df[col].isnull())
                    df.loc[nonsmoker_mask, col] = 0

                    smoker_mask = df['currentSmoker'] == 1
                    smoker_median = df.loc[smoker_mask, col].median()
                    missing_smoker_mask = (df['currentSmoker'] == 1) & (df[col].isnull())
                    df.loc[missing_smoker_mask, col] = smoker_median

                else:
                    median_value = df[col].median()
                    df[col].fillna(median_value, inplace=True)

        return df

    def review_and_remove_outliers(self, column_cat, df: pd.DataFrame = None) -> pd.DataFrame:
        """ Review and remove outliers from the DataFrame based on IQR method.
        Args:
            column_cat (_type_): Categorical Variables to be excluded from outlier detection.
            df (pd.DataFrame, optional):
        """
        if df is None:
            if self.df is None:
                raise ValueError("No DataFrame provided.")
            df = self.df

        df_copy = df.copy()
        original_len = len(df_copy)

        numeric_cols = [col for col in df_copy.select_dtypes(include='number').columns if col not in column_cat]

        indices_to_remove = set()
        outlier_counts = {}

        for col in numeric_cols:
            col_data = df_copy[col].dropna()
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_idx = df_copy[(df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)].index
            outlier_counts[col] = len(outlier_idx)
            indices_to_remove.update(outlier_idx)

        print("\nOutliers detected (rows affected):")
        for col, count in outlier_counts.items():
            print(f" - {col}: {count} records")

        total_to_remove = len(indices_to_remove)
        percent = 100 * total_to_remove / original_len
        print(f"\nTotal records to be removed: {total_to_remove} ({percent:.2f}% of dataset)")

        choice = input("Do you want to remove these outliers? Type 'yes' or 'no': ").strip().lower()

        if choice == 'yes':
            df_cleaned = df_copy.drop(index=indices_to_remove)
            print(f"{total_to_remove} records removed. Remaining: {len(df_cleaned)}")
            return df_cleaned
        else:
            print("No records were removed.")
            return df_copy

    def correlation_analysis(self, columns_cat, df: pd.DataFrame = None):
        """ Perform correlation analysis on numerical columns and plot a heatmap.
        Args:
            columns_cat (_type_): Categorical Columns
            df (pd.DataFrame) 
        """
        if df is None:
            if self.df is None:
                raise ValueError("No DataFrame provided.")
            df = self.df

        numerical_cols = [col for col in df.select_dtypes(include=['number']).columns if col not in columns_cat]
        numerical_df = df[numerical_cols]

        correlation_matrix = numerical_df.corr()
        print("Correlation Matrix:")
        print(correlation_matrix)

        plt.figure(figsize=(10, 8))
        custom_cmap = sns.color_palette(["orange", "lightgrey", "teal"])
        sns.heatmap(correlation_matrix, annot=True, cmap=custom_cmap, fmt=".2f", linewidths=0.5)
        plt.title("Heatmap of Numerical Feature Associations")
        plt.show()

        return correlation_matrix

    def compare_skew_between_columns(self, col1, col2, df: pd.DataFrame = None):
        """Compare skewness between two columns and print correlation.

        Args:
            col1 (_type_): First field to check multicolinearity
            col2 (_type_): Second field to check multicolinearity
            df (pd.DataFrame)
        """
        if df is None:
            if self.df is None:
                raise ValueError("No DataFrame provided.")
            df = self.df

        if col1 not in df.columns or col2 not in df.columns:
            raise ValueError("One or both columns not found in DataFrame.")

        data = df[[col1, col2]].dropna()

        correlation = data[col1].corr(data[col2])
        skew1 = abs(data[col1].skew())
        skew2 = abs(data[col2].skew())

        print(f"Correlation between '{col1}' and '{col2}': {correlation:.2f}")
        print(f"Skewness of '{col1}': {skew1:.2f}")
        print(f"Skewness of '{col2}': {skew2:.2f}")

        if skew1 > skew2:
            more_skewed = col1
        elif skew2 > skew1:
            more_skewed = col2
        else:
            more_skewed = "Both have equal skewness"

        print(f"→ More skewed column: {more_skewed}")
        return more_skewed

    def scale_numeric_columns(self, df: pd.DataFrame = None, columns_cat: list = []):
        """Scale numeric columns using StandardScaler.

        Args:
            df (pd.DataFrame)
            columns_cat (list): List of categorical columns to exclude from scaling. Defaults to [].
        """
        if df is None:
            if self.df is None:
                raise ValueError("No DataFrame provided.")
            df = self.df

        df_scaled = df.copy()

        # Select columns to scale (not in columns_cat and not of type object)
        columns_to_scale = [col for col in df.columns if col not in columns_cat and df[col].dtype != 'object']

        # Apply StandardScaler
        scaler = StandardScaler()
        df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])

        print(f"Scaled columns: {columns_to_scale}")
        return df_scaled
    
    
    def detect_and_report_outliers(self, df: pd.DataFrame):
        """Detect and report outliers in numerical columns, showing boxplots and summary stats.

        Args:
            df (pd.DataFrame)
        """
        outlier_report = []
        numeric_cols = df.select_dtypes(include=['number']).columns
        outlier_rows = set()

        for col in numeric_cols:
            # identifying the outliers
            col_data = df[col].dropna()
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # get outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            num_outliers = len(outliers)
            percent_outliers = 100 * num_outliers / len(col_data)

            # track rows with outliers
            outlier_rows.update(outliers.index)

            # this makes the boxplot
            plt.figure(figsize=(6, 1.5))
            plt.boxplot(col_data, vert=False)
            plt.title(f'Boxplot of {col}')
            plt.xlabel(col)
            plt.tight_layout()
            plt.show()

            outlier_report.append({
                'column': col,
                'num_outliers': num_outliers,
                'percent_outliers': round(percent_outliers, 2)
            })

        # Convert report to DataFrame
        report_df = pd.DataFrame(outlier_report)

        # Summary of total outlier rows
        total_outlier_rows = len(outlier_rows)
        total_rows = len(df)
        percent_removed = 100 * total_outlier_rows / total_rows

        print(f"\nSummary:")
        print(f"Total rows with at least one outlier: {total_outlier_rows}")
        print(f"If removed, rows removed: {total_outlier_rows}")
        print(f"Percentage of total records that could potentially be removed: {percent_removed:.2f}%")

        return report_df
