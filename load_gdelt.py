import pandas as pd
import os
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    load gdelt data from a csv file.
    Returns:
    pd.DataFrame: DataFrame containing the gdelt data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    # df = pd.read_csv(file_path, delimiter='\t', header=None)
    df = pd.read_csv(file_path)
    return df

def inspect_data(df):
    """
    Inspects the gdelt data by printing the first few rows and summary statistics.
    """
    print("First 5 rows of the DataFrame:")
    print(df.head())
    print("\nDataFrame Info:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe(include='all'))

def plot_events_per_year(df):
    """Plot number of events per year using the 'TotalEvents' column."""
    events_per_year = df.groupby("Year")["TotalEvents"].sum()

    events_per_year.plot(kind="bar", figsize=(12, 5))
    plt.title("Total Conflict Events per Year")
    plt.xlabel("Year")
    plt.ylabel("Total Conflict Events")
    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    file_path = '../data/gdelt_conflict_1_0.csv' 
    df = load_data(file_path)
    inspect_data(df)
    plot_events_per_year(df)
    