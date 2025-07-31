# This script loads FIFA match results data, shows basic info, and visualizes the data.
# It creates orange-themed plots: column distributions, a correlation matrix, and scatter/density plots.

from sklearn.preprocessing import StandardScaler  # for scaling data (not used in this script, but often useful)
import matplotlib.pyplot as plt  # for plotting graphs
import matplotlib as mpl  # for customizing plot styles
import numpy as np  # for numerical operations
import os  # for file operations (not used here)
import pandas as pd  # for data handling

# Set a beautiful orange theme for all plots
mpl.rcParams.update({
    'axes.prop_cycle': mpl.cycler(color=['#ff9800', '#ffa726', '#fb8c00', '#ffb300', '#ff7043', '#ffcc80']),
    'axes.facecolor': '#fff3e0',
    'figure.facecolor': '#fff3e0',
    'axes.edgecolor': '#ff9800',
    'xtick.color': '#fb8c00',
    'ytick.color': '#fb8c00',
    'axes.labelcolor': '#fb8c00',
    'text.color': '#e65100',
    'axes.titleweight': 'bold',
    'axes.titlesize': 16,
    'axes.titlecolor': '#e65100',
    'grid.color': '#ffb300',
    'grid.linestyle': '--',
    'grid.alpha': 0.5
})

# This function plots the distribution (histogram or bar) for each column in the DataFrame
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()  # get number of unique values in each column
    # keep only columns with more than 1 and less than 50 unique values
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]]
    nRow, nCol = df.shape  # get number of rows and columns
    columnNames = list(df)  # get column names
    nGraphRow = (nCol + nGraphPerRow - 1) // nGraphPerRow  # calculate number of rows for subplots
    plt.figure(num=None, figsize=(6 * nGraphPerRow, 8 * nGraphRow), dpi=80)
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)  # create subplot
        columnDf = df.iloc[:, i]  # get column data
        # If column is not numeric, plot bar chart
        if not np.issubdtype(type(columnDf.iloc[0]), np.number):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar(color='#ff9800', edgecolor='#e65100')
        else:
            columnDf.hist(color='#ff9800', edgecolor='#e65100')  # plot histogram for numeric columns
        plt.ylabel('counts')
        plt.xticks(rotation=90)
        plt.title(f'{columnNames[i]} (column {i})')
        plt.grid(True)
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show()

# This function plots a correlation matrix for numeric columns
def plotCorrelationMatrix(df, graphWidth):
    filename = getattr(df, 'dataframeName', 'DataFrame')  # get file name if available
    df = df.dropna(axis='columns')  # drop columns with NaN values
    df = df[[col for col in df if df[col].nunique() > 1]]  # drop constant columns
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()  # calculate correlation matrix
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80)
    corrMat = plt.matshow(corr, fignum=1, cmap='Oranges')  # show matrix with orange color map
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat, fraction=0.046, pad=0.04)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15, color='#e65100')
    plt.show()

# This function plots scatter and density plots for up to 10 numeric columns
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include=[np.number])  # keep only numeric columns
    df = df.dropna(axis='columns')  # drop columns with NaN values
    df = df[[col for col in df if df[col].nunique() > 1]]  # drop constant columns
    columnNames = list(df)
    if len(columnNames) > 10:
        columnNames = columnNames[:10]  # limit to 10 columns
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(
        df,
        alpha=0.75,
        figsize=[plotSize, plotSize],
        diagonal='kde',  # show density on diagonal
        color='#ff9800',
        hist_kwds={'color': '#ff9800', 'edgecolor': '#e65100'},
        marker='o'
    )
    corrs = df.corr().values  # get correlation values
    # Annotate upper triangle with correlation coefficients
    for i, j in zip(*np.triu_indices_from(ax, k=1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2),
                          xycoords='axes fraction', ha='center', va='center', size=textSize, color='#e65100')
    plt.suptitle('Scatter and Density Plot', color='#e65100')
    plt.show()

nRowsRead = 1000  # number of rows to read from the CSV file
df1 = pd.read_csv('Data/results.csv', delimiter=',', nrows=nRowsRead)  # load data
df1.dataframeName = 'results.csv'  # set a name for the DataFrame
nRow, nCol = df1.shape  # get shape of the DataFrame
print(f'There are {nRow} rows and {nCol} columns')  # print shape
print(df1.head(5))  # print first 5 rows

# Call the plotting functions
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 6, 15)