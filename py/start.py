# load pandas
import pandas as pd

# data location
url='../data/InputParameters.txt'

# load data
dataframe=pd.read_csv(url,header=None,sep='\s+ ',engine='python')

# dataframe.head(2)
dataframe[0].head(2)
