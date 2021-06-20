import pandas as pd
import yfinance as yf
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from natsort import index_natsorted
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

rnd_seed_state = 1


etf_df = pd.read_csv('ETFs.csv', low_memory=False)
mf_df = pd.read_csv('Mutual Funds.csv', low_memory=False)


#Throwing away 4 completely empty columns in df_etf - ['category_return_2019', 'category_return_2018', 'category_return_2017', 'category_return_2016']:

empty_cols2 = [col for col in etf_df.columns if etf_df[col].isnull().all()]
etf_df.drop(empty_cols2,
        axis=1,
        inplace=True)

##removing columns that are not useful to our analysis
throwaway_cols=[
    'quarters_up',
    'quarters_down',
    'years_up',
    'years_down',
    'currency'
]
mf_df.drop(columns=throwaway_cols,axis=1,inplace=True)
etf_df.drop(columns=throwaway_cols[2:],axis=1,inplace=True)


### Creates and returns a subset by keeping only the columns with 'keyword' in their column name. Iterating across the dataframe to make subsets and then merge theese together.
def clean_by_keyword(df, keyword):
    """
    Finds and creates a list of all of the column names in df that contain the keyword in the dfn.
    Subsets the df according to this list, then creates a new dataframe with this
    information and drops the corresponding columns from the original dataframe.

    Arguments:
        df: Dataset pandas DataFrame.
        cols: List of columns to be included in calculations.
        keyword: Word to be removed from column name in order to preserve clarity with labeling.


    Return: dataframe containing data matching keyword.
    """
    col_list = [i for i in df.columns if (keyword in i)]
    # print(col_list)
    df_new = df[col_list]
    df.drop(columns=col_list, axis=1, inplace=True)  # removes duplicate data from df_mf masterframe
    df_new.insert(0, 'fund_symbol', df.fund_symbol)  # inserts the fund symbols as the 1st column of the new data set
    return (df_new)


#dropping categorical data from out main dataframes.
mf_df_category_data=clean_by_keyword(mf_df,'category_')
etf_df_category_data=clean_by_keyword(etf_df,'category_')

#we do want to include ytd 3,5 and 10 years data in our data set so we re-add it to our df for analysis later

mf_df['category_return_ytd']=mf_df_category_data['category_return_ytd']
etf_df['category_return_ytd']=etf_df_category_data['category_return_ytd']

mf_df['category_return_3years']=mf_df_category_data['category_return_3years']
etf_df['category_return_3years']=etf_df_category_data['category_return_3years']

mf_df['category_return_5years']=mf_df_category_data['category_return_5years']
etf_df['category_return_5years']=etf_df_category_data['category_return_5years']

mf_df['category_return_10years']=mf_df_category_data['category_return_10years']
etf_df['category_return_10years']=etf_df_category_data['category_return_10years']



def str_to_float(df,column):
    '''
    Converts the values within a column from a string to float",
    '''
    list=[]
    col_id=df.columns.get_loc(column)
    for i in range(df.shape[0]):
        treynor=df.iloc[i,col_id]
        if type(treynor) == None:
            list.append(-999)
        if type(treynor) != None:
            if type(treynor) == str:
                treynor = treynor.replace(',','')
            f=float(treynor)
            list.append(f)
    return(list)


#Fills none types with the average value

def fill_with_mean(df,column):
    '''
    Use:
    This method is an addendum to the str_to_float method.
    '''
    col_id=df.columns.get_loc(column)
    mean=df[df[column]>-990][column].mean()
    for i in range(df.shape[0]):
        treynor=df.iloc[i,col_id]
        if treynor < -990:
            df.iloc[i,col_id]=mean
    return
#converting treynor columns from strings to floats Mutual Funds
mf_df['fund_treynor_ratio_3years'] = str_to_float(mf_df,'fund_treynor_ratio_3years')
fill_with_mean(mf_df,'fund_treynor_ratio_3years')
mf_df['fund_treynor_ratio_5years'] = str_to_float(mf_df,'fund_treynor_ratio_5years')
fill_with_mean(mf_df,'fund_treynor_ratio_5years')
mf_df['fund_treynor_ratio_10years'] = str_to_float(mf_df,'fund_treynor_ratio_10years')
fill_with_mean(mf_df,'fund_treynor_ratio_10years')

#converting treynor columns from strings to floats ETFs
etf_df['fund_treynor_ratio_3years'] = str_to_float(etf_df,'fund_treynor_ratio_3years')
fill_with_mean(etf_df,'fund_treynor_ratio_3years')
etf_df['fund_treynor_ratio_5years'] = str_to_float(etf_df,'fund_treynor_ratio_5years')
fill_with_mean(etf_df,'fund_treynor_ratio_5years')
etf_df['fund_treynor_ratio_10years'] = str_to_float(etf_df,'fund_treynor_ratio_10years')
fill_with_mean(etf_df,'fund_treynor_ratio_10years')


#Cleaning by fund_return
mf_df_return_history=clean_by_keyword(mf_df,'fund_return_')
etf_df_return_history=clean_by_keyword(etf_df,'fund_return_')

#Adds fund_return_ytd column to main datafrme and removes from return_history subframe
mf_df['fund_return_ytd']=mf_df_return_history['fund_return_ytd']
etf_df['fund_return_ytd']=etf_df_return_history['fund_return_ytd']
df_mf_return_history=mf_df_return_history.drop(columns=['fund_return_ytd'])
df_etf_return_history=etf_df_return_history.drop(columns=['fund_return_ytd'])


mf_df_sector=clean_by_keyword(mf_df,'sector')
etf_df_sector=clean_by_keyword(etf_df,'sector')
##print(etf_df_sector.head())

##standard deviation not required for inital analyis
mf_df_standard_deviations= clean_by_keyword(mf_df,'standard_deviation')
etf_df_standard_deviations=clean_by_keyword(etf_df,'standard_deviation').dropna()
##print(mf_df_standard_deviations.head(4))

mf_df_rsquare=clean_by_keyword(mf_df,'squared')
etf_df_rsquare=clean_by_keyword(etf_df,'squared')
##print(etf_df_rsquare.head(4))

mf_df_beta=clean_by_keyword(mf_df,'beta')
etf_df_beta=clean_by_keyword(etf_df,'beta')
##print(etf_df_beta.head())

mf_df_ratio=clean_by_keyword(mf_df,'_ratio')
etf_df_ratio=clean_by_keyword(etf_df,'_ratio')
##print(mf_df_ratio.head(2))

mf_df_alpha=clean_by_keyword(mf_df,'alpha')
etf_df_alpha=clean_by_keyword(etf_df,'alpha')
##print(mf_df_alpha.head(2))


##merging statisitcal columns as they may be used later in the project
stats_mf_df=mf_df_rsquare.merge(mf_df_alpha,how='left')
stats_mf_df=stats_mf_df.merge(mf_df_beta,how='left')
stats_mf_df=stats_mf_df.merge(mf_df_standard_deviations,how='left')
##print(stats_mf_df.sample(10))

stats_etf_df=etf_df_rsquare.merge(etf_df_alpha,how='left')
stats_etf_df=stats_etf_df.merge(etf_df_beta,how='left')
stats_etf_df=stats_etf_df.merge(etf_df_standard_deviations,how='left')
##print(stats_mf_df.sample(10))

###adapted the function so that each column would be divided by 100 - using this for the returns columns. Format in CSV file is 70, need the format as 0.70 for analysis later in script
def percentage_convert(df, keyword):
    """
    Adapted the clean_by_keyword function so that each column would be divided by 100 - using this for the returns columns.
    Format in CSV file is 70, need the format as 0.70 for analysis later in script

    Arguments:
        df: Dataset pandas DataFrame.
        cols: List of columns to be included in calculations.
        keyword: Word to be removed from column name in order to preserve clarity with labeling.
    Return: dataframe containing data matching keyword.
    """
    col_list = [i for i in df.columns if (keyword in i)]
    # print(col_list)
    df_new = df[col_list]/100
    df_new.insert(0, 'fund_symbol', df.fund_symbol)  # inserts the fund symbols as the 1st column of the new data set
    return (df_new)

mf_df_percent=percentage_convert(mf_df,'return')
etf_df_percent=percentage_convert(etf_df,'return')
##print(etf_df_percent.head())


##Scraping data using Yahoo Finance Api - pulling 2011-20 closing price data for NFLX that will be used later
data1 = yf.download(['NFLX'], start="2011-01-01", end="2020-01-01", interval='1mo', group_by='ticker')
#dropping rows with at least one null value - 37 in total dropped
Yahoo1 = data1.dropna(axis = 0, how ='any')
Yahoo1.columns = ['_'.join(col).rstrip('_') for col in Yahoo1.columns.values]
##print(Yahoo1.head())


##print("Original data frame length:", len(data1), "\nUpdated data frame length:",
       #len(Yahoo1), "\nNumber of rows with at least 1 NA value: ",
      #(len(data1)-len(Yahoo1)))

#regular expression function below to search for netflix in the top10 holdings column of the ETF file.

def NFLX_weight(holdings):
    x = re.findall('Netflix\sInc:\s([0-9]{1}.[0-9]{2})', holdings)
    return x

etf_df['NFLX_weight'] = etf_df['top10_holdings'].apply(NFLX_weight)

NFLX = etf_df['NFLX_weight'].sum()
##print(NFLX)
##print(len(NFLX)) ##check to see how many ETFs hold NFLX
##print(max(NFLX)) # check to find the biggest allocation


#### Exploratory data analysis to visualise how our data looks after cleaning ####

def comparison_pies(df1, df2, column, t1, t2):
    '''
    Creates two pie charts representing the relative distribution of funds in a certain category for both fund types

    Arguments:
        df: Pandas dataframe
        df2: Pandas dataframe
        column: column in df that is to be analyzed.
        t1:Title of pie chart 1
        t2: Title of pie chart 2

    '''
    y = df1.groupby(column).fund_symbol.count()
    x = df2.groupby(column).fund_symbol.count()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.pie(y, labels=y.index.to_numpy(), autopct='%.1f%%')
    ax2.pie(x, labels=x.index.to_numpy(), autopct='%.1f%%')

    ax1.set_title(t1)
    ax2.set_title(t2)
    plt.tight_layout()
    plt.show()

##print(comparison_pies(etf_df,mf_df,'size_type','ETFs','Mutual Funds'))
##print(comparison_pies(etf_df,mf_df,'investment_type','ETFs','Mutual Funds'))

##Creates a bar plot of the 25 most common values in the given column


def distribution_by_category(df):
    '''
    Use:
    Creates a barplot of percentage invested in each category

    Arguments
    df: dataframe of funds containing category type.

    '''
    cat_count = df['category'].value_counts()[:25]
    sns.barplot(x=cat_count.values, y=cat_count.index)
    plt.title('Distribution by Category')
    plt.tight_layout
    plt.show()

#ETF distribution by category
plt.xlabel('Number of ETFs')
##print(distribution_by_category(etf_df))

#Mutual funds distribution by category
plt.xlabel('Number of Mutual Funds')
##print(distribution_by_category(mf_df))

mfReturns=mf_df_return_history[['fund_return_1year','fund_return_3years','fund_return_5years','fund_return_10years']].mean()
etfReturns=etf_df_return_history[['fund_return_1year','fund_return_3years','fund_return_5years','fund_return_10years']].mean()


names=['1 year','3 years','5 years','10 years']
mfReturns.set_axis(names,axis=0,inplace=True)
etfReturns.set_axis(names,axis=0,inplace=True)


##plt.plot(mfReturns)
##plt.plot(etfReturns)

plt.title('Fund Returns Over Time')
plt.ylabel('Percent Returns')
plt.xlabel('Fund Investment Period')
plt.legend(['MF','ETF'])

###Time frame analysis, returns over time. How long to hold a position for

mfReturns=mf_df_return_history[['fund_return_1year','fund_return_3years','fund_return_5years','fund_return_10years']].mean()
etfReturns=etf_df_return_history[['fund_return_1year','fund_return_3years','fund_return_5years','fund_return_10years']].mean()


names=['1 year','3 years','5 years','10 years']
mfReturns.set_axis(names,axis=0,inplace=True)
etfReturns.set_axis(names,axis=0,inplace=True)


##plt.plot(mfReturns)
##plt.plot(etfReturns)

plt.title('Fund Returns Over Time')
plt.ylabel('Percent Returns')
plt.xlabel('Fund Investment Period')
plt.legend(['MF','ETF'])

##print(plt.show())

#ETF returns eclipse the returns of MF after 5 years - better for long term holding

###How do the returns over time compare between these fund investing styles?
#sorting by investment type

mfGrowth=mf_df.where(mf_df['investment_type']=='Growth')
mfValue=mf_df.where(mf_df['investment_type']=='Value')
mfBlend=mf_df.where(mf_df['investment_type']=='Blend')

etfGrowth=etf_df.where(etf_df['investment_type']=='Growth')
etfValue=etf_df.where(etf_df['investment_type']=='Value')
etfBlend=etf_df.where(etf_df['investment_type']=='Blend')

#merging fund history
etfGrowth=pd.merge(etfGrowth,df_etf_return_history).dropna()
etfValue=pd.merge(etfValue,df_etf_return_history).dropna()
etfBlend=pd.merge(etfBlend,df_etf_return_history).dropna()

mfGrowth=pd.merge(mfGrowth,df_mf_return_history).dropna()
mfValue=pd.merge(mfValue,df_mf_return_history).dropna()
mfBlend=pd.merge(mfBlend,df_mf_return_history).dropna()

etfGrowthReturns=etfGrowth[['fund_return_1year','fund_return_3years','fund_return_5years','fund_return_10years']].mean()
etfValueReturns=etfValue[['fund_return_1year','fund_return_3years','fund_return_5years','fund_return_10years']].mean()
etfBlendReturns=etfBlend[['fund_return_1year','fund_return_3years','fund_return_5years','fund_return_10years']].mean()

mfGrowthReturns=mfGrowth[['fund_return_1year','fund_return_3years','fund_return_5years','fund_return_10years']].mean()
mfValueReturns=mfValue[['fund_return_1year','fund_return_3years','fund_return_5years','fund_return_10years']].mean()
mfBlendReturns=mfBlend[['fund_return_1year','fund_return_3years','fund_return_5years','fund_return_10years']].mean()
names=['1 year','3 years','5 years','10 years']

etfGrowthReturns.set_axis(names,axis=0,inplace=True)
etfValueReturns.set_axis(names,axis=0,inplace=True)
etfBlendReturns.set_axis(names,axis=0,inplace=True)

mfGrowthReturns.set_axis(names,axis=0,inplace=True)
mfValueReturns.set_axis(names,axis=0,inplace=True)
mfBlendReturns.set_axis(names,axis=0,inplace=True)

##plt.plot(etfGrowthReturns,color = 'green')
##plt.plot(mfGrowthReturns,color = 'black')

##plt.plot(etfBlendReturns,color = 'blue',ls='dotted')
##plt.plot(mfBlendReturns,color = 'red',ls='dotted')

##plt.plot(etfValueReturns,color = 'orange',ls='dashed')
##plt.plot(mfValueReturns,color = 'purple',ls='dashed')

plt.title('Fund Returns Over Time based on Investment Strategy')
plt.ylabel('Percent Returns')
plt.xlabel('Fund Investment Period')
plt.legend(['ETF Growth','MF Growth','ETF Blend','MF Blend','ETF Value','MF Value'])

##print(plt.show())




