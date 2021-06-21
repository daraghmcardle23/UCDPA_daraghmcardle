
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import statsmodels.api as sm
rnd_seed_state = 1


etf_df = pd.read_csv('ETFs.csv', low_memory=False)
mf_df = pd.read_csv('Mutual Funds.csv', low_memory=False)




#Throwing away 4 completely empty columns in df_etf - ['category_return_2019', 'category_return_2018', 'category_return_2017', 'category_return_2016']: Same functions used on the script as part 1

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

def str_to_float(df,column):
    '''
    This method converts the values of a specific column in a given dataframe from strings to floats\n",
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
#I designed the str_to_float method to assign -100 for missing Treynor ratios so they could be exclded from calculations.
def fill_with_mean(df,column):
    '''
    Use:
    This method is an addendum to the str_to_float method. A lot of the
    Fills the specified column in df with
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



### Creates and returns a subset of df_mf by keeping only the columns with 'keyword' in their column name. Iterating across the dataframe to make subsets and then merge them together.
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


mf_df_ratio=clean_by_keyword(mf_df,'_ratio') #both datasets are used to analyse what ratios are important, treynor, sharpe for MFs and ETFs
etf_df_ratio=clean_by_keyword(etf_df,'_ratio')
##print(mf_df_ratio.head(2))


##Random Forest Regressor to find which ratios are most important to consider when investing in mutual funds.

new_mf_df_ratios = mf_df_ratio.drop(['fund_symbol'], axis = 1)
##print(new_mf_df_ratios.columns)
new_mf_df_ratios['fund_return_ytd'] = mf_df['fund_return_ytd'] ##adding fund return YTD to our dataframe, the dependent variable
##print(new_mf_df_ratios.columns)

##RFR to tell us which ratios are the most important to consider when investing in mutual funds and ETFs on a YTD basis

model1 = RandomForestRegressor()
new_mf_rats = new_mf_df_ratios.dropna()
y1 = new_mf_rats['fund_return_ytd']
X1 = new_mf_rats.drop(['fund_return_ytd'], axis = 1)
model1.fit(X1,y1)

mf_feat = pd.DataFrame(np.array(model1.feature_importances_).T, index = X1.columns, columns = ['feature_importances']).sort_values(ascending = False, by = 'feature_importances')
mf_feat['correlations'] = new_mf_rats.corr()['fund_return_ytd'].drop(['fund_return_ytd'])

exog1 = sm.add_constant(X1,prepend = False) ##fits an intercept
linearmodel1 = sm.OLS(y1, exog1)
result1 = linearmodel1.fit() ##creates a best line fit
##print(result1.summary())

mf_feat['p_values'] = pd.Series(result1.pvalues.round(4)).drop('const')
##print(mf_feat) ##prints a table with the feature importance, correlation and p values sorted in descending order by feature importance


new_etf_df_ratios = etf_df_ratio.drop(['fund_symbol'], axis = 1)
new_etf_df_ratios['fund_return_ytd'] = etf_df['fund_return_ytd']

model2 = RandomForestRegressor()
new_etf_rats = new_etf_df_ratios.dropna()
y2 = new_etf_rats['fund_return_ytd']
X2 = new_etf_rats.drop(['fund_return_ytd'], axis = 1)
model1.fit(X2,y2)

etf_feat = pd.DataFrame(np.array(model1.feature_importances_).T, index = X2.columns, columns = ['feature_importances']).sort_values(ascending = False, by = 'feature_importances')
etf_feat['correlations'] = new_mf_rats.corr()['fund_return_ytd'].drop(['fund_return_ytd'])

exog2 = sm.add_constant(X2,prepend = False)
linearmodel2 = sm.OLS(y2, exog2)
result2 = linearmodel2.fit()
##print(result2.summary())

etf_feat['p_values'] = pd.Series(result2.pvalues.round(4)).drop('const')
##print(etf_feat)


##running the models again to compare against fund returns across 10 years

new_mf_df_ratios['fund_return_10years'] = mf_df['fund_return_10years']


model3 = RandomForestRegressor()
new_mf_rats = new_mf_df_ratios.dropna()
y3 = new_mf_rats['fund_return_10years']
X3 = new_mf_rats.drop(['fund_return_10years'], axis = 1)
model3.fit(X3,y3)

mf_feat2 = pd.DataFrame(np.array(model3.feature_importances_).T, index = X3.columns, columns = ['feature_importances']).sort_values(ascending = False, by = 'feature_importances')
mf_feat2['correlations'] = new_mf_rats.corr()['fund_return_10years'].drop(['fund_return_10years'])

exog3 = sm.add_constant(X3,prepend = False)
linearmodel3 = sm.OLS(y3, exog3)
result3 = linearmodel3.fit()
##print(result3.summary())

mf_feat2['p_values'] = pd.Series(result3.pvalues.round(4)).drop('const')
##print(mf_feat2)


new_etf_df_ratios['fund_return_10years'] = etf_df['fund_return_10years']

model4 = RandomForestRegressor()
new_etf_rats = new_etf_df_ratios.dropna()
y4 = new_etf_rats['fund_return_10years']
X4 = new_etf_rats.drop(['fund_return_10years'], axis = 1)
model1.fit(X4,y4)

etf_feat2 = pd.DataFrame(np.array(model1.feature_importances_).T, index = X4.columns, columns = ['feature_importances']).sort_values(ascending = False, by = 'feature_importances')
etf_feat2['correlations'] = new_mf_rats.corr()['fund_return_10years'].drop(['fund_return_10years'])

exog4 = sm.add_constant(X4,prepend = False)
linearmodel4 = sm.OLS(y4, exog4)
result4 = linearmodel4.fit()
##print(result4.summary())

etf_feat2['p_values'] = pd.Series(result2.pvalues.round(4)).drop('const')
##print(etf_feat2)

