#%%
import dash
import dash_html_components as html
import matplotlib.pyplot as plt
import numpy as np
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import math
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import Counter
from scipy import stats
from sklearn.decomposition import PCA
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import StandardScaler
from numpy import linalg as LA
from normal_test import shapiro_test,ks_test,da_k_squared_test
#%%
def quartile(data):
    Q1 = np.percentile(data, 25, method='midpoint')
    Q3 = np.percentile(data, 75, method='midpoint')
    IQR = Q3 - Q1
    return Q1,Q3,IQR
#%%
df = pd.read_csv("/Users/atharvah/GWU/Sem 3 /Data Visualisation/Final Project/archive/fraudTrain.csv") # reading the train data
df = df.set_index(df.trans_date_trans_time)
df = df.drop(columns=["Unnamed: 0","trans_date_trans_time","cc_num","first","last","street","lat","long","dob","unix_time","merch_lat","merch_long"])
df2 = pd.read_csv("/Users/atharvah/GWU/Sem 3 /Data Visualisation/Final Project/archive/fraudTest.csv") # reading the test data
df2 = df2.set_index(df2.trans_date_trans_time)
df2 = df2.drop(columns=["Unnamed: 0","trans_date_trans_time","cc_num","first","last","street","lat","long","dob","unix_time","merch_lat","merch_long"])
fraud = df2[df2.is_fraud==1] # extracting all the 'fraud' transactions from the test dataset
fraud = fraud.append(df[df.is_fraud==1]) # combining all the fraud transactions from the test and the train dataset.
not_fraud = df[df.is_fraud==0] # Extracting all the "not fraud" data.
#%%
df_final = fraud
df_final= df_final.append(not_fraud) # to reduce the imbalance we only take 40500 "not fraud" transactions into consideration.
# Pre-processing the data
print("The number of missing values in the dataset:",df_final.isna().sum().sum()) # Counting the missing values in the data
print("The description of data\n",df_final.describe().to_string())
print("All the entries in the dataset are unique:",len(df_final)==len(set(df_final.trans_num)))
#%%
# Plotting first 100 samples for amount and populations
df_final[["amt","city_pop"]][:100].plot()
plt.title("Visualizing the first 100 samples")
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()
#%%
# z transformed data
def z_transform(df):
    mean = np.mean(df)
    std = np.std(df)
    trans = (df-mean)/std
    return trans

transformed_data = z_transform(df_final[["amt","city_pop"]])

transformed_data[:100].plot()
plt.title("Visualizing the first 100 samples after z transform")
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()
#%%
# Outlier detection for amount using interquartile method
Q1, Q3, IQR = quartile(df_final.amt)
print(f"Q1 and Q3 of the transaction amount is {round(Q1,2)} $ & {round(Q3,2)} $\n"
      f"IQR for the transaction amount is {round(IQR,2)} $\n"
      f"Any amount < {round(Q1-(1.5 * IQR),2)} $ and amount > {round(Q3+(1.5*IQR),2)} $ is an outlier")

#%%
print("The number of outliers present in the transaction amounts:",len(df_final.amt[(df_final.amt<-100.82)|(df_final.amt>193.79)]))
#%%
# Removing the outliers from the data
df_final = df_final[(df_final.amt<(Q3+(1.5*IQR)))&(df_final.amt>(Q1-(1.5 * IQR)))]
#%%
# Box plot for the transaction amounts after removing the outliers
sns.boxplot(df_final.amt)
plt.title("Boxplot for transaction amount outlier detection")
plt.show()
#%%
# From the above boxplot we can see that there are plenty of outlier which are yet to be removed,so we use the boxplot to find out upper limit.
# Removing the outliers from the data using boxplot
df_final = df_final[(df_final.amt< 170)&(df_final.amt>-94.77)]
sns.boxplot(df_final.amt)
plt.title("Boxplot for transaction amount outlier detection")
plt.show()
#%%
# Outlier detection for city population using interquartile method
Q1, Q3, IQR = quartile(df_final.city_pop)
print(f"Q1 and Q3 of the city population is {round(Q1,2)} & {round(Q3,2)}\n"
      f"IQR for the city population is {round(IQR,2)}\n"
      f"Any population < {round(Q1-(1.5 * IQR),2)} and population > {round(Q3+(1.5*IQR),2)} is an outlier")

#%%
print("The number of outliers present in the city population:",len(df_final.city_pop[(df_final.city_pop<-26768.5)|(df_final.city_pop>46547.5)]))
#%%
# Removing the outliers from the data
df_final = df_final[(df_final.city_pop<(Q3+(1.5*IQR)))&(df_final.city_pop>(Q1-(1.5 * IQR)))]
#%%
# Box plot for the city population after removing the outliers
sns.boxplot(df_final.city_pop)
plt.title("Boxplot for city population outlier detection")
plt.show()
#%%
# From the above boxplot we can see that there are plenty of outlier which are yet to be removed,so we use the boxplot to find out upper limit.
# Removing the outliers from the data using boxplot
df_final = df_final[(df_final.city_pop< 3900)&(df_final.city_pop>-94.77)]
sns.boxplot(df_final.city_pop)
plt.title("Boxplot for city populations outlier detection")
plt.show()
#%%
scaler = StandardScaler()
scaled = scaler.fit_transform(df_final[["amt","city_pop"]])
print(scaled)
plt.plot(scaled[:,0][:100],label = "Amount")
plt.plot(scaled[:,1][:100],label = "City Population")
plt.title("Visualizing the first 100 values of the standardised data")
plt.legend()
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.show()
#%%
H =np.matmul(df_final[["amt","city_pop"]].values.T,df_final[["amt","city_pop"]].values)
s, d, v = np.linalg.svd(H)
print("SingularValues = ",d)
print("The condition number for the features = ",LA.cond(df_final[["amt","city_pop"]].values))
#%%
pca = PCA(n_components="mle")
ScaledComponents = pca.fit_transform(scaled)
print("Explained Variance for Scaled Components",pca.explained_variance_ratio_)
cum_sum_eigenvalues = np.cumsum(ScaledComponents)
plt.plot(np.arange(1,len(np.cumsum(pca.explained_variance_ratio_))+1,1),np.cumsum((pca.explained_variance_ratio_)))
plt.xticks(np.arange(1,len(np.cumsum(pca.explained_variance_ratio_))+1,1))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title("cumulative explained variance VS number of components")
plt.grid()
plt.show()
#%%
df_pca = pd.DataFrame(ScaledComponents,columns=['Principal col %i' % i for i in range(ScaledComponents.shape[1])],index=df_final.index)
#%%
OriginalComponents = pca.fit_transform(df_final[["amt","city_pop"]])
print("Explained Variance for Original Components",pca.explained_variance_ratio_)
#%%
H =np.matmul(scaled.T,scaled)
s, d, v = np.linalg.svd(H)
print("SingularValues = ",d)
print("The condition number for the features = ",LA.cond(scaled))
#%%
#%%
# Normality test for transaction amount
qqplot(df_final.amt,line="s",ax=plt.subplot(2,1,1))
plt.title("qqplot for Transaction amount")

plt.subplot(2,1,2)
plt.hist(df_final.amt)
plt.title("Histogram for the transaction amount")
plt.xlabel("Samples")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
print(ks_test(df_final.amt,"Transaction Amount"))
#%%
# Normality test for transaction amount
qqplot(df_final.city_pop,line="s",ax=plt.subplot(2,1,1))
plt.title("qqplot for City Population")

plt.subplot(2,1,2)
plt.hist(df_final.city_pop)
plt.title("Histogram for the City Population")
plt.xlabel("Samples")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
print(ks_test(df_final.city_pop,"City Population"))
#%%
transformed_pop = z_transform(df_final.amt)

qqplot(transformed_pop,line="s",ax=plt.subplot(2,1,1))
plt.title("qqplot for Transaction amount")

plt.subplot(2,1,2)
plt.hist(transformed_pop)
plt.tight_layout()
plt.show()
print(ks_test(transformed_pop,"Transaction Amount"))
# qqplot(transformed_pop,line="s")
# plt.title("qqplot for City population")
# plt.show()
print(ks_test(df_final.city_pop,"City population"))
#%%
corr = df_final.corr()
sns.heatmap(corr,annot = True)
plt.title("Correlation heatmap")
plt.show()
#%%
#Line plots
# sns.lineplot(data = df_final,
#              x = "amt")
# plt.title("Line plot for the Transaction amounts using Seaborn")
# plt.show()

#%%
df_final.city_pop.plot(kind = "line")
plt.title("Line plot for city population")
plt.xticks(rotation = 90)
plt.tight_layout()
plt.show()
#%%
sns.barplot(data = df_final,
            y = "amt",
            x = "category")
plt.xticks(rotation = 90)
plt.title("Categories vs transactions")
plt.tight_layout()
plt.show()
#%%
sns.barplot(data = df_final,
            y = 'amt',
            x = "gender")
plt.title("Gender vs transactions")
plt.tight_layout()
plt.show()
#%%
sns.barplot(data = df_final,
            y = "amt",
            x = "state")
plt.title("Transaction amount vs State")
plt.xticks(rotation = 90)
plt.tight_layout()
plt.show()
#%%
sns.countplot(data = fraud,
              x = "is_fraud",
              hue = "gender")
plt.title("Fraud transaction Male vs Female")
plt.show()
#%%
sns.countplot(data = fraud,
              x = "is_fraud",
              hue = "category")
plt.legend(loc = "upper right")
plt.title("Fraud transaction for categories")
plt.show()
#%%
df_final.category.value_counts().plot(kind = "pie")
plt.title("Pie chart for categories")
plt.show()
#%%
df_final.gender.value_counts().plot(kind = "pie")
plt.title("Pie chart for genders")
plt.legend()
plt.show()
#%%
sns.catplot(data = df_final,
            x = "category",
            y = "amt",
            hue = "gender",
            kind="bar")
plt.title("Catplot for Category and amount with gender")
plt.xticks(rotation = 90)
plt.tight_layout()
plt.show()
#%%
sns.violinplot(data=df_final, x="category", y="city_pop", hue="gender",
               split=True)
plt.title("Violin plot for the state and it's population with gender")
plt.xticks(rotation = 90)
plt.tight_layout()
plt.show()
#%%
sns.violinplot(data=df_final, x="category", y="amt", hue="gender",
               split=True)
plt.title("Violin plot for the state and it's transaction amounts with gender")

plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()
#%%
sns.lmplot(
    data=df_final[:1000],
    x="amt", y="city_pop", hue="gender",
    height=5
)
plt.title("Scatter plot and regression line")
plt.show()
#%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plt.subplot(2,2,1)
df_final.city_pop.plot(kind = "line")
plt.title("Line plot for city population")
plt.xticks(rotation = 90)
plt.tight_layout()


sns.barplot(data = df_final,
            y = "amt",
            x = "category",
            ax = plt.subplot(2,2,2))
plt.xticks(rotation = 90)
plt.title("Categories vs transactions")
plt.tight_layout()


sns.barplot(data = df_final,
            y = 'amt',
            x = "gender",
            ax = plt.subplot(2,2,3))
plt.title("Gender vs transactions")
plt.tight_layout()


sns.barplot(data = df_final,
            y = "amt",
            x = "state",
            ax = plt.subplot(2,2,4))
plt.title("Transaction amount vs State")
plt.xticks(rotation = 90)
plt.tight_layout()
plt.show()
#%%

sns.countplot(data = fraud,
              x = "is_fraud",
              hue = "gender",
              ax = plt.subplot(2,2,1))
plt.title("Fraud transaction Male vs Female")


sns.countplot(data = fraud,
              x = "is_fraud",
              hue = "category",
              ax = plt.subplot(2,2,4))
plt.legend(loc = "upper right")
plt.title("Fraud transaction for categories")

plt.subplot(2,2,3)
df_final.category.value_counts().plot(kind = "pie")
plt.title("Pie chart for categories")


ax = plt.subplot(2,2,2)
df_final.gender.value_counts().plot(kind = "pie")
plt.title("Pie chart for genders")
plt.legend()
plt.show()
#%%
#%%
print("Recommendation\n 1: Making different plots \n 2: Understand the data with the help of plots \n 3:The dash app makes it easy in terms of the interface.")