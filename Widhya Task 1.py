#!/usr/bin/env python
# coding: utf-8

# ### Step 1:

# In[2]:


import pandas as pd
df=pd.read_csv("https://raw.githubusercontent.com/WidhyaOrg/datasets/master/bitcoin_dataset.csv")
df


# ### Step 2:

# In[3]:


df.head()


# ### Step 3: 

# In[4]:


df.loc[1023]["btc_market_price"]


# In[5]:


import seaborn as sns
from scipy.stats import pearsonr


# In[6]:


sns.jointplot(data=df,x="btc_total_bitcoins", y="btc_market_price",kind="scatter", stat_func=pearsonr)


# In[7]:


sns.jointplot(data=df,x="btc_market_cap", y="btc_market_price",kind="scatter", stat_func=pearsonr)


# In[8]:


sns.jointplot(data=df,x="btc_blocks_size", y="btc_market_price",kind="scatter", stat_func=pearsonr)


# In[9]:


sns.jointplot(data=df,x="btc_avg_block_size", y="btc_market_price",kind="scatter", stat_func=pearsonr)


# In[10]:


sns.jointplot(data=df,x="btc_n_transactions_per_block", y="btc_market_price",kind="scatter", stat_func=pearsonr)


# In[11]:


sns.jointplot(data=df,x="btc_median_confirmation_time", y="btc_market_price",kind="scatter", stat_func=pearsonr)


# In[12]:


sns.jointplot(data=df,x="btc_cost_per_transaction_percent", y="btc_market_price",kind="scatter", stat_func=pearsonr)


# In[13]:


sns.jointplot(data=df,x="btc_cost_per_transaction_percent", y="btc_market_price",kind="scatter", stat_func=pearsonr)


# In[14]:


sns.jointplot(data=df,x="btc_n_unique_addresses", y="btc_market_price",kind="scatter", stat_func=pearsonr)


# In[15]:


sns.jointplot(data=df,x="btc_n_transactions", y="btc_market_price",kind="scatter", stat_func=pearsonr)


# In[16]:


sns.jointplot(data=df,x="btc_n_transactions_total", y="btc_market_price",kind="scatter", stat_func=pearsonr)


# In[17]:


sns.jointplot(data=df,x="btc_n_transactions_excluding_popular", y="btc_market_price",kind="scatter", stat_func=pearsonr)


# In[18]:


sns.jointplot(data=df,x="btc_n_transactions_excluding_chains_longer_than_100", y="btc_market_price",kind="scatter", stat_func=pearsonr)


# In[19]:


sns.jointplot(data=df,x="btc_output_volume", y="btc_market_price",kind="scatter", stat_func=pearsonr)


# In[20]:


sns.jointplot(data=df,x="btc_estimated_transaction_volume", y="btc_market_price",kind="scatter", stat_func=pearsonr)


# In[21]:


sns.jointplot(data=df,x="btc_estimated_transaction_volume_usd", y="btc_market_price",kind="scatter", stat_func=pearsonr)


# In[22]:


df["btc_market_cap"].isna().sum()


# In[23]:


df["btc_n_transactions"].isna().sum()


# In[24]:


df["btc_miners_revenue"].isna().sum()


# In[25]:


df["btc_cost_per_transaction"].isna().sum()


# In[26]:


df["btc_difficulty"].isna().sum()


# In[27]:


df["btc_hash_rate"].isna().sum()


# In[28]:


df["btc_cost_per_transaction_percent"].isna().sum()


# In[29]:


test=df.fillna(df.mean())
test


# In[30]:


test.isna().sum()


# In[31]:


test["btc_difficulty"].isna().sum()


# In[32]:


from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


# In[34]:


feature_cols = ['btc_total_bitcoins','btc_market_cap','btc_trade_volume','btc_blocks_size','btc_avg_block_size','btc_n_orphaned_blocks','btc_n_transactions_per_block','btc_median_confirmation_time','btc_cost_per_transaction_percent','btc_cost_per_transaction',
            'btc_n_unique_addresses','btc_n_transactions','btc_n_transactions_total','btc_n_transactions_excluding_popular','btc_n_transactions_excluding_chains_longer_than_100','btc_output_volume','btc_estimated_transaction_volume','btc_estimated_transaction_volume_usd']
X=test[feature_cols]
Y=test["btc_market_price"]


# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y)


# In[37]:


from sklearn.linear_model import LinearRegression


# In[38]:


linreg = LinearRegression()


# In[39]:


df = pd.DataFrame(data=test)


# In[40]:


del df['Date']


# In[41]:


linreg.fit(X_train, Y_train)


# In[42]:


Y_pred = linreg.predict(X_test)


# In[43]:


print(Y_pred)


# In[44]:


print(mean_squared_error(Y_test,Y_pred))


# In[ ]:




