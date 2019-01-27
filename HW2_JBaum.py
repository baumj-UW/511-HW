'''
Created on Jan 24, 2019

@author: Jacqueline Baum
EE 511 -- HW 2
Computer Assignment with Ames Housing data
'''

import numpy as np
from sklearn import metrics, linear_model
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import pandas as pd


amesfile = "C:/Users/baumj/OneDrive/Documents/UW Courses/EE 511 -\
 Intro to Statistical Learning/Python Coding Problems/HW2/AmesHousing.txt"
#types = ()  
numerical_variables = ['Lot_Area', 'Lot_Frontage', 'Year_Built',\
'Mas_Vnr_Area', 'BsmtFin_SF_1', 'BsmtFin_SF_2',\
'Bsmt_Unf_SF', 'Total_Bsmt_SF', '1st_Flr_SF',\
'2nd_Flr_SF', 'Low_Qual_Fin_SF', 'Gr_Liv_Area',\
'Garage_Area', 'Wood_Deck_SF', 'Open_Porch_SF',\
'Enclosed_Porch', '3Ssn_Porch', 'Screen_Porch','Pool_Area']

discrete_variables = ['MS_SubClass','MS_Zoning','Street',\
                      'Alley','Lot_Shape','Land_Contour',\
                      'Utilities','Lot_Config','Land_Slope',\
                      'Neighborhood','Condition_1','Condition_2',\
                      'Bldg_Type','House_Style','Overall_Qual',\
                      'Overall_Cond','Roof_Style','Roof_Matl',\
                      'Exterior_1st','Exterior_2nd','Mas_Vnr_Type',\
                      'Exter_Qual','Exter_Cond','Foundation',\
                      'Bsmt_Qual','Bsmt_Cond','Bsmt_Exposure',\
                      'BsmtFin_Type_1','Heating','Heating_QC',\
                      'Central_Air','Electrical','Bsmt_Full_Bath',\
                      'Bsmt_Half_Bath','Full_Bath','Half_Bath',\
                      'Bedroom_AbvGr','Kitchen_AbvGr','Kitchen_Qual',\
                      'TotRms_AbvGrd','Functional','Fireplaces',\
                      'Fireplace_Qu','Garage_Type','Garage_Cars',\
                      'Garage_Qual','Garage_Cond','Paved_Drive',\
                      'Pool_QC','Fence','Sale_Type','Sale_Condition']

# numerical_variables = ['Lot_Area':0.0, 'Lot_Frontage':0.0, 'Year_Built':0.0,\
# 'Mas_Vnr_Area':0.0, 'BsmtFin_SF_1':0.0, 'BsmtFin_SF_2':0.0,\
# 'Bsmt_Unf_SF':0.0, 'Total_Bsmt_SF':0.0, '1st_Flr_SF':0.0,\
# '2nd_Flr_SF':0.0, 'Low_Qual_Fin_SF':0.0, 'Gr_Liv_Area':0.0,\
# 'Garage_Area':0.0, 'Wood_Deck_SF':0.0, 'Open_Porch_SF':0.0,\
# 'Enclosed_Porch':0.0, '3Ssn_Porch':0.0, 'Screen_Porch':0.0,'Pool_Area':0.0]

# discrete_variables = ['MS_SubClass':'null','MS_Zoning':'null','Street',\
# 'Alley','Lot_Shape':'null','Land_Contour',\
# 'Utilities':'null','Lot_Config':'null','Land_Slope',\
# 'Neighborhood':'null','Condition_1':'null','Condition_2',\
# 'Bldg_Type':'null','House_Style':'null','Overall_Qual',\
# 'Overall_Cond':'null','Roof_Style':'null','Roof_Matl',\
# 'Exterior_1st':'null','Exterior_2nd':'null','Mas_Vnr_Type',\
# 'Exter_Qual':'null','Exter_Cond':'null','Foundation',\
# 'Bsmt_Qual':'null','Bsmt_Cond':'null','Bsmt_Exposure',\
# 'BsmtFin_Type_1':'null','Heating':'null','Heating_QC',\
# 'Central_Air':'null','Electrical':'null','Bsmt_Full_Bath',\
# 'Bsmt_Half_Bath':'null','Full_Bath':'null','Half_Bath',\
# 'Bedroom_AbvGr':'null','Kitchen_AbvGr':'null','Kitchen_Qual',\
# 'TotRms_AbvGrd':'null','Functional':'null','Fireplaces',\
# 'Fireplace_Qu':'null','Garage_Type':'null','Garage_Cars',\
# 'Garage_Qual':'null','Garage_Cond':'null','Paved_Drive',\
# 'Pool_QC':'null','Fence':'null','Sale_Type':'null','Sale_Condition':'null']

amesdata = np.genfromtxt(amesfile,dtype=None,delimiter='\t', names=True,\
                          filling_values=0) #fills everything with zeros, may need to fix
#                               'Lot_Area':0.0, 'Lot_Frontage':0.0, 'Year_Built':0.0,\
# 'Mas_Vnr_Area':0.0, 'BsmtFin_SF_1':0.0, 'BsmtFin_SF_2':0.0,\
# 'Bsmt_Unf_SF':0.0, 'Total_Bsmt_SF':0.0, '1st_Flr_SF':0.0,\
# '2nd_Flr_SF':0.0, 'Low_Qual_Fin_SF':0.0, 'Gr_Liv_Area':0.0,\
# 'Garage_Area':0.0, 'Wood_Deck_SF':0.0, 'Open_Porch_SF':0.0,\
# 'Enclosed_Porch':0.0, '3Ssn_Porch':0.0, 'Screen_Porch':0.0,'Pool_Area':0.0})#\
# 'MS_SubClass':'null','MS_Zoning':'null','Street':'null',\
# 'Alley':'null','Lot_Shape':'null','Land_Contour':'null',\
# 'Utilities':'null','Lot_Config':'null','Land_Slope':'null',\
# 'Neighborhood':'null','Condition_1':'null','Condition_2':'null',\
# 'Bldg_Type':'null','House_Style':'null','Overall_Qual':'null',\
# 'Overall_Cond':'null','Roof_Style':'null','Roof_Matl':'null',\
# 'Exterior_1st':'null','Exterior_2nd':'null','Mas_Vnr_Type':'null',\
# 'Exter_Qual':'null','Exter_Cond':'null','Foundation':'null',\
# 'Bsmt_Qual':'null','Bsmt_Cond':'null','Bsmt_Exposure':'null',\
# 'BsmtFin_Type_1':'null','Heating':'null','Heating_QC':'null',\
# 'Central_Air':'null','Electrical':'null','Bsmt_Full_Bath':'null',\
# 'Bsmt_Half_Bath':'null','Full_Bath':'null','Half_Bath':'null',\
# 'Bedroom_AbvGr':'null','Kitchen_AbvGr':'null','Kitchen_Qual':'null',\
# 'TotRms_AbvGrd':'null','Functional':'null','Fireplaces':'null',\
# 'Fireplace_Qu':'null','Garage_Type':'null','Garage_Cars':'null',\
# 'Garage_Qual':'null','Garage_Cond':'null','Paved_Drive':'null',\
# 'Pool_QC':'null','Fence':'null','Sale_Type':'null','Sale_Condition':'null'})

# 'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2',
# 'Bsmt Unf SF', 'Total Bsmt SF', '1st Flr SF',
# '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area',
# 'Garage Area', 'Wood Deck SF', 'Open Porch SF',
# 'Enclosed Porch', '3Ssn Porch', 'Screen Porch',
# 'Pool Area'})

# with open(amesfile,'r') as f:
#     amesdata = f.read()

## Part 3 - split the data ##
training = np.delete(amesdata,np.s_[2::5],0) # Remove Ord mod 5=3
training = np.delete(training,np.s_[2::4],0) # Remove Ord mod5 = 4
validation = amesdata[2::5]
testdata = amesdata[3::5]


## Part 4 - Simple Linear Regression ##
feat1 = 'Gr_Liv_Area'
feat2 = 'SalePrice'
regr = linear_model.LinearRegression()
regr.fit(training[feat1].reshape(-1,1),training[feat2].reshape(-1,1))

print('Regression Equation','y=',regr.coef_,'*x+',regr.intercept_)

#Get line values
x_liv_area = np.linspace(np.min(training[feat1]),np.max(training[feat1]))
eqn_pred = regr.coef_*x_liv_area+regr.intercept_  # could change this to just predict(x)

plt.scatter(feat1,feat2,data=training,label='Training Data')
plt.plot(x_liv_area,eqn_pred.T,'r',label='Simple Regression')
plt.title('Simple Linear Regression')
plt.xlabel(feat1)
plt.ylabel(feat2)
plt.legend()
#plt.show()

valid_pred = regr.predict(validation[feat1].reshape(-1,1))
valid_rmse = np.sqrt(metrics.mean_squared_error(validation[feat2].reshape(-1,1),\
                                                 valid_pred))
print('RMSE of Validation set:',valid_rmse)


## Part 5 - Add more features ##

# Transform categorical to one-hot encoding
ames_enc = OneHotEncoder()
ames_enc.fit(training['Alley'].reshape(-1,1))
#make for loop to parse through categories and reconcat 
hot_train = pd.DataFrame(training.copy())
hot_valid = pd.DataFrame(validation.copy())


#hot_train = pd.DataFrame(training[numerical_variables].copy())

# test['Alley'] = pd.get_dummies(test['Alley'])
# test = pd.concat([test,pd.get_dummies(test['Alley'],prefix='Alley')],axis=1)

disc_hotnames= []
#Separate categorical variables into one-hot vectors
for i in discrete_variables:
    df_train = pd.get_dummies(hot_valid[i],prefix=i)
    disc_hotnames += df_train.columns.tolist()
    hot_train=pd.concat([hot_train,df_train],axis=1).drop([i],axis=1)

    df_valid=pd.get_dummies(hot_valid[i],prefix=i)
    hot_valid=pd.concat([hot_valid,df_valid],axis=1).drop([i],axis=1)
   
    #hot_train=pd.concat([hot_train,pd.get_dummies(hot_train[i],prefix=i)],axis=1).drop([i],axis=1)
    #hot_valid=pd.concat([hot_valid,pd.get_dummies(hot_valid[i],prefix=i)],axis=1).drop([i],axis=1)


## Part 5 - Full Regression ##

full_regr = linear_model.LinearRegression(normalize=True)  #check if normalize makes sense
full_regr.fit(hot_train[numerical_variables],hot_train[feat2])

# need to fix ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
# find columns with those things (fix input correctly?) 
    

hotvalid_pred = full_regr.predict(hot_valid[numerical_variables]) 
hotvalid_rmse = np.sqrt(metrics.mean_squared_error(hot_valid[feat2],hotvalid_pred))



## Part 6 - L1 Regularization (Lasso) ##
#Normalize features
#hot_train_scale = preprocessing.scale(hot_train)

l1_regr = linear_model.Lasso(alpha=250,normalize=True)
l1_regr.fit(hot_train[numerical_variables],hot_train[feat2]) # this should include all the features
# 
# l1valid_pred = l1_regr.predict(hot_valid[numerical_variables])
# l1valid_rmse = np.sqrt(metrics.mean_squared_error(hot_valid[feat2],hotvalid_pred))

l1valid_rmse = np.zeros(9)
i=0
for alpha in range(50,500,50):
    l1_regr = linear_model.Lasso(alpha,normalize=True)
    l1_regr.fit(hot_train[numerical_variables],hot_train[feat2])
    l1valid_pred = l1_regr.predict(hot_valid[numerical_variables])
    l1valid_rmse[i] = np.sqrt(metrics.mean_squared_error(hot_valid[feat2],hotvalid_pred))
    i +=1

plt.plot(range(50,500,50),l1valid_rmse)
plt.show()
   





