'''
Created on Jan 24, 2019

@author: Jacqueline Baum
EE 511 -- HW 2
Computer Assignment with Ames Housing data
'''

import numpy as np
from sklearn import metrics, linear_model
import matplotlib.pyplot as plt 


amesfile = "C:/Users/baumj/OneDrive/Documents/UW Courses/EE 511 -\
 Intro to Statistical Learning/Python Coding Problems/HW2/AmesHousing.txt"
#types = ()  
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

training = np.delete(amesdata,np.s_[2::5],0) # Remove Ord mod 5=3
training = np.delete(training,np.s_[2::4],0) # Remove Ord mod5 = 4
validation = amesdata[2::5]
testdata = amesdata[3::5]


## Part 3 - Simple Linear Regression ##
feat1 = 'Gr_Liv_Area'
feat2 = 'SalePrice'
regr = linear_model.LinearRegression()
regr.fit(training[feat1].reshape(-1,1),training[feat2].reshape(-1,1))

print('Regression Equation','y=',regr.coef_,'*x+',regr.intercept_)


x_liv_area = np.linspace(np.min(training[feat1]),np.max(training[feat1]))
eqn_pred = regr.coef_*x_liv_area+regr.intercept_

plt.scatter(feat1,feat2,data=training,label='Training Data')
plt.plot(x_liv_area,eqn_pred.T,'r',label='Simple Regression')
plt.title('Simple Linear Regression')
plt.xlabel(feat1)
plt.ylabel(feat2)
plt.legend()
plt.show()


print("did it work?")



