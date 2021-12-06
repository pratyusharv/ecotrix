# Import Required Libraries
import numpy as np
import pandas as pd 
import seaborn as sb
import matplotlib.pyplot as plt

# Read CSV file
data = pd.read_csv('data.csv')

# Initalising the required arrays
Y = np.array(data['Y']) # Response Matrix

X = np.array(data['X'], dtype=float)
Z = np.array(data['Z'], dtype=float)
F = np.array(data['F'], dtype=float)

X = np.c_[np.ones((100,1),dtype=float),  X, Z, F] # Design Matrix

# Solving to OLS Parameters:

B = (np.transpose(X))
#print("[X]'= ", B )

B = B@X
#print("[X]'*[X]= ", B)

B = np.linalg.inv(B)
#print("inv([X]'*[X]) ", B)

B = B@(np.transpose(X))
#print("inv([X]'*[X])*[X]' ", B)

B = B@Y
print("Parameter Matrix:  ", B)

e = Y - X@B

#print(np.mean(e))
#sb.displot(e, kind="kde", color='blue')
#plt.xlabel("Error e")
#plt.ylabel("Density")
#plt.title("KDE of Error Matrix")
#plt.show()

# Finding Standard Errors:
s2 = np.transpose(Y)@Y - (np.transpose(B)@np.transpose(X))@Y
s2 = s2/96

vcovar = s2*np.linalg.inv((np.transpose(X)@X))

print("SE in OLS: ", np.sqrt(np.diagonal(vcovar)))

## MLE:

s2_m = (s2*96)/100

vcovar_m = s2_m*np.linalg.inv((np.transpose(X)@X))

print("SE in MLE: ", np.sqrt(np.diagonal(vcovar_m)))





