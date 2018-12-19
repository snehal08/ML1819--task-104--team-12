import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error


def a(subdata):
    #
    Y = subdata["Purchase"]
    X = subdata.drop(["Purchase"], axis=1)

    #
    LE = LabelEncoder()
    X = X.apply(LE.fit_transform)
    X.Gender = pd.to_numeric(X.Gender)
    X.Age = pd.to_numeric(X.Age)
    X.Occupation = pd.to_numeric(X.Occupation)
    X.City_Category = pd.to_numeric(X.City_Category)
    X.Stay_In_Current_City_Years = pd.to_numeric(X.Stay_In_Current_City_Years)
    X.Marital_Status = pd.to_numeric(X.Marital_Status)
    X.Product_Category_1 = pd.to_numeric(X.Product_Category_1)
    X.Product_Category_2 = pd.to_numeric(X.Product_Category_2)
    X.Product_Category_3 = pd.to_numeric(X.Product_Category_3)

    #StandardScaler
    SS = StandardScaler()

    Xs = SS.fit_transform(X)
    pc = PCA(4)


    principalComponents = pc.fit_transform(X)
    # print(pc.explained_variance_ratio_)
    # array([7.35041374e-01, 2.64935995e-01, 1.10061180e-05, 6.21704987e-06])
    principalDf = pd.DataFrame(data = principalComponents, columns = ["component 1", "component 2", "component 3", "component 4"])

    kf = KFold(20)

    for a,b in kf.split(principalDf):
        X_train, X_test = Xs[a],Xs[b]
        y_train, y_test = Y[a],Y[b]


    lr = LinearRegression()
    dtr = DecisionTreeRegressor()
    rfr = RandomForestRegressor()
    gra = GradientBoostingRegressor()


    fit1 = lr.fit(X_train,y_train)#Here we fit training data to linear regressor
    fit2 = dtr.fit(X_train, y_train)  # Here we fit training data to Decision Tree Regressor
    fit3 = rfr.fit(X_train, y_train)  # Here we fit training data to Random Forest Regressor
    fit4 = gra.fit(X_train, y_train)
    y_pre1 = lr.predict(X_test)
    y_pre2 = dtr.predict(X_test)
    y_pre3 = rfr.predict(X_test)
    y_pre4 = gra.predict(X_test)

    print("Accuracy Score of Linear regression on train set",fit1.score(X_train,y_train)*100)
    print("Accuracy Score of Linear regression on test set",fit1.score(X_test,y_test)*100)
    print("MSE of Linear regression on test set",mean_squared_error(y_pre1,y_test))
    print("MAE of Linear regression on test set", mean_absolute_error(y_pre1, y_test))
    print('----------------------------------------------------')
    print("Accuracy Score of Decision Tree on train set", fit2.score(X_train, y_train) * 100)
    print("Accuracy Score of Decision Tree on test set", fit2.score(X_test, y_test) * 100)
    print("MSE of Descision Tree on test set",mean_squared_error(y_test , y_pre2))
    print("MAE of Descision Tree on test set", mean_absolute_error(y_test, y_pre2))
    print('----------------------------------------------------')
    print("Accuracy Score of Random Forests on train set", fit3.score(X_train, y_train) * 100)
    print("Accuracy Score of Random Forests on test set", fit3.score(X_test, y_test) * 100)
    print("MSE of Random Forests on test set", mean_squared_error(y_test, y_pre3))
    print("MAE of Random Forests on test set", mean_absolute_error(y_test, y_pre3))
    print('----------------------------------------------------')
    print("Accuracy Score of Gradient Boosting on train set", fit4.score(X_train, y_train) * 100)
    print("Accuracy Score of Gradient Boosting on testset", fit4.score(X_test, y_test) * 100)
    print("MSE of Gradient Boosting on test set", mean_squared_error(y_test, y_pre4))
    print("MAE of Gradient Boosting on test set", mean_absolute_error(y_test, y_pre4))
    print('************************************************************************')


if __name__=='__main__':

    bfriday = pd.read_csv("BlackFriday.csv")
    # print(bfriday.shape)
    # print(bfriday.info())
    # print(bfriday.head(10))
    Stay_in_city_years_counts = bfriday['Stay_In_Current_City_Years'].value_counts()
    non_value = bfriday.isnull().sum()
    # print(non_value)

    b = ['Product_Category_2', 'Product_Category_3']
    for i in b:
        exec ("bfriday.%s.fillna(bfriday.%s.value_counts().idxmax(), inplace=True)" % (i, i))

    df_1 = bfriday[:int((len(bfriday.index) / 5)/10)]  # 10k (approx)
    df_2 = bfriday[:int(3 * ((len(bfriday.index) / 5)/10))]  # 20k (approx)
    df_3 = bfriday[:int(10 * ((len(bfriday.index) / 5)/10))]  # 30k (approx)
    df_4 = bfriday[:int(16 * ((len(bfriday.index) / 5)/10))]  # 40k (approx)
    df_5 = bfriday[:int(25 * ((len(bfriday.index) / 5)/10))] # 50k (approx)

    setlist=[df_1,df_2,df_3,df_4,df_5]

    for x in setlist:
        a(x)