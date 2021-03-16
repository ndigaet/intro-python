import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

module_name = 'regression_helper_v2'

def get_linear_regression(filename, plot_data=False):
    df = pd.read_csv(filename)
    lr = LinearRegression()
    lr.fit(df['Peso'].values.reshape(-1, 1), df['Altura'].values.reshape(-1, 1))
    
    if(plot_data):
        x = np.linspace(20, 130, 100) # crea 100 valores entre 20 y 130        
        y = lr.predict(x.reshape(-1,1)) # es la predicción de los coeficientes
        plt.scatter(df['Peso'].values, df['Altura'].values)
        plt.plot(x, y, c='y')        

    return lr.coef_[0][0], lr.intercept_[0]

#if(__name__ == '__main__'):
#    print('Está corriendo como script')
#else:
#    print('Se está importando como módulo')

if(__name__ == '__main__'):
    print(get_linear_regression('data/alturas-pesos.csv'))