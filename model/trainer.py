import pandas as pd # Leer los csv
from sklearn.model_selection import train_test_split # dividir los datos para predecir
from sklearn.metrics import mean_absolute_error, r2_score #Medir resultados
from sklearn.ensemble import RandomForestRegressor # modelo de prediccion
import joblib # para guardar modelo
import os  # manejo de carpeta


from utils.cleaning import preprocesar

def entrenar_modelo(ruta_csv, ruta_guardado='modelo.pkl'):
    #1. Preprocesar datos
    df=preprocesar(ruta_csv)  #usamos el porcesamiento de cleaning
    
    #2. variable para el modelo

    X = df[['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Year', 'Month']]
    y = df['Weekly_Sales'] 
   

    #3. Separar datos en entrenamiento y prueba  
    X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42) #Divide en entreanmiento 80% y prueba 20% 

    #4. Modelo

    modelo = RandomForestRegressor(n_estimators=100, random_state=42) #ENtrenar random forest
    modelo.fit(X_train,y_train)

    #5.Evaluacion

    predicciones = modelo.predict(X_test)  #Mide el MSE
    mse = mean_absolute_error(y_test, predicciones)
    print(f"Error cuadratico medio (MSE): {mse:.2f}")

    # Guardar modelo

    joblib.dump(modelo, ruta_guardado)  # Guardar el modelo del archivo .pkl para usarlo despues
    print(f"Modelo guardado en {ruta_guardado}")