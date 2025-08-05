import pandas as pd
import joblib
from utils.cleaning import preprocesar

def predecir_ventas(ruta_csv_nuevo, ruta_model='modelo.pkl'):
    df_nuevo = preprocesar(ruta_csv_nuevo)

    #cargar nuevos datos
    modelo = joblib.load(ruta_model)

    #seleccionar mismas columnas que uso el entrenamiento

    X_nuevo = df_nuevo[['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Year', 'Month']]

    #hacer prediccion
    predeciciones = modelo.predict(X_nuevo)

    #Agregar predicciones al Dataframe
    df_nuevo['Prediccion_Ventas'] = predeciciones

    #Mostrar resultados

    print(df_nuevo[['Date','Store','Prediccion_Ventas']].head())

    return df_nuevo


