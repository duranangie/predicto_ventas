import pandas as pd

def cargar_datos(ruta_csv):
    """
    Carga el archivo csv y convierte la columna 'Date' a formato datetime
    """
    df = pd.read_csv(ruta_csv)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    return df


def crear_variables(df):
    """
    Crear variables útiles como el mes, año y semana
    """

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week


    return df


def preparar_dataset(df):
   
    """
    Elimina columnas innecesarias y deja solo las relevantes.
    """
    columnas_utiles = ['Date','Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Year', 'Month','Weekly_Sales']
    df = df[columnas_utiles]
    return df


def preprocesar(ruta_csv):
    """
    pipeline complejos: cargar, transformar y devolver el Dataframe final listo para el modelo
    """
    df = cargar_datos(ruta_csv)
    df= crear_variables(df)
    df = preparar_dataset(df)
    return df


