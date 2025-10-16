"""
Módulo para cargar y preparar los datos del dashboard de Minsait Motors
"""
import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    """
    Carga el archivo CSV y realiza las transformaciones iniciales necesarias.
    
    Returns:
        pd.DataFrame: DataFrame con los datos de ventas de vehículos
    """
    # Cargar el archivo CSV
    df = pd.read_csv('Car Sales.csv')
    
    # Convertir la columna Date a formato datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extraer año y mes para análisis temporal
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%B')
    df['Year_Month'] = df['Date'].dt.to_period('M')
    
    # Limpiar espacios en blanco en las columnas de texto
    df['Company'] = df['Company'].str.strip()
    df['Model'] = df['Model'].str.strip()
    df['Transmission'] = df['Transmission'].str.strip()
    df['Gender'] = df['Gender'].str.strip()
    
    return df


def calculate_kpis(df):
    """
    Calcula los indicadores clave de rendimiento (KPIs) del dashboard.
    Compara las ventas actuales con las ventas del año anterior.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de ventas
        
    Returns:
        dict: Diccionario con los KPIs calculados
    """
    # Obtener el año más reciente en los datos
    año_actual = df['Year'].max()
    año_anterior = año_actual - 1
    
    # Calcular ventas del año actual en millones de euros
    ventas_año_actual = df[df['Year'] == año_actual]['Price ($)'].sum() / 1_000_000
    
    # Calcular ventas del año anterior como objetivo (en millones de euros)
    ventas_año_anterior = df[df['Year'] == año_anterior]['Price ($)'].sum() / 1_000_000
    
    # Si no hay datos del año anterior, usar las ventas actuales como objetivo
    objetivo = ventas_año_anterior if ventas_año_anterior > 0 else ventas_año_actual
    
    # Calcular porcentaje de cumplimiento del objetivo (comparación con año anterior)
    if objetivo > 0:
        porcentaje_objetivo = ((ventas_año_actual - objetivo) / objetivo) * 100
    else:
        porcentaje_objetivo = 0
    
    # Calcular total de ventas (número de vehículos) del año actual
    total_ventas_año_actual = len(df[df['Year'] == año_actual])
    
    # Calcular total de ventas del año anterior
    total_ventas_año_anterior = len(df[df['Year'] == año_anterior])
    
    return {
        'ventas_anuales': ventas_año_actual,
        'objetivo': objetivo,
        'porcentaje_objetivo': porcentaje_objetivo,
        'total_ventas': total_ventas_año_actual,
        'año_actual': año_actual,
        'año_anterior': año_anterior,
        'total_ventas_año_anterior': total_ventas_año_anterior
    }


def filter_data(df, date_range=None, companies=None, transmissions=None, genders=None):
    """
    Filtra el DataFrame según los criterios especificados.
    
    Args:
        df (pd.DataFrame): DataFrame original
        date_range (tuple): Tupla con (fecha_inicio, fecha_fin)
        companies (list): Lista de compañías a incluir
        transmissions (list): Lista de tipos de transmisión
        genders (list): Lista de géneros
        
    Returns:
        pd.DataFrame: DataFrame filtrado
    """
    filtered_df = df.copy()
    
    # Filtrar por rango de fechas
    if date_range:
        # Convertir las fechas a datetime para comparación correcta
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        filtered_df = filtered_df[
            (filtered_df['Date'] >= start_date) & 
            (filtered_df['Date'] <= end_date)
        ]
    
    # Filtrar por compañías
    if companies:
        filtered_df = filtered_df[filtered_df['Company'].isin(companies)]
    
    # Filtrar por tipo de transmisión
    if transmissions:
        filtered_df = filtered_df[filtered_df['Transmission'].isin(transmissions)]
    
    # Filtrar por género
    if genders:
        filtered_df = filtered_df[filtered_df['Gender'].isin(genders)]
    
    return filtered_df
