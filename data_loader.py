"""
Módulo para cargar y preparar los datos del dashboard de Minsait Motors
"""
import pandas as pd
import streamlit as st
from auth_client import AuthClient

@st.cache_data(ttl=60)  # Cache se actualiza cada 60 segundos
def load_data(_token: str):
    """
    Carga los datos desde la API de Minsait Motors y realiza las transformaciones iniciales necesarias.
    
    Args:
        _token (str): Token de autenticación (el guion bajo evita que se use en el hash del cache)
    
    Returns:
        pd.DataFrame: DataFrame con los datos de ventas de vehículos
    """
    # Obtener datos desde la API (sin límite para obtener todos los registros)
    auth_client = AuthClient()
    result = auth_client.get_sales(_token)
    
    if not result["success"]:   
        st.error(f"Error al cargar datos desde la API: {result['error']}")
        st.info("Verifica que la API esté funcionando correctamente.")
        return pd.DataFrame()
    
    # Convertir a DataFrame
    df = pd.DataFrame(result["data"])
    
    if df.empty:
        st.warning("No hay datos disponibles en la base de datos.")
        return df
    
    # Renombrar columnas para mantener compatibilidad con el resto del código
    # La API devuelve 'id' pero el código espera 'Car_id'
    if 'id' in df.columns:
        df.rename(columns={'id': 'Car_id'}, inplace=True)
    
    # Convertir la columna date a formato datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Renombrar a Date para mantener compatibilidad
        df.rename(columns={'date': 'Date'}, inplace=True)
    else:
        st.error("Error: La columna 'date' no existe en los datos recibidos de la API")
        return pd.DataFrame()
    
    # Renombrar otras columnas para mantener compatibilidad con visualizaciones
    column_mapping = {
        'customer_name': 'Customer Name',
        'gender': 'Gender',
        'annual_income': 'Annual Income',
        'dealer_name': 'Dealer_Name',
        'company': 'Company',
        'model': 'Model',
        'engine': 'Engine',
        'transmission': 'Transmission',
        'color': 'Color',
        'price': 'Price ($)',
        'dealer_no': 'Dealer_No ',
        'body_style': 'Body Style',
        'phone': 'Phone',
        'dealer_region': 'Dealer_Region'
    }
    
    # Solo renombrar las columnas que existen
    existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
    df.rename(columns=existing_columns, inplace=True)
    
    # Extraer año y mes para análisis temporal
    if 'Date' in df.columns:
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Month_Name'] = df['Date'].dt.strftime('%B')
        df['Year_Month'] = df['Date'].dt.to_period('M')
    
    # Limpiar espacios en blanco en las columnas de texto (solo si existen)
    for col in ['Company', 'Model', 'Transmission', 'Gender']:
        if col in df.columns:
            df[col] = df[col].str.strip()
    
    return df


@st.cache_data
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
