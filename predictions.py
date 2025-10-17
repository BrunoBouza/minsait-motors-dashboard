"""
Módulo para realizar predicciones de ventas usando diferentes algoritmos de Machine Learning
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import warnings
warnings.filterwarnings('ignore')


def prepare_data_for_prediction(df):
    """
    Prepara los datos agregados por semana para el entrenamiento de modelos.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de ventas
        
    Returns:
        tuple: (X, y, dates) donde X son las features, y son las ventas, dates son las fechas
    """
    # Agrupar ventas por semana
    df_sorted = df.sort_values('Date')
    weekly_data = df_sorted.set_index('Date').resample('W')['Price ($)'].sum().reset_index()
    
    # Convertir a millones de euros
    weekly_data['Sales'] = weekly_data['Price ($)'] / 1_000_000
    
    # Crear features temporales
    weekly_data['Week_Num'] = range(len(weekly_data))
    weekly_data['Month'] = weekly_data['Date'].dt.month
    weekly_data['Quarter'] = weekly_data['Date'].dt.quarter
    weekly_data['Week_of_Year'] = weekly_data['Date'].dt.isocalendar().week
    
    # Features para el modelo
    X = weekly_data[['Week_Num', 'Month', 'Quarter', 'Week_of_Year']].values
    y = weekly_data['Sales'].values
    dates = weekly_data['Date'].values
    
    return X, y, dates


def create_acf_pacf_plot(y, max_lags=40):
    """
    Crea gráficas ACF y PACF para análisis de series temporales.
    Estas gráficas ayudan a determinar los parámetros p y q de ARIMA.
    
    Args:
        y (np.array): Serie temporal de ventas
        max_lags (int): Número máximo de lags a mostrar
        
    Returns:
        plotly.graph_objects.Figure: Figura con ACF y PACF
    """
    # Limitar max_lags según el tamaño de los datos
    max_lags = min(max_lags, len(y) // 2 - 1)
    
    # Calcular ACF y PACF
    acf_values = acf(y, nlags=max_lags, fft=False)
    pacf_values = pacf(y, nlags=max_lags, method='ywm')
    
    # Calcular intervalos de confianza (95%)
    confidence_interval = 1.96 / np.sqrt(len(y))
    
    # Crear subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Función de Autocorrelación (ACF)', 
                       'Función de Autocorrelación Parcial (PACF)'),
        vertical_spacing=0.12
    )
    
    # Gráfica ACF
    fig.add_trace(
        go.Bar(
            x=list(range(len(acf_values))),
            y=acf_values,
            name='ACF',
            marker=dict(color='#4A90E2'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Líneas de confianza ACF
    fig.add_hline(
        y=confidence_interval, line_dash="dash", line_color="red",
        opacity=0.5, row=1, col=1
    )
    fig.add_hline(
        y=-confidence_interval, line_dash="dash", line_color="red",
        opacity=0.5, row=1, col=1
    )
    fig.add_hline(
        y=0, line_color="black", line_width=1, row=1, col=1
    )
    
    # Gráfica PACF
    fig.add_trace(
        go.Bar(
            x=list(range(len(pacf_values))),
            y=pacf_values,
            name='PACF',
            marker=dict(color='#FF6B6B'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Líneas de confianza PACF
    fig.add_hline(
        y=confidence_interval, line_dash="dash", line_color="red",
        opacity=0.5, row=2, col=1
    )
    fig.add_hline(
        y=-confidence_interval, line_dash="dash", line_color="red",
        opacity=0.5, row=2, col=1
    )
    fig.add_hline(
        y=0, line_color="black", line_width=1, row=2, col=1
    )
    
    # Actualizar layout
    fig.update_xaxes(title_text="Lag", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=2, col=1)
    fig.update_yaxes(title_text="Autocorrelación", row=1, col=1)
    fig.update_yaxes(title_text="Autocorrelación Parcial", row=2, col=1)
    
    fig.update_layout(
        height=700,
        template='plotly_white',
        title_text="Análisis ACF/PACF para Selección de Parámetros ARIMA",
        title_x=0.5,
        showlegend=False
    )
    
    return fig


def predict_linear_regression(X, y, months_ahead=12):
    """
    Realiza predicciones usando Regresión Lineal.
    Modelo simple que asume una tendencia lineal en los datos.
    
    Args:
        X (np.array): Features de entrenamiento
        y (np.array): Variable objetivo (ventas)
        months_ahead (int): Número de meses a predecir (se convertirá a semanas)
        
    Returns:
        tuple: (predictions, model_name, r2_score)
    """
    # Entrenar modelo
    model = LinearRegression()
    model.fit(X, y)
    
    # Calcular R² score
    r2 = model.score(X, y)
    
    # Convertir meses a semanas (aproximadamente 4.3 semanas por mes)
    weeks_ahead = int(months_ahead * 4.3)
    
    # Generar features para predicción futura
    last_week = X[-1, 0]
    last_month = X[-1, 1]
    last_quarter = X[-1, 2]
    last_week_of_year = X[-1, 3]
    
    future_X = []
    
    for i in range(1, weeks_ahead + 1):
        week_num = last_week + i
        # Calcular mes y quarter basado en semanas
        week_of_year = ((last_week_of_year + i - 1) % 52) + 1
        month = ((week_of_year - 1) // 4) % 12 + 1
        quarter = ((month - 1) // 3) + 1
        future_X.append([week_num, month, quarter, week_of_year])
    
    future_X = np.array(future_X)
    
    # Realizar predicciones
    predictions = model.predict(future_X)
    
    return predictions, "Regresión Lineal", r2


def predict_random_forest(X, y, months_ahead=12):
    """
    Realiza predicciones usando Random Forest.
    Modelo más complejo que puede capturar patrones no lineales y estacionalidad.
    
    Args:
        X (np.array): Features de entrenamiento
        y (np.array): Variable objetivo (ventas)
        months_ahead (int): Número de meses a predecir (se convertirá a semanas)
        
    Returns:
        tuple: (predictions, model_name, r2_score)
    """
    # Entrenar modelo
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    # Calcular R² score
    r2 = model.score(X, y)
    
    # Convertir meses a semanas (aproximadamente 4.3 semanas por mes)
    weeks_ahead = int(months_ahead * 4.3)
    
    # Generar features para predicción futura
    last_week = X[-1, 0]
    last_month = X[-1, 1]
    last_quarter = X[-1, 2]
    last_week_of_year = X[-1, 3]
    
    future_X = []
    
    for i in range(1, weeks_ahead + 1):
        week_num = last_week + i
        # Calcular mes y quarter basado en semanas
        week_of_year = ((last_week_of_year + i - 1) % 52) + 1
        month = ((week_of_year - 1) // 4) % 12 + 1
        quarter = ((month - 1) // 3) + 1
        future_X.append([week_num, month, quarter, week_of_year])
    
    future_X = np.array(future_X)
    
    # Realizar predicciones
    predictions = model.predict(future_X)
    
    return predictions, "Random Forest", r2


def predict_moving_average(X, y, months_ahead=12, window=12):
    """
    Realiza predicciones usando Media Móvil con tendencia.
    Modelo simple basado en promedios históricos con ajuste de tendencia.
    
    Args:
        X (np.array): Features de entrenamiento
        y (np.array): Variable objetivo (ventas)
        months_ahead (int): Número de meses a predecir (se convertirá a semanas)
        window (int): Ventana de tiempo (semanas) para calcular la media móvil
        
    Returns:
        tuple: (predictions, model_name, r2_score)
    """
    # Calcular media móvil y tendencia
    if len(y) >= window:
        moving_avg = np.mean(y[-window:])
        # Calcular tendencia simple
        trend = (y[-1] - y[-window]) / window
    else:
        moving_avg = np.mean(y)
        trend = 0
    
    # Calcular factor estacional (promedio por semana del año)
    weeks_of_year = X[:, 3]
    seasonal_factors = {}
    for week in range(1, 53):
        week_data = y[weeks_of_year == week]
        if len(week_data) > 0:
            seasonal_factors[week] = np.mean(week_data) / np.mean(y)
        else:
            seasonal_factors[week] = 1.0
    
    # Convertir meses a semanas (aproximadamente 4.3 semanas por mes)
    weeks_ahead = int(months_ahead * 4.3)
    
    # Generar predicciones
    predictions = []
    last_week_of_year = X[-1, 3]
    
    for i in range(1, weeks_ahead + 1):
        week_of_year = ((last_week_of_year + i - 1) % 52) + 1
        # Predicción = media móvil + tendencia + ajuste estacional
        base_prediction = moving_avg + (trend * i)
        seasonal_prediction = base_prediction * seasonal_factors.get(week_of_year, 1.0)
        predictions.append(seasonal_prediction)
    
    predictions = np.array(predictions)
    
    # Calcular R² aproximado (comparando con últimos valores reales)
    if len(y) >= window:
        y_pred_train = np.full(len(y[-window:]), moving_avg)
        r2 = 1 - (np.sum((y[-window:] - y_pred_train) ** 2) / np.sum((y[-window:] - np.mean(y[-window:])) ** 2))
    else:
        r2 = 0.5
    
    return predictions, "Media Móvil con Tendencia", max(0, r2)


def predict_arima(X, y, months_ahead=12, order=(2, 1, 2)):
    """
    Realiza predicciones usando ARIMA (AutoRegressive Integrated Moving Average).
    Modelo básico de series temporales sin componente estacional.
    
    Args:
        X (np.array): Features de entrenamiento (no usado directamente por ARIMA)
        y (np.array): Variable objetivo (ventas) - serie temporal
        months_ahead (int): Número de meses a predecir (se convertirá a semanas)
        order (tuple): Orden (p, d, q) del modelo ARIMA
        
    Returns:
        tuple: (predictions, model_name, aic_score)
    """
    try:
        # Convertir meses a semanas (aproximadamente 4.3 semanas por mes)
        weeks_ahead = int(months_ahead * 4.3)
        
        # Asegurar que tenemos suficientes datos
        if len(y) < 20:
            raise ValueError("No hay suficientes datos para entrenar ARIMA")
        
        # ARIMA puro sin componente estacional
        model = SARIMAX(y, order=order, seasonal_order=(0, 0, 0, 0), 
                       enforce_stationarity=False, enforce_invertibility=False)
        fitted_model = model.fit(disp=False)
        
        # Realizar predicciones
        forecast = fitted_model.forecast(steps=weeks_ahead)
        predictions = np.array(forecast)
        
        # Asegurar que las predicciones no sean negativas
        predictions = np.maximum(predictions, 0)
        
        # Usar AIC (Akaike Information Criterion) como métrica
        aic = fitted_model.aic
        
        return predictions, f"ARIMA{order}", aic
        
    except Exception as e:
        print(f"Error en ARIMA: {e}")
        # Fallback simple
        weeks_ahead = int(months_ahead * 4.3)
        mean_value = np.mean(y[-min(len(y), 26):])
        predictions = np.full(weeks_ahead, mean_value)
        return predictions, f"ARIMA{order} (Fallback)", 0


def predict_sarima(X, y, months_ahead=12, order=(1, 0, 1)):
    """
    Realiza predicciones usando SARIMA (Seasonal ARIMA).
    Modelo avanzado de series temporales con componente estacional.
    
    Args:
        X (np.array): Features de entrenamiento (no usado directamente por SARIMA)
        y (np.array): Variable objetivo (ventas) - serie temporal
        months_ahead (int): Número de meses a predecir (se convertirá a semanas)
        order (tuple): Orden (p, d, q) del modelo ARIMA
        
    Returns:
        tuple: (predictions, model_name, aic_score)
    """
    try:
        # Convertir meses a semanas (aproximadamente 4.3 semanas por mes)
        weeks_ahead = int(months_ahead * 4.3)
        
        # Asegurar que tenemos suficientes datos
        if len(y) < 20:
            raise ValueError("No hay suficientes datos para entrenar SARIMA")
        
        # SARIMA con componente estacional semanal
        if len(y) >= 52:
            seasonal_order = (1, 0, 1, 52)  # Patrón anual en semanas
        else:
            seasonal_order = (1, 0, 0, 13)  # Patrón trimestral si hay menos datos
        
        model = SARIMAX(y, order=order, seasonal_order=seasonal_order, 
                       enforce_stationarity=False, enforce_invertibility=False)
        fitted_model = model.fit(disp=False)
        
        # Realizar predicciones
        forecast = fitted_model.forecast(steps=weeks_ahead)
        predictions = np.array(forecast)
        
        # Asegurar que las predicciones no sean negativas
        predictions = np.maximum(predictions, 0)
        
        # Usar AIC (Akaike Information Criterion) como métrica
        aic = fitted_model.aic
        
        return predictions, f"SARIMA{order}x{seasonal_order[:3]}", aic
        
    except Exception as e:
        print(f"Error en SARIMA: {e}")
        # Fallback con estacionalidad manual
        weeks_ahead = int(months_ahead * 4.3)
        
        # Calcular tendencia y estacionalidad
        window = min(52, len(y))
        recent_data = y[-window:]
        trend = np.mean(recent_data)
        seasonal_pattern = recent_data - trend
        
        # Generar predicciones
        predictions = []
        for i in range(weeks_ahead):
            seasonal_idx = i % len(seasonal_pattern)
            pred = trend + seasonal_pattern[seasonal_idx]
            predictions.append(max(pred, 0))
        
        return np.array(predictions), f"SARIMA{order} (Fallback)", 0


def create_prediction_plot(df, predictions, model_name, last_date):
    """
    Crea un gráfico interactivo con los datos históricos y las predicciones.
    
    Args:
        df (pd.DataFrame): DataFrame con datos históricos
        predictions (np.array): Array con las predicciones
        model_name (str): Nombre del modelo usado
        last_date (datetime): Última fecha en los datos históricos
        
    Returns:
        plotly.graph_objects.Figure: Figura con el gráfico
    """
    # Preparar datos históricos semanales
    df_sorted = df.sort_values('Date')
    weekly_historical = df_sorted.set_index('Date').resample('W')['Price ($)'].sum().reset_index()
    weekly_historical['Sales'] = weekly_historical['Price ($)'] / 1_000_000
    
    # Convertir last_date a pandas Timestamp si es numpy.datetime64
    if isinstance(last_date, np.datetime64):
        last_date = pd.Timestamp(last_date)
    
    # Generar fechas futuras (semanales)
    future_dates = []
    current_date = last_date
    for i in range(len(predictions)):
        # Avanzar una semana usando pandas
        current_date = current_date + pd.DateOffset(weeks=1)
        future_dates.append(current_date)
    
    # Crear figura
    fig = go.Figure()
    
    # Agregar datos históricos
    fig.add_trace(go.Scatter(
        x=weekly_historical['Date'],
        y=weekly_historical['Sales'],
        mode='lines',
        name='Ventas Históricas',
        line=dict(color='#4A90E2', width=2),
        fill='tozeroy',
        fillcolor='rgba(74, 144, 226, 0.1)'
    ))
    
    # Agregar predicciones
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        mode='lines',
        name='Predicción',
        line=dict(color='#FF6B6B', width=2, dash='dash'),
        fill='tozeroy',
        fillcolor='rgba(255, 107, 107, 0.1)'
    ))
    
    # Agregar línea de conexión
    fig.add_trace(go.Scatter(
        x=[weekly_historical['Date'].iloc[-1], future_dates[0]],
        y=[weekly_historical['Sales'].iloc[-1], predictions[0]],
        mode='lines',
        name='Conexión',
        line=dict(color='gray', width=1, dash='dot'),
        showlegend=False
    ))
    
    # Personalizar layout
    fig.update_layout(
        title=f'Predicción de Ventas - {model_name}',
        xaxis_title='Fecha',
        yaxis_title='Ventas (Millones €)',
        template='plotly_white',
        hovermode='x unified',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def get_prediction_summary(predictions, model_name, r2_score):
    """
    Genera un resumen estadístico de las predicciones.
    
    Args:
        predictions (np.array): Array con las predicciones
        model_name (str): Nombre del modelo usado
        r2_score (float): Coeficiente de determinación del modelo
        
    Returns:
        dict: Diccionario con estadísticas de las predicciones
    """
    total_predicted = np.sum(predictions)
    avg_monthly = np.mean(predictions)
    min_month = np.min(predictions)
    max_month = np.max(predictions)
    
    return {
        'model': model_name,
        'r2_score': r2_score,
        'total_year': total_predicted,
        'avg_monthly': avg_monthly,
        'min_month': min_month,
        'max_month': max_month,
        'predictions': predictions
    }
