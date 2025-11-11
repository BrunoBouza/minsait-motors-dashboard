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
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


@st.cache_data
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
    weekly_data = df_sorted.set_index('Date').resample('W-SUN')['Price ($)'].sum().reset_index()
    
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


@st.cache_data
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


@st.cache_data(ttl=300, show_spinner=False)
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


@st.cache_data(ttl=300, show_spinner=False)
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


@st.cache_data(ttl=300, show_spinner=False)
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


@st.cache_data(ttl=600, show_spinner=False)
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
        
        # Crear serie temporal con índice de fechas para evitar problemas con pandas 2.0+
        start_date = pd.Timestamp('2020-01-05')  # Fecha inicial arbitraria (un domingo)
        date_index = pd.date_range(start=start_date, periods=len(y), freq='W-SUN')
        y_series = pd.Series(y, index=date_index)
        
        # ARIMA puro sin componente estacional
        model = SARIMAX(y_series, order=order, seasonal_order=(0, 0, 0, 0), 
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


def perform_sarima_backtesting(y, order=(1, 0, 1), test_size=0.2, step_size=4, fixed_window=False, window_size=52):
    """
    Realiza backtesting REAL de SARIMA con ventana deslizante (rolling window).
    Reentrena el modelo en cada paso y evalúa su evolución temporal.
    
    Args:
        y (np.array): Serie temporal completa
        order (tuple): Parámetros (p, d, q) de SARIMA
        test_size (float): Proporción de datos para testing (0.2 = 20%)
        step_size (int): Tamaño del paso (semanas) para avanzar la ventana
        fixed_window (bool): Si True, usa ventana fija. Si False, ventana expandible (default)
        window_size (int): Tamaño de la ventana fija (solo si fixed_window=True)
        
    Returns:
        dict: Diccionario con métricas de error, evolución temporal y último modelo entrenado
    """
    try:
        # Calcular puntos de inicio y fin del backtesting
        split_point = int(len(y) * (1 - test_size))
        
        if split_point < 20:
            return None
        
        # Determinar orden estacional
        if split_point >= 52:
            seasonal_order = (1, 0, 1, 52)
        else:
            seasonal_order = (1, 0, 0, 13)
        
        # Crear fecha inicial para índices
        start_date = pd.Timestamp('2020-01-05')
        
        # Almacenar resultados de cada iteración
        all_predictions = []
        all_actuals = []
        iteration_errors = []  # MAE de cada iteración
        train_sizes = []
        last_fitted_model = None  # Guardar el último modelo entrenado
        
        # Ventana deslizante: desde split_point hasta el final
        current_pos = split_point
        
        while current_pos < len(y):
            # Determinar cuántas semanas predecir en esta iteración
            # (mínimo step_size o lo que quede hasta el final)
            forecast_horizon = min(step_size, len(y) - current_pos)
            
            # Datos de entrenamiento según el tipo de ventana
            if fixed_window:
                # Ventana FIJA: solo últimas window_size semanas
                train_start = max(0, current_pos - window_size)
                train_data = y[train_start:current_pos]
            else:
                # Ventana EXPANDIBLE: todo hasta current_pos
                train_data = y[:current_pos]
            
            # Datos reales para comparar (lo que vamos a predecir)
            actual_data = y[current_pos:current_pos + forecast_horizon]
            
            # Crear serie temporal con índice de fechas
            date_index_train = pd.date_range(start=start_date, periods=len(train_data), freq='W-SUN')
            train_series = pd.Series(train_data, index=date_index_train)
            
            try:
                # Entrenar modelo con datos de entrenamiento
                # Reducir maxiter para datasets grandes (optimización)
                model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order,
                               enforce_stationarity=False, enforce_invertibility=False)
                fitted_model = model.fit(disp=False, maxiter=50, method='lbfgs')
                
                # Guardar el último modelo entrenado
                last_fitted_model = fitted_model
                
                # Predecir el siguiente periodo
                predictions = fitted_model.forecast(steps=forecast_horizon)
                predictions = np.maximum(predictions, 0)  # No negativas
                
                # Guardar resultados de esta iteración
                all_predictions.extend(predictions)
                all_actuals.extend(actual_data)
                train_sizes.append(len(train_data))
                
                # Calcular error de esta iteración
                iter_mae = np.mean(np.abs(actual_data - predictions))
                iteration_errors.append(iter_mae)
                
            except Exception as iter_error:
                print(f"Error en iteración {current_pos}: {iter_error}")
                # Si falla una iteración, usar el último valor conocido
                fallback_pred = np.full(forecast_horizon, train_data[-1])
                all_predictions.extend(fallback_pred)
                all_actuals.extend(actual_data)
                iteration_errors.append(np.mean(np.abs(actual_data - fallback_pred)))
            
            # Avanzar la ventana
            current_pos += forecast_horizon
        
        # Convertir a arrays
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        
        # Calcular métricas globales sobre todas las predicciones
        mae = np.mean(np.abs(all_actuals - all_predictions))
        rmse = np.sqrt(np.mean((all_actuals - all_predictions) ** 2))
        mape = np.mean(np.abs((all_actuals - all_predictions) / (all_actuals + 1e-10))) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'predictions': all_predictions,
            'actuals': all_actuals,
            'train_size': split_point,
            'test_size': len(all_actuals),
            'iteration_errors': iteration_errors,  # Error por cada ventana
            'num_iterations': len(iteration_errors),  # Número de reentrenamientos
            'avg_iteration_mae': np.mean(iteration_errors),  # Promedio de errores
            'window_type': 'fixed' if fixed_window else 'expanding'  # Tipo de ventana usado
        }
    
    except Exception as e:
        print(f"Error en backtesting SARIMA: {e}")
        return None


@st.cache_data(ttl=600, show_spinner=False)
def predict_sarima(X, y, months_ahead=12, order=(1, 0, 1), _use_rolling_backtest=True):
    """
    Realiza predicciones usando SARIMA (Seasonal ARIMA).
    Modelo avanzado de series temporales con componente estacional.
    
    Args:
        X (np.array): Features de entrenamiento (no usado directamente por SARIMA)
        y (np.array): Variable objetivo (ventas) - serie temporal
        months_ahead (int): Número de meses a predecir (se convertirá a semanas)
        order (tuple): Orden (p, d, q) del modelo ARIMA
        _use_rolling_backtest (bool): Flag para invalidar caché tras cambios en backtesting
        
    Returns:
        tuple: (predictions, model_name, aic_score, backtest_results)
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
        
        # PASO 1: Realizar backtesting PRIMERO (entrena con ventana deslizante)
        # El backtesting se usa solo para validación, NO para predicciones
        backtest_results = perform_sarima_backtesting(y, order=order, test_size=0.2)
        
        # PASO 2: Entrenar modelo NUEVO con TODOS los datos para predicciones futuras
        # NO usamos el modelo del backtesting
        
        # Crear serie temporal con índice de fechas
        start_date = pd.Timestamp('2020-01-05')
        date_index = pd.date_range(start=start_date, periods=len(y), freq='W-SUN')
        y_series = pd.Series(y, index=date_index)
        
        # Entrenar modelo con TODOS los datos
        model = SARIMAX(y_series, order=order, seasonal_order=seasonal_order, 
                       enforce_stationarity=False, enforce_invertibility=False)
        fitted_model = model.fit(disp=False, maxiter=50, method='lbfgs')
        
        # PASO 3: Realizar predicciones futuras
        forecast = fitted_model.forecast(steps=weeks_ahead)
        predictions = np.array(forecast)
        
        # Asegurar que las predicciones no sean negativas
        predictions = np.maximum(predictions, 0)
        
        # Usar AIC (Akaike Information Criterion) como métrica
        aic = fitted_model.aic
        
        return predictions, f"SARIMA{order}x{seasonal_order[:3]}", aic, backtest_results
        
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
        
        return np.array(predictions), f"SARIMA{order} (Fallback)", 0, None


def create_backtest_plot(backtest_results, y):
    """
    Crea un gráfico comparando predicciones vs valores reales en el período de test.
    
    Args:
        backtest_results (dict): Resultados del backtesting con predicciones y valores reales
        y (np.array): Serie temporal completa para contexto
        
    Returns:
        plotly.graph_objects.Figure: Figura con comparación visual
    """
    if backtest_results is None:
        return None
    
    train_size = backtest_results['train_size']
    test_size = backtest_results['test_size']
    predictions = backtest_results['predictions']
    actuals = backtest_results['actuals']
    
    # Crear índices continuos (sin gaps)
    # Todo el rango de 0 a len(y)-1
    all_indices = list(range(len(y)))
    train_indices = all_indices[:train_size]
    
    # Los índices de test deben continuar inmediatamente después del entrenamiento
    # Esto asegura que no hay gap visual
    test_indices = list(range(train_size, train_size + test_size))
    
    fig = go.Figure()
    
    # Datos de entrenamiento (contexto)
    fig.add_trace(go.Scatter(
        x=train_indices,
        y=y[:train_size],
        mode='lines',
        name='Datos de Entrenamiento',
        line=dict(color='#4A90E2', width=2),
        opacity=0.6
    ))
    
    # Valores reales del período de test
    fig.add_trace(go.Scatter(
        x=test_indices,
        y=actuals,
        mode='lines+markers',
        name='Valores Reales (Test)',
        line=dict(color='#2ECC71', width=3),
        marker=dict(size=6)
    ))
    
    # Predicciones sobre el período de test
    fig.add_trace(go.Scatter(
        x=test_indices,
        y=predictions,
        mode='lines+markers',
        name='Predicciones del Modelo',
        line=dict(color='#E74C3C', width=2, dash='dash'),
        marker=dict(size=6, symbol='x')
    ))
    
    fig.update_layout(
        title='Backtesting: Comparación entre Predicciones y Valores Reales',
        xaxis_title='Semana',
        yaxis_title='Ventas (mill €)',
        template='plotly_white',
        hovermode='x unified',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_comparison_plot(df, pred_ma, pred_sarima, name_ma, name_sarima):
    """
    Crea un gráfico comparativo entre Media Móvil y SARIMA.
    
    Args:
        df (pd.DataFrame): DataFrame con datos históricos
        pred_ma (np.array): Predicciones de Media Móvil
        pred_sarima (np.array): Predicciones de SARIMA
        name_ma (str): Nombre del modelo Media Móvil
        name_sarima (str): Nombre del modelo SARIMA
        
    Returns:
        plotly.graph_objects.Figure: Figura comparativa
    """
    # Preparar datos históricos semanales
    df_sorted = df.sort_values('Date')
    weekly_historical = df_sorted.set_index('Date').resample('W-SUN')['Price ($)'].sum().reset_index()
    weekly_historical['Sales'] = weekly_historical['Price ($)'] / 1_000_000
    
    # Usar la última fecha del resample (no last_date original)
    # Esto asegura que las predicciones comiencen justo después del último punto histórico
    last_historical_date = weekly_historical['Date'].iloc[-1]
    
    # Generar fechas futuras (semanales) comenzando desde la semana siguiente
    future_dates = []
    for i in range(len(pred_ma)):
        future_date = last_historical_date + pd.Timedelta(weeks=i+1)
        future_dates.append(future_date)
    
    # Crear figura
    fig = go.Figure()
    
    # Datos históricos
    fig.add_trace(go.Scatter(
        x=weekly_historical['Date'],
        y=weekly_historical['Sales'],
        mode='lines',
        name='Datos Históricos',
        line=dict(color='#4A90E2', width=2),
        opacity=0.7
    ))
    
    # Predicciones Media Móvil
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=pred_ma,
        mode='lines+markers',
        name=name_ma,
        line=dict(color='#2ECC71', width=2, dash='dash'),
        marker=dict(size=5)
    ))
    
    # Predicciones SARIMA
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=pred_sarima,
        mode='lines+markers',
        name=name_sarima,
        line=dict(color='#E74C3C', width=2, dash='dot'),
        marker=dict(size=5, symbol='diamond')
    ))
    
    # Línea vertical para marcar el inicio de las predicciones
    # Usamos la última fecha histórica del resample
    fig.add_shape(
        type="line",
        x0=last_historical_date,
        x1=last_historical_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="gray", width=2, dash="solid"),
        opacity=0.5
    )
    
    # Añadir anotación manualmente
    fig.add_annotation(
        x=last_historical_date,
        y=1,
        yref="paper",
        text="Inicio de Predicciones",
        showarrow=False,
        yshift=10,
        font=dict(size=10, color="gray")
    )
    
    fig.update_layout(
        title='Comparación de Predicciones: Media Móvil vs SARIMA',
        xaxis_title='Fecha',
        yaxis_title='Ventas (mill €)',
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
    weekly_historical = df_sorted.set_index('Date').resample('W-SUN')['Price ($)'].sum().reset_index()
    weekly_historical['Sales'] = weekly_historical['Price ($)'] / 1_000_000
    
    # Usar la última fecha del resample para continuidad perfecta
    last_historical_date = weekly_historical['Date'].iloc[-1]
    
    # Generar fechas futuras (semanales) comenzando desde la semana siguiente
    future_dates = []
    for i in range(len(predictions)):
        future_date = last_historical_date + pd.Timedelta(weeks=i+1)
        future_dates.append(future_date)
    
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
