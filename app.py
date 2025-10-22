"""
Aplicación principal del Dashboard de Minsait Motors
Dashboard interactivo de análisis de ventas de vehículos usando Streamlit y Plotly
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from data_loader import load_data, calculate_kpis, filter_data
from visualizations import (
    create_sales_timeline,
    create_sales_by_company,
    create_monthly_sales,
    create_top_models,
    create_transmission_pie,
    create_gender_cards
)
from predictions import (
    prepare_data_for_prediction,
    predict_linear_regression,
    predict_random_forest,
    predict_moving_average,
    predict_arima,
    predict_sarima,
    create_prediction_plot,
    get_prediction_summary,
    create_acf_pacf_plot
)
from auth_client import init_session_state, login_page, logout, show_user_management

# Configuración de la página
st.set_page_config(
    page_title="Minsait Motors Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    /* Estilos para las métricas */
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric label {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #000000 !important;
    }
    h1 {
        color: #1f1f1f;
        font-size: 2.5rem;
        font-weight: 700;
    }
    h2 {
        color: #4A90E2;
        font-size: 1.5rem;
        margin-top: 2rem;
    }
    h3 {
        color: #000000 !important;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    """
    Función principal que ejecuta el dashboard de Minsait Motors.
    Carga los datos, crea la interfaz y muestra todas las visualizaciones.
    """
    
    # Inicializar estado de sesión
    init_session_state()
    
    # Verificar autenticación
    if not st.session_state.authenticated:
        login_page()
        return
    
    # Header del dashboard con botón de logout
    col_title, col_user = st.columns([3, 1])
    with col_title:
        st.title("🚗 MINSAIT MOTORS")
        st.markdown("Dashboard de Análisis de Ventas")
    with col_user:
        st.write("")  # Espaciado
        if st.session_state.user_info:
            st.markdown(f"**Usuario:** {st.session_state.user_info.get('username', 'N/A')}")
        if st.button("Cerrar Sesión", use_container_width=True):
            logout()
    
    # Cargar los datos
    with st.spinner('Cargando datos...'):
        df = load_data()
    
    # Guardar las fechas min/max
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    # Sidebar para filtros
    st.sidebar.header("Filtros")
    
    # Filtro de compañías
    st.sidebar.subheader("Compañías")
    all_companies = sorted(df['Company'].unique())
    selected_companies = st.sidebar.multiselect(
        "Seleccionar compañías",
        options=all_companies,
        default=None,
        placeholder="Todas las compañías"
    )
    
    # Filtro de transmisión
    st.sidebar.subheader("Tipo de Transmisión")
    transmission_options = df['Transmission'].unique()
    selected_transmissions = st.sidebar.multiselect(
        "Seleccionar transmisión",
        options=transmission_options,
        default=None,
        placeholder="Todos los tipos"
    )
    
    # Filtro de género
    st.sidebar.subheader("Género del Cliente")
    gender_options = df['Gender'].unique()
    selected_genders = st.sidebar.multiselect(
        "Seleccionar género",
        options=gender_options,
        default=None,
        placeholder="Todos"
    )
    
    # Slider de fechas en el área principal (ANTES de filtrar datos)
    st.markdown("##### Seleccionar Rango de Fechas")
    date_range_slider = st.slider(
        "Período de Análisis",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
        format="DD/MM/YYYY",
        label_visibility="collapsed",
        key="date_slider"
    )
    
    # Convertir a tupla de fechas
    date_range = (date_range_slider[0].date(), date_range_slider[1].date())
    
    # Mostrar fechas seleccionadas
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.caption(f"Desde: **{date_range[0].strftime('%d/%m/%Y')}**")
    with col_info2:
        st.caption(f"Hasta: **{date_range[1].strftime('%d/%m/%Y')}**")
    
    # Aplicar filtros
    filtered_df = filter_data(
        df,
        date_range=date_range,
        companies=selected_companies if selected_companies else None,
        transmissions=selected_transmissions if selected_transmissions else None,
        genders=selected_genders if selected_genders else None
    )
    
    # Calcular KPIs
    kpis = calculate_kpis(filtered_df)
    
    st.markdown("---")
    
    # Sección de KPIs principales
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    
    with kpi_col1:
        st.markdown("### Ventas Anuales")
        st.metric(
            label=f"Ventas {kpis['año_actual']} (Mill €)",
            value=f"{kpis['ventas_anuales']:.1f}",
            delta=f"{kpis['porcentaje_objetivo']:+.2f}%",
            delta_color="normal"
        )
        st.markdown(f"<p style='color: #666; font-size: 0.9rem;'>vs {kpis['año_anterior']}: {kpis['objetivo']:.1f} mill€</p>", unsafe_allow_html=True)
    
    with kpi_col2:
        st.markdown("### Total Ventas")
        # Calcular cambio porcentual en número de vehículos vendidos
        if kpis['total_ventas_año_anterior'] > 0:
            cambio_ventas = ((kpis['total_ventas'] - kpis['total_ventas_año_anterior']) / kpis['total_ventas_año_anterior']) * 100
            st.metric(
                label=f"Vehículos Vendidos {kpis['año_actual']}",
                value=f"{kpis['total_ventas']:,}",
                delta=f"{cambio_ventas:+.1f}%"
            )
        else:
            st.metric(
                label=f"Vehículos Vendidos {kpis['año_actual']}",
                value=f"{kpis['total_ventas']:,}",
                delta=None
            )
        st.markdown(f"<p style='color: #666; font-size: 0.9rem;'>vs {kpis['año_anterior']}: {kpis['total_ventas_año_anterior']:,} vehículos</p>", unsafe_allow_html=True)
    
    with kpi_col3:
        # Calcular estadísticas por género
        gender_stats = create_gender_cards(filtered_df)
        st.markdown("### Distribución por Género")
        for gender, stats in gender_stats.items():
            st.markdown(f"<p style='color: #000; font-size: 1.1rem;'><strong>{gender}</strong>: {stats['count']:,} ({stats['percentage']:.1f}%)</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Primera fila de visualizaciones
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Gráfico de ventas temporales
        st.plotly_chart(
            create_sales_timeline(filtered_df),
            use_container_width=True,
            key="sales_timeline"
        )
    
    with col2:
        # Gráfico de pastel de transmisiones
        st.plotly_chart(
            create_transmission_pie(filtered_df),
            use_container_width=True,
            key="transmission_pie"
        )
    
    st.markdown("---")
    
    # Segunda fila de visualizaciones
    col3, col4 = st.columns(2)
    
    with col3:
        # Gráfico de ventas por compañía
        st.plotly_chart(
            create_sales_by_company(filtered_df, top_n=10),
            use_container_width=True,
            key="sales_by_company"
        )
    
    with col4:
        # Gráfico de modelos más vendidos
        st.plotly_chart(
            create_top_models(filtered_df, top_n=10),
            use_container_width=True,
            key="top_models"
        )
    
    st.markdown("---")
    
    # Tercera fila - Gráfico de ventas mensuales
    st.plotly_chart(
        create_monthly_sales(filtered_df),
        use_container_width=True,
        key="monthly_sales"
    )
    
    st.markdown("---")
    
    # Sección de Predicciones
    st.header("Predicción de Ventas")
    st.markdown("Utiliza modelos de Machine Learning para predecir las ventas del próximo año.")
    
    # Sección de análisis ACF/PACF
    st.markdown("---")
    st.subheader("📊 Análisis de Autocorrelación (ACF/PACF)")
    
    # Mostrar gráficas ACF/PACF
    try:
        X_temp, y_temp, dates_temp = prepare_data_for_prediction(df)
        fig_acf_pacf = create_acf_pacf_plot(y_temp, max_lags=40)
        st.plotly_chart(fig_acf_pacf, use_container_width=True)
    except Exception as e:
        st.warning(f"No se pudieron generar las gráficas ACF/PACF: {str(e)}")
    
    st.markdown("---")
    
    # Controles de predicción
    pred_col1, pred_col2, pred_col3 = st.columns([2, 2, 1])
    
    with pred_col1:
        # Selector de algoritmo
        algorithm = st.selectbox(
            "Seleccionar Algoritmo de Predicción",
            options=[
                "Regresión Lineal",
                "Random Forest",
                "Media Móvil con Tendencia",
                "ARIMA",
                "SARIMA"
            ],
            help="Elige el algoritmo que se usará para predecir las ventas futuras"
        )
    
    with pred_col2:
        # Número de meses a predecir
        months_ahead = st.slider(
            "Meses a Predecir",
            min_value=3,
            max_value=24,
            value=12,
            help="Número de meses futuros a predecir"
        )
    
    with pred_col3:
        st.write("")  # Espaciado
        st.write("")  # Espaciado
    
    # Parámetros personalizados para ARIMA/SARIMA
    if algorithm in ["ARIMA", "SARIMA"]:
        st.markdown("#### Parámetros del Modelo")
        
        param_col1, param_col2, param_col3 = st.columns(3)
        
        with param_col1:
            p = st.number_input(
                "p (AR - Autoregresivo)",
                min_value=0,
                max_value=5,
                value=2 if algorithm == "ARIMA" else 1,
                help="Orden autoregresivo. Mira PACF: el lag donde corta abruptamente sugiere p"
            )
        
        with param_col2:
            d = st.number_input(
                "d (Diferenciación)",
                min_value=0,
                max_value=2,
                value=1 if algorithm == "ARIMA" else 0,
                help="Grado de diferenciación. Usa 1 o 2 para datos no estacionarios"
            )
        
        with param_col3:
            q = st.number_input(
                "q (MA - Media Móvil)",
                min_value=0,
                max_value=5,
                value=2 if algorithm == "ARIMA" else 1,
                help="Orden de media móvil. Mira ACF: el lag donde corta abruptamente sugiere q"
            )
        
        arima_order = (p, d, q)
    else:
        arima_order = None
    
    # Generar y mostrar predicción automáticamente
    with st.spinner('Generando predicción...'):
        try:
            # Preparar datos
            X, y, dates = prepare_data_for_prediction(df)
            
            # Seleccionar y ejecutar el algoritmo
            if algorithm == "Regresión Lineal":
                predictions, model_name, r2 = predict_linear_regression(X, y, months_ahead)
            elif algorithm == "Random Forest":
                predictions, model_name, r2 = predict_random_forest(X, y, months_ahead)
            elif algorithm == "ARIMA":
                predictions, model_name, r2 = predict_arima(X, y, months_ahead, order=arima_order)
            elif algorithm == "SARIMA":
                predictions, model_name, r2 = predict_sarima(X, y, months_ahead, order=arima_order)
            else:  # Media Móvil con Tendencia
                predictions, model_name, r2 = predict_moving_average(X, y, months_ahead)
            
            # Obtener resumen de predicciones
            summary = get_prediction_summary(predictions, model_name, r2)
            
            # Mostrar métricas de predicción
            st.markdown("### Resultados de la Predicción")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric(
                    label="Ventas Totales Predichas",
                    value=f"{summary['total_year']:.1f} M€",
                    help="Suma total de ventas predichas para el período"
                )
            
            with metric_col2:
                st.metric(
                    label="Promedio Mensual",
                    value=f"{summary['avg_monthly']:.1f} M€",
                    help="Promedio de ventas mensuales predichas"
                )
            
            with metric_col3:
                # Calcular ventas del último año para comparación
                last_year_sales = df[df['Year'] == df['Year'].max()]['Price ($)'].sum() / 1_000_000
                growth = ((summary['total_year'] - last_year_sales) / last_year_sales * 100)
                st.metric(
                    label="Crecimiento Esperado",
                    value=f"{growth:+.1f}%",
                    delta=f"{growth:+.1f}%",
                    help="Cambio porcentual respecto al último año"
                )
            
            # Mostrar gráfico de predicción
            last_date = dates[-1]
            fig_prediction = create_prediction_plot(df, predictions, model_name, last_date)
            st.plotly_chart(fig_prediction, use_container_width=True)
            
            st.success("Predicción generada exitosamente!")
            
        except Exception as e:
            st.error(f"Error al generar la predicción: {str(e)}")
            st.info("Asegúrate de tener suficientes datos históricos para entrenar el modelo.")
    
    # Panel de gestión de usuarios (solo para admin)
    st.sidebar.markdown("---")
    if st.session_state.user_info and st.session_state.user_info.get("role") == "admin":
        with st.sidebar.expander("👥 Gestionar Usuarios"):
            show_user_management()
    
    # Información adicional en el sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Información del Dataset")
    st.sidebar.info(f"""
    - **Total de registros**: {len(df):,}
    - **Registros filtrados**: {len(filtered_df):,}
    - **Período**: {df['Date'].min().strftime('%d/%m/%Y')} - {df['Date'].max().strftime('%d/%m/%Y')}
    - **Compañías**: {df['Company'].nunique()}
    - **Modelos**: {df['Model'].nunique()}
    """)
    
    # Opción para ver datos crudos
    st.sidebar.markdown("---")
    if st.sidebar.checkbox("Mostrar datos crudos"):
        st.subheader("Datos Crudos")
        st.dataframe(
            filtered_df.head(100),
            use_container_width=True,
            height=400
        )
        
        # Botón de descarga
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar datos filtrados (CSV)",
            data=csv,
            file_name=f"minsait_motors_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>Dashboard de Minsait Motors | Desarrollado con Streamlit y Plotly</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# Ejecutar la aplicación
if __name__ == "__main__":
    main()