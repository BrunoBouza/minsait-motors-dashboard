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
    create_backtest_plot,
    create_comparison_plot
)
from auth_client import init_session_state, login_page, logout, show_user_management, show_new_sale_form, AuthClient
from rbac_admin import show_rbac_admin

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
    col_title, col_user, col_refresh = st.columns([3, 1, 1])
    with col_title:
        st.title("🚗 MINSAIT MOTORS")
        st.markdown("Dashboard de Análisis de Ventas")
    with col_user:
        st.write("")  # Espaciado
        if st.session_state.user_info:
            st.markdown(f"**Usuario:** {st.session_state.user_info.get('username', 'N/A')}")
            st.caption(f"Rol: {st.session_state.user_info.get('role', 'N/A')}")
    with col_refresh:
        st.write("")  # Espaciado
        if st.button("🔄 Refrescar", use_container_width=True, help="Actualizar datos desde la base de datos"):
            st.cache_data.clear()
            st.rerun()
        if st.button("Cerrar Sesión", use_container_width=True):
            logout()
    
    st.markdown("---")
    
    # Pestañas principales con diseño mejorado
    user_role = st.session_state.user_info.get("role")
    
    # CSS personalizado para pestañas más notorias
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 60px;
            font-size: 18px;
            font-weight: 600;
            padding: 0 24px;
            background-color: white;
            border-radius: 8px;
            border: 2px solid #e0e0e0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4CAF50;
            color: white;
            border: 2px solid #4CAF50;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Mostrar pestaña de añadir venta solo para admin y writer
    if user_role in ["admin", "writer"]:
        if user_role == "admin":
            tab1, tab2, tab3 = st.tabs(["📊 DASHBOARD", "➕ NUEVA VENTA", "⚙️ ADMINISTRACIÓN RBAC"])
        else:
            tab1, tab2, tab3 = st.tabs(["📊 DASHBOARD", "➕ NUEVA VENTA", "ℹ️ INFORMACIÓN"])
    else:
        tab1, tab2, tab3 = st.tabs(["📊 DASHBOARD", "ℹ️ INFORMACIÓN", ""])  # Tab vacío
    
    # Cargar los datos (fuera de las pestañas para usarlos en ambas)
    with st.spinner('Cargando datos desde la API...'):
        df = load_data(st.session_state.token)
    
    # Verificar si hay datos
    if df.empty or 'Date' not in df.columns:
        st.error("⚠️ No se pudieron cargar los datos. Por favor verifica la conexión con la API.")
        st.stop()
    
    # TAB 1: Dashboard de Análisis
    with tab1:
        # Guardar las fechas min/max
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        
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
    
    # Sidebar para filtros (fuera de tabs para que sea visible siempre)
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
    
    # Continuar con tab1
    with tab1:
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
    
    # Determinar el año de corte para predicción
    max_year = df['Year'].max()
    min_year = df['Year'].min()
    
    # Verificar si el último año tiene datos completos (al menos 6 meses)
    year_counts = df.groupby('Year').size()
    latest_year_count = year_counts.get(max_year, 0)
    avg_year_count = year_counts.mean()
    
    # Si el último año tiene menos del 50% de los datos promedio, excluirlo
    if latest_year_count < avg_year_count * 0.5:
        cutoff_year = max_year - 1
        note_text = f"ℹ️ **Nota:** Las predicciones se realizan utilizando datos históricos hasta {cutoff_year}. El año {max_year} se excluye por tener datos incompletos ({latest_year_count} registros)."
    else:
        cutoff_year = max_year
        note_text = f"ℹ️ **Nota:** Las predicciones se realizan utilizando datos históricos hasta {cutoff_year}."
    
    st.info(note_text)
    
    # Filtrar datos para predicción
    df_prediction = df[df['Year'] <= cutoff_year].copy()
    
    st.markdown("---")
    
    # Controles de predicción
    pred_col1, pred_col2 = st.columns([2, 2])
    
    with pred_col1:
        # Número de meses a predecir
        months_ahead = st.slider(
            "Meses a Predecir",
            min_value=3,
            max_value=24,
            value=12,
            help="Número de meses futuros a predecir"
        )
    
    with pred_col2:
        st.write("")  # Espaciado
        st.write("")  # Espaciado
    
    # Parámetros personalizados para SARIMA
    st.markdown("#### Parámetros del Modelo SARIMA")
        
    
    param_col1, param_col2, param_col3 = st.columns(3)
    
    with param_col1:
        p = st.number_input(
            "p (AR - Autoregresivo)",
            min_value=0,
            max_value=5,
            value=1,
            help="Orden autoregresivo. Mira PACF: el lag donde corta abruptamente sugiere p"
        )
    
    with param_col2:
        d = st.number_input(
            "d (Diferenciación)",
            min_value=0,
            max_value=2,
            value=0,
            help="Grado de diferenciación. Usa 1 o 2 para datos no estacionarios"
        )
    
    with param_col3:
        q = st.number_input(
            "q (MA - Media Móvil)",
            min_value=0,
            max_value=5,
            value=1,
            help="Orden de media móvil. Mira ACF: el lag donde corta abruptamente sugiere q"
        )
    
    sarima_order = (p, d, q)
    
    # Generar y mostrar predicción automáticamente con ambos modelos
    with st.spinner('Generando predicciones con Media Móvil y SARIMA...'):
        try:
            # Preparar datos
            X, y, dates = prepare_data_for_prediction(df_prediction)
            
            # Ejecutar AMBOS modelos
            # 1. Media Móvil con Tendencia
            pred_ma, name_ma, r2_ma = predict_moving_average(X, y, months_ahead)
            
            # 2. SARIMA con backtesting
            pred_sarima, name_sarima, r2_sarima, backtest_results = predict_sarima(X, y, months_ahead, order=sarima_order)
            
            # Obtener resúmenes de ambos modelos
            summary_ma = get_prediction_summary(pred_ma, name_ma, r2_ma)
            summary_sarima = get_prediction_summary(pred_sarima, name_sarima, r2_sarima)
            
            # Mostrar métricas comparativas
            st.markdown("### 📊 Comparación de Modelos")
            
            # Tabla comparativa
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.markdown("#### 📈 Media Móvil con Tendencia")
                st.metric(
                    label="Ventas Totales Predichas",
                    value=f"{summary_ma['total_year']:.1f} M€"
                )
                st.metric(
                    label="Promedio Mensual",
                    value=f"{summary_ma['avg_monthly']:.1f} M€"
                )
            
            with comp_col2:
                st.markdown("#### 📉 SARIMA")
                st.metric(
                    label="Ventas Totales Predichas",
                    value=f"{summary_sarima['total_year']:.1f} M€"
                )
                st.metric(
                    label="Promedio Mensual",
                    value=f"{summary_sarima['avg_monthly']:.1f} M€"
                )
            
            # Mostrar métricas de backtesting de SARIMA
            if backtest_results is not None and 'num_iterations' in backtest_results:
                st.markdown("---")
                st.markdown("### 📊 Métricas de Backtesting SARIMA (Validación con Ventana Deslizante)")
                
                window_type_text = "expandible (crece con el tiempo)" if backtest_results.get('window_type') == 'expanding' else "fija"
                st.info(f"""
                **Backtesting Rolling Window ({window_type_text}):** El modelo fue reentrenado **{backtest_results['num_iterations']} veces** 
                usando ventana deslizante sobre **{backtest_results['test_size']} semanas** de datos de prueba.
                
                ℹ️ El backtesting se usa para **validación**. Las predicciones futuras se generan con un modelo SARIMA 
                entrenado con **todos los datos históricos** para máxima precisión.
                """)
                
                backtest_col1, backtest_col2, backtest_col3, backtest_col4 = st.columns(4)
                
                with backtest_col1:
                    st.metric(
                        label="MAE Global",
                        value=f"{backtest_results['mae']:.2f} M€",
                        help="Error absoluto medio sobre todas las predicciones"
                    )
                
                with backtest_col2:
                    st.metric(
                        label="RMSE Global",
                        value=f"{backtest_results['rmse']:.2f} M€",
                        help="Raíz del error cuadrático - penaliza errores grandes"
                    )
                
                with backtest_col3:
                    st.metric(
                        label="MAPE",
                        value=f"{backtest_results['mape']:.1f}%",
                        help="Error porcentual promedio"
                    )
                
                with backtest_col4:
                    st.metric(
                        label="Reentrenamientos",
                        value=f"{backtest_results['num_iterations']}",
                        help="Número de veces que se reentrenó el modelo"
                    )
                
                # Mostrar evolución de errores por iteración
                if 'iteration_errors' in backtest_results and len(backtest_results['iteration_errors']) > 0:
                    st.markdown("#### 📈 Evolución del Error por Iteración")
                    st.info("Cada punto representa el MAE de una ventana de predicción en el backtesting.")
                    
                    # Crear DataFrame para el gráfico
                    error_df = pd.DataFrame({
                        'Iteración': range(1, len(backtest_results['iteration_errors']) + 1),
                        'MAE (M€)': backtest_results['iteration_errors']
                    })
                    st.line_chart(error_df.set_index('Iteración'))
                
                # Gráfico simple: Predicciones vs Valores Reales
                st.markdown("#### 📊 Predicciones vs Valores Reales (Backtesting)")
                
                # Crear figura simple con matplotlib/plotly
                import plotly.graph_objects as go
                
                fig_comparison = go.Figure()
                
                # Valores reales
                fig_comparison.add_trace(go.Scatter(
                    y=backtest_results['actuals'],
                    mode='lines+markers',
                    name='Valores Reales',
                    line=dict(color='#2ECC71', width=3),
                    marker=dict(size=6)
                ))
                
                # Predicciones
                fig_comparison.add_trace(go.Scatter(
                    y=backtest_results['predictions'],
                    mode='lines+markers',
                    name='Predicciones',
                    line=dict(color='#E74C3C', width=2, dash='dash'),
                    marker=dict(size=5, symbol='x')
                ))
                
                fig_comparison.update_layout(
                    title='Comparación: Predicciones vs Valores Reales en periodo de test',
                    xaxis_title='Punto temporal',
                    yaxis_title='Ventas (M€)',
                    template='plotly_white',
                    hovermode='x unified',
                    height=350
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Mostrar gráfico de backtesting completo (con contexto de entrenamiento)
                st.markdown("#### 📈 Vista Completa del Backtesting")
                fig_backtest = create_backtest_plot(backtest_results, y)
                if fig_backtest:
                    st.plotly_chart(fig_backtest, use_container_width=True)
            elif backtest_results is None:
                st.warning("⚠️ No se pudo realizar el backtesting: datos insuficientes para validación.")
            
            # Crear gráfico comparativo entre Media Móvil y SARIMA
            st.markdown("---")
            st.markdown("### 📈 Comparación de Predicciones: Media Móvil vs SARIMA")
            
            # La función create_comparison_plot calcula internamente la última fecha desde el resample
            # No necesitamos pasar last_date porque usaría los datos originales (diarios) 
            # en lugar de los datos resampled (semanales), causando desalineación
            fig_comparison = create_comparison_plot(df_prediction, pred_ma, pred_sarima, name_ma, name_sarima)
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            st.success("✅ Predicciones generadas exitosamente con ambos modelos!")
            
        except Exception as e:
            import traceback
            st.error(f"❌ Error al generar la predicción: {str(e)}")
            
            # Mostrar detalles técnicos en un expander para debugging
            with st.expander("🔍 Ver detalles técnicos del error"):
                st.code(traceback.format_exc())
                st.markdown("**Información de contexto:**")
                if 'months_ahead' in locals():
                    st.write(f"- Meses a predecir: {months_ahead}")
                if 'sarima_order' in locals():
                    st.write(f"- Orden SARIMA: {sarima_order}")
                if 'df_prediction' in locals():
                    st.write(f"- Número de registros: {len(df_prediction)}")
                if 'y' in locals():
                    st.write(f"- Semanas de datos: {len(y)}")
            
            st.info("💡 Asegúrate de tener suficientes datos históricos para entrenar el modelo.")
    
    # Panel de gestión de usuarios (solo para admin)
    st.sidebar.markdown("---")
    if st.session_state.user_info and st.session_state.user_info.get("role") == "admin":
        with st.sidebar.expander("👥 Gestionar Usuarios"):
            show_user_management()
    
    # Botón de Sincronización con Snowflake (solo para admin)
    st.sidebar.markdown("---")
    if st.session_state.user_info and st.session_state.user_info.get("role") == "admin":
        if st.sidebar.button("❄️ Sincronizar con Snowflake", use_container_width=True):
            auth_client = AuthClient()
            with st.spinner("Sincronizando con Snowflake..."):
                sync_result = auth_client.trigger_snowflake_sync(st.session_state.token)
                
                if sync_result["success"]:
                    data = sync_result["data"]
                    if data.get("status") == "success":
                        sync_info = data.get("data", {})
                        st.sidebar.success(f"""
✅ Sincronización completada
- Batches: {sync_info.get('batches_processed', 0)}
- Sincronizados: {sync_info.get('total_synced', 0)}
- Fallidos: {sync_info.get('total_failed', 0)}
                        """)
                        st.rerun()
                    else:
                        st.sidebar.error(f"❌ {data.get('message', 'Error desconocido')}")
                else:
                    st.sidebar.error(f"❌ {sync_result['error']}")
    
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
        with tab1:
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
    
    # TAB 2: Nueva Venta o Info
    with tab2:
        if user_role in ["admin", "writer"]:
            show_new_sale_form()
        else:
            st.info("ℹ️ Solo los usuarios con rol **Admin** o **Writer** pueden añadir nuevas ventas.")
            st.markdown("""
            ### Roles y Permisos
            
            - **Reader**: Puede visualizar el dashboard y ver los datos
            - **Writer**: Puede visualizar el dashboard y añadir nuevas ventas
            - **Admin**: Acceso completo (visualizar, añadir, modificar y gestionar usuarios)
            
            Si necesitas permisos adicionales, contacta con un administrador.
            """)
    
    # TAB 3: Administración RBAC (solo para admin)
    with tab3:
        if user_role == "admin":
            show_rbac_admin()
        elif user_role == "writer":
            st.info("ℹ️ Esta funcionalidad está disponible solo para administradores.")
            st.markdown("""
            ### Administración RBAC
            
            Los administradores pueden gestionar:
            - **Usuarios**: Crear, editar, eliminar usuarios
            - **Grupos**: Organizar usuarios en equipos
            - **Roles**: Definir permisos y accesos
            - **Asignaciones**: Asignar usuarios a grupos y roles
            
            Contacta con un administrador si necesitas hacer cambios en el sistema de permisos.
            """)
        else:
            pass  # Tab vacío para readers
    
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