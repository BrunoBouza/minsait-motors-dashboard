"""
Módulo para crear las visualizaciones del dashboard usando Plotly
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def create_sales_timeline(df):
    """
    Crea un gráfico de área que muestra la evolución temporal de las ventas agrupadas por semana.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de ventas
        
    Returns:
        plotly.graph_objects.Figure: Figura de Plotly con el gráfico
    """
    # Crear una copia del dataframe para no modificar el original
    df_copy = df.copy()
    
    # Agrupar ventas por semana (resample requiere que Date sea el índice)
    df_copy = df_copy.set_index('Date')
    ventas_semanales = df_copy['Price ($)'].resample('W').sum().reset_index()
    ventas_semanales['Price (mill €)'] = ventas_semanales['Price ($)'] / 1_000_000
    
    # Formatear la fecha para el hover (mostrar inicio de semana)
    ventas_semanales['Week_Label'] = ventas_semanales['Date'].dt.strftime('%d/%m/%Y')
    
    # Crear gráfico de área
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=ventas_semanales['Date'],
        y=ventas_semanales['Price (mill €)'],
        mode='lines',
        fill='tozeroy',
        line=dict(color='#4A90E2', width=2),
        fillcolor='rgba(74, 144, 226, 0.3)',
        name='Ventas',
        hovertemplate='Semana del %{customdata}<br>Ventas: %{y:.2f} mill €<extra></extra>',
        customdata=ventas_semanales['Week_Label']
    ))
    
    # Personalizar el layout
    fig.update_layout(
        title='Ventas (Agrupadas por Semana)',
        xaxis_title='',
        yaxis_title='Ventas (mill €)',
        template='plotly_white',
        hovermode='x unified',
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    
    return fig


def create_sales_by_company(df, top_n=10):
    """
    Crea un gráfico de barras horizontales con las ventas por compañía.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de ventas
        top_n (int): Número de compañías principales a mostrar
        
    Returns:
        plotly.graph_objects.Figure: Figura de Plotly con el gráfico
    """
    # Agrupar ventas por compañía y contar vehículos vendidos
    ventas_compania = df.groupby('Company').size().reset_index(name='Recuento')
    ventas_compania = ventas_compania.sort_values('Recuento', ascending=True).tail(top_n)
    
    # Crear gráfico de barras horizontales
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=ventas_compania['Recuento'],
        y=ventas_compania['Company'],
        orientation='h',
        marker=dict(
            color='#4A90E2',
            line=dict(color='#4A90E2', width=1)
        ),
        text=ventas_compania['Recuento'],
        textposition='outside',
        hovertemplate='%{y}<br>Recuento: %{x}<extra></extra>'
    ))
    
    # Personalizar el layout
    fig.update_layout(
        title='Ventas por Compañía',
        xaxis_title='Recuento de Company',
        yaxis_title='Company',
        template='plotly_white',
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    
    return fig


def create_monthly_sales(df):
    """
    Crea un gráfico de barras con el total de coches vendidos por mes.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de ventas
        
    Returns:
        plotly.graph_objects.Figure: Figura de Plotly con el gráfico
    """
    # Definir el orden de los meses en español
    meses_orden = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
                   'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
    
    # Agrupar por año y mes
    ventas_mensuales = df.groupby(['Year', 'Month']).size().reset_index(name='Total_Coches')
    ventas_mensuales['Month_Name'] = pd.Categorical(
        ventas_mensuales['Month'].apply(lambda x: meses_orden[x-1]),
        categories=meses_orden,
        ordered=True
    )
    ventas_mensuales = ventas_mensuales.sort_values(['Year', 'Month'])
    
    # Crear etiquetas para el eje X
    ventas_mensuales['Label'] = ventas_mensuales['Month_Name'].str[:3]
    
    # Crear gráfico de barras
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=ventas_mensuales['Label'],
        y=ventas_mensuales['Total_Coches'],
        marker=dict(
            color='#4A90E2',
            line=dict(color='#3A7BC8', width=1)
        ),
        hovertemplate='%{x}<br>Total: %{y}<extra></extra>'
    ))
    
    # Personalizar el layout
    fig.update_layout(
        title='Total Coches Vendidos por Mes',
        xaxis_title='',
        yaxis_title='',
        template='plotly_white',
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        xaxis=dict(tickangle=-45)
    )
    
    return fig


def create_top_models(df, top_n=10):
    """
    Crea un gráfico de barras horizontales con los modelos más vendidos.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de ventas
        top_n (int): Número de modelos principales a mostrar
        
    Returns:
        plotly.graph_objects.Figure: Figura de Plotly con el gráfico
    """
    # Agrupar ventas por modelo
    ventas_modelo = df.groupby('Model').size().reset_index(name='Cantidad')
    ventas_modelo = ventas_modelo.sort_values('Cantidad', ascending=True).tail(top_n)
    
    # Crear gráfico de barras horizontales
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=ventas_modelo['Cantidad'],
        y=ventas_modelo['Model'],
        orientation='h',
        marker=dict(
            color='#4A90E2',
            line=dict(color='#3A7BC8', width=1)
        ),
        text=ventas_modelo['Cantidad'],
        textposition='outside',
        hovertemplate='%{y}<br>Cantidad: %{x}<extra></extra>'
    ))
    
    # Personalizar el layout
    fig.update_layout(
        title='Modelos más vendidos',
        xaxis_title='',
        yaxis_title='',
        template='plotly_white',
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    
    return fig


def create_transmission_pie(df):
    """
    Crea un gráfico de pastel (pie chart) con la distribución de transmisiones.
    Muestra la distribución entre transmisión automática y manual.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de ventas
        
    Returns:
        plotly.graph_objects.Figure: Figura de Plotly con el gráfico
    """
    # Contar vehículos por tipo de transmisión
    transmision_counts = df['Transmission'].value_counts().reset_index()
    transmision_counts.columns = ['Transmission', 'Count']
    
    # Calcular porcentajes
    total = transmision_counts['Count'].sum()
    transmision_counts['Percentage'] = (transmision_counts['Count'] / total * 100).round(2)
    
    # Definir colores (azul para Auto, marrón para Manual)
    colors = ['#6B7FB8', '#8B5A5A']
    
    # Crear gráfico de pastel
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=transmision_counts['Transmission'],
        values=transmision_counts['Count'],
        marker=dict(colors=colors),
        textinfo='label+value',
        textposition='inside',
        hovertemplate='%{label}<br>Cantidad: %{value}<br>Porcentaje: %{percent}<extra></extra>',
        hole=0  # 0 para pie chart completo, >0 para donut chart
    ))
    
    # Personalizar el layout
    fig.update_layout(
        title='Distribución por Tipo de Transmisión',
        template='plotly_white',
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )
    
    return fig


def create_gender_cards(df):
    """
    Calcula las estadísticas de ventas por género para mostrar en tarjetas.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de ventas
        
    Returns:
        dict: Diccionario con las estadísticas por género
    """
    gender_stats = df.groupby('Gender').size().reset_index(name='Count')
    
    stats = {}
    for _, row in gender_stats.iterrows():
        stats[row['Gender']] = {
            'count': row['Count'],
            'percentage': (row['Count'] / len(df) * 100)
        }
    
    return stats


def create_kpi_card_figure(value, title, objective=None, percentage=None):
    """
    Crea una figura de KPI estilo tarjeta para mostrar métricas clave.
    
    Args:
        value (float): Valor principal a mostrar
        title (str): Título de la métrica
        objective (float, optional): Valor objetivo
        percentage (float, optional): Porcentaje de cumplimiento
        
    Returns:
        plotly.graph_objects.Figure: Figura con la tarjeta KPI
    """
    fig = go.Figure()
    
    # Crear texto para mostrar
    main_text = f"{value:.1f} mill€" if 'Ventas' in title else f"{value:.1f} mill€"
    
    fig.add_trace(go.Indicator(
        mode="number+delta" if objective else "number",
        value=value,
        title={'text': title, 'font': {'size': 16}},
        number={'suffix': ' mill€', 'font': {'size': 40}},
        delta={'reference': objective, 'relative': False, 'valueformat': '.1f'} if objective else None,
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    
    fig.update_layout(
        height=150,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(240, 240, 240, 0.5)'
    )
    
    return fig
