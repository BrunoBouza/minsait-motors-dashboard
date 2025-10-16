# 🚗 Minsait Motors - Dashboard de Análisis de Ventas

Dashboard interactivo de análisis y predicción de ventas de vehículos desarrollado con Streamlit y Plotly.

## 📊 Características

### Análisis de Datos
- **KPIs dinámicos**: Ventas totales, ventas anuales, ventas promedio con comparación año a año
- **Gráficas interactivas**:
  - Línea temporal de ventas (agregación semanal)
  - Top 10 compañías por ventas
  - Ventas por mes
  - Top 10 modelos más vendidos
  - Distribución por tipo de transmisión
  - Ventas por género

### Filtros Interactivos
- Rango de fechas con slider
- Filtro por compañía
- Filtro por tipo de transmisión
- Filtro por género del comprador

### Predicción de Ventas 🔮
Múltiples algoritmos de Machine Learning para predicción:
- **Regresión Lineal**: Modelo simple para tendencias lineales
- **Random Forest**: Captura patrones no lineales y estacionalidad
- **Media Móvil con Tendencia**: Incluye factores estacionales semanales
- **ARIMA**: Modelo de series temporales sin componente estacional
- **SARIMA**: Modelo avanzado con estacionalidad (configurable)

### Análisis de Series Temporales
- **Gráficas ACF/PACF**: Para selección de parámetros ARIMA
- **Análisis de Diferencia Estacional**: Para SARIMA (y_t - y_(t-s))
- **Parámetros configurables**: Control completo sobre (p,d,q) y (P,D,Q,s)

## 🛠️ Tecnologías

- **Python 3.8+**
- **Streamlit**: Framework de aplicaciones web
- **Pandas**: Manipulación de datos
- **Plotly**: Visualizaciones interactivas
- **NumPy**: Cálculos numéricos
- **Scikit-learn**: Algoritmos de ML (Regresión Lineal, Random Forest)
- **Statsmodels**: Modelos de series temporales (ARIMA, SARIMA)

## 📦 Instalación

1. Clona el repositorio:
```bash
git clone <url-del-repositorio>
cd "Minsait Motors"
```

2. Crea un entorno virtual:
```bash
python -m venv venv
```

3. Activa el entorno virtual:
- Windows:
```bash
venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

4. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## 🚀 Uso

Ejecuta la aplicación:
```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📁 Estructura del Proyecto

```
Minsait Motors/
├── app.py                  # Aplicación principal de Streamlit
├── data_loader.py          # Carga de datos y funciones de negocio
├── visualizations.py       # Funciones de visualización con Plotly
├── predictions.py          # Modelos de predicción ML/Series Temporales
├── Car Sales.csv           # Dataset de ventas de vehículos
├── requirements.txt        # Dependencias del proyecto
├── README.md              # Este archivo
└── .gitignore             # Archivos a ignorar por Git
```

## 📈 Dataset

El dataset `Car Sales.csv` contiene información de ventas de vehículos con las siguientes columnas:
- **Date**: Fecha de la venta
- **Price ($)**: Precio de venta
- **Company**: Marca del vehículo
- **Model**: Modelo del vehículo
- **Transmission**: Tipo de transmisión (Automatic/Manual)
- **Gender**: Género del comprador

## 🎯 Modelos de Predicción

### ARIMA
Modelo de series temporales sin componente estacional. Parámetros configurables:
- **p**: Orden autoregresivo
- **d**: Grado de diferenciación
- **q**: Orden de media móvil

### SARIMA
Modelo avanzado con componente estacional. Parámetros configurables:
- **(p,d,q)**: Componente no estacional
- **(P,D,Q,s)**: Componente estacional
  - **s**: Período estacional (52 para datos semanales anuales)

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue primero para discutir los cambios que te gustaría realizar.

## 📝 Licencia

Este proyecto es un proyecto educativo desarrollado como parte del Streamlit Course.

## 👨‍💻 Autor

Desarrollado como proyecto práctico de análisis de datos y predicción de series temporales.
