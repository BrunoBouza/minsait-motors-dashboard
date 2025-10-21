# Minsait Motors - Dashboard de Ventas# Minsait Motors - Dashboard de Ventas# Minsait Motors - Dashboard de Ventas# 🚗 Minsait Motors - Dashboard de Análisis de Ventas



Dashboard de análisis de ventas de vehículos con predicciones usando Machine Learning.



## CaracterísticasDashboard de análisis de ventas de vehículos con predicciones usando Machine Learning.



- KPIs de ventas con comparación año anterior

- Gráficas interactivas con filtros

- Predicción de ventas con varios algoritmos (Regresión Lineal, Random Forest, ARIMA, SARIMA)## Estructura del ProyectoDashboard de análisis de ventas de vehículos con predicciones usando Machine Learning.Dashboard interactivo de análisis y predicción de ventas de vehículos desarrollado con Streamlit y Plotly.

- Análisis ACF/PACF para series temporales



## Instalación

```

```bash

git clone https://github.com/BrunoBouza/minsait-motors-dashboard.git├── streamlit/          # Aplicación Streamlit

cd minsait-motors-dashboard

pip install -r requirements.txt│   ├── app.py## Características## 📊 Características

```

│   ├── data_loader.py

Configurar `.streamlit/secrets.toml` con la conexión a PostgreSQL:

```toml│   ├── predictions.py

[connections.neon]

url = "postgresql://usuario:password@host/database"│   ├── visualizations.py

```

│   └── requirements.txt- KPIs de ventas con comparación año anterior### Análisis de Datos

## Uso

├── api/                # API REST (en desarrollo)

```bash

streamlit run app.py└── .streamlit/         # Configuración y secrets- Gráficas interactivas con filtros- **KPIs dinámicos**: Ventas totales, ventas anuales, ventas promedio con comparación año a año

```

```

## Tecnologías

- Predicción de ventas con varios algoritmos (Regresión Lineal, Random Forest, ARIMA, SARIMA)- **Gráficas interactivas**:

Python, Streamlit, Pandas, Plotly, Scikit-learn, Statsmodels, PostgreSQL

## Características

## Autor

- Análisis ACF/PACF para series temporales  - Línea temporal de ventas (agregación semanal)

Bruno Bouza Fernández

- KPIs de ventas con comparación año anterior

- Gráficas interactivas con filtros  - Top 10 compañías por ventas

- Predicción de ventas con varios algoritmos (Regresión Lineal, Random Forest, ARIMA, SARIMA)

- Análisis ACF/PACF para series temporales## Instalación  - Ventas por mes



## Instalación  - Top 10 modelos más vendidos



```bash```bash  - Distribución por tipo de transmisión

git clone https://github.com/BrunoBouza/minsait-motors-dashboard.git

cd minsait-motors-dashboard/streamlitgit clone https://github.com/BrunoBouza/minsait-motors-dashboard.git  - Ventas por género

pip install -r requirements.txt

```cd minsait-motors-dashboard



Configurar `.streamlit/secrets.toml` con la conexión a PostgreSQL:pip install -r requirements.txt### Filtros Interactivos

```toml

[connections.neon]```- Rango de fechas con slider

url = "postgresql://usuario:password@host/database"

```- Filtro por compañía



## UsoConfigurar `.streamlit/secrets.toml` con la conexión a PostgreSQL:- Filtro por tipo de transmisión



```bash```toml- Filtro por género del comprador

cd streamlit

streamlit run app.py[connections.neon]

```

url = "postgresql://usuario:password@host/database"### Predicción de Ventas 🔮

## Tecnologías

```Múltiples algoritmos de Machine Learning para predicción:

Python, Streamlit, Pandas, Plotly, Scikit-learn, Statsmodels, PostgreSQL

- **Regresión Lineal**: Modelo simple para tendencias lineales

## Autor

## Uso- **Random Forest**: Captura patrones no lineales y estacionalidad

Bruno Bouza Fernández

- **Media Móvil con Tendencia**: Incluye factores estacionales semanales

```bash- **ARIMA**: Modelo de series temporales sin componente estacional

streamlit run app.py- **SARIMA**: Modelo avanzado con estacionalidad (configurable)

```

### Análisis de Series Temporales

## Tecnologías- **Gráficas ACF/PACF**: Para selección de parámetros ARIMA

- **Análisis de Diferencia Estacional**: Para SARIMA (y_t - y_(t-s))

Python, Streamlit, Pandas, Plotly, Scikit-learn, Statsmodels, PostgreSQL- **Parámetros configurables**: Control completo sobre (p,d,q) y (P,D,Q,s)



## Autor## 🛠️ Tecnologías



Bruno Bouza Fernández- **Python 3.8+**

- **Streamlit**: Framework de aplicaciones web
- **Pandas**: Manipulación de datos
- **Plotly**: Visualizaciones interactivas
- **NumPy**: Cálculos numéricos
- **Scikit-learn**: Algoritmos de ML (Regresión Lineal, Random Forest)
- **Statsmodels**: Modelos de series temporales (ARIMA, SARIMA)
- **SQLAlchemy**: ORM para conexión a base de datos
- **Psycopg2**: Driver de PostgreSQL
- **Neon**: Base de datos PostgreSQL serverless en la nube

## 📦 Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/BrunoBouza/minsait-motors-dashboard.git
cd minsait-motors-dashboard
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

5. **Configura los Secrets** 🔒:
   - Copia el archivo de ejemplo:
     ```bash
     copy .streamlit\secrets.toml.example .streamlit\secrets.toml
     ```
   - Edita `.streamlit/secrets.toml` con tu URL de conexión a Neon PostgreSQL

## 🚀 Uso

Ejecuta la aplicación:
```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## ☁️ Deploy en Streamlit Cloud

1. Sube tu código a GitHub (el archivo `secrets.toml` NO se subirá)
2. Ve a [share.streamlit.io](https://share.streamlit.io/)
3. Conecta tu repositorio
4. En "Settings" > "Secrets", pega el contenido de tu `secrets.toml`:
   ```toml
   [connections.neon]
   url = "postgresql://USUARIO:CONTRASEÑA@ep-xxxx.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
   ```
5. ¡Despliega tu app!

## 📁 Estructura del Proyecto

```
minsait-motors-dashboard/
├── app.py                  # Aplicación principal de Streamlit
├── data_loader.py          # Carga de datos desde PostgreSQL
├── visualizations.py       # Funciones de visualización con Plotly
├── predictions.py          # Modelos de predicción ML/Series Temporales
├── Car Sales.csv           # Dataset original (respaldo)
├── requirements.txt        # Dependencias del proyecto
├── README.md              # Este archivo
├── .gitignore             # Archivos a ignorar por Git
└── .streamlit/
    ├── secrets.toml       # 🔒 Credenciales (NO subir a GitHub)
    └── secrets.toml.example  # Plantilla de configuración
```

## 📊 Base de Datos

El proyecto utiliza **Neon PostgreSQL** para almacenar los datos de ventas:
- **Tabla**: `car_sales`
- **Conexión**: Configurada en `.streamlit/secrets.toml`
- **Ventajas**: 
  - Datos centralizados en la nube
  - Actualización sin modificar código
  - Escalabilidad automática
  - Conexión segura con SSL

### Estructura de Datos

La tabla `car_sales` contiene información de ventas de vehículos con las siguientes columnas:
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



Desarrollado como proyecto práctico de análisis de datos y predicción de series temporales.
