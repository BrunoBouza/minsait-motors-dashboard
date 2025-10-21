# Configuración de Autenticación - Streamlit

## Variables de Entorno

Copia el archivo `.env.example` a `.env`:

```bash
cp .env.example .env
```

### Configuración de API

Por defecto, la aplicación se conecta a la API en `http://localhost:8000`.

Si despliegas la API en otro servidor, actualiza la variable en `.env`:

```
API_URL=https://tu-api.ejemplo.com
```

## Conexión a Neon PostgreSQL

La conexión a Neon ya está configurada en `.streamlit/secrets.toml` para cargar los datos de ventas.

La API usará la **misma base de datos** para almacenar usuarios.

## Instalación de Dependencias

```bash
pip install -r requirements.txt
```

## Ejecutar la Aplicación

```bash
streamlit run app.py
```

## Iniciar Sesión

1. Asegúrate de que la API esté ejecutándose (`http://localhost:8000`)
2. Abre la aplicación Streamlit
3. Verás la pantalla de login
4. Puedes:
   - **Iniciar sesión** con un usuario existente
   - **Registrarte** como nuevo usuario

## Flujo de Autenticación

1. El usuario ingresa credenciales en Streamlit
2. Streamlit envía las credenciales a la API
3. La API valida las credenciales contra la base de datos Neon
4. Si son válidas, la API devuelve un token JWT
5. Streamlit guarda el token en `session_state`
6. El token se usa para todas las peticiones subsiguientes

## Estructura de Sesión

La sesión de Streamlit mantiene:
- `authenticated`: Boolean - Si el usuario está autenticado
- `token`: String - Token JWT
- `user_info`: Dict - Información del usuario (username, email, etc.)

## Cerrar Sesión

Haz clic en el botón "Cerrar Sesión" en la esquina superior derecha del dashboard.
