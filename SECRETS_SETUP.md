# 🔐 Configuración de Secrets - Guía Rápida

## ✅ Archivos Creados

1. **`.streamlit/secrets.toml`** - Credenciales de Neon PostgreSQL (🔒 protegido en .gitignore)
2. **`.streamlit/secrets.toml.example`** - Plantilla pública para configuración

## 📝 Cambios Realizados

### 1. `data_loader.py`
**Antes:**
```python
connection_string = "postgresql://neondb_owner:contraseña@..."
```

**Después:**
```python
connection_string = st.secrets["connections"]["neon"]["url"]
```

### 2. `requirements.txt`
Agregadas dependencias:
- `sqlalchemy`
- `psycopg2-binary`

### 3. `.gitignore`
Ya incluye (verificado):
- `.streamlit/secrets.toml` ✅
- `temp.py` ✅

### 4. `README.md`
Agregadas secciones:
- Configuración de secrets
- Deploy en Streamlit Cloud
- Información sobre Neon PostgreSQL

## 🚀 Para Usar Localmente

```bash
# La app ya está configurada, solo ejecuta:
streamlit run app.py
```

## ☁️ Para Deploy en Streamlit Cloud

1. Sube el código a GitHub (secrets.toml NO se subirá)
2. Ve a https://share.streamlit.io/
3. Conecta tu repositorio
4. En **Settings > Secrets**, pega:
   ```toml
   [connections.neon]
   url = "postgresql://neondb_owner:npg_rfV6Ci8LAPeX@ep-falling-night-agmqu5t2-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
   ```

## ✨ Ventajas

- ✅ Credenciales protegidas (no en el código)
- ✅ No se suben a GitHub
- ✅ Fácil compartir código sin exponer credenciales
- ✅ Mismo código funciona local y en la nube
- ✅ Compatible con Streamlit Cloud

## 📌 Notas Importantes

- El archivo `secrets.toml` está en `.gitignore` y NUNCA se subirá a GitHub
- Para producción, considera usar variables de entorno adicionales
- La conexión usa SSL/TLS para seguridad
- Neon PostgreSQL es serverless y escala automáticamente
