"""
Cliente de autenticación para comunicarse con la API
"""
import requests
import streamlit as st
from typing import Optional, Dict
import os

# Intentar obtener la URL de la API desde diferentes fuentes
def get_api_url():
    """Obtiene la URL de la API desde secrets (producción) o .env (local)"""
    # Primero intentar desde secrets de Streamlit (producción)
    try:
        # Intentar desde connections.neon.API_URL primero
        if "connections" in st.secrets and "neon" in st.secrets["connections"]:
            if "API_URL" in st.secrets["connections"]["neon"]:
                return st.secrets["connections"]["neon"]["API_URL"]
    except:
        pass
    
    # Intentar desde la raíz de secrets
    try:
        api_url = st.secrets.get("API_URL")
        if api_url:
            return api_url
    except Exception as e:
        pass
    
    # Intentar como variable directa en secrets
    try:
        if "API_URL" in st.secrets:
            return st.secrets["API_URL"]
    except:
        pass
    
    # Si no, intentar desde variable de entorno
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_url = os.getenv("API_URL", "http://localhost:8000")
        return api_url
    except:
        return "http://localhost:8000"

class AuthClient:
    """Cliente para manejar autenticación con la API"""
    
    def __init__(self, api_url: str = None):
        self.api_url = api_url if api_url else get_api_url()
    
    def register(self, username: str, password: str, role: str, token: str) -> Dict:
        """Registra un nuevo usuario - Solo admins"""
        try:
            response = requests.post(
                f"{self.api_url}/register",
                json={
                    "username": username,
                    "password": password,
                    "role": role
                },
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            
            if response.status_code == 201:
                return {"success": True, "data": response.json()}
            else:
                error_detail = response.json().get("detail", "Error desconocido")
                return {"success": False, "error": error_detail}
                
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def login(self, username: str, password: str) -> Dict:
        """Inicia sesión y obtiene un token"""
        try:
            response = requests.post(
                f"{self.api_url}/login",
                json={
                    "username": username,
                    "password": password
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "token": data["access_token"],
                    "token_type": data["token_type"]
                }
            else:
                return {"success": False, "error": "Usuario o contraseña incorrectos"}
                
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def verify_token(self, token: str) -> Dict:
        """Verifica si un token es válido"""
        try:
            response = requests.get(
                f"{self.api_url}/verify-token",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": "Token inválido"}
                
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def get_current_user(self, token: str) -> Dict:
        """Obtiene información del usuario actual"""
        try:
            response = requests.get(
                f"{self.api_url}/users/me",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": "No se pudo obtener información del usuario"}
                
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def get_sales(self, token: str, skip: int = 0, limit: int = None) -> Dict:
        """Obtiene la lista de ventas - Todos los usuarios autenticados"""
        try:
            params = {"skip": skip}
            if limit is not None:
                params["limit"] = limit
            
            response = requests.get(
                f"{self.api_url}/sales",
                params=params,
                headers={"Authorization": f"Bearer {token}"},
                timeout=60  # Aumentar timeout para cargar todos los datos
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                error_detail = response.json().get("detail", "Error desconocido")
                return {"success": False, "error": error_detail}
                
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def create_sale(self, sale_data: Dict, token: str) -> Dict:
        """Crea una nueva venta - Solo writers y admins"""
        try:
            response = requests.post(
                f"{self.api_url}/sales",
                json=sale_data,
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            
            if response.status_code == 201:
                return {"success": True, "data": response.json()}
            else:
                error_detail = response.json().get("detail", "Error desconocido")
                return {"success": False, "error": error_detail}
                
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}


def init_session_state():
    """Inicializa el estado de sesión de Streamlit"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'token' not in st.session_state:
        st.session_state.token = None
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None


def login_page():
    """Muestra la página de login"""
    st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 2rem;
        }
        .stButton button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("🚗 Minsait Motors")
        st.markdown("### Iniciar Sesión")
        
        auth_client = AuthClient()
        
        # Formulario de Login
        with st.form("login_form"):
            username = st.text_input("Usuario", key="login_username")
            password = st.text_input("Contraseña", type="password", key="login_password")
            submit = st.form_submit_button("Iniciar Sesión", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("Por favor completa todos los campos")
                else:
                    with st.spinner("Autenticando..."):
                        result = auth_client.login(username, password)
                        
                        if result["success"]:
                            st.session_state.authenticated = True
                            st.session_state.token = result["token"]
                            
                            # Obtener información del usuario
                            user_result = auth_client.get_current_user(result["token"])
                            if user_result["success"]:
                                st.session_state.user_info = user_result["data"]
                            
                            st.success("¡Inicio de sesión exitoso!")
                            st.rerun()
                        else:
                            st.error(f"Error: {result['error']}")


def logout():
    """Cierra la sesión del usuario"""
    st.session_state.authenticated = False
    st.session_state.token = None
    st.session_state.user_info = None
    st.rerun()


def show_user_management():
    """Muestra el panel de gestión de usuarios (solo para admin)"""
    st.markdown("### 👥 Gestión de Usuarios")
    
    # Verificar que el usuario sea admin
    if st.session_state.user_info.get("role") != "admin":
        st.warning("⚠️ Solo los administradores pueden gestionar usuarios.")
        return
    
    auth_client = AuthClient()
    
    with st.form("create_user_form"):
        st.markdown("#### Crear Nuevo Usuario")
        new_username = st.text_input("Nombre de usuario")
        new_password = st.text_input("Contraseña", type="password")
        new_role = st.selectbox(
            "Rol",
            options=["reader", "writer", "admin"],
            help="reader: Solo lectura | writer: Lectura y escritura | admin: Acceso completo"
        )
        submit = st.form_submit_button("Crear Usuario", use_container_width=True)
        
        if submit:
            if not new_username or not new_password:
                st.error("Por favor completa todos los campos")
            elif len(new_password) < 6:
                st.error("La contraseña debe tener al menos 6 caracteres")
            else:
                with st.spinner("Creando usuario..."):
                    result = auth_client.register(
                        new_username,
                        new_password,
                        new_role,
                        st.session_state.token
                    )
                    
                    if result["success"]:
                        st.success(f"✓ Usuario '{new_username}' creado exitosamente con rol '{new_role}'")
                    else:
                        st.error(f"Error: {result['error']}")


def show_new_sale_form():
    """Muestra el formulario para añadir una nueva venta (solo para admin y writer)"""
    st.markdown("### 🚗 Añadir Nueva Venta")
    
    # Verificar que el usuario tenga permisos
    user_role = st.session_state.user_info.get("role")
    if user_role not in ["admin", "writer"]:
        st.warning("⚠️ Solo los administradores y editores pueden añadir ventas.")
        return
    
    auth_client = AuthClient()
    
    with st.form("new_sale_form"):
        st.markdown("#### Información de la Venta")
        
        # Organizar en columnas para mejor UX
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Datos del Cliente**")
            customer_name = st.text_input("Nombre del Cliente *", placeholder="Ej: John Doe")
            gender = st.selectbox("Género *", options=["Male", "Female"])
            annual_income = st.number_input("Ingreso Anual ($) *", min_value=0.0, step=1000.0, format="%.2f")
            phone = st.number_input("Teléfono * (solo números)", min_value=0, step=1, format="%d", help="Ejemplo: 1234567890")
        
        with col2:
            st.markdown("**Datos del Distribuidor**")
            dealer_name = st.text_input("Nombre del Distribuidor *", placeholder="Ej: AutoMax Dealers")
            dealer_no = st.text_input("Número de Distribuidor *", placeholder="Ej: D001")
            dealer_region = st.selectbox(
                "Región del Distribuidor *",
                options=["North", "South", "East", "West", "Central"]
            )
            date = st.date_input("Fecha de Venta *")
        
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**Datos del Vehículo**")
            company = st.text_input("Marca *", placeholder="Ej: Toyota")
            model = st.text_input("Modelo *", placeholder="Ej: Camry")
            engine = st.text_input("Motor *", placeholder="Ej: 2.5L 4-Cylinder")
            transmission = st.selectbox("Transmisión *", options=["Automatic", "Manual"])
        
        with col4:
            st.markdown("**Características**")
            color = st.text_input("Color *", placeholder="Ej: Silver")
            body_style = st.selectbox(
                "Estilo de Carrocería *",
                options=["Sedan", "SUV", "Hatchback", "Coupe", "Truck", "Van", "Convertible"]
            )
            price = st.number_input("Precio ($) *", min_value=0.0, step=100.0, format="%.2f")
        
        st.markdown("---")
        st.caption("* Campos obligatorios")
        
        submit = st.form_submit_button("💾 Guardar Venta", use_container_width=True, type="primary")
        
        if submit:
            # Validar campos obligatorios
            if not all([customer_name, dealer_name, company, model, engine, color, dealer_no]):
                st.error("❌ Por favor completa todos los campos obligatorios")
            elif annual_income <= 0:
                st.error("❌ El ingreso anual debe ser mayor a 0")
            elif price <= 0:
                st.error("❌ El precio debe ser mayor a 0")
            elif phone <= 0:
                st.error("❌ El teléfono debe ser un número válido mayor a 0")
            else:
                # Preparar datos para enviar a la API
                sale_data = {
                    "date": date.isoformat(),
                    "customer_name": customer_name,
                    "gender": gender,
                    "annual_income": annual_income,
                    "dealer_name": dealer_name,
                    "company": company,
                    "model": model,
                    "engine": engine,
                    "transmission": transmission,
                    "color": color,
                    "price": price,
                    "dealer_no": dealer_no,
                    "body_style": body_style,
                    "phone": phone,
                    "dealer_region": dealer_region
                }
                
                with st.spinner("Guardando venta..."):
                    result = auth_client.create_sale(sale_data, st.session_state.token)
                    
                    if result["success"]:
                        st.success(f"✅ Venta registrada exitosamente!")
                        st.markdown(f"""
                        **Detalles de la venta:**
                        - Cliente: {customer_name}
                        - Vehículo: {company} {model}
                        - Precio: ${price:,.2f}
                        - Fecha: {date.strftime('%d/%m/%Y')}
                        """)
                    else:
                        st.error(f"❌ Error al guardar la venta: {result['error']}")


def require_auth(func):
    """Decorador para requerir autenticación"""
    def wrapper(*args, **kwargs):
        init_session_state()
        
        if not st.session_state.authenticated:
            login_page()
            return None
        
        # Verificar que el token siga siendo válido
        auth_client = AuthClient()
        result = auth_client.verify_token(st.session_state.token)
        
        if not result["success"]:
            st.warning("Tu sesión ha expirado. Por favor, inicia sesión nuevamente.")
            logout()
            return None
        
        return func(*args, **kwargs)
    
    return wrapper
