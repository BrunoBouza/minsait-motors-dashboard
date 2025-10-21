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

API_URL = get_api_url()

class AuthClient:
    """Cliente para manejar autenticación con la API"""
    
    def __init__(self, api_url: str = API_URL):
        self.api_url = api_url
    
    def register(self, username: str, password: str) -> Dict:
        """Registra un nuevo usuario"""
        try:
            response = requests.post(
                f"{self.api_url}/register",
                json={
                    "username": username,
                    "password": password
                },
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
    
    # DEBUG: Mostrar qué URL está usando (temporal)
    api_url_debug = get_api_url()
    st.sidebar.info(f"🔍 API URL: {api_url_debug}")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("🚗 Minsait Motors")
        st.markdown("### Iniciar Sesión")
        
        # Tabs para Login y Registro
        tab1, tab2 = st.tabs(["Login", "Registro"])
        
        auth_client = AuthClient()
        
        with tab1:
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
        
        with tab2:
            # Formulario de Registro
            with st.form("register_form"):
                new_username = st.text_input("Usuario", key="reg_username")
                new_password = st.text_input("Contraseña", type="password", key="reg_password")
                new_password_confirm = st.text_input("Confirmar Contraseña", type="password", key="reg_password_confirm")
                submit_register = st.form_submit_button("Registrarse", use_container_width=True)
                
                if submit_register:
                    if not new_username or not new_password:
                        st.error("Por favor completa todos los campos")
                    elif new_password != new_password_confirm:
                        st.error("Las contraseñas no coinciden")
                    elif len(new_password) < 6:
                        st.error("La contraseña debe tener al menos 6 caracteres")
                    else:
                        with st.spinner("Registrando usuario..."):
                            result = auth_client.register(
                                new_username,
                                new_password
                            )
                            
                            if result["success"]:
                                st.success("¡Registro exitoso! Ahora puedes iniciar sesión.")
                            else:
                                st.error(f"Error: {result['error']}")


def logout():
    """Cierra la sesión del usuario"""
    st.session_state.authenticated = False
    st.session_state.token = None
    st.session_state.user_info = None
    st.rerun()


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
