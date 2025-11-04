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
                try:
                    error_detail = response.json().get("detail", "Error desconocido")
                except:
                    error_detail = f"Error HTTP {response.status_code}: {response.text[:200]}"
                return {"success": False, "error": error_detail}
                
        except requests.exceptions.JSONDecodeError as e:
            return {"success": False, "error": f"Error al parsear respuesta JSON: {str(e)}"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def update_sale(self, sale_id: str, sale_data: Dict, token: str) -> Dict:
        """Actualiza una venta existente - Solo writers y admins"""
        try:
            response = requests.put(
                f"{self.api_url}/sales/{sale_id}",
                json=sale_data,
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                error_detail = response.json().get("detail", "Error desconocido")
                return {"success": False, "error": error_detail}
                
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def delete_sale(self, sale_id: str, token: str) -> Dict:
        """Elimina una venta - Solo writers y admins"""
        try:
            response = requests.delete(
                f"{self.api_url}/sales/{sale_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            
            if response.status_code == 204:
                return {"success": True}
            else:
                error_detail = response.json().get("detail", "Error desconocido")
                return {"success": False, "error": error_detail}
                
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def get_sale_by_id(self, sale_id: str, token: str) -> Dict:
        """Obtiene una venta por ID"""
        try:
            response = requests.get(
                f"{self.api_url}/sales/{sale_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                error_detail = response.json().get("detail", "Error desconocido")
                return {"success": False, "error": error_detail}
                
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    # ==================== ENDPOINTS SNOWFLAKE SYNC ====================
    
    def trigger_snowflake_sync(self, token: str) -> Dict:
        """Ejecuta manualmente la sincronización con Snowflake - Solo admins"""
        try:
            response = requests.post(
                f"{self.api_url}/sync/snowflake",
                headers={"Authorization": f"Bearer {token}"},
                timeout=60  # Timeout más largo para sincronización
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                error_detail = response.json().get("detail", "Error desconocido")
                return {"success": False, "error": error_detail}
                
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def get_sync_status(self, token: str) -> Dict:
        """Obtiene el estado de la sincronización con Snowflake"""
        try:
            response = requests.get(
                f"{self.api_url}/sync/snowflake/status",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                error_detail = response.json().get("detail", "Error desconocido")
                return {"success": False, "error": error_detail}
                
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    # ==================== ENDPOINTS RBAC ====================
    
    # Usuarios
    def get_usuarios(self, token: str) -> Dict:
        """Obtiene la lista de usuarios - Solo admins"""
        try:
            response = requests.get(
                f"{self.api_url}/usuarios",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.json().get("detail", "Error desconocido")}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def update_usuario(self, usuario_id: int, data: Dict, token: str) -> Dict:
        """Actualiza un usuario - Solo admins"""
        try:
            response = requests.put(
                f"{self.api_url}/usuarios/{usuario_id}",
                json=data,
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.json().get("detail", "Error desconocido")}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def delete_usuario(self, usuario_id: int, token: str) -> Dict:
        """Elimina un usuario - Solo admins"""
        try:
            response = requests.delete(
                f"{self.api_url}/usuarios/{usuario_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            if response.status_code == 204:
                return {"success": True}
            else:
                return {"success": False, "error": response.json().get("detail", "Error desconocido")}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    # Grupos
    def get_grupos(self, token: str) -> Dict:
        """Obtiene la lista de grupos - Solo admins"""
        try:
            response = requests.get(
                f"{self.api_url}/grupos",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.json().get("detail", "Error desconocido")}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def create_grupo(self, data: Dict, token: str) -> Dict:
        """Crea un nuevo grupo - Solo admins"""
        try:
            response = requests.post(
                f"{self.api_url}/grupos",
                json=data,
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            if response.status_code == 201:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.json().get("detail", "Error desconocido")}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def update_grupo(self, grupo_id: int, data: Dict, token: str) -> Dict:
        """Actualiza un grupo - Solo admins"""
        try:
            response = requests.put(
                f"{self.api_url}/grupos/{grupo_id}",
                json=data,
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.json().get("detail", "Error desconocido")}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def delete_grupo(self, grupo_id: int, token: str) -> Dict:
        """Elimina un grupo - Solo admins"""
        try:
            response = requests.delete(
                f"{self.api_url}/grupos/{grupo_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            if response.status_code == 204:
                return {"success": True}
            else:
                return {"success": False, "error": response.json().get("detail", "Error desconocido")}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    # Roles
    def get_roles(self, token: str) -> Dict:
        """Obtiene la lista de roles - Solo admins"""
        try:
            response = requests.get(
                f"{self.api_url}/roles",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.json().get("detail", "Error desconocido")}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def create_rol(self, data: Dict, token: str) -> Dict:
        """Crea un nuevo rol - Solo admins"""
        try:
            response = requests.post(
                f"{self.api_url}/roles",
                json=data,
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            if response.status_code == 201:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.json().get("detail", "Error desconocido")}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    # Asignaciones
    def asignar_usuario_grupo(self, usuario_id: int, grupo_id: int, token: str) -> Dict:
        """Asigna un usuario a un grupo - Solo admins"""
        try:
            response = requests.post(
                f"{self.api_url}/usuarios/{usuario_id}/grupos",
                json={"grupo_id": grupo_id},
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.json().get("detail", "Error desconocido")}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def remover_usuario_grupo(self, usuario_id: int, grupo_id: int, token: str) -> Dict:
        """Remueve un usuario de un grupo - Solo admins"""
        try:
            response = requests.delete(
                f"{self.api_url}/usuarios/{usuario_id}/grupos/{grupo_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.json().get("detail", "Error desconocido")}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def asignar_rol_grupo(self, grupo_id: int, rol_id: int, token: str) -> Dict:
        """Asigna un rol a un grupo - Solo admins"""
        try:
            response = requests.post(
                f"{self.api_url}/grupos/{grupo_id}/roles",
                json={"rol_id": rol_id},
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.json().get("detail", "Error desconocido")}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def asignar_rol_usuario(self, usuario_id: int, rol_id: int, token: str) -> Dict:
        """Asigna un rol directo a un usuario - Solo admins"""
        try:
            response = requests.post(
                f"{self.api_url}/usuarios/{usuario_id}/roles",
                json={"rol_id": rol_id},
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.json().get("detail", "Error desconocido")}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def get_permisos_usuario(self, usuario_id: int, token: str) -> Dict:
        """Obtiene todos los permisos de un usuario"""
        try:
            response = requests.get(
                f"{self.api_url}/usuarios/{usuario_id}/permisos",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.json().get("detail", "Error desconocido")}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def get_roles_usuario(self, usuario_id: int, token: str) -> Dict:
        """Obtiene todos los roles de un usuario (directos y por grupos)"""
        try:
            response = requests.get(
                f"{self.api_url}/usuarios/{usuario_id}/roles",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.json().get("detail", "Error desconocido")}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def get_roles_grupo(self, grupo_id: int, token: str) -> Dict:
        """Obtiene todos los roles asignados a un grupo"""
        try:
            response = requests.get(
                f"{self.api_url}/grupos/{grupo_id}/roles",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.json().get("detail", "Error desconocido")}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    # ==================== MÉTODOS DE ELIMINACIÓN ====================
    
    def delete_usuario(self, usuario_id: int, token: str) -> Dict:
        """Elimina un usuario - Solo admins"""
        try:
            response = requests.delete(
                f"{self.api_url}/usuarios/{usuario_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            if response.status_code == 204:
                return {"success": True}
            else:
                return {"success": False, "error": response.json().get("detail", "Error desconocido")}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def delete_grupo(self, grupo_id: int, token: str) -> Dict:
        """Elimina un grupo - Solo admins"""
        try:
            response = requests.delete(
                f"{self.api_url}/grupos/{grupo_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            if response.status_code == 204:
                return {"success": True}
            else:
                return {"success": False, "error": response.json().get("detail", "Error desconocido")}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def delete_rol(self, rol_id: int, token: str) -> Dict:
        """Elimina un rol - Solo admins"""
        try:
            response = requests.delete(
                f"{self.api_url}/roles/{rol_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            if response.status_code == 204:
                return {"success": True}
            else:
                return {"success": False, "error": response.json().get("detail", "Error desconocido")}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    # ==================== MÉTODOS PARA REMOVER ASIGNACIONES ====================
    
    def remover_usuario_grupo(self, usuario_id: int, grupo_id: int, token: str) -> Dict:
        """Remueve un usuario de un grupo - Solo admins"""
        try:
            response = requests.delete(
                f"{self.api_url}/usuarios/{usuario_id}/grupos/{grupo_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.json().get("detail", "Error desconocido")}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def remover_rol_grupo(self, grupo_id: int, rol_id: int, token: str) -> Dict:
        """Remueve un rol de un grupo - Solo admins"""
        try:
            response = requests.delete(
                f"{self.api_url}/grupos/{grupo_id}/roles/{rol_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.json().get("detail", "Error desconocido")}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}
    
    def remover_rol_usuario(self, usuario_id: int, rol_id: int, token: str) -> Dict:
        """Remueve un rol directo de un usuario - Solo admins"""
        try:
            response = requests.delete(
                f"{self.api_url}/usuarios/{usuario_id}/roles/{rol_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.json().get("detail", "Error desconocido")}
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


def get_default_sale_data():
    """Retorna datos por defecto para testing rápido"""
    from datetime import date
    return {
        "customer_name": "John Testing Doe",
        "gender": "Male",
        "annual_income": 75000.0,
        "phone": 5551234567,
        "dealer_name": "TestDrive Motors",
        "dealer_no": "T999",
        "dealer_region": "Central",
        "date": date.today(),
        "company": "Toyota",
        "model": "Corolla",
        "engine": "1.8L 4-Cylinder",
        "transmission": "Auto",
        "color": "Silver",
        "body_style": "Sedan",
        "price": 25000.0
    }


def show_new_sale_form():
    """Muestra el formulario para añadir una nueva venta (solo para admin y writer)"""
    st.markdown("### 🚗 Gestión de Ventas")
    
    # Verificar que el usuario tenga permisos
    user_role = st.session_state.user_info.get("role")
    if user_role not in ["admin", "writer"]:
        st.warning("⚠️ Solo los administradores y editores pueden gestionar ventas.")
        return
    
    auth_client = AuthClient()
    
    # Tabs para diferentes acciones
    tab_add, tab_edit, tab_delete = st.tabs(["➕ Nueva Venta", "✏️ Editar Venta", "🗑️ Eliminar Venta"])
    
    # TAB: AÑADIR NUEVA VENTA
    with tab_add:
        st.markdown("#### Añadir Nueva Venta")
        
        # Botón para cargar datos por defecto
        use_defaults = st.checkbox("🚀 Usar datos por defecto (Testing rápido)", key="use_defaults_add")
        
        if use_defaults:
            defaults = get_default_sale_data()
            st.info("✅ Datos de prueba cargados. Puedes modificarlos antes de guardar.")
        else:
            defaults = {}
    
        with st.form("new_sale_form"):
            st.markdown("#### Información de la Venta")
            
            # Organizar en columnas para mejor UX
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Datos del Cliente**")
                customer_name = st.text_input("Nombre del Cliente *", placeholder="Ej: John Doe", value=defaults.get("customer_name", ""))
                gender = st.selectbox("Género *", options=["Male", "Female"], index=0 if defaults.get("gender") == "Male" else 1 if defaults.get("gender") else 0)
                annual_income = st.number_input("Ingreso Anual ($) *", min_value=0.0, step=1000.0, format="%.2f", value=defaults.get("annual_income", 0.0))
                phone = st.number_input("Teléfono * (solo números)", min_value=0, step=1, format="%d", help="Ejemplo: 1234567890", value=defaults.get("phone", 0))
            
            with col2:
                st.markdown("**Datos del Distribuidor**")
                dealer_name = st.text_input("Nombre del Distribuidor *", placeholder="Ej: AutoMax Dealers", value=defaults.get("dealer_name", ""))
                dealer_no = st.text_input("Número de Distribuidor *", placeholder="Ej: D001", value=defaults.get("dealer_no", ""))
                regions = ["North", "South", "East", "West", "Central"]
                dealer_region = st.selectbox(
                    "Región del Distribuidor *",
                    options=regions,
                    index=regions.index(defaults.get("dealer_region")) if defaults.get("dealer_region") in regions else 0
                )
                date = st.date_input("Fecha de Venta *", value=defaults.get("date"))
            
            st.markdown("---")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("**Datos del Vehículo**")
                company = st.text_input("Marca *", placeholder="Ej: Toyota", value=defaults.get("company", ""))
                model = st.text_input("Modelo *", placeholder="Ej: Camry", value=defaults.get("model", ""))
                engine = st.text_input("Motor *", placeholder="Ej: 2.5L 4-Cylinder", value=defaults.get("engine", ""))
                transmissions = ["Auto", "Manual"]
                transmission = st.selectbox("Transmisión *", options=transmissions, index=transmissions.index(defaults.get("transmission")) if defaults.get("transmission") in transmissions else 0)
            
            with col4:
                st.markdown("**Características**")
                color = st.text_input("Color *", placeholder="Ej: Silver", value=defaults.get("color", ""))
                body_styles = ["Sedan", "SUV", "Hatchback", "Coupe", "Truck", "Van", "Convertible"]
                body_style = st.selectbox(
                    "Estilo de Carrocería *",
                    options=body_styles,
                    index=body_styles.index(defaults.get("body_style")) if defaults.get("body_style") in body_styles else 0
                )
                price = st.number_input("Precio ($) *", min_value=0.0, step=100.0, format="%.2f", value=defaults.get("price", 0.0))
            
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
    
    # TAB: EDITAR VENTA
    with tab_edit:
        st.markdown("#### Editar Venta Existente")
        
        # Buscar venta por ID
        sale_id_search = st.text_input("🔍 Ingresa el ID de la venta a editar", key="edit_sale_id", placeholder="Ej: 5f8d0d55b...") 
        
        if st.button("Buscar Venta", key="search_sale"):
            if not sale_id_search:
                st.warning("⚠️ Por favor ingresa un ID de venta")
            else:
                with st.spinner("Buscando venta..."):
                    result = auth_client.get_sale_by_id(sale_id_search, st.session_state.token)
                    
                    if result["success"]:
                        st.session_state.sale_to_edit = result["data"]
                        st.success("✅ Venta encontrada")
                    else:
                        st.error(f"❌ {result['error']}")
                        st.session_state.sale_to_edit = None
        
        # Mostrar formulario si hay una venta cargada
        if "sale_to_edit" in st.session_state and st.session_state.sale_to_edit:
            sale = st.session_state.sale_to_edit
            
            with st.form("edit_sale_form"):
                st.markdown(f"**Editando venta ID:** `{sale['id']}`")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Datos del Cliente**")
                    customer_name_edit = st.text_input("Nombre del Cliente *", value=sale.get("customer_name", ""))
                    gender_edit = st.selectbox("Género *", options=["Male", "Female"], index=0 if sale.get("gender") == "Male" else 1)
                    annual_income_edit = st.number_input("Ingreso Anual ($) *", value=float(sale.get("annual_income", 0)), format="%.2f")
                    phone_edit = st.number_input("Teléfono *", value=int(sale.get("phone", 0)), format="%d")
                
                with col2:
                    st.markdown("**Datos del Distribuidor**")
                    dealer_name_edit = st.text_input("Nombre del Distribuidor *", value=sale.get("dealer_name", ""))
                    dealer_no_edit = st.text_input("Número de Distribuidor *", value=sale.get("dealer_no", ""))
                    regions = ["North", "South", "East", "West", "Central"]
                    dealer_region_edit = st.selectbox("Región *", options=regions, index=regions.index(sale.get("dealer_region")) if sale.get("dealer_region") in regions else 0)
                    from datetime import datetime
                    try:
                        date_edit = st.date_input("Fecha *", value=datetime.fromisoformat(sale.get("date")).date() if sale.get("date") else datetime.now().date())
                    except:
                        date_edit = st.date_input("Fecha *")
                
                st.markdown("---")
                col3, col4 = st.columns(2)
                
                with col3:
                    st.markdown("**Datos del Vehículo**")
                    company_edit = st.text_input("Marca *", value=sale.get("company", ""))
                    model_edit = st.text_input("Modelo *", value=sale.get("model", ""))
                    engine_edit = st.text_input("Motor *", value=sale.get("engine", ""))
                    transmissions = ["Auto", "Manual"]
                    transmission_edit = st.selectbox("Transmisión *", options=transmissions, index=transmissions.index(sale.get("transmission")) if sale.get("transmission") in transmissions else 0)
                
                with col4:
                    st.markdown("**Características**")
                    color_edit = st.text_input("Color *", value=sale.get("color", ""))
                    body_styles = ["Sedan", "SUV", "Hatchback", "Coupe", "Truck", "Van", "Convertible"]
                    body_style_edit = st.selectbox("Estilo de Carrocería *", options=body_styles, index=body_styles.index(sale.get("body_style")) if sale.get("body_style") in body_styles else 0)
                    price_edit = st.number_input("Precio ($) *", value=float(sale.get("price", 0)), format="%.2f")
                
                submit_edit = st.form_submit_button("💾 Actualizar Venta", use_container_width=True, type="primary")
                
                if submit_edit:
                    if not all([customer_name_edit, dealer_name_edit, company_edit, model_edit, engine_edit, color_edit, dealer_no_edit]):
                        st.error("❌ Por favor completa todos los campos obligatorios")
                    elif annual_income_edit <= 0 or price_edit <= 0 or phone_edit <= 0:
                        st.error("❌ Los valores numéricos deben ser mayores a 0")
                    else:
                        sale_data_edit = {
                            "date": date_edit.isoformat(),
                            "customer_name": customer_name_edit,
                            "gender": gender_edit,
                            "annual_income": annual_income_edit,
                            "dealer_name": dealer_name_edit,
                            "company": company_edit,
                            "model": model_edit,
                            "engine": engine_edit,
                            "transmission": transmission_edit,
                            "color": color_edit,
                            "price": price_edit,
                            "dealer_no": dealer_no_edit,
                            "body_style": body_style_edit,
                            "phone": phone_edit,
                            "dealer_region": dealer_region_edit
                        }
                        
                        with st.spinner("Actualizando venta..."):
                            result = auth_client.update_sale(sale["id"], sale_data_edit, st.session_state.token)
                            
                            if result["success"]:
                                st.success(f"✅ Venta actualizada exitosamente!")
                                del st.session_state.sale_to_edit
                            else:
                                st.error(f"❌ Error: {result['error']}")
        else:
            st.info("ℹ️ Ingresa un ID de venta y haz clic en 'Buscar Venta' para comenzar")
    
    # TAB: ELIMINAR VENTA
    with tab_delete:
        st.markdown("#### Eliminar Venta")
        st.warning("⚠️ **ADVERTENCIA:** Esta acción NO se puede deshacer")
        
        sale_id_delete = st.text_input("🔍 Ingresa el ID de la venta a eliminar", key="delete_sale_id", placeholder="Ej: 5f8d0d55b...")
        
        if st.button("Buscar Venta para Eliminar", key="search_sale_delete"):
            if not sale_id_delete:
                st.warning("⚠️ Por favor ingresa un ID de venta")
            else:
                with st.spinner("Buscando venta..."):
                    result = auth_client.get_sale_by_id(sale_id_delete, st.session_state.token)
                    
                    if result["success"]:
                        st.session_state.sale_to_delete = result["data"]
                        st.success("✅ Venta encontrada")
                    else:
                        st.error(f"❌ {result['error']}")
                        st.session_state.sale_to_delete = None
        
        # Mostrar detalles y confirmar eliminación
        if "sale_to_delete" in st.session_state and st.session_state.sale_to_delete:
            sale = st.session_state.sale_to_delete
            
            st.markdown("---")
            st.markdown("### Detalles de la Venta a Eliminar:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **Cliente:** {sale.get('customer_name')}  
                **Género:** {sale.get('gender')}  
                **Teléfono:** {sale.get('phone')}  
                **Ingreso Anual:** ${sale.get('annual_income'):,.2f}
                """)
            
            with col2:
                st.markdown(f"""
                **Vehículo:** {sale.get('company')} {sale.get('model')}  
                **Color:** {sale.get('color')}  
                **Precio:** ${sale.get('price'):,.2f}  
                **Fecha:** {sale.get('date')}
                """)
            
            st.markdown("---")
            
            # Confirmación
            confirm = st.checkbox("✅ Confirmo que quiero eliminar esta venta", key="confirm_delete")
            
            col_cancel, col_delete = st.columns([1, 1])
            
            with col_cancel:
                if st.button("❌ Cancelar", use_container_width=True):
                    del st.session_state.sale_to_delete
                    st.rerun()
            
            with col_delete:
                if st.button("🗑️ ELIMINAR VENTA", use_container_width=True, type="primary", disabled=not confirm):
                    with st.spinner("Eliminando venta..."):
                        result = auth_client.delete_sale(sale["id"], st.session_state.token)
                        
                        if result["success"]:
                            st.success("✅ Venta eliminada exitosamente")
                            del st.session_state.sale_to_delete
                        else:
                            st.error(f"❌ Error: {result['error']}")
        else:
            st.info("ℹ️ Ingresa un ID de venta y haz clic en 'Buscar Venta para Eliminar' para ver los detalles")


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
