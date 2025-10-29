"""
Interfaz de administración RBAC para Streamlit
"""
import streamlit as st
import pandas as pd
from auth_client import AuthClient


def show_rbac_admin():
    """Muestra la interfaz completa de administración RBAC"""
    
    # Verificar que el usuario sea admin
    if not st.session_state.user_info or st.session_state.user_info.get("role") != "admin":
        st.error("⛔ Solo los administradores pueden acceder a esta sección")
        return
    
    st.title("⚙️ Administración RBAC")
    st.markdown("Sistema de Control de Acceso Basado en Roles")
    st.markdown("---")
    
    # Tabs para diferentes secciones
    tab1, tab2, tab3, tab4 = st.tabs([
        "👥 Usuarios",
        "👨‍👩‍👧‍👦 Grupos",
        "🔐 Roles",
        "🔗 Asignaciones"
    ])
    
    auth_client = AuthClient()
    
    # TAB 1: Gestión de Usuarios
    with tab1:
        show_usuarios_management(auth_client)
    
    # TAB 2: Gestión de Grupos
    with tab2:
        show_grupos_management(auth_client)
    
    # TAB 3: Gestión de Roles
    with tab3:
        show_roles_management(auth_client)
    
    # TAB 4: Asignaciones
    with tab4:
        show_asignaciones_management(auth_client)


def show_usuarios_management(auth_client: AuthClient):
    """Gestión de usuarios"""
    st.header("👥 Gestión de Usuarios")
    
    # Botón para refrescar
    col_refresh, col_create = st.columns([3, 1])
    with col_refresh:
        if st.button("🔄 Refrescar Lista", key="refresh_usuarios"):
            st.rerun()
    with col_create:
        crear_usuario = st.button("➕ Nuevo Usuario", key="btn_crear_usuario", type="primary")
    
    # Obtener lista de usuarios
    result = auth_client.get_usuarios(st.session_state.token)
    
    if not result["success"]:
        st.error(f"Error al cargar usuarios: {result['error']}")
        return
    
    usuarios = result["data"]
    
    if not usuarios:
        st.info("No hay usuarios registrados")
        return
    
    # Mostrar tabla de usuarios
    st.markdown("### Lista de Usuarios")
    
    df_usuarios = pd.DataFrame(usuarios)
    
    # Agregar columnas formateadas para grupos y roles
    df_usuarios['grupos_str'] = df_usuarios['grupos'].apply(
        lambda x: '👥 ' + ', '.join(x) if isinstance(x, list) and x else '-'
    )
    df_usuarios['roles_str'] = df_usuarios['roles'].apply(
        lambda x: '🔑 ' + ', '.join(x) if isinstance(x, list) and x else '-'
    )
    
    # Mostrar tabla con grupos y roles (sin is_active)
    st.dataframe(
        df_usuarios[['id', 'username', 'email', 'nombre_completo', 'grupos_str', 'roles_str']].rename(
            columns={
                'username': 'Usuario',
                'email': 'Email',
                'nombre_completo': 'Nombre',
                'grupos_str': 'Grupos',
                'roles_str': 'Roles'
            }
        ),
        width='stretch',
        hide_index=True
    )
    
    # Sección para eliminar usuario
    st.markdown("---")
    col_delete, col_spacer = st.columns([3, 1])
    with col_delete:
        st.markdown("### 🗑️ Eliminar Usuario")
        
        col_select, col_btn = st.columns([3, 1])
        with col_select:
            usuario_a_eliminar = st.selectbox(
                "Seleccionar usuario a eliminar",
                options=[u['username'] for u in usuarios],
                key="delete_usuario_select"
            )
        with col_btn:
            if st.button("🗑️ Eliminar", key="btn_delete_usuario", type="secondary"):
                usuario = next((u for u in usuarios if u['username'] == usuario_a_eliminar), None)
                if usuario:
                    if st.session_state.user_info.get('username') == usuario['username']:
                        st.error("❌ No puedes eliminarte a ti mismo")
                    else:
                        result = auth_client.delete_usuario(usuario['id'], st.session_state.token)
                        if result["success"]:
                            st.success(f"✅ Usuario '{usuario['username']}' eliminado exitosamente")
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {result['error']}")
    
    st.markdown("---")
    
    # Formulario para crear nuevo usuario
    if crear_usuario or st.session_state.get('show_create_usuario', False):
        st.session_state.show_create_usuario = True
        
        with st.form("form_crear_usuario"):
            st.markdown("### ➕ Crear Nuevo Usuario")
            
            col1, col2 = st.columns(2)
            with col1:
                nuevo_username = st.text_input("Username *", key="nuevo_username")
                nuevo_email = st.text_input("Email", key="nuevo_email")
            with col2:
                nuevo_password = st.text_input("Contraseña *", type="password", key="nuevo_password")
                nuevo_nombre = st.text_input("Nombre Completo", key="nuevo_nombre")
            
            col_submit, col_cancel = st.columns(2)
            with col_submit:
                submit = st.form_submit_button("💾 Crear Usuario", width='stretch', type="primary")
            with col_cancel:
                cancel = st.form_submit_button("❌ Cancelar", width='stretch')
            
            if cancel:
                st.session_state.show_create_usuario = False
                st.rerun()
            
            if submit:
                if not nuevo_username or not nuevo_password:
                    st.error("Username y contraseña son obligatorios")
                else:
                    result = auth_client.register(
                        nuevo_username,
                        nuevo_password,
                        "reader",  # Rol por defecto
                        st.session_state.token
                    )
                    
                    if result["success"]:
                        st.success(f"✅ Usuario '{nuevo_username}' creado exitosamente")
                        st.session_state.show_create_usuario = False
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {result['error']}")
    
    # Sección para ver roles de un usuario
    st.markdown("---")
    st.markdown("### 🔍 Ver Roles de Usuario")
    
    usuario_seleccionado = st.selectbox(
        "Seleccionar usuario",
        options=[u['username'] for u in usuarios],
        key="select_usuario_roles"
    )
    
    if st.button("Ver Roles", key="btn_ver_roles_usuario"):
        usuario = next((u for u in usuarios if u['username'] == usuario_seleccionado), None)
        if usuario:
            result = auth_client.get_roles_usuario(usuario['id'], st.session_state.token)
            if result["success"]:
                roles_data = result["data"]
                
                st.info(f"**Total de roles únicos:** {roles_data['total_roles']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Roles Directos:**")
                    if roles_data['roles_directos']:
                        for rol in roles_data['roles_directos']:
                            st.markdown(f"- 🔑 **{rol['nombre']}** (`{rol['codigo']}`)")
                            if rol.get('descripcion'):
                                st.caption(f"  _{rol['descripcion']}_")
                    else:
                        st.info("Sin roles directos")
                
                with col2:
                    st.markdown("**Roles por Grupo:**")
                    if roles_data['roles_por_grupo']:
                        grupos_vistos = set()
                        for rol in roles_data['roles_por_grupo']:
                            grupo_nombre = rol['grupo_nombre']
                            if grupo_nombre not in grupos_vistos:
                                st.markdown(f"**Grupo: {grupo_nombre}**")
                                grupos_vistos.add(grupo_nombre)
                            st.markdown(f"- 👥 **{rol['nombre']}** (`{rol['codigo']}`)")
                            if rol.get('descripcion'):
                                st.caption(f"  _{rol['descripcion']}_")
                    else:
                        st.info("Sin roles por grupos")
                
                st.markdown("**Lista consolidada de roles:**")
                st.code(", ".join(roles_data['todos_los_roles']))
            else:
                st.error(f"Error: {result['error']}")


def show_grupos_management(auth_client: AuthClient):
    """Gestión de grupos"""
    st.header("👨‍👩‍👧‍👦 Gestión de Grupos")
    
    # Botones de acción
    col_refresh, col_create = st.columns([3, 1])
    with col_refresh:
        if st.button("🔄 Refrescar Lista", key="refresh_grupos"):
            st.rerun()
    with col_create:
        crear_grupo = st.button("➕ Nuevo Grupo", key="btn_crear_grupo", type="primary")
    
    # Obtener lista de grupos
    result = auth_client.get_grupos(st.session_state.token)
    
    if not result["success"]:
        st.error(f"Error al cargar grupos: {result['error']}")
        return
    
    grupos = result["data"]
    
    if not grupos:
        st.info("No hay grupos registrados")
    else:
        # Mostrar tabla de grupos
        st.markdown("### Lista de Grupos")
        
        df_grupos = pd.DataFrame(grupos)
        
        # Agregar columnas formateadas
        df_grupos['roles_str'] = df_grupos['roles'].apply(
            lambda x: '🔐 ' + ', '.join(x) if isinstance(x, list) and x else '-'
        )
        
        st.dataframe(
            df_grupos[['id', 'nombre', 'descripcion', 'roles_str', 'usuarios_count']].rename(
                columns={
                    'nombre': 'Grupo',
                    'descripcion': 'Descripción',
                    'roles_str': 'Roles',
                    'usuarios_count': 'Usuarios'
                }
            ),
            width='stretch',
            hide_index=True
        )
    
    # Sección para eliminar grupo
    st.markdown("---")
    col_delete, col_spacer = st.columns([3, 1])
    with col_delete:
        st.markdown("### 🗑️ Eliminar Grupo")
        
        if grupos:
            col_select, col_btn = st.columns([3, 1])
            with col_select:
                grupo_a_eliminar = st.selectbox(
                    "Seleccionar grupo a eliminar",
                    options=[g['nombre'] for g in grupos],
                    key="delete_grupo_select"
                )
            with col_btn:
                if st.button("🗑️ Eliminar", key="btn_delete_grupo", type="secondary"):
                    grupo = next((g for g in grupos if g['nombre'] == grupo_a_eliminar), None)
                    if grupo:
                        result = auth_client.delete_grupo(grupo['id'], st.session_state.token)
                        if result["success"]:
                            st.success(f"✅ Grupo '{grupo['nombre']}' eliminado exitosamente")
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {result['error']}")
    
    st.markdown("---")
    
    # Formulario para crear nuevo grupo
    if crear_grupo or st.session_state.get('show_create_grupo', False):
        st.session_state.show_create_grupo = True
        
        with st.form("form_crear_grupo"):
            st.markdown("### ➕ Crear Nuevo Grupo")
            
            col1, col2 = st.columns(2)
            with col1:
                nuevo_nombre = st.text_input("Nombre *", key="nuevo_grupo_nombre", help="Nombre único del grupo (ej: Ventas, Soporte)")
            with col2:
                nuevo_descripcion = st.text_area("Descripción", key="nuevo_grupo_descripcion")
            
            col_submit, col_cancel = st.columns(2)
            with col_submit:
                submit = st.form_submit_button("💾 Crear Grupo", width='stretch', type="primary")
            with col_cancel:
                cancel = st.form_submit_button("❌ Cancelar", width='stretch')
            
            if cancel:
                st.session_state.show_create_grupo = False
                st.rerun()
            
            if submit:
                if not nuevo_nombre:
                    st.error("El nombre es obligatorio")
                else:
                    data = {
                        "nombre": nuevo_nombre,
                        "descripcion": nuevo_descripcion if nuevo_descripcion else None
                    }
                    
                    result = auth_client.create_grupo(data, st.session_state.token)
                    
                    if result["success"]:
                        st.success(f"✅ Grupo '{nuevo_nombre}' creado exitosamente")
                        st.session_state.show_create_grupo = False
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {result['error']}")
    
    # Sección para ver roles de un grupo
    st.markdown("---")
    st.markdown("### 🔍 Ver Roles de Grupo")
    
    if grupos:
        grupo_seleccionado = st.selectbox(
            "Seleccionar grupo",
            options=[g['nombre'] for g in grupos],
            key="select_grupo_roles"
        )
        
        if st.button("Ver Roles", key="btn_ver_roles_grupo"):
            grupo = next((g for g in grupos if g['nombre'] == grupo_seleccionado), None)
            if grupo:
                result = auth_client.get_roles_grupo(grupo['id'], st.session_state.token)
                if result["success"]:
                    roles_grupo = result["data"]
                    
                    if roles_grupo:
                        st.success(f"**Total de roles asignados:** {len(roles_grupo)}")
                        for rol in roles_grupo:
                            st.markdown(f"- 🔐 **{rol['nombre']}** (`{rol['codigo']}`)")
                            if rol.get('descripcion'):
                                st.caption(f"  _{rol['descripcion']}_")
                    else:
                        st.info("Este grupo no tiene roles asignados")
                else:
                    st.error(f"Error: {result['error']}")


def show_roles_management(auth_client: AuthClient):
    """Gestión de roles"""
    st.header("🔐 Gestión de Roles")
    
    # Botones de acción
    col_refresh, col_create = st.columns([3, 1])
    with col_refresh:
        if st.button("🔄 Refrescar Lista", key="refresh_roles"):
            st.rerun()
    with col_create:
        crear_rol = st.button("➕ Nuevo Rol", key="btn_crear_rol", type="primary")
    
    # Obtener lista de roles
    result = auth_client.get_roles(st.session_state.token)
    
    if not result["success"]:
        st.error(f"Error al cargar roles: {result['error']}")
        return
    
    roles = result["data"]
    
    if not roles:
        st.info("No hay roles registrados")
    else:
        # Mostrar tabla de roles (sin is_active)
        st.markdown("### Lista de Roles")
        
        df_roles = pd.DataFrame(roles)
        st.dataframe(
            df_roles[['id', 'codigo', 'nombre', 'descripcion']].rename(
                columns={
                    'codigo': 'Código',
                    'nombre': 'Nombre',
                    'descripcion': 'Descripción'
                }
            ),
            width='stretch',
            hide_index=True
        )
    
    # Sección para eliminar rol
    st.markdown("---")
    col_delete, col_spacer = st.columns([3, 1])
    with col_delete:
        st.markdown("### 🗑️ Eliminar Rol")
        
        if roles:
            col_select, col_btn = st.columns([3, 1])
            with col_select:
                rol_a_eliminar = st.selectbox(
                    "Seleccionar rol a eliminar",
                    options=[r['nombre'] for r in roles],
                    key="delete_rol_select"
                )
            with col_btn:
                if st.button("🗑️ Eliminar", key="btn_delete_rol", type="secondary"):
                    rol = next((r for r in roles if r['nombre'] == rol_a_eliminar), None)
                    if rol:
                        result = auth_client.delete_rol(rol['id'], st.session_state.token)
                        if result["success"]:
                            st.success(f"✅ Rol '{rol['nombre']}' eliminado exitosamente")
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {result['error']}")
    
    st.markdown("---")
    
    # Formulario para crear nuevo rol
    if crear_rol or st.session_state.get('show_create_rol', False):
        st.session_state.show_create_rol = True
        
        with st.form("form_crear_rol"):
            st.markdown("### ➕ Crear Nuevo Rol")
            
            col1, col2 = st.columns(2)
            with col1:
                nuevo_codigo = st.text_input("Código *", key="nuevo_rol_codigo", help="Código único del rol (ej: manager)")
            with col2:
                nuevo_nombre = st.text_input("Nombre *", key="nuevo_rol_nombre", help="Nombre descriptivo del rol")
            
            nuevo_descripcion = st.text_area("Descripción", key="nuevo_rol_descripcion")
            
            col_submit, col_cancel = st.columns(2)
            with col_submit:
                submit = st.form_submit_button("💾 Crear Rol", width='stretch', type="primary")
            with col_cancel:
                cancel = st.form_submit_button("❌ Cancelar", width='stretch')
            
            if cancel:
                st.session_state.show_create_rol = False
                st.rerun()
            
            if submit:
                if not nuevo_codigo or not nuevo_nombre:
                    st.error("Código y nombre son obligatorios")
                else:
                    data = {
                        "codigo": nuevo_codigo,
                        "nombre": nuevo_nombre,
                        "descripcion": nuevo_descripcion if nuevo_descripcion else None
                    }
                    
                    result = auth_client.create_rol(data, st.session_state.token)
                    
                    if result["success"]:
                        st.success(f"✅ Rol '{nuevo_nombre}' creado exitosamente")
                        st.session_state.show_create_rol = False
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {result['error']}")


def show_asignaciones_management(auth_client: AuthClient):
    """Gestión de asignaciones"""
    st.header("🔗 Gestión de Asignaciones")
    
    st.markdown("""
    Aquí puedes asignar:
    - **Usuarios a Grupos**: Un usuario puede pertenecer a múltiples grupos
    - **Roles a Grupos**: Un grupo puede tener múltiples roles
    - **Roles a Usuarios**: Roles asignados directamente (sin grupo)
    """)
    
    # Obtener datos necesarios
    usuarios_result = auth_client.get_usuarios(st.session_state.token)
    grupos_result = auth_client.get_grupos(st.session_state.token)
    roles_result = auth_client.get_roles(st.session_state.token)
    
    if not all([usuarios_result["success"], grupos_result["success"], roles_result["success"]]):
        st.error("Error al cargar datos")
        return
    
    usuarios = usuarios_result["data"]
    grupos = grupos_result["data"]
    roles = roles_result["data"]
    
    # Tabs para diferentes tipos de asignaciones
    tab1, tab2, tab3 = st.tabs([
        "👤➡️👨‍👩‍👧‍👦 Usuario → Grupo",
        "👨‍👩‍👧‍👦➡️🔐 Grupo → Rol",
        "👤➡️🔐 Usuario → Rol (Directo)"
    ])
    
    # TAB 1: Asignar Usuario a Grupo
    with tab1:
        col_asignar, col_remover = st.columns(2)
        
        with col_asignar:
            st.markdown("### ➕ Asignar Usuario a Grupo")
            
            with st.form("form_asignar_usuario_grupo"):
                usuario_sel = st.selectbox(
                    "Usuario",
                    options=[u['username'] for u in usuarios],
                    key="asign_usuario_grupo_usuario"
                )
                grupo_sel = st.selectbox(
                    "Grupo",
                    options=[g['nombre'] for g in grupos],
                    key="asign_usuario_grupo_grupo"
                )
                
                submit = st.form_submit_button("➕ Asignar", width='stretch', type="primary")
                
                if submit:
                    usuario = next((u for u in usuarios if u['username'] == usuario_sel), None)
                    grupo = next((g for g in grupos if g['nombre'] == grupo_sel), None)
                    
                    if usuario and grupo:
                        result = auth_client.asignar_usuario_grupo(
                            usuario['id'],
                            grupo['id'],
                            st.session_state.token
                        )
                        
                        if result["success"]:
                            st.success(f"✅ {result['data']['message']}")
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {result['error']}")
        
        with col_remover:
            st.markdown("### ➖ Remover Usuario de Grupo")
            
            with st.form("form_remover_usuario_grupo"):
                usuario_rem = st.selectbox(
                    "Usuario",
                    options=[u['username'] for u in usuarios],
                    key="rem_usuario_grupo_usuario"
                )
                grupo_rem = st.selectbox(
                    "Grupo",
                    options=[g['nombre'] for g in grupos],
                    key="rem_usuario_grupo_grupo"
                )
                
                submit_rem = st.form_submit_button("➖ Remover", width='stretch', type="secondary")
                
                if submit_rem:
                    usuario = next((u for u in usuarios if u['username'] == usuario_rem), None)
                    grupo = next((g for g in grupos if g['nombre'] == grupo_rem), None)
                    
                    if usuario and grupo:
                        result = auth_client.remover_usuario_grupo(
                            usuario['id'],
                            grupo['id'],
                            st.session_state.token
                        )
                        
                        if result["success"]:
                            st.success(f"✅ {result['data']['message']}")
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {result['error']}")
    
    # TAB 2: Asignar Rol a Grupo
    with tab2:
        col_asignar, col_remover = st.columns(2)
        
        with col_asignar:
            st.markdown("### ➕ Asignar Rol a Grupo")
            
            with st.form("form_asignar_rol_grupo"):
                grupo_sel = st.selectbox(
                    "Grupo",
                    options=[g['nombre'] for g in grupos],
                    key="asign_rol_grupo_grupo"
                )
                rol_sel = st.selectbox(
                    "Rol",
                    options=[r['nombre'] for r in roles],
                    key="asign_rol_grupo_rol"
                )
                
                submit = st.form_submit_button("➕ Asignar", width='stretch', type="primary")
                
                if submit:
                    grupo = next((g for g in grupos if g['nombre'] == grupo_sel), None)
                    rol = next((r for r in roles if r['nombre'] == rol_sel), None)
                    
                    if grupo and rol:
                        result = auth_client.asignar_rol_grupo(
                            grupo['id'],
                            rol['id'],
                            st.session_state.token
                        )
                        
                        if result["success"]:
                            st.success(f"✅ {result['data']['message']}")
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {result['error']}")
        
        with col_remover:
            st.markdown("### ➖ Remover Rol de Grupo")
            
            with st.form("form_remover_rol_grupo"):
                grupo_rem = st.selectbox(
                    "Grupo",
                    options=[g['nombre'] for g in grupos],
                    key="rem_rol_grupo_grupo"
                )
                rol_rem = st.selectbox(
                    "Rol",
                    options=[r['nombre'] for r in roles],
                    key="rem_rol_grupo_rol"
                )
                
                submit_rem = st.form_submit_button("➖ Remover", width='stretch', type="secondary")
                
                if submit_rem:
                    grupo = next((g for g in grupos if g['nombre'] == grupo_rem), None)
                    rol = next((r for r in roles if r['nombre'] == rol_rem), None)
                    
                    if grupo and rol:
                        result = auth_client.remover_rol_grupo(
                            grupo['id'],
                            rol['id'],
                            st.session_state.token
                        )
                        
                        if result["success"]:
                            st.success(f"✅ {result['data']['message']}")
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {result['error']}")
    
    # TAB 3: Asignar Rol directo a Usuario
    with tab3:
        col_asignar, col_remover = st.columns(2)
        
        with col_asignar:
            st.markdown("### ➕ Asignar Rol Directo a Usuario")
            st.info("Los roles directos se asignan sin grupo intermedio")
            
            with st.form("form_asignar_rol_usuario"):
                usuario_sel = st.selectbox(
                    "Usuario",
                    options=[u['username'] for u in usuarios],
                    key="asign_rol_usuario_usuario"
                )
                rol_sel = st.selectbox(
                    "Rol",
                    options=[r['nombre'] for r in roles],
                    key="asign_rol_usuario_rol"
                )
                
                submit = st.form_submit_button("➕ Asignar", width='stretch', type="primary")
                
                if submit:
                    usuario = next((u for u in usuarios if u['username'] == usuario_sel), None)
                    rol = next((r for r in roles if r['nombre'] == rol_sel), None)
                    
                    if usuario and rol:
                        result = auth_client.asignar_rol_usuario(
                            usuario['id'],
                            rol['id'],
                            st.session_state.token
                        )
                        
                        if result["success"]:
                            st.success(f"✅ {result['data']['message']}")
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {result['error']}")
        
        with col_remover:
            st.markdown("### ➖ Remover Rol de Usuario")
            
            with st.form("form_remover_rol_usuario"):
                usuario_rem = st.selectbox(
                    "Usuario",
                    options=[u['username'] for u in usuarios],
                    key="rem_rol_usuario_usuario"
                )
                rol_rem = st.selectbox(
                    "Rol",
                    options=[r['nombre'] for r in roles],
                    key="rem_rol_usuario_rol"
                )
                
                submit_rem = st.form_submit_button("➖ Remover", width='stretch', type="secondary")
                
                if submit_rem:
                    usuario = next((u for u in usuarios if u['username'] == usuario_rem), None)
                    rol = next((r for r in roles if r['nombre'] == rol_rem), None)
                    
                    if usuario and rol:
                        result = auth_client.remover_rol_usuario(
                            usuario['id'],
                            rol['id'],
                            st.session_state.token
                        )
                        
                        if result["success"]:
                            st.success(f"✅ {result['data']['message']}")
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {result['error']}")
