import streamlit as st
import io
import zipfile
import time

# Configuración de página principal
st.set_page_config(page_title="Procesador Múltiple de Imágenes", layout="centered", page_icon="🖼️")

def procesar_imagenes_mock(archivos):
    """
    Simula el procesamiento de imágenes en el backend.
    En el futuro, esto se conectará con la lógica real de CPU/GPU.
    """
    imagenes_procesadas = []
    # Barra de progreso para darle feedback al usuario
    progress_bar = st.progress(0)
    
    for i, archivo in enumerate(archivos):
        # Simulamos el tiempo de trabajo por cada imagen (ej: 1 segundo)
        time.sleep(1)
        
        # Para esta etapa, solo copiamos la imagen original como "resultado"
        imagenes_procesadas.append({
            "nombre": archivo.name,
            "bytes": archivo.getvalue(),
            "tipo": archivo.type
        })
        progress_bar.progress((i + 1) / len(archivos))
        
    return imagenes_procesadas

def generar_zip(imagenes_procesadas):
    """Genera un archivo ZIP en memoria con todas las imágenes procesadas listas para descarga."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for img in imagenes_procesadas:
            # En un entorno real, la imagen tendría otro nombre o vendría modificada
            # Acá mantenemos el original o le agregamos el prefijo
            zip_file.writestr(img['nombre'], img['bytes'])
    return zip_buffer.getvalue()

def main():
    # Título y descripción
    st.title("⚡ Procesador de Imágenes")
    st.write("Sube tus imágenes para aplicarles transformaciones de forma rápida. Más adelante el sistema aprovechará tu CPU o GPU disponible para acelerar las tareas.")
    
    # Nota requerida sobre el límite de archivos
    st.info("💡 **Nota:** Puedes subir desde **1** hasta un máximo de **10** imágenes al mismo tiempo.")
    
    # Botón/Área para subir las imágenes
    archivos_subidos = st.file_uploader(
        "Selecciona tus imágenes", 
        type=["png", "jpg", "jpeg", "webp", "bmp"], 
        accept_multiple_files=True
    )
    
    # Procesar si hay archivos subidos
    if archivos_subidos:
        if len(archivos_subidos) > 10:
            st.error("⚠️ Has superado el límite máximo de 10 archivos. Por favor, remueve algunas imágenes.")
        else:
            st.write("### 🖼️ Archivos seleccionados:")
            
            # Ventana/Galería con los archivos seleccionados
            cols = st.columns(3)
            for idx, archivo in enumerate(archivos_subidos):
                with cols[idx % 3]:
                    st.image(archivo, caption=archivo.name, use_container_width=True)
            
            st.divider()
            
            # Botón de transformación
            if st.button("Transformar Imágenes", type="primary"):
                with st.spinner("Transformando las imágenes, por favor espera..."):
                    # Acá llamamos a nuestra simulación del backend
                    resultados = procesar_imagenes_mock(archivos_subidos)
                
                st.success("¡Transformación finalizada con éxito!")
                
                # Preparamos el ZIP con todos los archivos transformados
                zip_data = generar_zip(resultados)
                
                # Botón para descargar, mostrando los nombres originales dentro del zip
                st.download_button(
                    label="⬇️ Descargar todas las imágenes transformadas",
                    data=zip_data,
                    file_name="imagenes_transformadas.zip",
                    mime="application/zip"
                )

if __name__ == "__main__":
    main()
