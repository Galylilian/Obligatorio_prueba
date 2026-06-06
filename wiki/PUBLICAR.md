# Publicar la wiki en GitHub

La wiki de GitHub vive en un repositorio **separado**:
`https://github.com/Galylilian/Obligatorio_prueba.wiki.git`

## Paso 1 — Habilitar Wiki en GitHub

1. Ir a https://github.com/Galylilian/Obligatorio_prueba/settings
2. Sección **Features** → marcar **Wikis**
3. Guardar

## Paso 2 — Subir estas páginas

```powershell
cd c:\Users\usuario\Desktop\proyectos_cursor\MLP_caidas\obligatorio\obligatorio

# Clonar repo wiki (vacío la primera vez)
git clone https://github.com/Galylilian/Obligatorio_prueba.wiki.git _wiki_push

# Copiar páginas
Copy-Item wiki\*.md _wiki_push\ -Force

# Commit y push
cd _wiki_push
git add .
git commit -m "Wiki inicial: deteccion de caidas MLP"
git push origin master
```

> Si la rama por defecto es `main`: usar `git push origin main`

## Páginas incluidas

| Archivo | Título en GitHub Wiki |
|---------|------------------------|
| Home.md | Inicio |
| Instalacion-y-configuracion.md | Instalación y configuración |
| Pipeline-offline.md | Pipeline offline |
| Modelo-y-entrenamiento.md | Modelo y entrenamiento |
| API-e-inferencia.md | API e inferencia |
| Streamlit.md | Streamlit |
| EDA-y-datos.md | EDA y datos |
| Docker.md | Docker |
| Solucion-de-problemas.md | Solución de problemas |

URL final: https://github.com/Galylilian/Obligatorio_prueba/wiki
