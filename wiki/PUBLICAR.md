# Publicar la wiki en GitHub

> **Importante:** primero tenés que **activar Wikis** en el repo, si no GitHub responde "Repository not found".

## Paso 1 — Habilitar Wiki (solo una vez)

1. Ir a https://github.com/Galylilian/Obligatorio_prueba/settings
2. **Features** → marcar **Wikis** ✅
3. Guardar

## Paso 2 — Subir (ya tenemos los archivos listos)

Desde la carpeta del proyecto:

```powershell
cd c:\Users\usuario\Desktop\proyectos_cursor\MLP_caidas\obligatorio\obligatorio\_wiki_push
$env:GIT_SSL_NO_VERIFY='true'
git push -u origin master
```

Si la rama por defecto es `main`:

```powershell
git branch -M main
git push -u origin main
```

**Alternativa** (copiar de nuevo desde `wiki/`):

```powershell
cd c:\Users\usuario\Desktop\proyectos_cursor\MLP_caidas\obligatorio\obligatorio
git clone https://github.com/Galylilian/Obligatorio_prueba.wiki.git _wiki_push
Copy-Item wiki\*.md _wiki_push\ -Exclude PUBLICAR.md -Force
cd _wiki_push
git add .
git commit -m "Wiki humanizada"
git push origin master
```

## URL final

https://github.com/Galylilian/Obligatorio_prueba/wiki
