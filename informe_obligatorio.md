# Informe — Obligatorio Machine Learning en Producción

**Proyecto:** Fall Detector — detección de caídas en imágenes y video
**Materia:** Machine Learning en Producción — Máster en Big Data, Universidad ORT

---

## 1. Introducción

Este informe describe el proceso de construcción de un sistema de Machine Learning de punta a punta: desde la recolección de datos hasta una API en producción, pasando por el entrenamiento del modelo y las decisiones tomadas para que el sistema se comporte de forma consistente entre el entrenamiento y la inferencia real.

Elegimos como problema la **detección de caídas de personas** a partir de imágenes y video. Es un problema con valor real (monitoreo de adultos mayores, seguridad en espacios públicos), con datos no triviales de conseguir (no hay un dataset "de caídas" grande y limpio disponible), y que obliga a pensar en varios de los desafíos que la materia puso el foco: cómo se define el target, cómo se evita que el dataset tenga fugas de información, y cómo se evita que el modelo vea en producción algo distinto de lo que vio en entrenamiento.

El resto de este informe sigue, en la medida de lo posible, el mismo orden que la consigna del obligatorio: dataset, representación del problema, ambiente, versionado, desafíos generales (data leakage y training-serving skew), API, despliegue, y por último los requerimientos electivos que decidimos implementar.

---

## 2. Dataset

### 2.1 Por qué dos fuentes, y por qué no una carpeta de "subida manual"

Al principio del proyecto teníamos una vía adicional para sumar imágenes al dataset: subirlas manualmente a una carpeta, sin pasar por scraper ni por extracción de video. La sacamos del diseño. El motivo es simple: esa vía no dejaba ningún rastro de dónde venía la imagen (`source=unknown`), y eso rompe cualquier intento serio de trazabilidad o de análisis de calidad del dataset más adelante. Decidimos que **todo dato tiene que entrar por una de dos fuentes bien diferenciadas y comparables entre sí**:

- **Pexels** (`scripts/scrape_dataset.py`): scraping vía la API oficial de Pexels, con un conjunto de queries genéricas en `queries/people.txt` (personas paradas, caminando, en el piso, etc.).
- **Video propio** (`scripts/extract_video_frames.py`): extracción de frames de un video real, guardando **todos** los frames sin usar el modelo para preseleccionar cuáles guardar. Esto fue una decisión deliberada: si dejáramos que un modelo (o cualquier heurística) decidiera qué frames vale la pena guardar, estaríamos contaminando el dataset con el sesgo de ese mismo modelo antes incluso de entrenar nada.

Ambas fuentes caen en un mismo pool sin etiqueta (`data/raw/pool/`), y cada imagen queda registrada en `pool_log.csv` con su procedencia real (`source=pexels` o `source=video`). Esa procedencia se propaga después hasta el dataset final (`dataset_labels.csv`), así que en cualquier momento se puede responder "¿de dónde salió esta imagen que el modelo clasificó mal?".

### 2.2 Etiquetado: por qué por persona y no por imagen

El etiquetado lo hace un humano, mirando cada imagen, con una herramienta propia (`scripts/label_tool.py`) — nunca se asigna una clase a partir del texto de la query de búsqueda ni de la predicción de un modelo (eso sería *weak labeling*, y arrastra ruido sistemático al dataset).

La decisión más importante que tomamos acá fue diseñar el etiquetado **por persona, con bounding box, y no por imagen completa**. La razón es doble:

1. **Una imagen puede tener más de una persona**, y no necesariamente todas están en el mismo estado — puede haber alguien caído y alguien parado al lado. Etiquetar la imagen entera como una sola clase hubiera sido incorrecto en esos casos, o nos hubiera obligado a descartar imágenes perfectamente válidas.
2. **El modelo debería mirar a la persona, no el fondo.** Si entrenamos con la imagen completa, el modelo puede aprender atajos espurios (el tipo de piso, la iluminación, el ángulo de cámara) en vez de la postura de la persona. Recortar al bounding box de cada persona fuerza al modelo a enfocarse en la señal real.

Cada persona confirmada (`F`=fall / `N`=no_fall) queda registrada como una fila en `data/raw/bbox_log.csv` (filename, label, box normalizado 0–1, timestamp). Cuando la imagen completa está lista (al menos una persona marcada), se mueve a `data/raw/labeled/`. Las imágenes ambiguas (mala calidad, postura no clara, persona cortada) se descartan o se saltan explícitamente, en vez de forzar una clase con baja confianza.

Esta decisión de "etiquetar por persona con box" tuvo un costo: hubo que reescribir el etiquetador dos veces durante el proyecto (primero para soportar un solo box por imagen, después para soportar múltiples personas por imagen con estados independientes), y en el camino encontramos y corregimos un bug real de la herramienta donde, al pasar de una imagen a la siguiente, quedaban *listeners* de mouse duplicados en el navegador que corrompían el dibujo del box a partir de la segunda imagen. Lo mencionamos porque es un buen ejemplo de que construir la herramienta de etiquetado no es un detalle menor del proyecto: es donde se define la calidad de todo lo que viene después.

### 2.3 Análisis Exploratorio de Datos (EDA)

El EDA (`notebooks/eda.ipynb`) se armó en cuatro bloques, y no fue un ejercicio cosmético: encontramos y corregimos dos problemas reales del dataset a partir de él.

1. **Auditoría de archivos y metadata**: formatos, imágenes corruptas o de 0 KB, distribución de resoluciones y aspect ratio.
2. **Contenido visual**: brillo, contraste y nitidez (Laplaciano vía OpenCV), distribución de canales RGB.
3. **Balance de clases**: conteo de `fall`/`no_fall`, cruce con la fuente (Pexels vs. video), verificación de que la estratificación del split se cumple, personas por imagen, y relación entre aspect ratio/área del bounding box y la clase.
4. **Duplicados y dimensionalidad**: detección de imágenes casi-idénticas (perceptual hashing) y una proyección t-SNE de los embeddings de la penúltima capa de la ResNet18 entrenada, coloreada por clase, para ver si el modelo separa razonablemente bien las dos clases en su espacio latente.

Los hallazgos importantes del punto 4 se explican en detalle en la sección de Data Leakage más abajo, porque terminaron siendo el hallazgo más significativo de todo el análisis.

Además, después de construir el dataset final hicimos una revisión de calidad puntual: al inspeccionar visualmente una muestra del EDA detectamos casos de imágenes con boxes duplicados sobre la misma persona (de una etapa anterior del etiquetador, donde un fallo silencioso hacía que una confirmación se guardara más de una vez) y un caso de etiqueta incorrecta (una persona acostada en una cama, etiquetada como `fall` cuando el criterio del proyecto es "caída al piso", no "postura acostada"). Escribimos `scripts/find_inconsistent_duplicates.py` específicamente para automatizar la detección de este tipo de conflictos: agrupa imágenes casi-duplicadas por el mismo hash que se usa para evitar la fuga entre splits, y marca los grupos donde aparecen a la vez `fall` y `no_fall` sobre la misma persona. Corregimos manualmente los casos reales (uno de relabeling, varios de deduplicación) y volvimos a construir el dataset y reentrenar sobre esa versión más limpia.

---

## 3. Representación del problema

Definimos el problema como **clasificación binaria a nivel de persona**: `fall` vs. `no_fall`. Consideramos brevemente un esquema multiclase (por ejemplo, distinguir "cayendo", "caído", "sentado", "parado"), pero lo descartamos por dos razones: primero, el valor práctico del sistema está en la pregunta binaria ("¿hay que alertar o no?"); segundo, con el tamaño de dataset que podíamos conseguir en el tiempo disponible, dividir en más clases iba a dejar cada una con muy pocos ejemplos, especialmente en la clase crítica (`fall`).

El desbalance entre clases (la clase `fall` es minoritaria, como es de esperar en cualquier fuente real de datos — la gente pasa la mayor parte del tiempo sin estar caída) lo resolvimos con **class weights** calculados automáticamente desde la distribución real del set de entrenamiento (inversamente proporcionales a la frecuencia de cada clase), en vez de fijar un valor a mano. Esto penaliza más los errores sobre `fall`, que es la clase que realmente importa detectar bien en este problema.

---

## 4. Ambiente

Separamos las dependencias de desarrollo (`dev-requirements.txt`) de las de producción (`requirements.txt`). En producción, `requirements.txt` instala la build CPU-only de PyTorch (`torch==2.7.1+cpu`) en lugar de la build con soporte CUDA — evita descargar ~5 GB de paquetes que no sirven de nada en una instancia sin GPU y que pueden llenar el disco de una instancia chica de EC2.

Todo el sistema online (API, Streamlit, PostgreSQL, Grafana) se levanta con `docker-compose.yml`, y cada servicio corre en su propio contenedor:

| Servicio | Rol |
|---|---|
| `fastapi` | Sirve el modelo vía API REST |
| `streamlit` | Interfaz visual sobre la API |
| `postgres` | Persiste cada predicción hecha por la API |
| `grafana` | Dashboard operacional, lee Postgres directamente |

Los pesos de la ResNet18 (`models/resnet18.pth`) no se descargan: son un artefacto local que se copia a la imagen y además se monta como volumen en `docker-compose.yml`, para poder reemplazar el modelo sin reconstruir la imagen. Los únicos pesos que sí se descargan de internet son los del detector de personas (COCO, `ssdlite320_mobilenet_v3_large`), y el `Dockerfile` los precachea explícitamente durante el build, para que el contenedor no dependa de conectividad a internet al arrancar en producción.

---

## 5. Versionado de código

El código se versiona con Git y se comparte en GitHub (`Galylilian/Obligatorio_prueba`). El `.gitignore` excluye explícitamente todo lo que no debería vivir en el repositorio: el dataset (`data/`), los modelos entrenados (`models/*.pth`, que pesan varios MB y se regeneran con `train.py`), la base de datos de test (`test.db`) y el archivo `.env` con credenciales (la Pexels API key y la connection string de la base). Esta separación es intencional: el repositorio versiona **cómo se construye y se sirve el sistema**, no los artefactos pesados o sensibles que ese proceso genera — esos se reconstruyen a partir del código, o se transfieren por fuera de Git (por ejemplo, copiando `models/` a la instancia de producción vía `scp`).

Antes del primer commit, revisamos que el `.gitignore` realmente hiciera lo que decía: dos patrones (`data/processed/**` y `data/video/**`) no dejaban pasar los `.gitkeep` de las subcarpetas porque les faltaba negar explícitamente el directorio intermedio (el mismo problema que sí estaba resuelto para `data/raw/**`). Lo corregimos antes de subir nada, para que la estructura de carpetas del proyecto quede visible en GitHub aunque su contenido no se versione.

Al conectar el repositorio local con el remoto encontramos que ya existía trabajo del equipo: la rama `main` tenía commits recientes de un compañero con una base muy similar a la nuestra, y una rama personal (`marcelo2`) tenía una versión mucho más vieja del proyecto (otra estructura de datos, sin bounding boxes). Antes de hacer push comparamos el historial de ambas ramas para confirmar que no había trabajo único en la versión vieja que se fuera a perder, y recién ahí se actualizó `marcelo2` con la versión completa desarrollada en esta etapa, quedando pendiente integrarla a `main` mediante un pull request.

---

## 6. Desafíos generales

Esta es la sección donde más tiempo invertimos, porque son los dos problemas que la consigna pide prevenir explícitamente, y en los dos casos terminamos encontrando evidencia concreta de que, si no los atacábamos, el sistema iba a fallar en producción de una forma que las métricas offline no iban a delatar.

### 6.1 Data Leakage

El split train/valid/test se hace una única vez, con semilla fija (`42`), estratificado por `(clase, fuente)` — es decir, garantizamos que la proporción de `fall`/`no_fall` y de `pexels`/`video` sea similar en los tres splits, para que el modelo no aprenda a distinguir "pinta de frame de video" en vez de la caída en sí.

Eso resuelve la fuga *obvia* (la misma imagen en dos splits), pero el EDA nos hizo notar una fuga mucho más sutil: como una de las dos fuentes son frames extraídos de un video, hay pares de frames que son **casi idénticos** entre sí (frames consecutivos o muy cercanos en el tiempo). Si el split estratifica solo por clase y fuente, nada impide que un frame termine en train y su casi-gemelo termine en test — y en ese caso, el modelo no está siendo evaluado sobre datos nuevos, está siendo evaluado sobre una imagen que ya vio (o una prácticamente idéntica) durante el entrenamiento. Eso infla la accuracy de test de forma artificial, y es exactamente el tipo de problema que en producción se manifiesta como "funcionaba genial en la demo y anda mal con datos reales".

Para medirlo, implementamos un **perceptual hash** casero (dHash sobre 8×8 píxeles, sin depender de la librería `imagehash`) y agrupamos imágenes cuya distancia de Hamming es ≤ 4 sobre 64 bits. El EDA mostró que, sin corregir nada, **~47% de los pares casi-duplicados terminaban repartidos entre splits distintos**. Es un número demasiado alto para ignorarlo.

La solución fue agrupar las imágenes casi-duplicadas *antes* de estratificar: cada grupo de duplicados se trata como una única unidad indivisible, se le asigna una clase dominante y una fuente dominante (la más frecuente dentro del grupo), y el split se decide a nivel de grupo, no de imagen individual — todas las imágenes de un mismo grupo van al mismo split, sin excepción. Después de aplicar esto, volvimos a correr la misma verificación del EDA y el cruce entre splits bajó a 0 pares. Reentrenamos el modelo sobre este dataset corregido; las métricas finales reportadas en este informe ya reflejan esa versión, no la original.

### 6.2 Training-Serving Skew

El segundo desafío es evitar que el modelo, en producción, reciba una entrada distinta de la que vio durante el entrenamiento. Encontramos dos fuentes posibles de esta divergencia y las resolvimos de la misma manera en ambos casos: **un único punto de verdad para la lógica, reusado literalmente por el pipeline offline y por la API online.**

- **Preprocesamiento de imagen**: `src/core/preprocessing/transforms.py` define `get_test_transforms()` una sola vez. Tanto `evaluate.py` (offline) como `ImageClassifier` en la API (online) usan exactamente esa función — no hay una copia de las transformaciones "para producción" que se pueda desincronizar de la de evaluación.

- **Recorte a la persona**: esta fue la fuente de skew más interesante del proyecto, porque no es un problema que existiera desde el principio — lo introdujimos nosotros mismos al decidir recortar el dataset al bounding box de cada persona (sección 2.2). El dataset de entrenamiento se recorta con un box **dibujado a mano** por el etiquetador. En producción, evidentemente, no hay un humano dibujando un box antes de cada predicción. Si no resolvíamos esto, la API iba a clasificar la imagen completa mientras el modelo fue entrenado para clasificar recortes de personas — un skew grave y automático, garantizado en el 100% de las predicciones reales.

  La solución fue agregar un **detector de personas automático** (`src/core/detector.py`, `PersonDetector`) que corre *antes* de la clasificación en producción, y que usa exactamente la misma función de recorte con margen (`crop_to_box()`, `src/core/preprocessing/cropping.py`, ~15% de margen) que ya usa `convert_dataset.py` con el box dibujado a mano. No es el detector el que se comparte entre offline y online (en el dataset no hace falta detectar nada, el humano ya marcó a la persona) — lo que se comparte es la función de recorte, para que un box de origen humano y un box de origen automático produzcan el mismo tipo de imagen recortada a la entrada del clasificador. Elegimos `ssdlite320_mobilenet_v3_large` (preentrenado en COCO, sin fine-tuning) por ser el detector más liviano disponible en `torchvision`, priorizando latencia sobre precisión de localización — el proyecto corre inferencia en CPU (pensado para EC2 sin GPU), y un detector más pesado hubiera dominado la latencia total de `/predict`.

  También decidimos explícitamente qué hacer cuando el detector no encuentra a nadie en la imagen: en vez de forzar una clasificación sobre una imagen sin personas (lo que hubiera producido una predicción sin sentido, pero con una confianza numérica que parece legítima), la API devuelve `person_detected: false` y `label: null`. Es una decisión de diseño más amplia que el skew en sí, pero nace del mismo lugar: si el modelo nunca fue entrenado para clasificar "ausencia de persona", no le pedimos que invente una respuesta para ese caso.

### 6.3 Calidad del etiquetado (relacionado)

Aunque no es uno de los dos desafíos formales de la consigna, lo tratamos con el mismo criterio: `scripts/find_inconsistent_duplicates.py` reutiliza el mismo agrupado por dHash de la sección 6.1 para detectar señales de etiquetado inconsistente (el mismo grupo de imágenes casi-idénticas con labels contradictorios), como una forma de trazabilidad y control de calidad continuo sobre el dataset, no solo sobre el split.

---

## 7. API — predicciones online y batch

La API está construida con FastAPI y expone:

- **`POST /predict`**: clasifica una imagen. Corre el detector de personas, recorta, clasifica, y devuelve `label`, `confidence`, `person_detected` y `bbox`. Solo persiste en la base de datos cuando efectivamente se detectó una persona (si no hay nada que clasificar, no tiene sentido loguear una predicción).
- **`POST /predict/batch`**: la misma lógica sobre múltiples imágenes en una sola llamada.
- **`POST /gradcam`**: devuelve un heatmap de explicabilidad (ver sección de electivos) sobre el recorte de la persona detectada.
- **`POST /predict/video`**: analiza un video frame a frame (uno cada 5 segundos, para no reprocesar redundantemente un video largo), aplicando el mismo pipeline de detección + clasificación a cada frame.
- **`GET /dashboard/stats`**: combina estadísticas operacionales de PostgreSQL (predicciones totales, caídas del día/semana) con las métricas offline del modelo (`metrics.json`).

La documentación completa de contratos de request/response está en `docs/endpoints.md`, con ejemplos de `curl` para cada endpoint, y también queda expuesta automáticamente vía Swagger en `/docs` gracias a FastAPI.

---

## 8. Despliegue

El despliegue local se resuelve completamente con `docker-compose up --build`, levantando los cuatro servicios (API, Streamlit, PostgreSQL, Grafana) con un solo comando. Para producción real, el proyecto está preparado para correr sobre una instancia EC2 de AWS Academy sin GPU, usando la misma imagen Docker CPU-only — la única diferencia operativa es copiar los pesos entrenados (`models/*.pth`, que no viajan por Git) a la instancia antes de levantar los contenedores. Los pasos concretos están documentados en el README, sección "Despliegue en AWS".

### 8.1 Validación end-to-end

No nos quedamos con que el build terminara sin errores: levantamos los cuatro contenedores desde cero (incluida la instalación completa de PyTorch/OpenCV dentro de la imagen) y probamos cada endpoint contra el sistema realmente corriendo, no solo contra los tests automatizados:

- `GET /health` y `GET /dashboard/stats` respondiendo con las métricas reales de `metrics.json`.
- `POST /predict` con una foto real de una persona: detectó, recortó y clasificó correctamente, devolviendo `bbox` y `confidence`.
- `POST /predict` con una imagen sin ninguna persona: devolvió `person_detected: false` y `label: null`, tal como se diseñó en la sección 6.2, en vez de forzar una clasificación.
- `POST /predict/batch` mezclando ambos casos en la misma llamada, cada imagen resuelta de forma independiente.
- `POST /gradcam` devolviendo un JPEG válido para el caso con persona, y el JSON de error esperado para el caso sin persona.
- Confirmamos en `PostgreSQL` (vía `/dashboard/stats`) que **solo las predicciones con persona detectada quedaron persistidas** — la imagen sin persona no generó ninguna fila, validando en un entorno real (no solo en el código) la decisión de no loguear una clasificación que no existió.

Esta validación importa porque varias de las decisiones de diseño de este informe (el detector de personas, el `person_detected: false`, la persistencia condicional) son invisibles en una lectura del código: solo se confirman corriendo el sistema completo y mirando las respuestas reales.

---

## 9. Requerimientos electivos implementados

La consigna pide un mínimo de 3 categorías electivas; implementamos 4:

### 9.1 Scraper de datos
`scripts/scrape_dataset.py` scrapea imágenes de Pexels vía su API oficial, a partir de un conjunto de queries genéricas (sección 2.1). Es una de las dos fuentes del dataset, no un complemento.

### 9.2 Explicabilidad
Implementamos GradCAM sobre `layer4[-1]` de la ResNet18 (la última capa convolucional), usando hooks de forward/backward para generar el mapa de activación. Una decisión importante acá fue que el heatmap se genera **sobre el recorte de la persona**, no sobre la imagen completa — es coherente con todo lo dicho en la sección de training-serving skew: si el modelo nunca ve la imagen completa, no tiene sentido explicar una predicción sobre la imagen completa tampoco.

### 9.3 Visualización
Interfaz en Streamlit con tres vistas: un dashboard con métricas operacionales y del modelo, una vista de predicción de imágenes (individual y batch) con GradCAM opcional, y una vista de predicción de video. Además, Grafana lee directamente la tabla de predicciones en PostgreSQL para un dashboard operacional con 7 paneles, complementando la visualización más orientada al usuario final que ofrece Streamlit.

**Revisión de honestidad de los datos mostrados.** Al pulir la interfaz encontramos dos problemas que no eran errores de código en el sentido estricto (no rompían nada, no tiraban excepciones) pero sí eran datos falsos disfrazados de datos reales:

- Un campo `analytics_enabled` en `/dashboard/stats` que devolvía `True` siempre, sin chequear nada — una tarjeta en la UI mostraba "Analytics: ON" permanentemente, sugiriendo una funcionalidad que no existía.
- Un campo `high_risk_persons` que en realidad era una copia literal de `falls_today` con otro nombre — dos tarjetas del dashboard mostraban el mismo número disfrazado de dos métricas distintas.

Los eliminamos de la API y de la interfaz en vez de dejarlos o de simular que medían algo. Lo mencionamos en este informe porque es exactamente el tipo de "deuda silenciosa" que puede colarse en un sistema de ML en producción: un dashboard que "se ve bien" pero contiene números que no significan lo que dicen medir es, en la práctica, peor que no tener esa métrica — genera confianza infundada en quien lo consulta.

**Revisión de correctitud visual.** También encontramos que el color del resultado estaba invertido: la interfaz mostraba `fall` (caída) en verde y `no_fall` en rojo, exactamente al revés de la convención esperada (rojo = alerta/crítico, verde = seguro). Es un bug menor en términos de código, pero con impacto real en un sistema pensado para alertar sobre caídas: un color invertido en la señal más visible de la UI puede hacer que alguien interprete una alerta real como una situación segura.

**Bug de duplicación de predicciones (detectado mirando Grafana, no el código).** Después de dejar el sistema corriendo y usarlo desde la UI, el dashboard de Grafana mostró números que no cerraban: 12 predicciones totales y 8 caídas "hoy", cuando solo habíamos probado un puñado de imágenes reales. La causa no estaba en la lógica de clasificación sino en cómo Streamlit ejecuta la página: **reejecuta el script completo ante cualquier interacción** (tocar el uploader de video, cualquier botón), y el bloque que procesa las imágenes subidas (`if files: for file in files: ...`) no tenía ninguna guarda contra eso — cada re-ejecución volvía a mandar las mismas imágenes ya subidas a `/predict`, insertando una fila nueva en PostgreSQL por cada rerun de la página, no por cada imagen realmente nueva. La solución fue cachear la llamada a la API por el contenido binario del archivo (`@st.cache_data`), para que una misma imagen dispare como máximo una predicción real sin importar cuántas veces se vuelva a correr el script. Truncamos la tabla `predictions` para sacar los duplicados ya insertados durante las pruebas. Es un buen ejemplo de un bug que ninguna revisión de código iba a detectar por sí sola — solo apareció al observar el sistema corriendo con datos reales durante un rato, que es exactamente el tipo de verificación que describe la sección 8.1.

**Foco en el usuario final, no solo en el desarrollador.** Reescribimos las etiquetas y textos de la interfaz para que alguien sin conocimiento técnico entienda qué está viendo: "Acierta en general" en vez de "Accuracy", con una aclaración de una línea debajo de cada métrica del modelo (por ejemplo, para *recall*: "De las caídas reales, cuántas no se le escapan"); una barra de confianza visual en vez de solo un número; mensajes de ayuda cuando no se detecta ninguna persona ("Probá con una foto donde se vea el cuerpo completo"); y, en el video, una tabla con columnas en español (`¿Es una caída?`, `Segundo del video`) en lugar de los nombres de campo técnicos de la respuesta JSON (`is_fall`, `time_sec`). Ninguno de estos cambios afecta la lógica de predicción — es, en esencia, aplicar el mismo criterio de "no mostrar algo que confunda o engañe" que motivó sacar las métricas falsas.

### 9.4 Optimización de modelos
Implementamos dos técnicas, cumpliendo el mínimo pedido:

- **Data augmentation**: `get_train_transforms()` aplica augmentation solo sobre el conjunto de entrenamiento (nunca sobre test/valid, para no invalidar la evaluación).
- **Quantization dinámica** (`torch.ao.quantization.quantize_dynamic`, int8): la implementamos y, siguiendo lo que pide la consigna, medimos su impacto real en latencia (`scripts/benchmark_quantization.py`) en vez de asumir que funciona. El resultado fue negativo: la latencia no mejora (e incluso empeora levemente). La razón es que la cuantización dinámica de PyTorch solo cuantiza capas `Linear`, y en una ResNet18 la única capa `Linear` es la `fc` final (512→2), una capa minúscula comparada con el backbone convolucional donde realmente está el costo computacional — verificamos empíricamente que las capas `Conv2d` no se cuantizan con esta técnica, incluso forzándolas explícitamente en el set de módulos a cuantizar. Nos pareció más honesto reportar este resultado negativo, con su explicación técnica, que descartarlo o simular una mejora que no existe. La alternativa real para reducir latencia sería cuantización estática (PTQ) con fusión Conv+BN+ReLU y calibración con datos reales, que queda fuera del alcance de este obligatorio.

También comparamos las predicciones del modelo normal contra el cuantizado sobre el set de test completo (`scripts/compare_models.py` → `compare_models.json`), para confirmar que, aunque la cuantización no mejora la latencia, tampoco degrada la calidad de las predicciones.

---

## 10. Resultados

> **Nota sobre reproducibilidad**: el dataset (`data/`) y los modelos entrenados (`models/*.pth`) no se versionan en Git por su tamaño — se reconstruyen corriendo el pipeline (`convert_dataset.py` → `train.py` → `evaluate.py`). Los números de esta sección corresponden a la corrida de entrenamiento real hecha sobre el dataset completo durante el desarrollo. El `metrics.json` que quede efectivamente commiteado en un momento dado del repositorio puede no reflejar esta corrida si no se reentrenó después del último cambio al pipeline — antes de citar estos números en una entrega, conviene correr `evaluate.py` de nuevo y confirmar que `metrics.json` los reproduce.

Métricas finales sobre el conjunto de test (96 personas: 25 `fall`, 71 `no_fall`), con el dataset ya corregido (sin fuga entre splits, sin boxes duplicados ni etiquetas en conflicto):

| Métrica | Valor |
|---|---|
| Accuracy | 88.54% |
| Precision (no_fall) | 92.86% |
| Recall (no_fall) | 91.55% |
| F1 Score | 92.20% |

Matriz de confusión (fall=0, no_fall=1):

| Real / Predicho | Pred. fall | Pred. no_fall |
|---|---|---|
| Real fall | 20 | 5 |
| Real no_fall | 6 | 65 |

Las métricas de la tabla principal son las de `no_fall` (default de scikit-learn, que toma la clase de índice 1 como positiva), pero la clase que realmente importa en este problema es `fall`: **recall de fall = 20/25 = 80%** (se pierden 5 caídas reales de 25) y **precision de fall = 20/26 = 76.9%**.

Es un resultado razonable para un dataset construido y etiquetado íntegramente por el equipo en el tiempo del obligatorio, pero el recall de `fall` — la métrica crítica del problema, porque una caída no detectada es el error costoso — todavía tiene margen de mejora. La palanca más directa para subirlo no es ajustar hiperparámetros, sino sumar más ejemplos etiquetados de la clase minoritaria: con solo 25 casos de `fall` en test, cada error individual pesa 4 puntos porcentuales de recall.

---

## 11. Discusión: alternativas consideradas y limitaciones

- **Bounding boxes dibujados a mano vs. un detector entrenado desde el principio**: podríamos haber entrenado (o fine-tuneado) un detector de personas específico para este dominio en vez de usar uno pre-entrenado en COCO sin ajustar. Decidimos no hacerlo porque hubiera significado etiquetar bounding boxes de "persona" además de las etiquetas de "caída", duplicando el esfuerzo de etiquetado manual sin necesidad — COCO ya tiene una clase "persona" robusta y bien entrenada, y lo que necesitábamos era localización, no una clasificación fina.
- **Cuantización dinámica vs. estática**: como se explicó en la sección 9.4, la cuantización dinámica fue la opción más simple de implementar (no requiere calibración con datos), pero resultó inefectiva para esta arquitectura. Documentamos el resultado negativo en vez de forzar una alternativa más compleja (PTQ estática) dentro del tiempo disponible, dejándola explícitamente como mejora futura.
- **Multiclase vs. binario**: como se discutió en la sección 3, priorizamos tener suficientes ejemplos por clase por sobre una granularidad mayor del target.
- **Explicabilidad**: GradCAM es una técnica basada en gradientes, relativamente barata de computar, pero es una explicación aproximada (a nivel de mapa de activación de una capa convolucional), no una explicación causal de la predicción. Nos pareció suficiente para el objetivo de este obligatorio, pero una alternativa más rigurosa (SHAP, por ejemplo) queda como posible extensión.
- **Limitación principal del dataset**: el tamaño total sigue siendo chico para un problema de visión por computadora (decenas, no miles, de personas etiquetadas por clase), y una fuente importante de datos (los frames de video) tiene correlación temporal entre imágenes cercanas — la controlamos para que no cause fuga entre splits, pero sigue significando que gran parte de la clase `fall` proviene de un único evento filmado, lo cual limita cuánta variedad real de "formas de caerse" ve el modelo.

---

## 12. Uso de Inteligencia Artificial Generativa

Se utilizó **Claude Code** (Anthropic, modelo Claude Sonnet 5) como asistente durante todo el desarrollo del proyecto, en las siguientes instancias:

- Redacción y refactor de código (scripts de scraping, extracción de frames, herramienta de etiquetado, pipeline de conversión del dataset, routers de la API, utilidades de detección de duplicados).
- Diseño y ejecución del análisis exploratorio de datos (notebook `eda.ipynb`), incluyendo la implementación del perceptual hashing y la detección del problema de fuga de datos entre splits.
- Debugging de errores puntuales (bug de listeners duplicados en el etiquetador, errores de encoding en consola de Windows, filas huérfanas en `bbox_log.csv`).
- Redacción inicial de la documentación técnica (`README.md`, `docs/arquitectura.md`, `docs/endpoints.md`) y de este informe.
- Configuración del repositorio Git/GitHub (`.gitignore`, primer commit, resolución de la divergencia con el trabajo previo del equipo).
- Validación end-to-end del despliegue con Docker Compose (sección 8.1) y revisión de la interfaz de Streamlit, incluyendo la detección de las dos métricas falsas, el bug de color y el bug de duplicación de predicciones por reruns de Streamlit descriptos en la sección 9.3.

Todo el contenido generado fue revisado, ejecutado y validado por el equipo antes de incorporarlo al proyecto — en particular, los hallazgos de calidad de datos (fuga entre splits, etiquetas en conflicto) se verificaron con inspección visual manual de las imágenes involucradas antes de aplicar cualquier corrección, y las decisiones de diseño con impacto en el criterio de etiquetado (por ejemplo, qué hacer con casos ambiguos) fueron decisiones explícitas del equipo, no delegadas a la IA.

---

## 13. Conclusiones

El aprendizaje más importante de este obligatorio no fue de modelado, sino de **proceso**: los dos problemas más serios que encontramos (la fuga de datos entre splits por frames casi-duplicados, y el riesgo de skew al introducir el recorte por persona) no eran visibles mirando solo las métricas de accuracy — hacía falta un EDA deliberadamente diseñado para buscarlos, y en el caso del skew, hacía falta anticiparlo en el diseño antes de que el sistema estuviera en producción. Priorizamos, como sugiere la consigna, tener un sistema end-to-end funcionando primero (scraping → etiquetado → dataset → entrenamiento → API → UI) y recién después invertir tiempo en estos desafíos y en las optimizaciones electivas, lo cual nos permitió detectar y corregir estos problemas sobre un sistema real en vez de sobre un diseño en el papel.
