import cv2
import json
from pathlib import Path
from PIL import Image

from src.core.classification import ImageClassifier


def detect_falls_from_video(
    video_path: str,
    output_path: str = "data/video/results.json",
    save_frames: bool = True
):
    # =============================
    # CARGAR MODELO ✅
    # =============================
    classifier = ImageClassifier()

    cap = cv2.VideoCapture(video_path)
    results = []

    # =============================
    # FPS DEL VIDEO ✅
    # =============================
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 25  # fallback seguro

    # 👉 cada 5 segundos
    frames_interval = int(fps * 5)

    # =============================
    # CARPETA DE FRAMES ✅
    # =============================
    frames_dir = Path("data/video/frames")
    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0

    # =============================
    # LOOP SOBRE VIDEO ✅
    # =============================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ✅ PROCESAR CADA 5 SEGUNDOS
        if frame_idx % frames_interval == 0:

            # convertir a PIL
            img = Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            # ✅ INFERENCIA (detecta persona, recorta y clasifica)
            prediction = classifier.predict([img])["images"][0]

            label = prediction["label"]
            confidence = prediction["confidence"]
            person_detected = prediction["person_detected"]

            # ✅ TIEMPO EN SEGUNDOS
            time_sec = round(frame_idx / fps, 2)

            # ✅ RESULTADO FINAL
            # is_fall solo si ademas se detecto una persona: un frame sin
            # nadie no cuenta como caida aunque label sea None.
            result = {
                "frame": frame_idx,
                "time_sec": time_sec,
                "label": label,
                "confidence": confidence,
                "person_detected": person_detected,
                "is_fall": person_detected and label == "fall"
            }

            results.append(result)

            # ✅ GUARDAR FRAME SI HAY CAÍDA
            if save_frames and result["is_fall"]:
                out_path = frames_dir / f"fall_{frame_idx}.jpg"
                cv2.imwrite(str(out_path), frame)

            print(f"Frame {frame_idx} (~{time_sec}s): {result}")

        frame_idx += 1

    cap.release()

    # =============================
    # GUARDAR RESULTADOS ✅
    # =============================
    Path(output_path).write_text(json.dumps(results, indent=2))

    print(f"\n✅ Resultados guardados en {output_path}")
