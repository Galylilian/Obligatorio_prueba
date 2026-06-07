"""Detecta caidas frame a frame en un video (primeros N segundos)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from PIL import Image

from src.analytics.repository import record_prediction
from src.core.inference import (
    get_eval_transform,
    get_inference_model,
    get_label_encoder,
    predict_with_confidence,
)
from src.settings.config import DEVICE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deteccion de caidas en video")
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("data/demo/fall_video.mp4"),
        help="Ruta al video MP4",
    )
    parser.add_argument("--seconds", type=float, default=10.0, help="Segundos a analizar")
    parser.add_argument("--fps-sample", type=float, default=2.0, help="Frames por segundo a evaluar")
    parser.add_argument(
        "--person-id",
        type=str,
        default="VIDEO_DEMO",
        help="person_id para analytics/Grafana",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/demo/fall_video_results.json"),
        help="JSON con resultados",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Guardar frames donde se detecta fall",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"No existe el video: {args.video}")

    model = get_inference_model()
    label_encoder = get_label_encoder()
    transform = get_eval_transform()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir: {args.video}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    max_frame = int(native_fps * args.seconds)
    step = max(int(native_fps / args.fps_sample), 1)

    frames_dir = args.video.parent / "frames"
    if args.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    frame_idx = 0

    while frame_idx < max_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = transform(pil).unsqueeze(0).to(DEVICE)

        _, label, confidence, probabilities = predict_with_confidence(
            model, tensor, label_encoder
        )
        t_sec = round(frame_idx / native_fps, 2)

        entry = {
            "time_sec": t_sec,
            "frame": frame_idx,
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
        }
        results.append(entry)

        try:
            record_prediction(
                label=label,
                confidence=confidence,
                person_id=args.person_id,
                source=f"video:{args.video.name}@{t_sec}s",
            )
        except Exception:
            pass

        if args.save_frames and label == "fall":
            out = frames_dir / f"fall_{t_sec:.2f}s.jpg"
            cv2.imwrite(str(out), bgr)

        print(f"[{t_sec:5.2f}s] {label:8} conf={confidence*100:.1f}%")

        frame_idx += step

    cap.release()

    summary = {
        "video": str(args.video),
        "seconds_analyzed": args.seconds,
        "frames_evaluated": len(results),
        "falls_detected": sum(1 for r in results if r["label"] == "fall"),
        "person_id": args.person_id,
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nResumen: {summary['falls_detected']}/{len(results)} frames con caida")
    print(f"Resultados: {args.output}")


if __name__ == "__main__":
    main()
