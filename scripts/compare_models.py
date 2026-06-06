import requests
import os

API_URL = "http://localhost:8080"  #  8080 si usas Docker

test_folder = "data/processed/test/acostado"

results = []

for file in os.listdir(test_folder)[:10]:
    filepath = os.path.join(test_folder, file)

    files = {"file": open(filepath, "rb")}

    # CNN
    cnn_resp = requests.post(f"{API_URL}/predict", files=files)
    cnn_pred = cnn_resp.json()

    # YOLO
    files = {"file": open(filepath, "rb")}
    yolo_resp = requests.post(f"{API_URL}/predict_yolo", files=files)
    yolo_pred = yolo_resp.json()

    results.append({
        "image": file,
        "cnn": cnn_pred,
        "yolo": yolo_pred
    })

    print(f"{file} -> CNN: {cnn_pred} | YOLO: {yolo_pred}")

print("\n✅ Comparación final:")
for r in results:
    print(r)