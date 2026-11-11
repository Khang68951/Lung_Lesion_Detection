from __future__ import annotations

import base64
import io
import threading

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from unsupervised_lesion.production_v75 import VERSION, predict_rgb


app = FastAPI(title="Lung lesion local review")
lock = threading.Lock()

HTML = """<!doctype html><html lang='vi'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>Lung lesion V75</title><style>body{font:16px system-ui;margin:2rem;max-width:1100px;background:#10141b;color:#eef}
button,input{font:inherit;padding:.7rem}button{background:#52b788;border:0;border-radius:.4rem}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-top:1rem}
img{width:100%;background:#000;border-radius:.5rem}.card{background:#19212d;padding:1rem;border-radius:.6rem}small{color:#aab}</style></head>
<body><h1>Phân vùng phổi và tổn thương</h1><p><small>V75 chạy local. Kết quả chỉ hỗ trợ nghiên cứu, không phải chẩn đoán lâm sàng.</small></p>
<input id='f' type='file' accept='image/*'><button onclick='run()'>Phân tích</button><span id='s'></span><div id='r'></div>
<script>async function run(){let f=document.getElementById('f').files[0];if(!f)return;let d=new FormData();d.append('file',f);s.textContent=' Đang chạy GPU…';r.innerHTML='';
let x=await fetch('/predict',{method:'POST',body:d});let j=await x.json();if(!x.ok){s.textContent=' Lỗi: '+(j.detail||'unknown');return}s.textContent=' Hoàn tất';r.innerHTML=`<div class=card><b>Tỷ lệ tổn thương / phổi: ${j.ratio_percent.toFixed(2)}%</b><br><small>${j.version}</small></div><div class=grid><div>Ảnh gốc<img src='${j.original}'></div><div>Phổi (xanh)<img src='${j.lung_overlay}'></div><div>Tổn thương (đỏ)<img src='${j.lesion_overlay}'></div></div>`}</script></body></html>"""


def encoded(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.post("/predict")
def predict(file: UploadFile = File(...)) -> dict:
    try:
        raw = file.file.read()
        rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception as exc:
        raise HTTPException(400, f"Không đọc được ảnh: {exc}") from exc
    if rgb.shape[0] < 64 or rgb.shape[1] < 64:
        raise HTTPException(400, "Ảnh quá nhỏ")
    with lock:
        result = predict_rgb(rgb)
    lung = result["lung"].astype(bool)
    lesion = result["lesion"].astype(bool)
    lung_overlay = rgb.copy()
    lung_overlay[lung] = (.45 * lung_overlay[lung] + .55 * np.array([30, 210, 160])).astype(np.uint8)
    lesion_overlay = rgb.copy()
    lesion_overlay[lesion] = (.35 * lesion_overlay[lesion] + .65 * np.array([245, 55, 70])).astype(np.uint8)
    return {"version": VERSION, "ratio_percent": 100 * float(result["ratio"]),
            "original": encoded(rgb), "lung_overlay": encoded(lung_overlay),
            "lesion_overlay": encoded(lesion_overlay)}
