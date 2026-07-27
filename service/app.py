"""FastAPI service for demand forecasting."""
from __future__ import annotations

import io
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import predict as eng

BASE = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, eng.warmup)
    yield


app = FastAPI(title="Прогноз спроса", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE / "templates"))

LABELS = {"week": "1 неделя", "month": "1 месяц", "3month": "3 месяца"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    catalog = eng.get_catalog()
    papa1   = sorted(catalog["Папка1"].dropna().unique().tolist())
    p2map   = (
        catalog.groupby("Папка1")["Папка2"]
        .apply(lambda s: sorted(s.dropna().unique().tolist()))
        .to_dict()
    )
    return templates.TemplateResponse(request, "index.html", {
        "papa1_opts": papa1,
        "papa2_json": json.dumps(p2map, ensure_ascii=False),
    })


@app.post("/predict", response_class=HTMLResponse)
def predict_route(
    request: Request,
    horizon:     str        = Form(...),
    papa1:       list[str]  = Form(default=[]),
    papa2:       list[str]  = Form(default=[]),
    name_search: str        = Form(default=""),
    ids:         str        = Form(default=""),
):
    import time
    print(f"[route] start horizon={horizon} ids={ids!r} name={name_search!r}", flush=True)
    t0 = time.time()
    filters = {
        "papa1":       papa1 or None,
        "papa2":       papa2 or None,
        "name_search": name_search.strip() or None,
        "ids":         [x.strip() for x in ids.split(",") if x.strip()] or None,
    }
    results, period = eng.predict(horizon, filters)
    print(f"[route] predict done in {time.time()-t0:.2f}s", flush=True)

    if results.empty:
        catalog = eng.get_catalog()
        return templates.TemplateResponse(request, "index.html", {
            "error":      "Нет товаров по выбранным фильтрам.",
            "papa1_opts": sorted(catalog["Папка1"].dropna().unique().tolist()),
            "papa2_json": "{}",
        })

    rows = results.sort_values("Прогноз", ascending=False).to_dict(orient="records")
    return templates.TemplateResponse(request, "results.html", {
        "rows":          rows,
        "horizon":       horizon,
        "horizon_label": LABELS.get(horizon, horizon),
        "period":        period,
        "total":         int(results["Прогноз"].sum()),
        "count":         len(rows),
    })


@app.post("/export")
async def export(request: Request):
    body   = await request.json()
    rows   = body.get("rows", [])
    period = body.get("period", "forecast")

    if not rows:
        return {"error": "no data"}

    df = pd.DataFrame(rows)[["КодТовара", "Номенклатура", "Папка1", "Папка2", "Прогноз", "к_заказу"]]
    df.columns = ["Код товара", "Название", "Категория", "Подкатегория", "Прогноз (шт)", "К заказу (шт)"]

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Заявка на закупку")
        ws = writer.sheets["Заявка на закупку"]
        for col in ws.columns:
            max_w = max(len(str(c.value or "")) for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_w + 4, 55)

    buf.seek(0)
    fname = f"zakupka_{period.replace(' ', '_')}.xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )
