import os
from pathlib import Path
import sys

base_path = Path(__file__).resolve().parent
ml_path = base_path.parent
root_path = base_path.parent.parent
for p in (root_path, ml_path, base_path):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

import numpy as np
import plotly.graph_objects as go
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from plotly.subplots import make_subplots
from typing import Optional
from contextlib import asynccontextmanager
from constant import TARGETS
from preprocess import add_moving_average_features
from predict import load_models_and_scalers, get_predictions


# $env:PYTHONPATH="$(Get-Location)\ml;$env:PYTHONPATH"
# python -u "ml/attention_lstm_rolling/dashboard.py"


app = FastAPI(title="환율 예측 API")
df_master, models, scalers, features_map = None, None, None, None


def generate_metrics_table(metrics_dict: dict) -> str:
    rows = []
    for tgt, m in metrics_dict.items():
        if m is None:
            rows.append(
                f"<tr><td colspan='4' style='border:1px solid black; text-align:center;'>{tgt.upper()}: 데이터 부족</td></tr>"
            )
        else:
            rows.append(
                f"<tr>"
                f"<td style='border:1px solid black; padding:6px; text-align:center;'>{tgt.upper()}</td>"
                f"<td style='border:1px solid black; padding:6px; text-align:center;'>{m['rmse']:.4f}</td>"
                f"<td style='border:1px solid black; padding:6px; text-align:center;'>{m['r2']:.4f}</td>"
                f"<td style='border:1px solid black; padding:6px; text-align:center;'>{m['mape']:.2f}%</td>"
                f"</tr>"
            )
    return (
        "<div style='display:flex; justify-content:flex-end; margin-top:30px;'>"
        "<table style='border-collapse: collapse; font-size: 16px; margin-bottom: 40px;'>"
        "<thead>"
        "<tr><th>통화</th><th>RMSE</th><th>R²</th><th>MAPE</th></tr>"
        "</thead><tbody>"
        f"{''.join(rows)}"
        "</tbody></table></div>"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global df_master, models, scalers, features_map
    try:
        df_master, models, scalers, features_map = load_models_and_scalers()
    except Exception as e:
        print(f"[ERROR] 모델 로딩 실패: {e}")
    yield


app = FastAPI(title="환율 예측 API", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    html = """
    <html><head><title>환율 예측 API</title></head>
    <body>
        <h2>환율 예측 API</h2>
        <p><a href='/train'>/train</a> - 전체 예측</p>
        <p><a href='/train?target=usd&start_date=2024-01-01&end_date=2025-09-01'>/train?target=usd&start_date=2024-01-01&end_date=2025-09-01</a></p>
        <p><a href='/predict'>/predict</a> - 1일 후 예측</p>
        <p><a href='/train_visual'>/train_visual</a> - 성능 시각화</p>
    </body></html>
    """
    return HTMLResponse(html)


@app.get("/train", response_class=HTMLResponse)
async def train(
    target: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> HTMLResponse:
    if df_master is None or not models:
        return HTMLResponse("모델 없음.", status_code=500)

    selected_targets = (
        [target.lower()] if target and target.lower() in TARGETS else TARGETS
    )
    fig = make_subplots(
        rows=len(selected_targets),
        cols=1,
        subplot_titles=[f"{t.upper()} 환율 예측" for t in selected_targets],
    )
    metrics_dict = {}

    for i, tgt in enumerate(selected_targets, 1):
        y_true, y_pred, dates, metrics = get_predictions(
            df_master, models, scalers, features_map, tgt, start_date, end_date
        )
        metrics_dict[tgt] = metrics

        if y_true is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=dates, y=y_true.flatten(), mode="lines", name=f"{tgt.upper()} 실제값"
            ),
            row=i,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dates, y=y_pred.flatten(), mode="lines", name=f"{tgt.upper()} 예측값"
            ),
            row=i,
            col=1,
        )

    fig.update_layout(
        height=600 * len(selected_targets), title="환율 예측 결과", showlegend=True
    )
    return HTMLResponse(
        fig.to_html(full_html=False, include_plotlyjs="cdn")
        + generate_metrics_table(metrics_dict)
    )


@app.get("/predict", response_class=HTMLResponse)
async def predict(future_steps: int = 1) -> HTMLResponse:
    if df_master is None or not models:
        return HTMLResponse("로딩 실패", status_code=500)

    fig = make_subplots(rows=len(TARGETS), cols=1)
    for i, tgt in enumerate(TARGETS, 1):
        _, y_pred, dates, _ = get_predictions(
            df_master, models, scalers, features_map, tgt, future_steps=future_steps
        )
        if y_pred is None:
            continue
        df = add_moving_average_features(df_master, tgt)
        fig.add_trace(
            go.Scatter(
                x=[dates[0]],
                y=[y_pred[0]],
                mode="markers+lines",
                name=f"{tgt.upper()} 예측",
                marker=dict(size=8, color="red"),
            ),
            row=i,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index[-100:],
                y=df[tgt].iloc[-100:],
                mode="lines",
                name=f"{tgt.upper()} 실제값",
            ),
            row=i,
            col=1,
        )

    fig.update_layout(
        height=400 * len(TARGETS),
        title=f"{future_steps}일 후 환율 예측",
        showlegend=True,
    )
    return HTMLResponse(fig.to_html(full_html=True))


@app.get("/train_visual", response_class=HTMLResponse)
async def train_visual() -> HTMLResponse:
    if df_master is None or not models:
        return HTMLResponse("로딩 실패", status_code=500)

    heat_values, residual_traces, scatter_traces = (
        {"target": [], "rmse": [], "mape": [], "r2": []},
        [],
        [],
    )
    y_true_all, y_pred_all = [], []

    for tgt in TARGETS:
        y_true, y_pred, _, metrics = get_predictions(
            df_master, models, scalers, features_map, tgt
        )
        if y_true is None:
            continue
        heat_values["target"].append(tgt.upper())
        heat_values["rmse"].append(metrics["rmse"])
        heat_values["mape"].append(metrics["mape"])
        heat_values["r2"].append(metrics["r2"])

        residual_traces.append(go.Box(y=(y_true - y_pred).flatten(), name=tgt.upper()))
        scatter_traces.append(
            go.Scatter(
                x=y_true.flatten(),
                y=y_pred.flatten(),
                mode="markers",
                name=tgt.upper(),
                opacity=0.5,
            )
        )
        y_true_all.extend(y_true.flatten()), y_pred_all.extend(y_pred.flatten())

    z_values = np.array([heat_values["rmse"], heat_values["mape"], heat_values["r2"]])
    text_values = np.round(z_values, 3).astype(str)
    heatmap_fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=heat_values["target"],
            y=["RMSE", "MAPE", "R²"],
            colorscale="RdYlGn_r",
            text=text_values,
            texttemplate="%{text}",
            showscale=True,
        )
    )
    heatmap_fig.update_layout(title="모델 성능 지표", height=400)

    box_fig = go.Figure(data=residual_traces)
    box_fig.update_layout(title="Residual 분포", height=400)

    scatter_fig = go.Figure(data=scatter_traces)
    min_val, max_val = min(y_true_all + y_pred_all), max(y_true_all + y_pred_all)
    scatter_fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Ideal",
            line=dict(color="gray", dash="dash"),
        )
    )
    scatter_fig.update_layout(
        title="예측 vs 실제", xaxis_title="실제값", yaxis_title="예측값", height=500
    )

    return HTMLResponse(
        "<h2>결과 분석 시각화</h2>"
        + heatmap_fig.to_html(full_html=False, include_plotlyjs="cdn")
        + box_fig.to_html(full_html=False, include_plotlyjs=False)
        + scatter_fig.to_html(full_html=False, include_plotlyjs=False)
    )


if __name__ == "__main__":
    import uvicorn
    import os

    print(os.getcwd())
    uvicorn.run("dashboard:app", host="127.0.0.1", port=8000, reload=True)
