import pandas as pd
import os
import io
import traceback
import json
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .constant import BASE_DIR, PRED_TRUE_DIR

# 경로 상수
RESULTS_JSON_PATH = BASE_DIR / "train_results.json"


# 통화 리스트 (pred_true 폴더의 파일명에서 추출)
def get_targets():
    files = os.listdir(PRED_TRUE_DIR)
    targets = sorted([f.split("_")[0] for f in files if f.endswith(".csv")])
    return list(dict.fromkeys(targets))  # 중복 제거


TARGETS = get_targets()
MA_PERIODS = [5, 20, 60]

app = FastAPI(title="환율 예측 API (파일 기반)")


def load_metrics():
    with open(RESULTS_JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_pred_true(target):
    path = (
        PRED_TRUE_DIR / f"{target}_mha_pred_true.csv"
    )  # mha 모델로 학습된것만 불러오기
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


@app.get("/predict", response_class=HTMLResponse)
def get_prediction_chart():
    try:
        all_metrics = load_metrics()
        subplot_titles = [f"{t.upper()} 환율 예측 비교" for t in TARGETS]
        fig = make_subplots(rows=len(TARGETS), cols=1, subplot_titles=subplot_titles)

        metrics_texts = []

        for i, target in enumerate(TARGETS, 1):
            df = load_pred_true(target)

            # 실제값
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["true"],
                    name="실제",
                    mode="lines",
                    line=dict(width=2.5, color="royalblue"),
                    legendgrouptitle_text=target.upper(),
                ),
                row=i,
                col=1,
            )

            # 예측값
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["pred"],
                    name="예측",
                    mode="lines",
                    line=dict(width=2, color="red", dash="dash"),
                ),
                row=i,
                col=1,
            )

            # MA/EMA
            for period in MA_PERIODS:
                if f"MA{period}" in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[f"MA{period}"],
                            name=f"MA{period}",
                            mode="lines",
                            line=dict(width=1, dash="dot"),
                            visible="legendonly",
                        ),
                        row=i,
                        col=1,
                    )
                if f"EMA{period}" in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[f"EMA{period}"],
                            name=f"EMA{period}",
                            mode="lines",
                            line=dict(width=1, dash="dot"),
                            visible="legendonly",
                        ),
                        row=i,
                        col=1,
                    )

            fig.update_xaxes(title_text="날짜", row=i, col=1)
            fig.update_yaxes(title_text="환율", row=i, col=1)

            # 성능지표
            metrics = all_metrics.get(target, {})
            if metrics:
                metrics_texts.append(
                    f"<b>{target.upper()}</b> | RMSE: {metrics.get('rmse', 0):.2f} | MAPE: {metrics.get('mape', 0):.2f}% | R²: {metrics.get('r2', 0):.4f}"
                )

        annotation_text = "<br>".join(metrics_texts)
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=1.0,
            y=-0.1,
            text=annotation_text,
            showarrow=False,
            align="left",
            xanchor="right",
            yanchor="bottom",
            font=dict(size=14, color="black"),
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="black",
            borderwidth=1,
        )

        fig.update_layout(
            height=500 * len(TARGETS),
            title_text="통화별 환율 예측 결과 비교 (파일 기반)",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                tracegroupgap=25,
            ),
            margin=dict(b=200),
        )
        return HTMLResponse(fig.to_html())

    except Exception:
        buffer = io.StringIO()
        traceback.print_exc(file=buffer)
        return PlainTextResponse(
            f"파일 기반 예측 결과 시각화 중 오류가 발생했습니다:\n\n{buffer.getvalue()}",
            status_code=500,
        )
