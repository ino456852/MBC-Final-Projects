import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import io
import traceback
import json
from types import SimpleNamespace
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from training.data import get_master_dataframe, prepare_data_for_target, TARGETS, LOOK_BACK, MA_WINDOWS
from training.models import CustomAttention, SumOverTime, MODELS_TO_TRAIN

app = FastAPI(title="환율 예측 API")
CUSTOM_OBJECTS = {'CustomAttention': CustomAttention, 'SumOverTime': SumOverTime}

def run_prediction_and_analysis():
    master_data = get_master_dataframe()
    all_metrics = {}
    
    params_path = 'best_params.json'
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"'{params_path}' 파일을 찾을 수 없습니다.")
    with open(params_path, 'r') as f:
        all_best_params = json.load(f)

    final_df = None

    for target_name in TARGETS:
        print(f"\n===== '{target_name}' 예측 및 분석 시작 =====")
        
        _, _, X_test, test_dates, _, target_scaler = prepare_data_for_target(master_data, target_name)

        if final_df is None:
            final_df = pd.DataFrame(index=test_dates)

        y_true_values = master_data.loc[test_dates, target_name].values
        final_df[f'Actual_{target_name}'] = y_true_values

        df_with_ma = master_data.copy()
        for window in MA_WINDOWS:
            df_with_ma[f'{target_name}_MA{window}'] = df_with_ma[target_name].rolling(window).mean()
            df_with_ma[f'{target_name}_EMA{window}'] = df_with_ma[target_name].ewm(span=window, adjust=False).mean()
        
        for col in df_with_ma.columns:
            if '_MA' in col or '_EMA' in col:
                if target_name in col:
                    final_df[col] = df_with_ma.loc[test_dates, col]

        target_metrics = {}
        for model_key in MODELS_TO_TRAIN.keys():
            model_path = os.path.join("models", f"{target_name}_{model_key}_attention.keras")
            if not os.path.exists(model_path):
                final_df[f'Predicted_{model_key.title()}_{target_name}'] = np.nan
                continue

            model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
            pred_scaled = model.predict(X_test)
            pred_inversed = target_scaler.inverse_transform(pred_scaled).flatten()
            final_df[f'Predicted_{model_key.title()}_{target_name}'] = pred_inversed

            mse = mean_squared_error(y_true_values, pred_inversed)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_values, pred_inversed)
            r2 = r2_score(y_true_values, pred_inversed)
            mape = mean_absolute_percentage_error(y_true_values, pred_inversed) * 100
            target_metrics[model_key] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}
        
        all_metrics[target_name] = target_metrics

    return final_df, all_metrics


@app.get("/predict", response_class=HTMLResponse)
def get_prediction_chart():
    try:
        final_df, all_metrics = run_prediction_and_analysis()
        
        subplot_titles = [f"{t.upper()} 환율 예측 비교" for t in TARGETS]
        fig = make_subplots(rows=len(TARGETS), cols=1, subplot_titles=subplot_titles)

        model_styles = [
            {'key': 'Custom', 'name': 'Custom', 'color': 'red', 'dash': 'dash'},
            {'key': 'Mha', 'name': 'Multi-Head', 'color': 'green', 'dash': 'dot'},
            {'key': 'Dot', 'name': 'Dot-Product', 'color': 'orange', 'dash': 'dashdot'},
            {'key': 'Cross', 'name': 'Cross-Attention', 'color': 'purple', 'dash': 'longdash'},
            {'key': 'Causal', 'name': 'Causal Self', 'color': 'cyan', 'dash': 'longdashdot'},
            {'key': 'Self', 'name': 'Self-Attention', 'color': 'brown', 'dash': 'solid'}
        ]
        
        metrics_texts = []
        for i, target in enumerate(TARGETS, 1):
            legend_group_name = f'group{i}'
            
            fig.add_trace(go.Scatter(x=final_df.index, y=final_df[f'Actual_{target}'], name='실제', mode='lines', line=dict(width=2.5, color='royalblue'), legendgroup=legend_group_name, legendgrouptitle_text=target.upper()), row=i, col=1)

            for style in model_styles:
                col_name = f'Predicted_{style["key"]}_{target}'
                if col_name in final_df and not final_df[col_name].isnull().all():
                    fig.add_trace(go.Scatter(x=final_df.index, y=final_df[col_name], name=style['name'], mode='lines', line=dict(dash=style['dash'], color=style['color']), legendgroup=legend_group_name), row=i, col=1)
            
            for window in MA_WINDOWS:
                ma_col = f'{target}_MA{window}'
                ema_col = f'{target}_EMA{window}'
                fig.add_trace(go.Scatter(x=final_df.index, y=final_df[ma_col], name=f'MA{window}', mode='lines', line=dict(width=1, dash='dot'), legendgroup=legend_group_name, visible='legendonly'), row=i, col=1)
                fig.add_trace(go.Scatter(x=final_df.index, y=final_df[ema_col], name=f'EMA{window}', mode='lines', line=dict(width=1, dash='dot'), legendgroup=legend_group_name, visible='legendonly'), row=i, col=1)

            fig.update_xaxes(title_text="날짜", row=i, col=1)
            fig.update_yaxes(title_text="환율", row=i, col=1)

            metrics = all_metrics.get(target, {}).get('mha', {})
            if metrics:
                metrics_texts.append(f"<b>{target.upper()} (MHA)</b> | MSE: {metrics['MSE']:.2f} | RMSE: {metrics['RMSE']:.2f} | MAE: {metrics['MAE']:.2f} | MAPE: {metrics['MAPE']:.2f}% | R²: {metrics['R2']:.4f}")

        annotation_text = "<br>".join(metrics_texts)
        fig.add_annotation(xref="paper", yref="paper", x=1.0, y=-0.1, text=annotation_text, showarrow=False, align="left", xanchor="right", yanchor="bottom", font=dict(size=14, color="black"), bgcolor="rgba(255,255,255,0.7)", bordercolor="black", borderwidth=1)

        fig.update_layout(height=2200, title_text='통화별 환율 예측 결과 비교 (Optuna 최적화)', legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, tracegroupgap=25), margin=dict(b=200))
        return HTMLResponse(fig.to_html())
        
    except Exception:
        buffer = io.StringIO()
        traceback.print_exc(file=buffer)
        return PlainTextResponse(f"모델 예측 또는 시각화 중 오류가 발생했습니다:\n\n{buffer.getvalue()}", status_code=500)