import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from typing import Optional
from contextlib import asynccontextmanager

# 경로 설정 (attention_lstm.py 방식 적용)
base_path = os.path.abspath(os.path.dirname(__file__))
ml_path = os.path.join(base_path, 'ml')
for p in [base_path, ml_path]:
    if p not in os.sys.path:
        os.sys.path.insert(0, p)

from data_merge import create_merged_dataset

# 상수 정의 (원본 유지)
TARGETS = ['usd', 'cny', 'jpy', 'eur', 'gbp']

# ------------------------------------------------
# 전역 변수 (predict.py 방식 적용)
# ------------------------------------------------
app = FastAPI(title="환율 예측 시각화 API")

# ------------------------------------------------
# 원본 시각화 함수들 (기능 유지)
# ------------------------------------------------
def load_2024_predictions(csv_file='2024_predictions.csv'):
    """2024년 예측 결과 로드"""
    try:
        return pd.read_csv(csv_file, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"'{csv_file}' 파일을 찾을 수 없습니다.")
        return None

def create_2024_prediction_chart(csv_file='2024_predictions.csv'):
    """2024년 실제값 vs 예측값 비교 차트"""
    predictions_df = load_2024_predictions(csv_file)
    
    if predictions_df is None:
        return None
    
    fig = make_subplots(
        rows=len(TARGETS), cols=1, 
        subplot_titles=[f"{t.upper()} 환율 실제값 vs 예측값 (2024년~)" for t in TARGETS]
    )

    for i, target in enumerate(TARGETS, 1):
        actual_col = f'Actual_{target.upper()}'
        predicted_col = f'Predicted_{target.upper()}'
        
        # 실제값
        fig.add_trace(
            go.Scatter(
                x=predictions_df.index, 
                y=predictions_df[actual_col], 
                name=f'실제 {target.upper()}', 
                line=dict(width=2, color='blue'), 
                showlegend=(i==1)
            ), row=i, col=1
        )
        
        # 예측값
        fig.add_trace(
            go.Scatter(
                x=predictions_df.index, 
                y=predictions_df[predicted_col], 
                name=f'예측 {target.upper()}', 
                line=dict(width=2, color='red', dash='dash'), 
                showlegend=(i==1)
            ), row=i, col=1
        )

    fig.update_layout(
        title='2024년 환율 실제값 vs 예측값 비교', 
        height=1500,
        hovermode='x unified'
    )
    return fig

def create_performance_metrics_table(csv_file='2024_predictions.csv'):
    """성능 지표 테이블"""
    predictions_df = load_2024_predictions(csv_file)
    
    if predictions_df is None:
        return None
    
    performance_data = []
    for target in TARGETS:
        actual_col = f'Actual_{target.upper()}'
        predicted_col = f'Predicted_{target.upper()}'
        
        actual = predictions_df[actual_col].values
        predicted = predictions_df[predicted_col].values
        
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        performance_data.append({
            '통화': target.upper(),
            'MAE': f"{mae:.4f}",
            'RMSE': f"{rmse:.4f}",
            'R²': f"{r2:.4f}",
            'MAPE(%)': f"{mape:.2f}%",
            '예측 정확도': f"{max(0, (1 - mape/100) * 100):.1f}%"
        })
    
    performance_df = pd.DataFrame(performance_data)
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(performance_df.columns),
            fill_color='lightblue',
            align='center',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=[performance_df[col] for col in performance_df.columns],
            fill_color='white',
            align='center',
            font=dict(size=11)
        )
    )])
    
    fig.update_layout(
        title="2024년 예측 성능 지표",
        height=300
    )
    
    return fig

def create_correlation_scatter_plots(csv_file='2024_predictions.csv'):
    """실제값 vs 예측값 산점도"""
    predictions_df = load_2024_predictions(csv_file)
    
    if predictions_df is None:
        return None
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"{t.upper()}" for t in TARGETS],
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]]
    )
    
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
    
    for idx, target in enumerate(TARGETS):
        row, col = positions[idx]
        
        actual_col = f'Actual_{target.upper()}'
        predicted_col = f'Predicted_{target.upper()}'
        
        actual = predictions_df[actual_col]
        predicted = predictions_df[predicted_col]
        
        # 산점도
        fig.add_trace(
            go.Scatter(
                x=actual,
                y=predicted,
                mode='markers',
                name=f'{target.upper()}',
                marker=dict(size=4, opacity=0.6),
                showlegend=False
            ), row=row, col=col
        )
        
        # 완벽한 예측선 (y=x)
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='완벽한 예측',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ), row=row, col=col
        )
        
        # R² 점수 추가 (subplot에서 annotation 올바른 방식)
        r2 = r2_score(actual, predicted)
        # subplot의 경우 xref를 직접 지정해야 함
        if idx == 0:  # 첫 번째 subplot
            xref, yref = "x domain", "y domain"
        elif idx == 1:  # 두 번째 subplot
            xref, yref = "x2 domain", "y2 domain"
        elif idx == 2:  # 세 번째 subplot
            xref, yref = "x3 domain", "y3 domain"
        elif idx == 3:  # 네 번째 subplot
            xref, yref = "x4 domain", "y4 domain"
        else:  # 다섯 번째 subplot
            xref, yref = "x5 domain", "y5 domain"
            
        fig.add_annotation(
            x=0.05, y=0.95,
            text=f'R² = {r2:.3f}',
            xref=xref, yref=yref,
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)"
        )
    
    fig.update_layout(
        title="실제값 vs 예측값 산점도 (2024년~)",
        height=800
    )
    
    fig.update_xaxes(title_text="실제값")
    fig.update_yaxes(title_text="예측값")
    
    return fig

def print_2024_analysis(csv_file='2024_predictions.csv'):
    """2024년 예측 분석 결과 출력"""
    predictions_df = load_2024_predictions(csv_file)
    
    if predictions_df is None:
        return
    
    print("\n" + "="*60)
    print(" 2024년 환율 예측 성능 분석")
    print("="*60)
    
    start_date = predictions_df.index[0].strftime('%Y-%m-%d')
    end_date = predictions_df.index[-1].strftime('%Y-%m-%d')
    print(f"\n 분석 기간: {start_date} ~ {end_date}")
    print(f" 총 예측 일수: {len(predictions_df)}일")
    
    print(f"\n{'='*60}")
    print("통화별 예측 성능")
    print(f"{'='*60}")
    
    for target in TARGETS:
        actual_col = f'Actual_{target.upper()}'
        predicted_col = f'Predicted_{target.upper()}'
        
        actual = predictions_df[actual_col].values
        predicted = predictions_df[predicted_col].values
        
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        accuracy = max(0, (1 - mape/100) * 100)
        
        print(f"\n {target.upper()}:")
        print(f"   MAE: {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R²: {r2:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   예측 정확도: {accuracy:.1f}%")

# ------------------------------------------------
# FastAPI 엔드포인트 (predict.py 방식 적용)
# ------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("환율 예측 시각화 API 시작")
    except Exception as e:
        print(f"[ERROR] 초기화 실패: {e}")
    yield

app = FastAPI(title="환율 예측 시각화 API", lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    html = """
    <html><head><title>환율 예측 시각화 API</title></head>
    <body>
        <h2>환율 예측 시각화 API</h2>
        <p><a href='/chart'>/chart</a> - 2024년 예측 차트</p>
        <p><a href='/table'>/table</a> - 성능 지표 테이블</p>
        <p><a href='/scatter'>/scatter</a> - 산점도 분석</p>
        <p><a href='/analysis'>/analysis</a> - 전체 분석</p>
    </body></html>
    """
    return HTMLResponse(html)

@app.get("/chart", response_class=HTMLResponse)
async def chart(csv_file: str = '2024_predictions.csv') -> HTMLResponse:
    """2024년 예측 차트 표시"""
    try:
        fig = create_2024_prediction_chart(csv_file)
        if fig:
            return HTMLResponse(fig.to_html(full_html=True, include_plotlyjs="cdn"))
        else:
            return HTMLResponse("<h1>데이터를 찾을 수 없습니다</h1><p>2024_predictions.csv 파일이 필요합니다.</p>", status_code=404)
    except Exception as e:
        return HTMLResponse(f"<h1>오류 발생</h1><p>{str(e)}</p>", status_code=500)

@app.get("/table", response_class=HTMLResponse)
async def table(csv_file: str = '2024_predictions.csv') -> HTMLResponse:
    """성능 테이블 표시"""
    try:
        fig = create_performance_metrics_table(csv_file)
        if fig:
            return HTMLResponse(fig.to_html(full_html=True, include_plotlyjs="cdn"))
        else:
            return HTMLResponse("<h1>데이터를 찾을 수 없습니다</h1><p>2024_predictions.csv 파일이 필요합니다.</p>", status_code=404)
    except Exception as e:
        return HTMLResponse(f"<h1>오류 발생</h1><p>{str(e)}</p>", status_code=500)

@app.get("/scatter", response_class=HTMLResponse)
async def scatter(csv_file: str = '2024_predictions.csv') -> HTMLResponse:
    """산점도 표시"""
    try:
        fig = create_correlation_scatter_plots(csv_file)
        if fig:
            return HTMLResponse(fig.to_html(full_html=True, include_plotlyjs="cdn"))
        else:
            return HTMLResponse("<h1>데이터를 찾을 수 없습니다</h1><p>2024_predictions.csv 파일이 필요합니다.</p>", status_code=404)
    except Exception as e:
        return HTMLResponse(f"<h1>오류 발생</h1><p>{str(e)}</p>", status_code=500)

@app.get("/analysis", response_class=HTMLResponse)
async def analysis(csv_file: str = '2024_predictions.csv') -> HTMLResponse:
    """전체 분석 결과"""
    predictions_df = load_2024_predictions(csv_file)
    if predictions_df is None:
        return HTMLResponse("데이터를 찾을 수 없습니다.", status_code=404)
    
    # 차트들 생성
    chart_fig = create_2024_prediction_chart(csv_file)
    table_fig = create_performance_metrics_table(csv_file)
    scatter_fig = create_correlation_scatter_plots(csv_file)
    
    html_content = "<h1>2024년 환율 예측 분석 결과</h1>"
    
    if chart_fig:
        html_content += "<h2>예측 차트</h2>" + chart_fig.to_html(full_html=False, include_plotlyjs="cdn")
    
    if table_fig:
        html_content += "<h2>성능 지표</h2>" + table_fig.to_html(full_html=False, include_plotlyjs=False)
    
    if scatter_fig:
        html_content += "<h2>산점도 분석</h2>" + scatter_fig.to_html(full_html=False, include_plotlyjs=False)
    
    return HTMLResponse(html_content)

# 간단한 표시 함수들 (원본 기능 유지)
def show_2024_prediction_chart(csv_file='2024_predictions.csv'):
    """2024년 예측 차트 표시"""
    fig = create_2024_prediction_chart(csv_file)
    if fig:
        fig.show()

def show_performance_table(csv_file='2024_predictions.csv'):
    """성능 테이블 표시"""
    fig = create_performance_metrics_table(csv_file)
    if fig:
        fig.show()

def show_scatter_plots(csv_file='2024_predictions.csv'):
    """산점도 표시"""
    fig = create_correlation_scatter_plots(csv_file)
    if fig:
        fig.show()

def show_all_2024_visualizations(csv_file='2024_predictions.csv'):
    """모든 2024년 예측 시각화 표시"""
    predictions_df = load_2024_predictions(csv_file)
    if predictions_df is None:
        return
    
    print("📊 2024년 환율 예측 시각화 생성 중...")
    
    print("1. 2024년 예측 분석 출력...")
    print_2024_analysis(csv_file)
    
    print("\n2. 실제값 vs 예측값 비교 차트...")
    show_2024_prediction_chart(csv_file)
    
    print("3. 성능 지표 테이블...")
    show_performance_table(csv_file)
    
    print("4. 실제값 vs 예측값 산점도...")
    show_scatter_plots(csv_file)
    
    print("✅ 2024년 예측 시각화 완료!")

if __name__ == "__main__":
    import sys
    
    try:
        import uvicorn
    except ImportError:
        print("❌ uvicorn이 설치되지 않았습니다.")
        print("다음 명령어로 설치해주세요: pip install uvicorn")
        sys.exit(1)
    
    # 명령행 인자 확인
    if len(sys.argv) > 1 and sys.argv[1] == "--api-only":
        # FastAPI 서버만 실행
        print("🚀 FastAPI 서버만 시작합니다...")
        print("📍 서버 주소: http://localhost:8002")
        print("📖 API 문서: http://localhost:8002/docs")
        try:
            uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")
        except Exception as e:
            print(f"❌ 서버 시작 실패: {e}")
    else:
        # 기본: 시각화 출력 후 FastAPI 서버 실행
        print("=== 2024년 환율 예측 결과 시각화 ===")
        
        try:
            # 분석 결과 출력
            print_2024_analysis()
            
            # 모든 시각화 표시
            show_all_2024_visualizations()
            
        except Exception as e:
            print(f"⚠️ 시각화 생성 중 오류 발생: {e}")
            print("CSV 파일이 없어도 FastAPI 서버는 시작됩니다.")
        
        # FastAPI 서버 실행
        print("\n🚀 FastAPI 서버를 시작합니다...")
        print("📍 서버 주소: http://localhost:8002")
        print("📖 API 문서: http://localhost:8002/docs")
        print("⏹️ 서버 종료: Ctrl+C")
        
        try:
            uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")
        except Exception as e:
            print(f"❌ 서버 시작 실패: {e}")
            print("💡 다른 포트로 시도해보세요: uvicorn predict_visualization2:app --port 8003")