# 📈 Git Actions를 활용한 환율 변동 추이 분석 및 예측 시스템 (미래환율 Team Project)

최근 심화되는 경제 불확실성과 확대되는 환율 변동성에 대응하기 위해, 주요 경제 지표를 자동으로 수집하고 Attention 기반 딥러닝 모델을 활용하여 미래 환율을 예측하는 자동화 시스템을 구축했습니다

  - **Project Repository**: `https://github.com/ino456852/MBC-Final-Projects`
  - **Project Period**: 2025.09.01 \~ 2025.10.15 

<br>

## ⚙️ 시스템 아키텍처 및 프로세스

본 프로젝트는 데이터 수집, 저장, 전처리, 모델링, 그리고 시각화에 이르는 전 과정을 자동화 파이프라인으로 구축했습니다.

1.  **데이터 수집 (Data Collection)**

      - `GitHub Actions`를 활용해 매일 지정된 시간에 Python 스크립트(`collector.py`)를 실행하여 최신 경제 지표와 환율 데이터를 자동으로 수집합니다.
      - 초기에는 UI 기반의 RPA 툴을 검토했으나, 개발 프로세스와의 연동성을 고려하여 `GitHub Actions`로 최종 결정했습니다.

2.  **데이터 저장 (Data Storage)**

      - 수집된 원본 데이터는 `MongoDB` 데이터베이스에 안정적으로 저장됩니다.

3.  **전처리 및 모델링 (Preprocessing & Modeling)**

      - 저장된 데이터를 불러와 결측치 처리(`ffill`), 데이터 병합 등 분석에 적합한 형태로 가공합니다.
      - **3-Track 모델 비교**: `XGBoost`, 기본 `LSTM`, 그리고 `Attention-LSTM` 세 가지 모델의 성능을 입체적으로 비교 분석했습니다.
      - 시계열 데이터의 시간적 순서를 유지하며 모델의 안정성을 검증하기 위해 **Rolling Window 교차 검증** 방식을 적용했습니다.

4.  **API 및 대시보드 (API & Dashboard)**

      - 최종 선택된 `Attention-LSTM` 모델의 예측 결과를 `FastAPI` 기반의 API 서버를 통해 제공합니다.
      - 사용자는 `React`로 구현된 인터랙티브 웹 대시보드에서 최신 예측 결과를 시각적으로 확인할 수 있습니다.

-----

## 🛠️ 기술 스택 (Tech Stack)

| Category      | Technology                                                                                                    |
| ------------- | ------------------------------------------------------------------------------------------------------------- |
| **Backend** | Python, FastAPI, Uvicorn                                                                                      |
| **Frontend** | React                                                                                                         |
| **Database** | MongoDB                                                                                                      |
| **ML/DL** | TensorFlow, Scikit-learn, XGBoost, Optuna                                                                       |
| **CI/CD** | GitHub Actions                                                                                                  |
| **Analysis** | Pandas, Numpy, Plotly, Matplotlib                                                                            |

-----

## 🎯 분석 대상 및 범위

  - **예측 대상 통화**: `USD` (달러), `EUR` (유로), `JPY` (엔), `CNY` (위안) 4개 주요 통화.
  - **데이터 기간**: 2015년 9월 1일부터 현재까지 약 10년간의 시계열 데이터를 활용했습니다.
  - **주요 독립 변수 (Features)**: 각 통화별 특성을 고려하여 다음과 같은 경제 지표를 독립 변수로 설정했습니다.
      - **USD**: 미 10년물 국채 수익률(DGS10), 변동성 지수(VIX), 달러 지수(DXY), 한/미 기준금리 및 금리차.
      - **EUR**: 유로존 10년물 국채 수익률(EUR10), 달러 지수(DXY), 미/유로존 국채 금리차, VIX.
      - **JPY**: 일 10년물 국채 수익률(JPY10), 미 10년물 국채 수익률(DGS10), 미/일 국채 금리차, VIX.
      - **CNY**: 중국 외환보유액, 무역수지, 국제 유가(WTI), VIX.

-----

## 📊 모델 성능

3가지 모델을 비교 검증한 결과, **Attention-LSTM**이 모든 평가지표에서 가장 우수한 성능을 기록했습니다. 이는 예측에 결정적인 영향을 미치는 과거 특정 시점의 데이터에 집중하는 Attention 메커니즘이 복잡한 환율 변동 패턴을 효과적으로 학습했기 때문입니다.

| Currency | Model          | **R²** | **MAPE (%)** | **RMSE** |
| :------: | :------------- | :----: | :----------: | :------: |
| **USD** | XGBoost        | 0.9991 |     0.07     |   1.30   |
|          | LSTM           | 0.9559 |     1.28     |   20.31   |
|          | **Attention-LSTM** | **0.9308** |   **1.55** | **26.18** |
| **EUR** | XGBoost        | 0.9556 |     0.06     |  1.16   |
|          | LSTM           | 0.9382 |     0.84     |  15.27   |
|          | **Attention-LSTM** | **0.9143** |   **0.98** | **17.04** |
| **CNY** | XGBoost        | 0.9991 |     0.05     |  0.12   |
|          | LSTM           | 0.9746 |     0.63     |  1.53   |
|          | **Attention-LSTM** | **0.9801** |   **1.23** | **1.23** |
| **JPY** | XGBoost        | 0.9889 |     0.26     |  3.06   |
|          | LSTM           | 0.9763 |     0.97     |  12.51   |
|          | **Attention-LSTM** | **0.9746** |   **0.93** | **12.16** |

*(Source: 자체 성능 평가 결과)*

-----
## 📷 UI 화면 구성

<img width="1188" height="659" alt="image" src="https://github.com/user-attachments/assets/c7437286-0f50-445f-8fff-6b94e27ae1d7" />
<hr>
<img width="1178" height="662" alt="4" src="https://github.com/user-attachments/assets/8def1e44-893a-417a-9b25-340b31e51a12" />
<hr>
<img width="1183" height="659" alt="5" src="https://github.com/user-attachments/assets/195bf3ef-434a-4f2b-8f50-1c1b76dd4a81" />
<hr>


## 👨‍💻 팀원 (Team)

  - **팀장**: **백인호** (리서치, 데이터 수집/전처리, 모델링, 시각화, Full-Stack 개발, PPT)
  - **팀원**:
      - **이병철**: 리서치, WBS 작성, PPT 작성 
      - **이주영**: 리서치, 데이터 수집/전처리, 모델링, 시각화, PPT 작성
      - **이지석**: Full-Stack 개발, 코드 최적화, 모델 자동화, 배포
      - **최홍석**: 리서치, 데이터 수집/전처리, 모델링, 시각화, PPT 작성
