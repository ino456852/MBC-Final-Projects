import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 한글 폰트 설정 (Windows, Linux, macOS 환경에 따라 다르게 설정)
plt.rcParams["font.family"] = "Malgun Gothic"  # Windows
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 부호 깨짐 방지

# 에러나면 RUN
# $env:PYTHONPATH="$(Get-Location)\ml;$env:PYTHONPATH"

from ml.data_merge import create_merged_dataset


def calculate_vif(df):
    """VIF 계산"""
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df.values, i) for i in range(df.shape[1])
    ]
    return vif_data


def main():
    # 1. 데이터 로드
    df = create_merged_dataset()
    if df is None:
        print("데이터 로딩 실패")
        return

    # 2. 파생변수 추가
    df["kr_us_diff"] = df["kr_rate"] - df["us_rate"]

    # 3. 변수 정의
    targets = ["usd", "eur", "jpy", "gbp", "cny"]
    features = ["wti", "dxy", "vix", "kr_rate", "us_rate", "kr_us_diff", "dgs10"]

    df_subset = df[targets + features].dropna()

    # 4. 상관계수 계산
    corr_matrix = df_subset.corr(method="pearson")
    print("피어슨 상관계수:")
    print(corr_matrix)

    # 5. 상관관계 높은 변수만 필터링 (절댓값 0.5 이상)
    # Note: 이 코드를 사용하여 필터링하면 일부 변수가 제외될 수 있습니다.
    # 모든 수치를 히트맵에 표시하려면 필터링하지 않은 전체 행렬을 사용해야 합니다.
    # corr_thresh = 0.5
    # high_corr_vars = corr_matrix.columns[(corr_matrix.abs() >= corr_thresh).any()]
    # filtered_corr = corr_matrix.loc[high_corr_vars, high_corr_vars]

    # 히트맵 시각화
    plt.figure(figsize=(15, 12))

    # 히트맵을 절반만 보이도록 마스크 생성
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        linecolor="black",
        annot_kws={"size": 9},
    )
    plt.title("전체 변수 상관관계 히트맵 (상단 삼각형)", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # 6. VIF 계산 및 출력
    vif_df = calculate_vif(df_subset[features])
    print("\n독립변수 VIF 점수:")
    print(vif_df)

    # 7. 타겟별 산점도 + 회귀선 그리기 (타겟별 분할하여 출력)
    important_features = ["dxy", "vix", "dgs10", "wti", "kr_us_diff"]

    for target in targets:
        fig, axes = plt.subplots(1, len(important_features), figsize=(18, 4))
        fig.suptitle(f"{target.upper()}와 주요 독립 변수 간 산점도", fontsize=16)

        for j, feat in enumerate(important_features):
            sns.scatterplot(
                x=df_subset[feat], y=df_subset[target], alpha=0.6, ax=axes[j]
            )
            sns.regplot(
                x=df_subset[feat],
                y=df_subset[target],
                scatter=False,
                color="red",
                ax=axes[j],
            )
            axes[j].set_title(f"{target.upper()} vs {feat.upper()}")
            axes[j].set_xlabel(feat)
            axes[j].set_ylabel(target)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    # 8. USD 환율과 DXY 동조화 시각화
    df_sync = df[["usd", "dxy"]].dropna()

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # 왼쪽 y축: USD 환율
    color_usd = "tab:blue"
    ax1.set_xlabel("날짜")
    ax1.set_ylabel("USD 환율", color=color_usd)
    ax1.plot(df_sync.index, df_sync["usd"], color=color_usd, label="USD 환율")
    ax1.tick_params(axis="y", labelcolor=color_usd)

    # 오른쪽 y축: DXY
    ax2 = ax1.twinx()
    color_dxy = "tab:red"
    ax2.set_ylabel("DXY 지수", color=color_dxy)
    ax2.plot(df_sync.index, df_sync["dxy"], color=color_dxy, label="DXY 지수")
    ax2.tick_params(axis="y", labelcolor=color_dxy)

    # 제목 및 범례
    plt.title("USD 환율과 DXY 지수 동조화 시각화", fontsize=16)
    fig.tight_layout()
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
