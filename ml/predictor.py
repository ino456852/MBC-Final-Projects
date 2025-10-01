from datetime import datetime
import pandas as pd
import ml.lstm_mha_new.predict as lstm_predict
import ml.attention_lstm_rolling_new.predict as attention_lstm_rolling_predict
import ml.attention_lstm.predict as XGBoost_predict
from .database import MongoDB


def insert_predicted_price():

    # 각 모델별 예측 결과에 index(모델명) 추가
    results = []
    today = datetime.now().strftime("%Y-%m-%d")

    results.append(
        {"date": today, "model": "LSTM-Rolling", **lstm_predict.predict_next_day()}
    )
    results.append(
        {"date": today, "model": "Attention_LSTM-Rolling", **attention_lstm_rolling_predict.predict_next_day()}
    )
    results.append(
        {"date": today, "model": "XRGBBoost-Rolling", **XGBoost_predict.predict_next_day()}
    )

    df = pd.DataFrame(results)
    df.set_index("date", inplace=True)

    print(df)
    
    MongoDB.connect()
    db = MongoDB.get_database()
    db["predicted_price"].insert_many(df.reset_index().to_dict("records"))
    MongoDB.close()


if __name__ == "__main__":
    insert_predicted_price()
