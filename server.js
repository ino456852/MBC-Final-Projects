const express = require("express");
const { MongoClient } = require("mongodb");
const cors = require("cors");

const app = express();
const port = 5001; // API 서버가 사용할 포트 번호

// 1. MongoDB 연결 정보
const uri = "mongodb+srv://mbc_final_user:UqAM7Z2eZ6ZQ11Ra@mbc-final-cluster.g9pivwk.mongodb.net/?retryWrites=true&w=majority&appName=mbc-final-cluster"; // 사용자의 로컬 MongoDB 주소
const dbName = "mbc_final_project_db";   // 데이터베이스 이름
const collectionName = "predicted_price"; // 컬렉션 이름

const client = new MongoClient(uri);

app.use(cors()); // 다른 주소(React 앱)에서의 요청을 허용

// 2. API 경로 설정: '/api/exchange-rate/usd' 와 같은 요청을 처리
app.get("/api/exchange-rate/:currency", async (req, res) => {
    const { currency } = req.params; // URL에서 'usd' 같은 통화 코드를 가져옴
    const today = new Date().toISOString().split("T")[0]; // 오늘 날짜

    try {
        await client.connect();
        const database = client.db(dbName);
        const collection = database.collection(collectionName);

        // 3. DB에서 실제값(actual) 데이터 조회 (과거 5일치)
        const actualDocs = await collection
            .find({ model: "actual", date: { $lt: today } })
            .sort({ date: -1 })
            .limit(5)
            .toArray();

        // 4. DB에서 예측값(models) 데이터 조회 (오늘 이후)
        const modelsDocs = await collection
            .find({ model: { $ne: "actual" }, date: { $gte: today } })
            .toArray();

        // 5. 조회한 데이터를 프론트엔드가 사용하기 좋은 형태로 가공
        const actual = actualDocs.map(doc => ({
            date: doc.date,
            value: doc[currency] // 'usd'가 요청되면 doc.usd 값을 추출
        })).reverse(); // 날짜순 정렬

        const models = modelsDocs.reduce((acc, doc) => {
            if (doc[currency] !== undefined) {
                acc[doc.model] = {
                    date: doc.date,
                    value: doc[currency] // 'mha' 모델의 'usd' 값을 추출
                };
            }
            return acc;
        }, {});

        // 6. 최종 데이터를 JSON 형태로 응답
        res.json({ actual, models });

    } catch (error) {
        console.error("API 서버 오류:", error);
        res.status(500).json({ message: "서버에서 데이터를 가져오지 못했습니다." });
    }
});

app.listen(port, () => {
    console.log(`✅ API 서버가 http://localhost:${port} 에서 실행되었습니다.`);
});