import { useChartOptionsStore } from "@/store/chartOptionsStore"
import { Card, CardContent } from "@/components/ui/card"
import { Line } from "react-chartjs-2"
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Legend,
  Tooltip,
  ChartData,
  ChartOptions,
} from "chart.js"
import { useMemo, type JSX } from "react"

ChartJS.register(LineElement, PointElement, CategoryScale, LinearScale, Legend, Tooltip)


interface ActualData {
  date: string
  value: number
}

interface ModelData {
  [key: string]: {
    date: string
    value: number
  }
}

interface DummyData {
  [key: string]: {
    actual: ActualData[]
    models: ModelData
  }
}

const dummyData: DummyData = {
  usd: {
    actual: [
      { date: "2025-09-22", value: 1340 },
      { date: "2025-09-23", value: 1342 },
      { date: "2025-09-24", value: 1345 },
      { date: "2025-09-25", value: 1347 },
      { date: "2025-09-26", value: 1349 },
    ],
    models: {
      "lstm_attention": { date: "2025-09-27", value: 1352 },
      "lstm_attention_rolling": { date: "2025-09-27", value: 1350 },
      "lstm_mha": { date: "2025-09-27", value: 1348 },
    },
  },
  eur: {
    actual: [
      { date: "2025-09-22", value: 1430 },
      { date: "2025-09-23", value: 1432 },
      { date: "2025-09-24", value: 1435 },
      { date: "2025-09-25", value: 1437 },
      { date: "2025-09-26", value: 1439 },
    ],
    models: {
      "lstm_attention": { date: "2025-09-27", value: 1442 },
      "lstm_attention_rolling": { date: "2025-09-27", value: 1440 },
      "lstm_mha": { date: "2025-09-27", value: 1438 },
    },
  },
  cny: {
    actual: [
      { date: "2025-09-22", value: 190 },
      { date: "2025-09-23", value: 191 },
      { date: "2025-09-24", value: 192 },
      { date: "2025-09-25", value: 193 },
      { date: "2025-09-26", value: 194 },
    ],
    models: {
      "lstm_attention": { date: "2025-09-27", value: 195 },
      "lstm_attention_rolling": { date: "2025-09-27", value: 194 },
      "lstm_mha": { date: "2025-09-27", value: 193 },
    },
  },
  jpy: {
    actual: [
      { date: "2025-09-22", value: 9.1 },
      { date: "2025-09-23", value: 9.2 },
      { date: "2025-09-24", value: 9.3 },
      { date: "2025-09-25", value: 9.4 },
      { date: "2025-09-26", value: 9.5 },
    ],
    models: {
      "lstm_attention": { date: "2025-09-27", value: 9.6 },
      "lstm_attention_rolling": { date: "2025-09-27", value: 9.5 },
      "lstm_mha": { date: "2025-09-27", value: 9.4 },
    },
  },
  gbp: {
    actual: [
      { date: "2025-09-22", value: 1700 },
      { date: "2025-09-23", value: 1702 },
      { date: "2025-09-24", value: 1705 },
      { date: "2025-09-25", value: 1707 },
      { date: "2025-09-26", value: 1709 },
    ],
    models: {
      "lstm_attention": { date: "2025-09-27", value: 1712 },
      "lstm_attention_rolling": { date: "2025-09-27", value: 1710 },
      "lstm_mha": { date: "2025-09-27", value: 1708 },
    },
  },
}

// modelLabelMap 제거

export function ChartArea(): JSX.Element {
  const { model, currency } = useChartOptionsStore()

  const chartData: ChartData<"line"> | null = useMemo(() => {
    if (!currency || !dummyData[currency]) return null
    const actual = dummyData[currency].actual
    const modelPred = model && dummyData[currency].models[model]
    const labels = actual.map(d => d.date)
    if (modelPred) labels.push(modelPred.date)

    return {
      labels,
      datasets: [
        {
          label: "실제값",
          data: [
            ...actual.map(d => d.value),
            null,
          ],
          borderColor: "#6366f1",
          backgroundColor: "#6366f1",
          tension: 0.3,
          spanGaps: true,
        },
        modelPred && {
          label: `${model} 예측값`,
          data: [
            ...Array(actual.length).fill(null),
            modelPred.value,
          ],
          borderColor: "#f59e42",
          backgroundColor: "#f59e42",
          borderDash: [6, 6],
          pointStyle: "rectRot",
          tension: 0,
        },
      ].filter(Boolean) as any,
    }
  }, [model, currency])

  const options: ChartOptions<"line"> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: true },
      tooltip: { mode: "index", intersect: false },
    },
    scales: {
      x: { title: { display: true, text: "날짜" } },
      y: { title: { display: true, text: "환율" } },
    },
  }

  return (
    <div className="flex-1 p-4 min-w-0 overflow-hidden">
      <Card className="h-full">
        <CardContent className="h-full p-4">
          {chartData ? (
            <div className="w-full h-full">
              <Line data={chartData} options={options} />
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-center text-muted-foreground">
              통화와 모델을 선택하세요
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}