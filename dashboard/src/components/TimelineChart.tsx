import React from 'react';
import ReactECharts from 'echarts-for-react';
import { CloudRain, TrendingUp, AlertTriangle } from 'lucide-react';

interface BacktestTraceRow {
  date: string;
  days_before_event: number;
  api_mm: number;
  forecast_next_7d_mm: number;
  susceptibility_prob: number;
  combined_score: number;
  risk_level: string;
  insar_displacement_mm: number;
}

interface BacktestData {
  summary: {
    would_have_warned: boolean;
    max_lead_time_days: number;
    message: string;
  };
  trace: BacktestTraceRow[];
}

interface TimelineChartProps {
  data: BacktestData | null;
}

export const TimelineChart: React.FC<TimelineChartProps> = ({ data }) => {
  if (!data) {
    return (
      <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl h-[400px] flex items-center justify-center p-6 shadow-sm">
        <div className="flex flex-col items-center">
          <svg className="animate-spin h-8 w-8 text-emerald-600" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          <span className="text-zinc-500 mt-4 text-sm">Loading early warning trace...</span>
        </div>
      </div>
    );
  }

  const { summary, trace } = data;

  // Prepare chart series data
  const dates = trace.map((t) => t.date);
  const daysBefore = trace.map((t) => `${t.days_before_event}d before`);
  const rainfall = trace.map((t) => t.forecast_next_7d_mm);
  const api = trace.map((t) => t.api_mm);
  const combinedScore = trace.map((t) => t.combined_score);
  const displacement = trace.map((t) => t.insar_displacement_mm);

  // ECharts configuration options
  const option = {
    color: ["#3b82f6", "#06b6d4", "#ef4444", "#8b5cf6"], // blue, cyan, red, purple
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(255, 255, 255, 0.95)',
      borderColor: '#e2e8f0',
      borderWidth: 1,
      textStyle: {
        color: '#0f172a',
        fontFamily: 'Inter, sans-serif',
      },
      axisPointer: {
        type: 'cross',
        crossStyle: {
          color: '#94a3b8'
        }
      }
    },
    legend: {
      data: ['7d Forecast Rainfall (mm)', 'API Saturation Index (mm)', 'Combined Risk Score (0-1)', 'InSAR Displacement (mm)'],
      textStyle: {
        color: '#64748b',
        fontFamily: 'Inter, sans-serif',
      },
      bottom: 0
    },
    grid: {
      top: '12%',
      left: '6%',
      right: '6%',
      bottom: '15%',
      containLabel: true
    },
    xAxis: [
      {
        type: 'category',
        data: dates,
        axisLabel: {
          formatter: (value: string, idx: number) => {
            const row = trace[idx];
            return `${value}\n(${row.days_before_event}d)`;
          },
          textStyle: {
            color: '#64748b',
            fontFamily: 'Inter, sans-serif',
          }
        },
        axisPointer: {
          type: 'shadow'
        }
      }
    ],
    yAxis: [
      {
        type: 'value',
        name: 'Rainfall / Saturation (mm)',
        min: 0,
        axisLabel: {
          formatter: '{value} mm',
          textStyle: {
            color: '#64748b',
          }
        },
        nameTextStyle: {
          color: '#64748b',
          fontFamily: 'Inter, sans-serif',
        },
        splitLine: {
          lineStyle: {
            color: '#e2e8f0',
            type: 'dashed'
          }
        }
      },
      {
        type: 'value',
        name: 'Risk Score / Displacement',
        min: 0,
        max: 30, // Fits both combined score (0-1 scale, mapped) and displacement (up to 30mm)
        axisLabel: {
          formatter: (value: number) => {
            if (value <= 1.0) {
              return `Risk: ${value.toFixed(1)}`;
            }
            return `${value} mm`;
          },
          textStyle: {
            color: '#64748b',
          }
        },
        nameTextStyle: {
          color: '#64748b',
          fontFamily: 'Inter, sans-serif',
        },
        splitLine: {
          show: false
        }
      }
    ],
    series: [
      {
        name: '7d Forecast Rainfall (mm)',
        type: 'bar',
        barWidth: '40%',
        data: rainfall,
        itemStyle: {
          opacity: 0.8,
          borderRadius: [4, 4, 0, 0]
        }
      },
      {
        name: 'API Saturation Index (mm)',
        type: 'line',
        yAxisIndex: 0,
        data: api,
        symbol: 'circle',
        symbolSize: 6,
        lineStyle: {
          width: 2.5
        }
      },
      {
        name: 'Combined Risk Score (0-1)',
        type: 'line',
        yAxisIndex: 1,
        data: combinedScore,
        symbol: 'triangle',
        symbolSize: 8,
        lineStyle: {
          width: 3,
          type: 'dashed'
        }
      },
      {
        name: 'InSAR Displacement (mm)',
        type: 'line',
        yAxisIndex: 1,
        data: displacement,
        symbol: 'square',
        symbolSize: 7,
        lineStyle: {
          width: 3
        }
      }
    ]
  };

  return (
    <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-5 shadow-sm space-y-4">
      {/* Header and alerts */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-3 pb-3 border-b border-zinc-100 dark:border-zinc-800">
        <div>
          <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-50 tracking-tight flex items-center gap-1.5">
            <CloudRain className="w-5 h-5 text-emerald-600 dark:text-emerald-500" />
            Early-Warning Trigger Backtest
          </h2>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
            10-Day simulation before the historical event on {trace.length > 0 ? trace[trace.length - 1].date : "the historical event"}
          </p>
        </div>
      </div>

      {/* Warning status message */}
      <div className={`p-4 rounded-lg border flex gap-3 ${
        summary.would_have_warned
          ? 'bg-rose-50 dark:bg-rose-950/20 border-rose-200 dark:border-rose-800/30 text-rose-800 dark:text-rose-300'
          : 'bg-amber-50 dark:bg-amber-950/20 border-amber-200 dark:border-amber-800/30 text-amber-800 dark:text-amber-300'
      }`}>
        <AlertTriangle className="w-5 h-5 shrink-0" />
        <div className="text-sm">
          <p className="font-bold">Historical Backtest Result</p>
          <p className="mt-0.5">{summary.message}</p>
        </div>
      </div>

      {/* Multi-Axis Chart */}
      <div className="h-[360px] w-full">
        <ReactECharts option={option} style={{ height: '100%', width: '100%' }} />
      </div>
    </div>
  );
};
