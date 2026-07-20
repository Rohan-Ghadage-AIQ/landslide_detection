import React from 'react';
import { Award, BarChart3, Database, ShieldAlert } from 'lucide-react';

interface FeatureImportance {
  feature: string;
  importance: number;
}

interface MetricsData {
  pr_auc: number;
  roc_auc: number;
  total_rows: number;
  positive_rows: number;
  feature_importances: FeatureImportance[];
}

interface MetricsPanelProps {
  data: MetricsData | null;
}

export const MetricsPanel: React.FC<MetricsPanelProps> = ({ data }) => {
  if (!data) {
    return (
      <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl h-[350px] flex items-center justify-center p-6 shadow-sm">
        <div className="flex flex-col items-center">
          <svg className="animate-spin h-8 w-8 text-emerald-600" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          <span className="text-zinc-500 mt-4 text-sm">Loading validation metrics...</span>
        </div>
      </div>
    );
  }

  const { pr_auc, roc_auc, total_rows, positive_rows, feature_importances } = data;

  const getFeatureLabel = (feature: string) => {
    const labels: Record<string, string> = {
      elevation: 'Elevation (DEM height)',
      slope: 'Slope Gradient (Horn)',
      aspect: 'Slope Aspect (direction)',
      curvature: 'Profile Curvature',
      tri: 'Terrain Ruggedness (TRI)',
      flow_accum: 'D8 Flow Accumulation',
      twi: 'Topographic Wetness (TWI)',
      spi: 'Stream Power Index (SPI)',
      dist_to_drainage: 'Distance to Drainage',
      ndvi: 'NDVI Vegetation Baseline',
    };
    return labels[feature] || feature;
  };

  return (
    <div className="space-y-6">
      {/* KPI Cards Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Total dataset rows */}
        <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-4 shadow-sm flex items-center space-x-3.5">
          <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
            <Database className="w-5 h-5 text-blue-600 dark:text-blue-400" />
          </div>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">Total Samples</p>
            <h3 className="text-lg font-bold text-zinc-800 dark:text-zinc-50 mt-0.5">{total_rows}</h3>
          </div>
        </div>

        {/* Positive landslide events */}
        <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-4 shadow-sm flex items-center space-x-3.5">
          <div className="p-2 bg-rose-100 dark:bg-rose-900/30 rounded-lg">
            <ShieldAlert className="w-5 h-5 text-rose-600 dark:text-rose-400" />
          </div>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">Landslide Events</p>
            <h3 className="text-lg font-bold text-zinc-800 dark:text-zinc-50 mt-0.5">{positive_rows}</h3>
          </div>
        </div>

        {/* Spatial PR-AUC */}
        <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-4 shadow-sm flex items-center space-x-3.5">
          <div className="p-2 bg-emerald-100 dark:bg-emerald-900/30 rounded-lg">
            <Award className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
          </div>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">PR-AUC (CV)</p>
            <h3 className="text-lg font-bold text-zinc-800 dark:text-zinc-50 mt-0.5">{(pr_auc * 100).toFixed(0)}%</h3>
          </div>
        </div>

        {/* Spatial ROC-AUC */}
        <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-4 shadow-sm flex items-center space-x-3.5">
          <div className="p-2 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
            <Award className="w-5 h-5 text-purple-600 dark:text-purple-400" />
          </div>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">ROC-AUC (CV)</p>
            <h3 className="text-lg font-bold text-zinc-800 dark:text-zinc-50 mt-0.5">{(roc_auc * 100).toFixed(0)}%</h3>
          </div>
        </div>
      </div>

      {/* Feature Importance Panel */}
      <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-5 shadow-sm space-y-4">
        <div className="flex items-center space-x-2 pb-3 border-b border-zinc-100 dark:border-zinc-800">
          <BarChart3 className="w-5 h-5 text-emerald-600 dark:text-emerald-500" />
          <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-50 tracking-tight">
            Spatial Susceptibility Feature Importances
          </h2>
        </div>

        <div className="space-y-3.5">
          {feature_importances.map((item, idx) => {
            const percentage = item.importance * 100;
            // Map color based on index or value
            let barColor = 'bg-emerald-500';
            if (idx === 0 || idx === 1) barColor = 'bg-rose-500';
            else if (idx === 2 || idx === 3) barColor = 'bg-orange-400';

            return (
              <div key={item.feature} className="space-y-1">
                <div className="flex justify-between text-xs font-semibold">
                  <span className="text-zinc-700 dark:text-zinc-300">
                    {getFeatureLabel(item.feature)}
                  </span>
                  <span className="font-mono text-zinc-500 dark:text-zinc-400">
                    {percentage.toFixed(1)}%
                  </span>
                </div>
                {/* Horizontal Progress Bar */}
                <div className="w-full bg-zinc-100 dark:bg-zinc-800 rounded-full h-2">
                  <div
                    className={`${barColor} h-2 rounded-full transition-all duration-500`}
                    style={{ width: `${Math.max(1, percentage)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
