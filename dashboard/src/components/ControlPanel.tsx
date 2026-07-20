import React from 'react';
import { Play, Settings, RefreshCw } from 'lucide-react';

export interface PipelineParams {
  model_type: string;
  api_decay_k: number;
  api_window_days: number;
  pseudo_absence_ratio: number;
  pseudo_absence_min_dist_m: number;
  spatial_block_size_m: number;
  min_precision_for_threshold: number;
  api_saturation_reference_mm: number;
  forecast_trigger_reference_mm: number;
}

interface ControlPanelProps {
  params: PipelineParams;
  onChange: (key: keyof PipelineParams, value: any) => void;
  onRun: () => void;
  isRunning: boolean;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  params,
  onChange,
  onRun,
  isRunning,
}) => {
  return (
    <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-5 shadow-sm space-y-6">
      <div className="flex items-center space-x-2 pb-3 border-b border-zinc-100 dark:border-zinc-800">
        <Settings className="w-5 h-5 text-emerald-600 dark:text-emerald-500" />
        <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-50 tracking-tight">
          Pipeline Parameters
        </h2>
      </div>

      <div className="space-y-4">
        {/* Model Type */}
        <div className="flex flex-col">
          <label className="text-xs font-semibold text-zinc-500 dark:text-zinc-400 mb-1">
            CLASSIFICATION MODEL
          </label>
          <select
            value={params.model_type}
            onChange={(e) => onChange('model_type', e.target.value)}
            disabled={isRunning}
            className="bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
          >
            <option value="xgboost">XGBoost (Default)</option>
            <option value="random_forest">Random Forest</option>
          </select>
        </div>

        {/* API Decay Constant */}
        <div className="flex flex-col">
          <div className="flex justify-between text-xs mb-1">
            <span className="font-semibold text-zinc-500 dark:text-zinc-400">
              API DECAY CONSTANT (k)
            </span>
            <span className="font-mono text-emerald-600 dark:text-emerald-400 font-bold">
              {params.api_decay_k}
            </span>
          </div>
          <input
            type="range"
            min="0.5"
            max="0.99"
            step="0.01"
            value={params.api_decay_k}
            onChange={(e) => onChange('api_decay_k', parseFloat(e.target.value))}
            disabled={isRunning}
            className="w-full h-1.5 bg-zinc-200 dark:bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-600"
          />
          <span className="text-[10px] text-zinc-400 mt-0.5">
            Daily rainfall accumulation weight retention rate
          </span>
        </div>

        {/* API Saturation Reference */}
        <div className="flex flex-col">
          <div className="flex justify-between text-xs mb-1">
            <span className="font-semibold text-zinc-500 dark:text-zinc-400">
              API SATURATION LIMIT
            </span>
            <span className="font-mono text-emerald-600 dark:text-emerald-400 font-bold">
              {params.api_saturation_reference_mm} mm
            </span>
          </div>
          <input
            type="range"
            min="50"
            max="300"
            step="10"
            value={params.api_saturation_reference_mm}
            onChange={(e) => onChange('api_saturation_reference_mm', parseInt(e.target.value))}
            disabled={isRunning}
            className="w-full h-1.5 bg-zinc-200 dark:bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-600"
          />
          <span className="text-[10px] text-zinc-400 mt-0.5">
            Rainfall sum at which soil is fully saturated
          </span>
        </div>

        {/* Forecast Trigger Reference */}
        <div className="flex flex-col">
          <div className="flex justify-between text-xs mb-1">
            <span className="font-semibold text-zinc-500 dark:text-zinc-400">
              FORECAST TRIGGER THRESHOLD
            </span>
            <span className="font-mono text-emerald-600 dark:text-emerald-400 font-bold">
              {params.forecast_trigger_reference_mm} mm
            </span>
          </div>
          <input
            type="range"
            min="50"
            max="300"
            step="10"
            value={params.forecast_trigger_reference_mm}
            onChange={(e) => onChange('forecast_trigger_reference_mm', parseInt(e.target.value))}
            disabled={isRunning}
            className="w-full h-1.5 bg-zinc-200 dark:bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-600"
          />
          <span className="text-[10px] text-zinc-400 mt-0.5">
            7-day forecasted cumulative rainfall alert level
          </span>
        </div>

        {/* Pseudo-Absence Ratio */}
        <div className="flex flex-col">
          <div className="flex justify-between text-xs mb-1">
            <span className="font-semibold text-zinc-500 dark:text-zinc-400">
              PSEUDO-ABSENCE RATIO
            </span>
            <span className="font-mono text-emerald-600 dark:text-emerald-400 font-bold">
              {params.pseudo_absence_ratio}:1
            </span>
          </div>
          <input
            type="range"
            min="1"
            max="5"
            step="1"
            value={params.pseudo_absence_ratio}
            onChange={(e) => onChange('pseudo_absence_ratio', parseInt(e.target.value))}
            disabled={isRunning}
            className="w-full h-1.5 bg-zinc-200 dark:bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-600"
          />
          <span className="text-[10px] text-zinc-400 mt-0.5">
            Number of negatives sampled per landslide positive point
          </span>
        </div>

        {/* Spatial Block Size */}
        <div className="flex flex-col">
          <div className="flex justify-between text-xs mb-1">
            <span className="font-semibold text-zinc-500 dark:text-zinc-400">
              SPATIAL BLOCK SIZE
            </span>
            <span className="font-mono text-emerald-600 dark:text-emerald-400 font-bold">
              {params.spatial_block_size_m} m
            </span>
          </div>
          <input
            type="range"
            min="1000"
            max="10000"
            step="500"
            value={params.spatial_block_size_m}
            onChange={(e) => onChange('spatial_block_size_m', parseInt(e.target.value))}
            disabled={isRunning}
            className="w-full h-1.5 bg-zinc-200 dark:bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-600"
          />
          <span className="text-[10px] text-zinc-400 mt-0.5">
            Cross-validation spatial block grid resolution
          </span>
        </div>

        {/* Min Precision for Threshold */}
        <div className="flex flex-col">
          <div className="flex justify-between text-xs mb-1">
            <span className="font-semibold text-zinc-500 dark:text-zinc-400">
              MIN ACCEPTABLE PRECISION
            </span>
            <span className="font-mono text-emerald-600 dark:text-emerald-400 font-bold">
              {Math.round(params.min_precision_for_threshold * 100)}%
            </span>
          </div>
          <input
            type="range"
            min="0.1"
            max="0.8"
            step="0.05"
            value={params.min_precision_for_threshold}
            onChange={(e) => onChange('min_precision_for_threshold', parseFloat(e.target.value))}
            disabled={isRunning}
            className="w-full h-1.5 bg-zinc-200 dark:bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-600"
          />
          <span className="text-[10px] text-zinc-400 mt-0.5">
            Enforced precision target for threshold tuning
          </span>
        </div>
      </div>

      <button
        onClick={onRun}
        disabled={isRunning}
        className="w-full bg-[#059669] hover:bg-[#047857] disabled:bg-zinc-300 dark:disabled:bg-zinc-800 text-white font-semibold py-2.5 px-4 rounded-lg flex items-center justify-center space-x-2 transition-all shadow-sm duration-150 cursor-pointer disabled:cursor-not-allowed"
      >
        {isRunning ? (
          <>
            <RefreshCw className="w-5 h-5 animate-spin" />
            <span>Training Model...</span>
          </>
        ) : (
          <>
            <Play className="w-5 h-5 fill-current" />
            <span>Execute Pipeline Run</span>
          </>
        )}
      </button>
    </div>
  );
};
