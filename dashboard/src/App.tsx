import { useState, useEffect } from 'react';
import axios from 'axios';
import { ShieldAlert, Compass, AlertCircle } from 'lucide-react';
import { ControlPanel } from './components/ControlPanel';
import type { PipelineParams } from './components/ControlPanel';
import { SusceptibilityMap } from './components/SusceptibilityMap';
import { TimelineChart } from './components/TimelineChart';
import { MetricsPanel } from './components/MetricsPanel';

export default function App() {
  const [params, setParams] = useState<PipelineParams>({
    model_type: 'xgboost',
    api_decay_k: 0.9,
    api_window_days: 15,
    pseudo_absence_ratio: 2,
    pseudo_absence_min_dist_m: 100,
    spatial_block_size_m: 500,
    min_precision_for_threshold: 0.3,
    api_saturation_reference_mm: 150,
    forecast_trigger_reference_mm: 100,
  });

  const [mapData, setMapData] = useState<any>(null);
  const [backtestData, setBacktestData] = useState<any>(null);
  const [metricsData, setMetricsData] = useState<any>(null);
  
  const [selectedCell, setSelectedCell] = useState<any>(null);
  const [isPipelineRunning, setIsPipelineRunning] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Explicitly remove dark mode class to force light theme
  useEffect(() => {
    document.body.classList.remove('dark');
  }, []);

  // Fetch all dashboard data
  const fetchAllData = async () => {
    try {
      setErrorMsg(null);
      const [mapRes, backtestRes, metricsRes] = await Promise.all([
        axios.get('/api/map-data'),
        axios.get('/api/backtest'),
        axios.get('/api/metrics'),
      ]);
      setMapData(mapRes.data);
      setBacktestData(backtestRes.data);
      setMetricsData(metricsRes.data);
    } catch (err: any) {
      console.error(err);
      setErrorMsg(err.response?.data?.detail || 'Failed to connect to the backend server. Please verify FastAPI is running.');
    }
  };

  // Initial config load
  useEffect(() => {
    const loadConfig = async () => {
      try {
        const configRes = await axios.get('/api/config');
        const cfg = configRes.data;
        setParams({
          model_type: cfg.model_type,
          api_decay_k: cfg.api_decay_k,
          api_window_days: cfg.api_window_days,
          pseudo_absence_ratio: cfg.pseudo_absence_ratio,
          pseudo_absence_min_dist_m: cfg.pseudo_absence_min_dist_m,
          spatial_block_size_m: cfg.spatial_block_size_m,
          min_precision_for_threshold: cfg.min_precision_for_threshold,
          api_saturation_reference_mm: cfg.api_saturation_reference_mm,
          forecast_trigger_reference_mm: cfg.forecast_trigger_reference_mm,
        });
      } catch (err) {
        console.error('Failed to load initial configuration.', err);
      }
      fetchAllData();
    };

    loadConfig();
  }, []);

  const handleParamChange = (key: keyof PipelineParams, value: any) => {
    setParams((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const handleRunPipeline = async () => {
    setIsPipelineRunning(true);
    setErrorMsg(null);
    try {
      await axios.post('/api/run-pipeline', params);
      await fetchAllData();
      setSelectedCell(null); // Reset selection after raster updates
    } catch (err: any) {
      console.error(err);
      setErrorMsg(err.response?.data?.detail || 'An error occurred while running the pipeline.');
    } finally {
      setIsPipelineRunning(false);
    }
  };

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-900 transition-colors duration-200">
      
      {/* Top Header */}
      <header className="border-b border-zinc-200 bg-white sticky top-0 z-40 backdrop-blur shadow-sm">
        <div className="max-w-[1600px] mx-auto px-6 py-4 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-emerald-600 rounded-lg text-white">
              <ShieldAlert className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-black tracking-tight text-zinc-950">
                LANDSLIDE DETECTION MODEL
              </h1>
              <p className="text-[10px] font-bold tracking-wider text-zinc-500 uppercase">
                Early Warning System &bull; Pune District R&D Scaffold
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Layout */}
      <main className="max-w-[1600px] mx-auto p-6 space-y-6">
        
        {/* Error Alert Box */}
        {errorMsg && (
          <div className="bg-rose-50 border border-rose-200 text-rose-800 rounded-xl p-4 flex gap-3 shadow-sm animate-pulse">
            <AlertCircle className="w-5 h-5 shrink-0" />
            <div className="text-sm">
              <p className="font-bold">System Connection Issue</p>
              <p className="mt-0.5">{errorMsg}</p>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 items-start">
          
          {/* Parameters Control Panel Side Column */}
          <div className="lg:col-span-1">
            <ControlPanel
              params={params}
              onChange={handleParamChange}
              onRun={handleRunPipeline}
              isRunning={isPipelineRunning}
            />
          </div>

          {/* Interactive Map & Charts Main Column */}
          <div className="lg:col-span-3 space-y-6">
            
            {/* Susceptibility Map Canvas */}
            <SusceptibilityMap
              data={mapData}
              selectedCell={selectedCell}
              onSelectCell={setSelectedCell}
            />

            {/* Warning timelines */}
            <TimelineChart data={backtestData} />

            {/* CV Stats and Feature Importance list */}
            <MetricsPanel data={metricsData} />

          </div>
        </div>
      </main>

      {/* Bottom Citation footer */}
      <footer className="border-t border-zinc-200 bg-white py-6 text-center text-xs text-zinc-500">
        <div className="max-w-[1600px] mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-3">
          <div className="flex items-center space-x-1.5">
            <Compass className="w-4 h-4 text-emerald-500" />
            <span className="font-semibold">AIQ Space Ventures, Mumbai</span>
          </div>
          <p>&copy; 2026 AIQ Space Ventures. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
