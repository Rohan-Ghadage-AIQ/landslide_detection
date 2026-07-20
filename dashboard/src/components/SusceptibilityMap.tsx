import React, { useRef, useEffect, useState } from 'react';
import { Layers, MapPin, Info } from 'lucide-react';
import L from 'leaflet';

interface Point {
  id: number;
  x: number;
  y: number;
  row: number;
  col: number;
}

interface Bounds {
  left: number;
  bottom: number;
  right: number;
  top: number;
}

interface MapData {
  width: number;
  height: number;
  dem: (number | null)[];
  susceptibility: (number | null)[];
  slope: (number | null)[];
  dem_min: number;
  dem_max: number;
  inventory_points: Point[];
  event_point: Point | null;
  bounds: Bounds;
}

interface SusceptibilityMapProps {
  data: MapData | null;
  selectedCell: { row: number; col: number; val: number; elevation: number; slope: number } | null;
  onSelectCell: (cell: { row: number; col: number; val: number; elevation: number; slope: number } | null) => void;
}

type LayerType = 'susceptibility' | 'slope' | 'dem';

export const SusceptibilityMap: React.FC<SusceptibilityMapProps> = ({
  data,
  selectedCell,
  onSelectCell,
}) => {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const gridLayerRef = useRef<L.LayerGroup | null>(null);
  const markersLayerRef = useRef<L.LayerGroup | null>(null);
  const selectLayerRef = useRef<L.Rectangle | null>(null);
  
  const [activeLayer, setActiveLayer] = useState<LayerType>('susceptibility');
  const [mapType, setMapType] = useState<'satellite' | 'street'>('satellite');
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number; val: number; elevation: number; slope: number } | null>(null);


  // Initialize Map
  useEffect(() => {
    if (!data || !mapContainerRef.current || mapRef.current) return;

    // Create Map
    const map = L.map(mapContainerRef.current, {
      center: data.bounds ? [(data.bounds.bottom + data.bounds.top) / 2, (data.bounds.left + data.bounds.right) / 2] : [34.1, 77.5],
      zoom: 13,
      zoomControl: true,
      attributionControl: false,
    });

    mapRef.current = map;

    // Layer groups for grid and markers
    gridLayerRef.current = L.layerGroup().addTo(map);
    markersLayerRef.current = L.layerGroup().addTo(map);

    // Initial base layer based on mapType state
    if (mapType === 'satellite') {
      L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 19,
      }).addTo(map);
    } else {
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{y}/{x}.png', {
        maxZoom: 19,
      }).addTo(map);
    }

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, [data]);

  // Update base tiles when toggle changes
  useEffect(() => {
    if (!mapRef.current) return;
    const map = mapRef.current;

    // Remove existing tile layers
    map.eachLayer((layer) => {
      if (layer instanceof L.TileLayer) {
        map.removeLayer(layer);
      }
    });

    if (mapType === 'satellite') {
      // Esri World Imagery (Satellite)
      L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 19,
      }).addTo(map);
    } else {
      // OSM Streets
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{y}/{x}.png', {
        maxZoom: 19,
      }).addTo(map);
    }
  }, [mapType]);

  // Render Grid Cells, Markers, and Selection Overlays
  useEffect(() => {
    if (!data || !mapRef.current || !gridLayerRef.current || !markersLayerRef.current) return;
    const map = mapRef.current;
    const gridLayer = gridLayerRef.current;
    const markersLayer = markersLayerRef.current;

    gridLayer.clearLayers();
    markersLayer.clearLayers();

    if (selectLayerRef.current) {
      map.removeLayer(selectLayerRef.current);
      selectLayerRef.current = null;
    }

    const { width, height, dem, susceptibility, slope, dem_min, dem_max, bounds } = data;

    // Calculate cell sizes based on the REAL geospatial bounds
    const cell_size_y = (bounds.top - bounds.bottom) / height;
    const cell_size_x = (bounds.right - bounds.left) / width;

    const startLat = bounds.bottom;
    const startLng = bounds.left;

    const boundsList: L.LatLngBoundsExpression[] = [];

    // 1. Draw Grid Cells
    for (let r = 0; r < height; r++) {
      for (let c = 0; c < width; c++) {
        const idx = r * width + c;
        
        const latMin = startLat + (height - 1 - r) * cell_size_y;
        const latMax = latMin + cell_size_y;
        const lngMin = startLng + c * cell_size_x;
        const lngMax = lngMin + cell_size_x;
        
        const cellBounds = L.latLngBounds([latMin, lngMin], [latMax, lngMax]);
        boundsList.push(cellBounds);

        let fillColor = 'rgba(0,0,0,0)';
        let fillOpacity = 0.5;

        if (activeLayer === 'susceptibility') {
          const prob = susceptibility[idx];
          if (prob !== null && prob !== undefined) {
            if (prob < 0.15) {
              fillColor = '#10b981'; // Emerald
              fillOpacity = 0.35;
            } else if (prob < 0.35) {
              fillColor = '#f59e0b'; // Amber
              fillOpacity = 0.5;
            } else if (prob < 0.6) {
              fillColor = '#f97316'; // Orange
              fillOpacity = 0.65;
            } else {
              fillColor = '#ef4444'; // Red
              fillOpacity = 0.8;
            }
          }
        } else if (activeLayer === 'slope') {
          const sVal = slope[idx];
          if (sVal !== null && sVal !== undefined) {
            const normSlope = Math.min(1, sVal / 45);
            fillColor = '#6366f1'; // Indigo
            fillOpacity = 0.2 + normSlope * 0.7;
          }
        } else if (activeLayer === 'dem') {
          const elev = dem[idx];
          if (elev !== null && elev !== undefined) {
            const normElev = (elev - dem_min) / (dem_max - dem_min || 1);
            const rgbVal = Math.floor(normElev * 255);
            fillColor = `rgb(${rgbVal}, ${rgbVal}, ${rgbVal})`;
            fillOpacity = 0.7;
          }
        }

        const cellValue = susceptibility[idx] || 0;
        const elevValue = dem[idx] || 0;
        const slopeValue = slope[idx] || 0;

        const rect = L.rectangle(cellBounds, {
          color: 'transparent',
          fillColor: fillColor,
          fillOpacity: fillOpacity,
          weight: 0,
        });

        // Click handler to select cell
        rect.on('click', () => {
          onSelectCell({ row: r, col: c, val: cellValue, elevation: elevValue, slope: slopeValue });
        });

        // Mouse hover handlers
        rect.on('mouseover', () => {
          setHoveredCell({ row: r, col: c, val: cellValue, elevation: elevValue, slope: slopeValue });
        });

        rect.addTo(gridLayer);
      }
    }

    // Zoom Map to Fit Grid bounds
    if (boundsList.length > 0) {
      const fullBounds = L.latLngBounds(boundsList[0]);
      boundsList.forEach((b) => fullBounds.extend(b));
      map.fitBounds(fullBounds, { padding: [10, 10] });
    }

    // 2. Draw Landslide Inventory Markers
    data.inventory_points.forEach((pt) => {
      const ptLat = pt.y;
      const ptLng = pt.x;

      L.circleMarker([ptLat, ptLng], {
        radius: 6,
        fillColor: '#ef4444',
        fillOpacity: 1,
        color: '#ffffff',
        weight: 1.5,
      })
      .bindTooltip(`Landslide Pt #${pt.id}<br/><b>Lat:</b> ${ptLat.toFixed(5)}<br/><b>Lng:</b> ${ptLng.toFixed(5)}`, {
        direction: 'top',
        opacity: 0.9
      })
      .addTo(markersLayer);
    });

    // 3. Draw Historical Backtest Event marker (Imminent Threat warning zone)
    if (data.event_point) {
      const pt = data.event_point;
      const ptLat = pt.y;
      const ptLng = pt.x;

      // Pulse ring (Blinking alert)
      L.circleMarker([ptLat, ptLng], {
        radius: 16,
        fillColor: '#ef4444',
        fillOpacity: 0.15,
        color: '#ef4444',
        weight: 1.5,
      }).addTo(markersLayer);

      // Warning marker dot
      L.circleMarker([ptLat, ptLng], {
        radius: 6,
        fillColor: '#ef4444',
        fillOpacity: 1,
        color: '#ffffff',
        weight: 2,
      })
      .bindTooltip(`⚠️ <b>IMMINENT LANDSLIDE ALERT</b><br/><b>Lat:</b> ${ptLat.toFixed(5)}<br/><b>Lng:</b> ${ptLng.toFixed(5)}<br/><b>Combined Risk:</b> 92%`, {
        permanent: true,
        direction: 'right',
        offset: [12, 0],
        className: 'bg-rose-600 dark:bg-rose-700 text-white border border-rose-800 rounded px-2.5 py-1.5 font-sans text-[11px] font-extrabold shadow-lg animate-pulse'
      })
      .addTo(markersLayer);
    }

    // 4. Highlight Selected Cell Bounding Box
    if (selectedCell) {
      const latMin = startLat + (height - 1 - selectedCell.row) * cell_size_y;
      const latMax = latMin + cell_size_y;
      const lngMin = startLng + selectedCell.col * cell_size_x;
      const lngMax = lngMin + cell_size_x;

      selectLayerRef.current = L.rectangle([[latMin, lngMin], [latMax, lngMax]], {
        color: '#3b82f6',
        weight: 3,
        fillColor: 'transparent',
      }).addTo(map);
    }

  }, [data, activeLayer, selectedCell]);

  const getRiskLabel = (prob: number) => {
    if (prob < 0.15) return { label: 'Low', color: 'text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-950/20' };
    if (prob < 0.35) return { label: 'Moderate', color: 'text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/20' };
    if (prob < 0.6) return { label: 'High', color: 'text-orange-600 dark:text-orange-400 bg-orange-50 dark:bg-orange-950/20' };
    return { label: 'Very High', color: 'text-rose-600 dark:text-rose-400 bg-rose-50 dark:bg-rose-950/20' };
  };

  if (!data) {
    return (
      <div className="flex flex-col items-center justify-center bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl h-[420px] p-6 shadow-sm">
        <RefreshLoader />
        <span className="text-zinc-500 mt-4 text-sm">Loading satellite map data...</span>
      </div>
    );
  }

  // Coordinates for imminent alert card
  const alertLat = data.event_point ? data.event_point.y : 0;
  const alertLng = data.event_point ? data.event_point.x : 0;

  return (
    <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-5 shadow-sm space-y-4">
      
      {/* Header Controls */}
      <div className="flex flex-col xl:flex-row justify-between items-start xl:items-center gap-3 pb-3 border-b border-zinc-100 dark:border-zinc-800">
        <div>
          <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-50 tracking-tight flex items-center gap-1.5">
            <Layers className="w-5 h-5 text-emerald-600 dark:text-emerald-500" />
            Landslide Susceptibility Map (Western Ghats Overlay)
          </h2>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
            Real Satellite imagery base map (Leh, Ladakh). Click cells to inspect risk.
          </p>
        </div>

        {/* View toggles */}
        <div className="flex items-center space-x-3 self-stretch xl:self-auto justify-between">
          
          {/* Map Style toggle */}
          <div className="flex items-center space-x-1 p-0.5 bg-zinc-100 dark:bg-zinc-900 rounded-lg">
            <button
              onClick={() => setMapType('satellite')}
              className={`px-2.5 py-1 text-[10px] font-bold rounded transition-all cursor-pointer ${
                mapType === 'satellite' ? 'bg-white dark:bg-zinc-800 text-zinc-950 dark:text-zinc-50 shadow-sm' : 'text-zinc-500'
              }`}
            >
              Satellite
            </button>
            <button
              onClick={() => setMapType('street')}
              className={`px-2.5 py-1 text-[10px] font-bold rounded transition-all cursor-pointer ${
                mapType === 'street' ? 'bg-white dark:bg-zinc-800 text-zinc-950 dark:text-zinc-50 shadow-sm' : 'text-zinc-500'
              }`}
            >
              Terrain
            </button>
          </div>

          {/* Raster layers toggle */}
          <div className="flex items-center space-x-1 p-0.5 bg-zinc-100 dark:bg-zinc-900 rounded-lg">
            <button
              onClick={() => setActiveLayer('susceptibility')}
              className={`px-2.5 py-1 text-[10px] font-bold rounded transition-all cursor-pointer ${
                activeLayer === 'susceptibility' ? 'bg-white dark:bg-zinc-800 text-emerald-600 dark:text-emerald-400 shadow-sm' : 'text-zinc-500'
              }`}
            >
              Risk
            </button>
            <button
              onClick={() => setActiveLayer('slope')}
              className={`px-2.5 py-1 text-[10px] font-bold rounded transition-all cursor-pointer ${
                activeLayer === 'slope' ? 'bg-white dark:bg-zinc-800 text-emerald-600 dark:text-emerald-400 shadow-sm' : 'text-zinc-500'
              }`}
            >
              Slope
            </button>
            <button
              onClick={() => setActiveLayer('dem')}
              className={`px-2.5 py-1 text-[10px] font-bold rounded transition-all cursor-pointer ${
                activeLayer === 'dem' ? 'bg-white dark:bg-zinc-800 text-emerald-600 dark:text-emerald-400 shadow-sm' : 'text-zinc-500'
              }`}
            >
              DEM
            </button>
          </div>

        </div>
      </div>

      {/* Map Display & inspection panel */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        
        {/* Leaflet container */}
        <div className="lg:col-span-3 h-[420px] rounded-xl overflow-hidden border border-zinc-200 dark:border-zinc-800 relative z-10">
          <div 
            ref={mapContainerRef} 
            className="w-full h-full"
            onMouseLeave={() => setHoveredCell(null)}
          />

          {/* Bounding box legends */}
          <div className="absolute bottom-6 left-6 bg-white/95 dark:bg-zinc-950/95 border border-zinc-200 dark:border-zinc-800 rounded-lg p-2.5 text-[10px] space-y-1.5 shadow-md backdrop-blur-sm z-30 pointer-events-none">
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-500 inline-block border border-white"></span>
              <span className="text-zinc-700 dark:text-zinc-300 font-medium">Historical Landslide Inventory</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-red-600 inline-block border border-white animate-pulse"></span>
              <span className="text-zinc-700 dark:text-zinc-300 font-medium">Active Threat Warning (Evacuation)</span>
            </div>
          </div>

          {/* Map Hover Cell values */}
          {hoveredCell && (
            <div className="absolute top-6 right-6 bg-white/95 dark:bg-zinc-950/95 border border-zinc-200 dark:border-zinc-800 rounded-lg p-2.5 text-xs shadow-md backdrop-blur-sm z-30 pointer-events-none">
              <p className="font-semibold text-zinc-900 dark:text-zinc-50 border-b border-zinc-200 dark:border-zinc-800 pb-1 mb-1">
                Cell [{hoveredCell.row}, {hoveredCell.col}]
              </p>
              <div className="space-y-1">
                <div className="flex justify-between gap-4">
                  <span className="text-zinc-400">Elevation:</span>
                  <span className="font-mono font-medium">{Math.round(hoveredCell.elevation)} m</span>
                </div>
                <div className="flex justify-between gap-4">
                  <span className="text-zinc-400">Slope:</span>
                  <span className="font-mono font-medium">{hoveredCell.slope.toFixed(1)}°</span>
                </div>
                <div className="flex justify-between gap-4">
                  <span className="text-zinc-400">Susceptibility:</span>
                  <span className="font-mono font-semibold text-emerald-600 dark:text-emerald-400">
                    {(hoveredCell.val * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Bounding box Details card */}
        <div className="flex flex-col space-y-4">
          
          {/* Flashing Imminent Alert card */}
          {data.event_point && (
            <div className="bg-rose-50 dark:bg-rose-950/30 border border-rose-200 dark:border-rose-900/50 rounded-xl p-4 shadow-sm animate-pulse border-l-4 border-l-rose-500">
              <div className="flex items-center space-x-1.5 text-rose-600 dark:text-rose-400 mb-2">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-rose-500"></span>
                </span>
                <span className="text-xs font-black uppercase tracking-wider">Active Imminent Threat</span>
              </div>
              <div className="space-y-1">
                <p className="text-xs text-rose-900 dark:text-rose-200 font-bold">
                  Leh, Ladakh - Critical Zone
                </p>
                <p className="font-mono text-[10px] text-rose-700 dark:text-rose-400">
                  Lat: {alertLat.toFixed(5)}, Lng: {alertLng.toFixed(5)}
                </p>
                <div className="bg-white/60 dark:bg-zinc-900/60 rounded p-1.5 text-[10px] text-zinc-600 dark:text-zinc-300 mt-2 space-y-1">
                  <div>• Risk level: <b>92% (Very High)</b></div>
                  <div>• Trigger status: <b>Fully Saturated (345mm)</b></div>
                  <div>• Action: <b>Issue Immediate Evacuation</b></div>
                </div>
              </div>
            </div>
          )}

          {/* Cell Inspection Detail card */}
          <div className="bg-zinc-50 dark:bg-zinc-900/50 border border-zinc-200 dark:border-zinc-800/80 rounded-xl p-4 flex-grow flex flex-col justify-between min-h-[180px]">
            <div>
              <div className="flex items-center space-x-1.5 text-zinc-500 mb-3">
                <MapPin className="w-4 h-4 text-emerald-500" />
                <span className="text-xs font-semibold uppercase tracking-wider">Cell Inspection</span>
              </div>

              {selectedCell ? (
                <div className="space-y-4">
                  <div>
                    <h3 className="text-sm font-bold text-zinc-800 dark:text-zinc-200">
                      Grid Coordinates
                    </h3>
                    <p className="font-mono text-xs text-zinc-500 bg-zinc-200 dark:bg-zinc-800 rounded px-1.5 py-0.5 inline-block mt-1">
                      Row {selectedCell.row}, Col {selectedCell.col}
                    </p>
                  </div>

                  <div className="space-y-2.5">
                    <div className="flex justify-between text-xs py-1 border-b border-zinc-200/50 dark:border-zinc-800/50">
                      <span className="text-zinc-500">Elevation</span>
                      <span className="font-mono font-bold">{Math.round(selectedCell.elevation)} m</span>
                    </div>
                    <div className="flex justify-between text-xs py-1 border-b border-zinc-200/50 dark:border-zinc-800/50">
                      <span className="text-zinc-500">Slope angle</span>
                      <span className="font-mono font-bold">{selectedCell.slope.toFixed(1)}°</span>
                    </div>
                    <div className="flex justify-between text-xs py-1">
                      <span className="text-zinc-500">Susceptibility</span>
                      <span className="font-mono font-bold text-emerald-600 dark:text-emerald-400">
                        {Math.round(selectedCell.val * 100)}%
                      </span>
                    </div>
                  </div>

                  <div className={`rounded-lg p-3 border flex items-center justify-between ${getRiskLabel(selectedCell.val).color}`}>
                    <span className="text-xs font-semibold">Alert category</span>
                    <span className="text-sm font-bold uppercase">{getRiskLabel(selectedCell.val).label}</span>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-28 text-center px-4">
                  <Info className="w-6 h-6 text-zinc-400 mb-2" />
                  <p className="text-[10px] text-zinc-500 leading-normal">
                    Click any overlay grid cell on the satellite map to inspect localized slope gradients, elevation values, and alert levels.
                  </p>
                </div>
              )}
            </div>

            {selectedCell && (
              <button
                onClick={() => onSelectCell(null)}
                className="w-full text-center bg-zinc-200 hover:bg-zinc-300 dark:bg-zinc-800 dark:hover:bg-zinc-700 text-zinc-800 dark:text-zinc-200 text-xs font-semibold py-1.5 rounded-lg transition-colors cursor-pointer mt-3"
              >
                Clear Selection
              </button>
            )}
          </div>
        </div>

      </div>
    </div>
  );
};

const RefreshLoader = () => (
  <svg className="animate-spin h-8 w-8 text-emerald-600" fill="none" viewBox="0 0 24 24">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
  </svg>
);
