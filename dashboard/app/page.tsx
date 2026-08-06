"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import * as maplibregl from "maplibre-gl";
import type { Map as MapLibreMap } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { cellToBoundary, cellToLatLng, gridDisk, latLngToCell } from "h3-js";

type WindowKey = "1m" | "5m" | "30m";
type Mode = "live" | "demo";

type Cell = {
  h3_cell: string;
  resolution: number;
  stage1_probability: number;
  stage1_prediction: number;
  stage2_probability: number;
  stage2_prediction: number;
  stage2_executed: boolean;
  combined_prediction: number;
  strike_counts: Record<WindowKey, number>;
  strike_count_1m: number;
  strike_count_5m: number;
  strike_count_30m: number;
  processing_ms: number;
  e2e_ms: number;
  event_type: "strike" | "candidate";
  age_seconds: number;
  updated_at: string;
};

type Snapshot = {
  generated_at: string;
  summary: {
    active_cells: number;
    total_prediction_cells: number;
    stage1_positive_cells: number;
    stage2_positive_cells: number;
    candidate_cells: number;
    observed_cells: number;
    strikes_by_window: Record<WindowKey, number>;
    strikes_1m: number;
    strikes_5m: number;
    strikes_30m: number;
    stage2_skip_rate: number;
    e2e_p95_ms: number;
  };
  cells: Cell[];
};

const API_URL = process.env.NEXT_PUBLIC_LIGHTNING_API_URL ?? "http://127.0.0.1:8765/api/state";
const WINDOWS: { key: WindowKey; label: string; detail: string }[] = [
  { key: "1m", label: "1 MIN", detail: "Immediate" },
  { key: "5m", label: "5 MIN", detail: "Tactical" },
  { key: "30m", label: "30 MIN", detail: "Trend" },
];

function demoSnapshot(): Snapshot {
  const origin = latLngToCell(28.42, -81.53, 7);
  const now = new Date().toISOString();
  const cells = gridDisk(origin, 3).slice(0, 37).map((cell, index) => {
    const radial = Math.abs(index - 9);
    const stage1 = Math.max(0.08, Math.min(0.97, 0.95 - radial * 0.047));
    const executed = stage1 >= 0.5;
    const stage2 = executed ? Math.max(0.11, 0.91 - radial * 0.055) : 0;
    const count5 = Math.max(0, 24 - radial * 2);
    const counts = {
      "1m": Math.max(0, Math.round(count5 * 0.24)),
      "5m": count5,
      "30m": count5 * 4 + (index % 5),
    };
    return {
      h3_cell: cell,
      resolution: 7,
      stage1_probability: stage1,
      stage1_prediction: Number(executed),
      stage2_probability: stage2,
      stage2_prediction: Number(stage2 >= 0.78),
      stage2_executed: executed,
      combined_prediction: Number(executed && stage2 >= 0.78),
      strike_counts: counts,
      strike_count_1m: counts["1m"],
      strike_count_5m: counts["5m"],
      strike_count_30m: counts["30m"],
      processing_ms: 3.9 + index * 0.19,
      e2e_ms: 11.2 + index * 0.43,
      event_type: index % 3 ? "candidate" as const : "strike" as const,
      age_seconds: index * 2.7,
      updated_at: now,
    };
  });
  const stage2Calls = cells.filter((cell) => cell.stage2_executed).length;
  const strikesByWindow = Object.fromEntries(
    WINDOWS.map(({ key }) => [key, cells.reduce((sum, cell) => sum + cell.strike_counts[key], 0)]),
  ) as Record<WindowKey, number>;
  return {
    generated_at: now,
    summary: {
      active_cells: cells.length,
      total_prediction_cells: cells.length,
      stage1_positive_cells: stage2Calls,
      stage2_positive_cells: cells.filter((cell) => cell.stage2_prediction).length,
      candidate_cells: cells.filter((cell) => cell.event_type === "candidate").length,
      observed_cells: cells.filter((cell) => cell.event_type === "strike").length,
      strikes_by_window: strikesByWindow,
      strikes_1m: strikesByWindow["1m"],
      strikes_5m: strikesByWindow["5m"],
      strikes_30m: strikesByWindow["30m"],
      stage2_skip_rate: 1 - stage2Calls / cells.length,
      e2e_p95_ms: 22.8,
    },
    cells,
  };
}

function riskLabel(probability: number) {
  if (probability >= 0.8) return { label: "CRITICAL", className: "critical" };
  if (probability >= 0.5) return { label: "LIKELY", className: "likely" };
  if (probability >= 0.3) return { label: "WATCH", className: "watch" };
  return { label: "LOW", className: "low" };
}

function cellsGeoJson(cells: Cell[], windowKey: WindowKey) {
  return {
    type: "FeatureCollection" as const,
    features: cells.map((cell) => {
      const boundary = cellToBoundary(cell.h3_cell).map(([lat, lng]) => [lng, lat]);
      return {
        type: "Feature" as const,
        geometry: { type: "Polygon" as const, coordinates: [[...boundary, boundary[0]]] },
        properties: {
          h3_cell: cell.h3_cell,
          probability: cell.stage1_probability,
          intensity: cell.stage2_probability,
          window_count: cell.strike_counts[windowKey] ?? 0,
          stage2_executed: Number(cell.stage2_executed),
          combined_prediction: cell.combined_prediction,
        },
      };
    }),
  };
}

function H3Map({
  cells,
  windowKey,
  fitKey,
  selectedCell,
  onSelect,
}: {
  cells: Cell[];
  windowKey: WindowKey;
  fitKey: string | null;
  selectedCell: string | null;
  onSelect: (cell: string) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const lastFitKeyRef = useRef("");
  const onSelectRef = useRef(onSelect);

  useEffect(() => {
    onSelectRef.current = onSelect;
  }, [onSelect]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      center: [-81.53, 28.42],
      zoom: 6.7,
      minZoom: 1.5,
      attributionControl: false,
      style: {
        version: 8,
        sources: {
          osm: {
            type: "raster",
            tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            attribution: "© OpenStreetMap contributors",
          },
        },
        layers: [{
          id: "osm",
          type: "raster",
          source: "osm",
          paint: {
            "raster-saturation": -0.96,
            "raster-brightness-min": 0.05,
            "raster-brightness-max": 0.31,
            "raster-contrast": 0.34,
          },
        }],
      },
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-left");
    map.on("load", () => {
      map.addSource("h3-cells", { type: "geojson", data: cellsGeoJson([], "5m") });
      map.addLayer({
        id: "h3-glow",
        type: "line",
        source: "h3-cells",
        paint: {
          "line-color": [
            "case",
            [">=", ["get", "probability"], 0.8], "#ff2448",
            [">=", ["get", "probability"], 0.5], "#ff654f",
            [">=", ["get", "probability"], 0.3], "#f2bd43",
            "#37a8ff",
          ],
          "line-width": ["interpolate", ["linear"], ["zoom"], 3, 1, 9, 6],
          "line-opacity": 0.24,
          "line-blur": 4,
        },
      });
      map.addLayer({
        id: "h3-fill",
        type: "fill",
        source: "h3-cells",
        paint: {
          "fill-color": [
            "case",
            [">=", ["get", "probability"], 0.8], "#ff2448",
            [">=", ["get", "probability"], 0.5], "#ff654f",
            [">=", ["get", "probability"], 0.3], "#f2bd43",
            "#278fdb",
          ],
          "fill-opacity": [
            "interpolate", ["linear"], ["get", "window_count"],
            0, 0.25,
            1, 0.5,
            8, 0.8,
            30, 0.92,
          ],
        },
      });
      map.addLayer({
        id: "h3-outline",
        type: "line",
        source: "h3-cells",
        paint: {
          "line-color": ["case", ["==", ["get", "h3_cell"], ""], "#ffffff", "#dbe9ff"],
          "line-opacity": 0.78,
          "line-width": 1,
        },
      });
      map.on("click", "h3-fill", (event) => {
        const cell = event.features?.[0]?.properties?.h3_cell;
        if (cell) onSelectRef.current(String(cell));
      });
      map.on("mouseenter", "h3-fill", () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", "h3-fill", () => { map.getCanvas().style.cursor = ""; });
    });
    mapRef.current = map;
    return () => { map.remove(); mapRef.current = null; };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const update = () => {
      const source = map.getSource("h3-cells") as maplibregl.GeoJSONSource | undefined;
      source?.setData(cellsGeoJson(cells, windowKey));
      if (map.getLayer("h3-outline")) {
        map.setPaintProperty("h3-outline", "line-color", [
          "case",
          ["==", ["get", "h3_cell"], selectedCell ?? ""], "#ffffff",
          "#dbe9ff",
        ]);
        map.setPaintProperty("h3-outline", "line-width", [
          "case",
          ["==", ["get", "h3_cell"], selectedCell ?? ""], 3,
          1,
        ]);
      }
      if (fitKey && fitKey !== lastFitKeyRef.current && cells.length) {
        const focus = [...cells].sort((a, b) => {
          const countDelta = (b.strike_counts[windowKey] ?? 0) - (a.strike_counts[windowKey] ?? 0);
          return countDelta || b.stage1_probability - a.stage1_probability;
        })[0];
        const [lat, lng] = cellToLatLng(focus.h3_cell);
        map.easeTo({ center: [lng, lat], zoom: 7.4, duration: 900 });
        lastFitKeyRef.current = fitKey;
      }
    };
    if (map.isStyleLoaded()) update(); else map.once("load", update);
  }, [cells, windowKey, fitKey, selectedCell]);

  useEffect(() => {
    if (!selectedCell || !mapRef.current) return;
    const [lat, lng] = cellToLatLng(selectedCell);
    mapRef.current.easeTo({ center: [lng, lat], zoom: Math.max(8.2, mapRef.current.getZoom()), duration: 650 });
  }, [selectedCell]);

  return <div ref={containerRef} className="map-canvas" aria-label="Interactive map of live H3 risk cells" />;
}

function metric(value: number, digits = 0) {
  return Number.isFinite(value)
    ? value.toLocaleString(undefined, { maximumFractionDigits: digits })
    : "—";
}

function utcTime(value: string) {
  return value ? `${value.slice(11, 19)}Z` : "—";
}

export default function Home() {
  const [mode, setMode] = useState<Mode>("live");
  const [windowKey, setWindowKey] = useState<WindowKey>("5m");
  const [snapshot, setSnapshot] = useState<Snapshot>(() => demoSnapshot());
  const [connected, setConnected] = useState(false);
  const [lastError, setLastError] = useState("");
  const [selectedCell, setSelectedCell] = useState<string | null>(null);
  const [focusPulse, setFocusPulse] = useState(0);

  useEffect(() => {
    if (mode === "demo") return;
    let cancelled = false;
    const load = async () => {
      try {
        const response = await fetch(API_URL, { cache: "no-store" });
        if (!response.ok) throw new Error(`API returned ${response.status}`);
        const next = await response.json() as Snapshot;
        if (!cancelled) {
          setSnapshot(next);
          setConnected(true);
          setLastError("");
        }
      } catch {
        if (!cancelled) {
          setConnected(false);
          setLastError("Live API unavailable. Start the private SSM tunnel on port 8765.");
        }
      }
    };
    load();
    const timer = window.setInterval(load, 2_000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [mode]);

  const sortedCells = useMemo(
    () => [...snapshot.cells].sort((a, b) => {
      const riskDelta = b.stage1_probability - a.stage1_probability;
      return Math.abs(riskDelta) > 0.01
        ? riskDelta
        : (b.strike_counts?.[windowKey] ?? 0) - (a.strike_counts?.[windowKey] ?? 0);
    }),
    [snapshot.cells, windowKey],
  );
  const selected = snapshot.cells.find((cell) => cell.h3_cell === selectedCell) ?? sortedCells[0] ?? null;
  const strikes = snapshot.summary.strikes_by_window?.[windowKey]
    ?? snapshot.summary[`strikes_${windowKey}` as "strikes_1m" | "strikes_5m" | "strikes_30m"]
    ?? 0;
  const likelyCells = snapshot.cells.filter((cell) => cell.stage1_probability >= 0.5).length;
  const fitKey = mode === "demo"
    ? `demo:${windowKey}:${focusPulse}`
    : connected ? `live:${windowKey}:${focusPulse}` : null;

  const activateDemo = () => {
    setMode("demo");
    setSnapshot(demoSnapshot());
    setConnected(false);
    setLastError("");
  };

  return (
    <main>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"><i /></span>
          <div>
            <strong>STORMSIGNAL</strong>
            <span>Geospatial prediction system</span>
          </div>
        </div>
        <div className="top-meta">
          <span>H3 RES 7</span>
          <span>15M FORECAST</span>
          <span>UPDATED {utcTime(snapshot.generated_at)}</span>
        </div>
        <div className={`status ${mode === "demo" ? "preview" : connected ? "online" : "offline"}`}>
          <i />{mode === "demo" ? "SCENARIO" : connected ? "LIVE" : "OFFLINE"}
        </div>
      </header>

      <section className="command-row">
        <div>
          <p className="eyebrow">REAL-TIME LIGHTNING INTELLIGENCE</p>
          <h1>Forecast surface</h1>
          <p className="lede">Kafka-streamed observations, H3 spatial features, and gated XGBoost inference.</p>
        </div>
        <div className="controls">
          <div className="segmented" role="group" aria-label="Data source">
            <button className={mode === "live" ? "active" : ""} onClick={() => setMode("live")}>AWS LIVE</button>
            <button className={mode === "demo" ? "active" : ""} onClick={activateDemo}>SCENARIO</button>
          </div>
          <button className="focus-button" onClick={() => setFocusPulse((value) => value + 1)}>FOCUS ACTIVITY</button>
        </div>
      </section>

      {lastError && mode === "live" && <div className="connection-note"><b>PRIVATE LINK OFFLINE</b><span>{lastError}</span></div>}

      <section className="metric-grid" aria-label="Prediction metrics">
        <article>
          <span>STRIKES / {windowKey.toUpperCase()}</span>
          <strong>{metric(strikes)}</strong>
          <small>{snapshot.summary.observed_cells ?? 0} observed cells</small>
        </article>
        <article>
          <span>LIKELY CELLS</span>
          <strong>{metric(likelyCells)}</strong>
          <small>Probability ≥ 50%</small>
        </article>
        <article className="accent-metric">
          <span>CANDIDATE GATE SKIP</span>
          <strong>{metric(snapshot.summary.stage2_skip_rate * 100, 1)}<em>%</em></strong>
          <small>{snapshot.summary.candidate_cells ?? 0} candidates evaluated</small>
        </article>
        <article>
          <span>INFERENCE P95</span>
          <strong>{metric(snapshot.summary.e2e_p95_ms, 1)}<em>ms</em></strong>
          <small>Ingestion → prediction</small>
        </article>
        <article>
          <span>DISPLAYED CELLS</span>
          <strong>{metric(snapshot.summary.active_cells)}</strong>
          <small>{metric(snapshot.summary.total_prediction_cells ?? snapshot.summary.active_cells)} recent predictions</small>
        </article>
      </section>

      <section className="window-row" aria-label="Observation window">
        <div><span className="kicker">OBSERVATION WINDOW</span><p>Reweight the surface by recent strike density.</p></div>
        <div className="window-switch">
          {WINDOWS.map((window) => (
            <button key={window.key} className={windowKey === window.key ? "active" : ""} onClick={() => setWindowKey(window.key)}>
              <strong>{window.label}</strong><span>{window.detail}</span>
            </button>
          ))}
        </div>
        <div className="risk-legend">
          <span><i className="low" />LOW</span>
          <span><i className="watch" />WATCH</span>
          <span><i className="likely" />LIKELY ≥50%</span>
          <span><i className="critical" />CRITICAL ≥80%</span>
        </div>
      </section>

      <section className="workspace">
        <article className="map-panel">
          <div className="panel-head">
            <div><span className="kicker">GEOSPATIAL RISK SURFACE</span><h2>Live H3 forecast cells</h2></div>
            <div className="panel-stat"><span>MODEL GATE</span><b>0.50</b></div>
          </div>
          <H3Map
            cells={snapshot.cells}
            windowKey={windowKey}
            fitKey={fitKey}
            selectedCell={selected?.h3_cell ?? null}
            onSelect={setSelectedCell}
          />
          <div className="map-caption">
            <span>CLICK A HEX CELL TO INSPECT</span>
            <span>RED = P(STRIKE) ≥ 50%</span>
          </div>
        </article>

        <aside className="intel-panel">
          <div className="panel-head"><div><span className="kicker">CELL INTELLIGENCE</span><h2>Selected forecast</h2></div></div>
          {selected && (
            <section className="selected-card">
              <div className="selected-top">
                <span className={`risk-chip ${riskLabel(selected.stage1_probability).className}`}>{riskLabel(selected.stage1_probability).label}</span>
                <span>{selected.event_type === "candidate" ? "CANDIDATE GRID" : "OBSERVED STRIKE"}</span>
              </div>
              <strong className="cell-id">{selected.h3_cell}</strong>
              <div className="probability-readout">
                <div><span>STRIKE PROBABILITY</span><strong>{Math.round(selected.stage1_probability * 100)}<em>%</em></strong></div>
                <div><span>INTENSITY</span><strong>{selected.stage2_executed ? Math.round(selected.stage2_probability * 100) : "—"}<em>{selected.stage2_executed ? "%" : ""}</em></strong></div>
              </div>
              <div className="window-counts">
                {WINDOWS.map((window) => <div key={window.key}><span>{window.label}</span><b>{selected.strike_counts?.[window.key] ?? 0}</b></div>)}
              </div>
              <div className={`gate-result ${selected.stage2_executed ? "passed" : "skipped"}`}>
                <i />
                <div><span>CASCADE DECISION</span><strong>{selected.stage2_executed ? "Stage 2 executed" : "Stage 2 skipped"}</strong></div>
                <b>{metric(selected.e2e_ms, 1)} ms</b>
              </div>
            </section>
          )}

          <div className="ranking-head"><span>HIGHEST SIGNAL</span><span>{windowKey.toUpperCase()} WINDOW</span></div>
          <div className="cell-list">
            {sortedCells.slice(0, 7).map((cell, index) => {
              const risk = riskLabel(cell.stage1_probability);
              return (
                <button className={`cell-row ${selected?.h3_cell === cell.h3_cell ? "selected" : ""}`} key={cell.h3_cell} onClick={() => setSelectedCell(cell.h3_cell)}>
                  <span className={`rank ${risk.className}`}>{String(index + 1).padStart(2, "0")}</span>
                  <span className="cell-main"><strong>{cell.h3_cell}</strong><small>{cell.strike_counts?.[windowKey] ?? 0} strikes · {cell.age_seconds.toFixed(0)}s ago</small></span>
                  <span className="prob"><strong>{Math.round(cell.stage1_probability * 100)}%</strong><small>{risk.label}</small></span>
                </button>
              );
            })}
          </div>
        </aside>
      </section>

      <section className="pipeline-strip">
        <div><span>01</span><b>INGEST</b><small>Blitzortung → MSK</small></div><i />
        <div><span>02</span><b>FEATURES</b><small>Redis temporal windows</small></div><i />
        <div><span>03</span><b>GATE</b><small>XGBoost activity filter</small></div><i />
        <div><span>04</span><b>INTENSITY</b><small>Conditional stage two</small></div>
      </section>

      <footer><span>STORMSIGNAL / ENGINEERING PROOF OF CONCEPT</span><span>NOT A SAFETY-GRADE WARNING SYSTEM</span></footer>
    </main>
  );
}
