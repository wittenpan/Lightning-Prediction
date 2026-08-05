"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import * as maplibregl from "maplibre-gl";
import type { Map as MapLibreMap } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { cellToBoundary, cellToLatLng, gridDisk, latLngToCell } from "h3-js";

type Cell = {
  h3_cell: string;
  resolution: number;
  stage1_probability: number;
  stage1_prediction: number;
  stage2_probability: number;
  stage2_prediction: number;
  stage2_executed: boolean;
  combined_prediction: number;
  strike_count_5m: number;
  processing_ms: number;
  e2e_ms: number;
  age_seconds: number;
  updated_at: string;
};

type Snapshot = {
  generated_at: string;
  summary: {
    active_cells: number;
    stage1_positive_cells: number;
    stage2_positive_cells: number;
    strikes_5m: number;
    stage2_skip_rate: number;
    e2e_p95_ms: number;
  };
  cells: Cell[];
};

const API_URL = "http://127.0.0.1:8765/api/state";
const DEMO_GENERATED_AT = "2026-08-05T16:00:00.000Z";

function demoSnapshot(): Snapshot {
  const origin = latLngToCell(28.42, -81.53, 7);
  const now = DEMO_GENERATED_AT;
  const cells = gridDisk(origin, 2).slice(0, 16).map((cell, index) => {
    const stage1 = Math.max(0.05, 0.94 - index * 0.061);
    const executed = stage1 >= 0.71;
    const stage2 = executed ? Math.max(0.08, 0.9 - index * 0.12) : 0;
    return {
      h3_cell: cell,
      resolution: 7,
      stage1_probability: stage1,
      stage1_prediction: Number(executed),
      stage2_probability: stage2,
      stage2_prediction: Number(stage2 >= 0.78),
      stage2_executed: executed,
      combined_prediction: Number(executed && stage2 >= 0.78),
      strike_count_5m: Math.max(0, 28 - index * 2),
      processing_ms: 4.8 + index * 0.34,
      e2e_ms: 10.2 + index * 0.83,
      age_seconds: index * 3.1,
      updated_at: now,
    };
  });
  const stage2Calls = cells.filter((cell) => cell.stage2_executed).length;
  return {
    generated_at: now,
    summary: {
      active_cells: cells.length,
      stage1_positive_cells: stage2Calls,
      stage2_positive_cells: cells.filter((cell) => cell.stage2_prediction).length,
      strikes_5m: cells.reduce((sum, cell) => sum + cell.strike_count_5m, 0),
      stage2_skip_rate: 1 - stage2Calls / cells.length,
      e2e_p95_ms: 18.6,
    },
    cells,
  };
}

function cellsGeoJson(cells: Cell[]) {
  return {
    type: "FeatureCollection" as const,
    features: cells.map((cell) => {
      const boundary = cellToBoundary(cell.h3_cell).map(([lat, lng]) => [lng, lat]);
      return {
        type: "Feature" as const,
        geometry: {
          type: "Polygon" as const,
          coordinates: [[...boundary, boundary[0]]],
        },
        properties: {
          ...cell,
          activity: Math.round(cell.stage1_probability * 100),
          intensity: Math.round(cell.stage2_probability * 100),
        },
      };
    }),
  };
}

function H3Map({ cells }: { cells: Cell[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const fittedRef = useRef(false);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      center: [-81.53, 28.42],
      zoom: 7.2,
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
        layers: [
          { id: "osm", type: "raster", source: "osm", paint: { "raster-saturation": -0.88, "raster-brightness-max": 0.43, "raster-contrast": 0.2 } },
        ],
      },
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
    map.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-left");
    map.on("load", () => {
      map.addSource("h3-cells", { type: "geojson", data: cellsGeoJson([]) });
      map.addLayer({
        id: "h3-fill",
        type: "fill",
        source: "h3-cells",
        paint: {
          "fill-color": [
            "case",
            ["==", ["get", "combined_prediction"], 1], "#ff4d5f",
            ["==", ["get", "stage1_prediction"], 1], "#ffb72e",
            ["interpolate", ["linear"], ["get", "stage1_probability"], 0, "#182b47", 0.7, "#3d65ed"],
          ],
          "fill-opacity": ["interpolate", ["linear"], ["get", "stage1_probability"], 0, 0.28, 1, 0.82],
        },
      });
      map.addLayer({
        id: "h3-outline",
        type: "line",
        source: "h3-cells",
        paint: { "line-color": "#d9e8ff", "line-opacity": 0.62, "line-width": 1.1 },
      });
      map.on("click", "h3-fill", (event) => {
        const props = event.features?.[0]?.properties;
        if (!props) return;
        new maplibregl.Popup({ closeButton: false, offset: 8 })
          .setLngLat(event.lngLat)
          .setHTML(`<div class="map-popup"><b>${props.h3_cell}</b><span>Activity ${props.activity}%</span><span>Intensity ${props.intensity}%</span><span>${props.strike_count_5m} strikes / 5m</span></div>`)
          .addTo(map);
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
      source?.setData(cellsGeoJson(cells));
      if (cells.length && !fittedRef.current) {
        const bounds = new maplibregl.LngLatBounds();
        cells.forEach((cell) => {
          const [lat, lng] = cellToLatLng(cell.h3_cell);
          bounds.extend([lng, lat]);
        });
        map.fitBounds(bounds, { padding: 80, maxZoom: 10.5, duration: 800 });
        fittedRef.current = true;
      }
    };
    if (map.isStyleLoaded()) update(); else map.once("load", update);
  }, [cells]);

  return <div ref={containerRef} className="map-canvas" aria-label="Map of live H3 prediction cells" />;
}

function metric(value: number, digits = 0) {
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function utcTime(value: string) {
  return `${value.slice(11, 19)} UTC`;
}

export default function Home() {
  const [mode, setMode] = useState<"live" | "demo">("demo");
  const [snapshot, setSnapshot] = useState<Snapshot>(() => demoSnapshot());
  const [connected, setConnected] = useState(false);
  const [lastError, setLastError] = useState("");

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
          setLastError(window.location.protocol === "https:"
            ? "Live AWS mode runs at localhost:3000 with the SSM tunnel; hosted preview stays self-contained."
            : "Start the SSM tunnel to connect live AWS state.");
        }
      }
    };
    load();
    const timer = window.setInterval(load, 2_000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [mode]);

  const sortedCells = useMemo(
    () => [...snapshot.cells].sort((a, b) => b.stage1_probability - a.stage1_probability),
    [snapshot.cells],
  );

  return (
    <main>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">ϟ</span>
          <div><strong>StormSignal</strong><span>Lightning intelligence</span></div>
        </div>
        <div className="source-switch" role="group" aria-label="Dashboard data source">
          <button className={mode === "demo" ? "active" : ""} onClick={() => {
            setMode("demo");
            setSnapshot(demoSnapshot());
            setConnected(false);
            setLastError("");
          }}>Preview</button>
          <button className={mode === "live" ? "active" : ""} onClick={() => setMode("live")}>Live AWS</button>
        </div>
        <div className={`status ${mode === "demo" ? "preview" : connected ? "online" : "offline"}`}>
          <i />{mode === "demo" ? "Simulated feed" : connected ? "Streaming" : "Tunnel offline"}
        </div>
      </header>

      <section className="hero-row">
        <div>
          <p className="eyebrow">15-minute horizon · H3 resolution 7</p>
          <h1>Live convective activity,<br />cell by cell.</h1>
        </div>
        <div className="pipeline">
          <span>Kafka ingest</span><b>→</b><span>Redis features</span><b>→</b><span>XGBoost cascade</span>
        </div>
      </section>

      {lastError && mode === "live" && <div className="connection-note"><span>Private connection required</span>{lastError}</div>}

      <section className="metrics" aria-label="Prediction metrics">
        <article><span>Strikes · 5 min</span><strong>{metric(snapshot.summary.strikes_5m)}</strong><small>Across active H3 cells</small></article>
        <article><span>Active cells</span><strong>{metric(snapshot.summary.active_cells)}</strong><small>{snapshot.summary.stage1_positive_cells} passed stage one</small></article>
        <article><span>Stage-two skip</span><strong>{metric(snapshot.summary.stage2_skip_rate * 100, 1)}%</strong><small>Compute gate efficiency</small></article>
        <article><span>Prediction p95</span><strong>{metric(snapshot.summary.e2e_p95_ms, 1)}<em>ms</em></strong><small>Ingestion to prediction</small></article>
      </section>

      <section className="workspace">
        <article className="map-panel">
          <div className="panel-head">
            <div><span className="kicker">Geospatial activity</span><h2>{mode === "demo" ? "Central Florida preview" : "Live H3 cell field"}</h2></div>
            <div className="legend"><span><i className="quiet" />Low signal</span><span><i className="active" />Activity</span><span><i className="intense" />Intensity</span></div>
          </div>
          <H3Map cells={snapshot.cells} />
          <div className="map-caption"><span>H3 / res 7</span><span>Updated {utcTime(snapshot.generated_at)}</span></div>
        </article>

        <aside className="cell-panel">
          <div className="panel-head"><div><span className="kicker">Highest signal</span><h2>Active cells</h2></div><span className="count">{sortedCells.length}</span></div>
          <div className="cell-list">
            {sortedCells.slice(0, 9).map((cell, index) => (
              <article className="cell-row" key={cell.h3_cell}>
                <div className={`rank ${cell.combined_prediction ? "danger" : cell.stage1_prediction ? "warn" : ""}`}>{String(index + 1).padStart(2, "0")}</div>
                <div className="cell-main"><strong>{cell.h3_cell}</strong><span>{cell.strike_count_5m} strikes · {cell.age_seconds.toFixed(0)}s ago</span></div>
                <div className="prob"><strong>{Math.round(cell.stage1_probability * 100)}%</strong><span>{cell.stage2_executed ? `S2 ${Math.round(cell.stage2_probability * 100)}%` : "S2 skipped"}</span></div>
              </article>
            ))}
          </div>
          <div className="cascade-card">
            <div><span>Stage one</span><strong>Activity gate</strong></div><b>0.71</b>
            <div className="cascade-line"><i style={{ width: `${Math.max(2, (1 - snapshot.summary.stage2_skip_rate) * 100)}%` }} /></div>
            <p>Only high-signal cells reach the intensity model.</p>
          </div>
        </aside>
      </section>

      <footer><span>Engineering PoC · Not a safety-grade warning system</span><span>MSK · ElastiCache · EC2 · XGBoost</span></footer>
    </main>
  );
}
