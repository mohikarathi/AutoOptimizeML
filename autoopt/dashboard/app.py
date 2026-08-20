"""Interactive Web Dashboard for AutoOptimizeML."""

import os
import json
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from autoopt.hardware import HardwareProfiler
from autoopt.utils.storage import load_run_artifact, list_runs, DEFAULT_RUNS_DIR


def create_dashboard_app(runs_dir: str = DEFAULT_RUNS_DIR) -> FastAPI:
    app = FastAPI(title="AutoOptimizeML Dashboard")

    @app.get("/api/hardware")
    async def get_hardware():
        p = HardwareProfiler.profile()
        return p.to_dict()

    @app.get("/api/runs")
    async def get_runs():
        return list_runs(runs_dir)

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str):
        data = load_run_artifact(run_id, runs_dir)
        if data is None:
            return JSONResponse(status_code=404, content={"error": f"Run {run_id} not found"})
        return data

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_page():
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoOptimizeML — Optimization & Deployment Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #334155;
            --text-primary: #f8fafc;
            --text-muted: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-green: #4ade80;
            --accent-purple: #c084fc;
            --accent-orange: #fb923c;
            --accent-red: #f87171;
            --border-color: #475569;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background-color: var(--bg-primary); color: var(--text-primary); padding: 24px; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border-color); }
        .header h1 { font-size: 24px; font-weight: 700; color: var(--accent-blue); display: flex; align-items: center; gap: 8px; }
        .header .subtitle { color: var(--text-muted); font-size: 14px; margin-top: 4px; }
        .run-selector { display: flex; gap: 12px; align-items: center; }
        select { background: var(--bg-secondary); color: var(--text-primary); border: 1px solid var(--border-color); padding: 8px 16px; border-radius: 6px; font-size: 14px; }
        .grid-4 { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 20px; margin-bottom: 24px; }
        .card { background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
        .card h3 { font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-muted); margin-bottom: 12px; display: flex; justify-content: space-between; }
        .stat-value { font-size: 28px; font-weight: 700; margin-bottom: 6px; }
        .stat-sub { font-size: 13px; color: var(--text-muted); }
        .highlight-green { color: var(--accent-green); }
        .highlight-blue { color: var(--accent-blue); }
        .highlight-purple { color: var(--accent-purple); }
        .highlight-orange { color: var(--accent-orange); }
        .badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; text-transform: uppercase; }
        .badge-accepted { background: rgba(74, 222, 128, 0.2); color: var(--accent-green); border: 1px solid rgba(74, 222, 128, 0.4); }
        .badge-rejected { background: rgba(251, 146, 60, 0.2); color: var(--accent-orange); border: 1px solid rgba(251, 146, 60, 0.4); }
        .badge-failed { background: rgba(248, 113, 113, 0.2); color: var(--accent-red); border: 1px solid rgba(248, 113, 113, 0.4); }
        table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }
        th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border-color); }
        th { color: var(--text-muted); font-weight: 600; }
        tr:hover { background: rgba(255,255,255,0.02); }
        .chart-container { position: relative; height: 280px; width: 100%; }
        .info-list { display: flex; flex-direction: column; gap: 8px; font-size: 14px; }
        .info-row { display: flex; justify-content: space-between; border-bottom: 1px dashed rgba(255,255,255,0.1); padding-bottom: 4px; }
        .info-label { color: var(--text-muted); }
        .info-val { font-weight: 600; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>⚡ AutoOptimizeML Platform</h1>
            <div class="subtitle">Hardware-Agnostic Model Profiling, Bottleneck Analysis & Execution Discovery</div>
        </div>
        <div class="run-selector">
            <label for="runSelect" style="font-size: 14px; color: var(--text-muted);">Optimization Run:</label>
            <select id="runSelect" onchange="loadRunDetails(this.value)">
                <option value="latest">Latest Run</option>
            </select>
        </div>
    </div>

    <!-- Top Key Metric Cards -->
    <div class="grid-4">
        <div class="card">
            <h3>Throughput Speedup</h3>
            <div class="stat-value highlight-green" id="throughputGain">+0.0%</div>
            <div class="stat-sub" id="throughputCompare">Baseline: 0 → Optimized: 0 samples/s</div>
        </div>
        <div class="card">
            <h3>Latency Reduction</h3>
            <div class="stat-value highlight-blue" id="latencyReduction">0.0%</div>
            <div class="stat-sub" id="latencyCompare">Baseline: 0 ms → Optimized: 0 ms</div>
        </div>
        <div class="card">
            <h3>Accuracy Retention</h3>
            <div class="stat-value highlight-purple" id="accuracyVal">100.0%</div>
            <div class="stat-sub" id="accuracyDelta">Accuracy delta: +0.0%</div>
        </div>
        <div class="card">
            <h3>Primary Bottleneck</h3>
            <div class="stat-value highlight-orange" style="font-size: 18px;" id="bottleneckName">Analyzing...</div>
            <div class="stat-sub" id="bottleneckDesc">Profiling stage distribution</div>
        </div>
    </div>

    <!-- Middle: System & Model Info + Bottleneck Breakdown -->
    <div class="grid-2">
        <div class="card">
            <h3>Hardware & Model Topology</h3>
            <div class="info-list" style="margin-top: 8px;">
                <div class="info-row"><span class="info-label">Host CPU</span><span class="info-val" id="cpuName">-</span></div>
                <div class="info-row"><span class="info-label">CPU Cores</span><span class="info-val" id="cpuCores">-</span></div>
                <div class="info-row"><span class="info-label">Hardware Accelerators</span><span class="info-val" id="gpuName">-</span></div>
                <div class="info-row"><span class="info-label">Model Framework</span><span class="info-val" id="modelFw">-</span></div>
                <div class="info-row"><span class="info-label">Model Architecture</span><span class="info-val" id="modelType">-</span></div>
                <div class="info-row"><span class="info-label">Total Parameters</span><span class="info-val" id="modelParams">-</span></div>
                <div class="info-row"><span class="info-label">Best Configuration</span><span class="info-val highlight-green" id="bestConfig">-</span></div>
            </div>
        </div>
        <div class="card">
            <h3>Baseline Stage Timing Breakdown</h3>
            <div class="chart-container">
                <canvas id="stageChart"></canvas>
            </div>
        </div>
    </div>

    <!-- Charts: Batch Size Scaling & Search Space Pareto -->
    <div class="grid-2">
        <div class="card">
            <h3>Batch Size vs Throughput Scaling</h3>
            <div class="chart-container">
                <canvas id="throughputChart"></canvas>
            </div>
        </div>
        <div class="card">
            <h3>Batch Size vs Latency</h3>
            <div class="chart-container">
                <canvas id="latencyChart"></canvas>
            </div>
        </div>
    </div>

    <!-- Candidates Table -->
    <div class="card">
        <h3>Optimization Candidate Evaluations</h3>
        <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Candidate ID</th>
                        <th>Device</th>
                        <th>Precision</th>
                        <th>Batch Size</th>
                        <th>Workers</th>
                        <th>JIT / Native</th>
                        <th>Latency (ms)</th>
                        <th>Throughput (samp/s)</th>
                        <th>Accuracy</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="candidatesTableBody">
                    <tr><td colspan="10" style="text-align: center; color: var(--text-muted);">Loading candidates...</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        let stageChart = null;
        let throughputChart = null;
        let latencyChart = null;

        async function init() {
            try {
                const res = await fetch('/api/runs');
                const runs = await res.json();
                const sel = document.getElementById('runSelect');
                sel.innerHTML = '<option value="latest">Latest Run</option>';
                runs.forEach(r => {
                    const opt = document.createElement('option');
                    opt.value = r.replace('.json', '');
                    opt.textContent = r.replace('.json', '');
                    sel.appendChild(opt);
                });
            } catch(e) {}
            loadRunDetails('latest');
        }

        async function loadRunDetails(runId) {
            try {
                const res = await fetch(`/api/runs/${runId}`);
                if (!res.ok) return;
                const data = await res.json();
                renderData(data);
            } catch(e) {
                console.error(e);
            }
        }

        function renderData(data) {
            // Metrics
            const imp = data.improvement || {};
            const base = data.baseline || {};
            const bestM = data.best_metrics || {};
            const bestC = data.best_configuration || {};
            const model = data.model || {};
            const hw = data.hardware || {};
            const bneck = data.bottleneck || {};

            document.getElementById('throughputGain').textContent = `+${imp.throughput_gain_pct || 0}%`;
            document.getElementById('throughputCompare').textContent = `Baseline: ${base.throughput_samples_per_sec || 0} → Optimized: ${bestM.throughput_samples_per_sec || 0} samp/s`;

            document.getElementById('latencyReduction').textContent = `${imp.latency_reduction_pct || 0}% faster`;
            document.getElementById('latencyCompare').textContent = `Baseline: ${base.total_latency_ms || 0} ms → Optimized: ${bestM.mean_latency_ms || 0} ms`;

            document.getElementById('accuracyVal').textContent = `${((bestM.accuracy || base.accuracy || 1.0) * 100).toFixed(1)}%`;
            document.getElementById('accuracyDelta').textContent = `Accuracy delta: ${imp.accuracy_delta_pct >= 0 ? '+' : ''}${imp.accuracy_delta_pct || 0}%`;

            document.getElementById('bottleneckName').textContent = bneck.primary_bottleneck || 'Balanced';
            document.getElementById('bottleneckDesc').textContent = (bneck.primary_bottleneck_description || '').substring(0, 50) + '...';

            document.getElementById('cpuName').textContent = hw.cpu ? hw.cpu.model_name : 'Host CPU';
            document.getElementById('cpuCores').textContent = hw.cpu ? `${hw.cpu.physical_cores} Physical / ${hw.cpu.logical_cores} Logical (${hw.cpu.total_ram_gb} GB RAM)` : '-';
            document.getElementById('gpuName').textContent = hw.gpus && hw.gpus.length ? hw.gpus.map(g => g.device_name).join(', ') : 'CPU Host Execution';
            document.getElementById('modelFw').textContent = (model.framework || '').toUpperCase();
            document.getElementById('modelType').textContent = model.model_type || '-';
            document.getElementById('modelParams').textContent = (model.parameters || 0).toLocaleString() + ` (${model.model_size_mb || 0} MB)`;

            const jit = bestC.compile_graph ? ' + JIT' : '';
            const native = bestC.native_preprocessing ? ' + Native' : '';
            document.getElementById('bestConfig').textContent = bestC.device ? `${bestC.device.toUpperCase()} | ${bestC.precision.toUpperCase()} | Batch=${bestC.batch_size} | Workers=${bestC.workers}${jit}${native}` : 'No valid candidate';

            // Render Stage Chart
            renderStageChart(bneck.stage_percentages || {});

            // Render Candidates Table and Trend Charts
            renderCandidatesAndTrends(data.evaluations || []);
        }

        function renderStageChart(stages) {
            const ctx = document.getElementById('stageChart').getContext('2d');
            if (stageChart) stageChart.destroy();

            stageChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Preprocessing', 'H2D Transfer', 'Inference Compute', 'D2H Transfer', 'Postprocessing'],
                    datasets: [{
                        data: [
                            stages.preprocessing || 0,
                            stages.h2d_transfer || 0,
                            stages.inference || 0,
                            stages.d2h_transfer || 0,
                            stages.postprocessing || 0
                        ],
                        backgroundColor: ['#38bdf8', '#fb923c', '#4ade80', '#c084fc', '#f87171']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'right', labels: { color: '#94a3b8' } }
                    }
                }
            });
        }

        function renderCandidatesAndTrends(evals) {
            const tbody = document.getElementById('candidatesTableBody');
            tbody.innerHTML = '';

            const batchMap = {};

            evals.forEach(ev => {
                const c = ev.candidate;
                const m = ev.metrics;
                const tr = document.createElement('tr');

                let badgeClass = 'badge-rejected';
                if (ev.status === 'ACCEPTED') badgeClass = 'badge-accepted';
                else if (ev.status === 'FAILED') badgeClass = 'badge-failed';

                const jit = c.compile_graph ? 'JIT' : '';
                const nat = c.native_preprocessing ? 'Native' : '';
                const accel = [jit, nat].filter(Boolean).join(' + ') || 'Standard';

                tr.innerHTML = `
                    <td style="font-weight:600;">${c.candidate_id}</td>
                    <td>${c.device.toUpperCase()}</td>
                    <td>${c.precision.toUpperCase()}</td>
                    <td>${c.batch_size}</td>
                    <td>${c.workers}</td>
                    <td>${accel}</td>
                    <td>${m ? m.mean_latency_ms.toFixed(2) : '-'}</td>
                    <td style="font-weight:600; color: var(--accent-green);">${m ? m.throughput_samples_per_sec.toFixed(1) : '-'}</td>
                    <td>${m ? (m.accuracy * 100).toFixed(1) + '%' : '-'}</td>
                    <td><span class="badge ${badgeClass}">${ev.status}</span></td>
                `;
                tbody.appendChild(tr);

                if (m && ev.status === 'ACCEPTED') {
                    if (!batchMap[c.batch_size] || batchMap[c.batch_size].throughput < m.throughput_samples_per_sec) {
                        batchMap[c.batch_size] = {
                            throughput: m.throughput_samples_per_sec,
                            latency: m.mean_latency_ms
                        };
                    }
                }
            });

            // Trend Charts
            const sortedBatches = Object.keys(batchMap).map(Number).sort((a,b) => a - b);
            const tpData = sortedBatches.map(b => batchMap[b].throughput);
            const latData = sortedBatches.map(b => batchMap[b].latency);

            // Throughput Chart
            const ctxTp = document.getElementById('throughputChart').getContext('2d');
            if (throughputChart) throughputChart.destroy();
            throughputChart = new Chart(ctxTp, {
                type: 'line',
                data: {
                    labels: sortedBatches.map(b => `Batch ${b}`),
                    datasets: [{
                        label: 'Throughput (samples/s)',
                        data: tpData,
                        borderColor: '#4ade80',
                        backgroundColor: 'rgba(74, 222, 128, 0.1)',
                        fill: true,
                        tension: 0.3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
                        y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } }
                    },
                    plugins: { legend: { labels: { color: '#94a3b8' } } }
                }
            });

            // Latency Chart
            const ctxLat = document.getElementById('latencyChart').getContext('2d');
            if (latencyChart) latencyChart.destroy();
            latencyChart = new Chart(ctxLat, {
                type: 'line',
                data: {
                    labels: sortedBatches.map(b => `Batch ${b}`),
                    datasets: [{
                        label: 'Latency (ms)',
                        data: latData,
                        borderColor: '#38bdf8',
                        backgroundColor: 'rgba(56, 189, 248, 0.1)',
                        fill: true,
                        tension: 0.3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
                        y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } }
                    },
                    plugins: { legend: { labels: { color: '#94a3b8' } } }
                }
            });
        }

        window.onload = init;
    </script>
</body>
</html>"""
        return HTMLResponse(content=html_content)

    return app
