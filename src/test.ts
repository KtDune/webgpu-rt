import { vec3, mat4 } from "gl-matrix";
import { ArcballCamera } from "./class/ArcballCamera";
import { Controller } from "./class/Controller";
import { uploadGLBModel, Triangle } from "./uploadGlb";

// ─── Shaders ──────────────────────────────────────────────────────────────────

const VERTEX_SHADER = /* wgsl */`
    struct ViewParams {
        projView: mat4x4<f32>,
    }
    struct VertexInput {
        @location(0) position: vec3<f32>,
        @location(1) normal:   vec3<f32>,
        @location(2) uv:       vec2<f32>,
    }
    struct VertexOutput {
        @builtin(position) Position: vec4<f32>,
        @location(0) uv:     vec2<f32>,
        @location(1) normal: vec3<f32>,
    }
    @group(0) @binding(0) var<uniform> viewParams: ViewParams;
    @vertex
    fn vs_main(in: VertexInput) -> VertexOutput {
        var out: VertexOutput;
        out.Position = viewParams.projView * vec4<f32>(in.position, 1.0);
        out.uv       = in.uv;
        out.normal   = in.normal;
        return out;
    }
`;

const FRAGMENT_SHADER_TEXTURED = /* wgsl */`
    @group(0) @binding(1) var base_color_sampler: sampler;
    @group(0) @binding(2) var base_color_texture: texture_2d<f32>;
    struct VertexOutput {
        @builtin(position) Position: vec4<f32>,
        @location(0) uv:     vec2<f32>,
        @location(1) normal: vec3<f32>,
    }
    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
        let lightDir  = normalize(vec3<f32>(1.0, 1.0, 1.0));
        let diff      = max(dot(normalize(in.normal), lightDir), 0.0);
        let texColor  = textureSample(base_color_texture, base_color_sampler, in.uv).rgb;
        let shaded    = texColor * (0.2 + 0.8 * diff);
        return vec4<f32>(shaded, 1.0);
    }
`;

const FRAGMENT_SHADER_FALLBACK = /* wgsl */`
    struct VertexOutput {
        @builtin(position) Position: vec4<f32>,
        @location(0) uv:     vec2<f32>,
        @location(1) normal: vec3<f32>,
    }
    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
        let lightDir = normalize(vec3<f32>(1.0, 1.0, 1.0));
        let diff     = max(dot(normalize(in.normal), lightDir), 0.0);
        return vec4<f32>(1.0, 0.0, 0.0, 1.0) * (0.3 + 0.7 * diff);
    }
`;

// ─── Geometry Validation ──────────────────────────────────────────────────────

interface ValidationResult {
    label: string;
    pass: boolean;
    details: string;
}

interface GeometryReport {
    triangleCount: number;
    checks: ValidationResult[];
    overallPass: boolean;
    stats: {
        posMin: [number, number, number];
        posMax: [number, number, number];
        normalLengthMin: number;
        normalLengthMax: number;
        uvMin: [number, number];
        uvMax: [number, number];
        nanPositions: number;
        nanNormals: number;
        nanUVs: number;
        denormalizedNormals: number;
        outOfRangeUVs: number;
        degenerateTriangles: number;
    };
}

function validateGeometry(triangles: Triangle[]): GeometryReport {
    const stats = {
        posMin: [Infinity, Infinity, Infinity] as [number, number, number],
        posMax: [-Infinity, -Infinity, -Infinity] as [number, number, number],
        normalLengthMin: Infinity,
        normalLengthMax: -Infinity,
        uvMin: [Infinity, Infinity] as [number, number],
        uvMax: [-Infinity, -Infinity] as [number, number],
        nanPositions: 0,
        nanNormals: 0,
        nanUVs: 0,
        denormalizedNormals: 0,
        outOfRangeUVs: 0,
        degenerateTriangles: 0,
    };

    for (const tri of triangles) {
        // ── positions ────────────────────────────────────────────────────────
        for (const pos of tri.positions) {
            const [x, y, z] = pos;
            if (!isFinite(x) || !isFinite(y) || !isFinite(z)) {
                stats.nanPositions++;
            } else {
                stats.posMin[0] = Math.min(stats.posMin[0], x);
                stats.posMin[1] = Math.min(stats.posMin[1], y);
                stats.posMin[2] = Math.min(stats.posMin[2], z);
                stats.posMax[0] = Math.max(stats.posMax[0], x);
                stats.posMax[1] = Math.max(stats.posMax[1], y);
                stats.posMax[2] = Math.max(stats.posMax[2], z);
            }
        }

        // ── normals ───────────────────────────────────────────────────────────
        for (const n of tri.normals) {
            const [nx, ny, nz] = n;
            if (!isFinite(nx) || !isFinite(ny) || !isFinite(nz)) {
                stats.nanNormals++;
            } else {
                const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
                stats.normalLengthMin = Math.min(stats.normalLengthMin, len);
                stats.normalLengthMax = Math.max(stats.normalLengthMax, len);
                if (Math.abs(len - 1.0) > 0.01) stats.denormalizedNormals++;
            }
        }

        // ── UVs ───────────────────────────────────────────────────────────────
        for (const uv of tri.uv) {
            const [u, v] = uv;
            if (!isFinite(u) || !isFinite(v)) {
                stats.nanUVs++;
            } else {
                stats.uvMin[0] = Math.min(stats.uvMin[0], u);
                stats.uvMin[1] = Math.min(stats.uvMin[1], v);
                stats.uvMax[0] = Math.max(stats.uvMax[0], u);
                stats.uvMax[1] = Math.max(stats.uvMax[1], v);
                // UVs outside [0,1] are not errors per se (tiling), but flag extreme values
                if (u < -10 || u > 11 || v < -10 || v > 11) stats.outOfRangeUVs++;
            }
        }

        // ── degenerate triangles (zero area) ──────────────────────────────────
        const a = tri.positions[0];
        const b = tri.positions[1];
        const c = tri.positions[2];
        const ab = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
        const ac = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
        const cross = [
            ab[1]*ac[2] - ab[2]*ac[1],
            ab[2]*ac[0] - ab[0]*ac[2],
            ab[0]*ac[1] - ab[1]*ac[0],
        ];
        const area = Math.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2) * 0.5;
        if (area < 1e-10) stats.degenerateTriangles++;
    }

    const fmt3 = (v: [number,number,number]) =>
        `(${v[0].toFixed(3)}, ${v[1].toFixed(3)}, ${v[2].toFixed(3)})`;

    const checks: ValidationResult[] = [
        {
            label: "No NaN/Inf positions",
            pass: stats.nanPositions === 0,
            details: stats.nanPositions === 0
                ? `All ${triangles.length * 3} vertices finite`
                : `${stats.nanPositions} vertices with NaN or Inf`,
        },
        {
            label: "No NaN/Inf normals",
            pass: stats.nanNormals === 0,
            details: stats.nanNormals === 0
                ? `All normals finite`
                : `${stats.nanNormals} normals with NaN or Inf`,
        },
        {
            label: "No NaN/Inf UVs",
            pass: stats.nanUVs === 0,
            details: stats.nanUVs === 0
                ? `All UVs finite`
                : `${stats.nanUVs} UVs with NaN or Inf`,
        },
        {
            label: "Normals are unit length",
            pass: stats.denormalizedNormals === 0,
            details: stats.denormalizedNormals === 0
                ? `Length range [${stats.normalLengthMin.toFixed(4)}, ${stats.normalLengthMax.toFixed(4)}]`
                : `${stats.denormalizedNormals} normals not unit-length (range [${stats.normalLengthMin.toFixed(4)}, ${stats.normalLengthMax.toFixed(4)}])`,
        },
        {
            label: "No degenerate triangles",
            pass: stats.degenerateTriangles === 0,
            details: stats.degenerateTriangles === 0
                ? `All triangles have non-zero area`
                : `${stats.degenerateTriangles} zero-area triangles`,
        },
        {
            label: "UVs in sane range",
            pass: stats.outOfRangeUVs === 0,
            details: stats.outOfRangeUVs === 0
                ? `UV range [${stats.uvMin[0].toFixed(3)}, ${stats.uvMax[0].toFixed(3)}] × [${stats.uvMin[1].toFixed(3)}, ${stats.uvMax[1].toFixed(3)}]`
                : `${stats.outOfRangeUVs} UVs outside [-10, 11] — possible data corruption`,
        },
        {
            label: "Position bounds sane",
            pass: stats.posMin[0] !== Infinity,
            details: `Min: ${fmt3(stats.posMin)}  Max: ${fmt3(stats.posMax)}`,
        },
    ];

    return {
        triangleCount: triangles.length,
        checks,
        overallPass: checks.every(c => c.pass),
        stats,
    };
}

// ─── Diagnostic UI ────────────────────────────────────────────────────────────

function buildDiagnosticUI(): {
    panel: HTMLElement;
    setStatus: (msg: string, type: "loading" | "ok" | "warn" | "error") => void;
    showReport: (report: GeometryReport, hasTex: boolean) => void;
} {
    // inject fonts + global styles
    const style = document.createElement("style");
    style.textContent = `
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@700;800&display=swap');

        :root {
            --bg: #0a0c0f;
            --panel-bg: #0e1117;
            --border: #1e2530;
            --border-bright: #2d3748;
            --text: #c9d1d9;
            --dim: #4a5568;
            --pass: #39d353;
            --fail: #f85149;
            --warn: #e3b341;
            --info: #58a6ff;
            --accent: #58a6ff;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            background: var(--bg);
            font-family: 'JetBrains Mono', monospace;
            color: var(--text);
            min-height: 100vh;
        }

        #diag-panel {
            position: fixed;
            top: 0; right: 0;
            width: 360px;
            height: 100vh;
            background: var(--panel-bg);
            border-left: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            z-index: 100;
            font-size: 11px;
        }

        #diag-header {
            padding: 18px 20px 14px;
            border-bottom: 1px solid var(--border);
            flex-shrink: 0;
        }

        #diag-header h1 {
            font-family: 'Syne', sans-serif;
            font-size: 13px;
            font-weight: 800;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            color: var(--accent);
            margin-bottom: 4px;
        }

        #diag-subtitle {
            color: var(--dim);
            font-size: 10px;
            letter-spacing: 0.05em;
        }

        #diag-status-bar {
            padding: 10px 20px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 8px;
            flex-shrink: 0;
            min-height: 40px;
        }

        .status-dot {
            width: 7px; height: 7px;
            border-radius: 50%;
            flex-shrink: 0;
            animation: none;
        }
        .status-dot.loading { background: var(--warn); animation: pulse 1s ease-in-out infinite; }
        .status-dot.ok      { background: var(--pass); }
        .status-dot.warn    { background: var(--warn); }
        .status-dot.error   { background: var(--fail); }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50%       { opacity: 0.3; }
        }

        #diag-status-text { color: var(--text); flex: 1; line-height: 1.4; }

        #diag-body {
            flex: 1;
            overflow-y: auto;
            padding: 16px 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        #diag-body::-webkit-scrollbar { width: 4px; }
        #diag-body::-webkit-scrollbar-track { background: transparent; }
        #diag-body::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 2px; }

        .diag-section-title {
            font-family: 'Syne', sans-serif;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--dim);
            margin-bottom: 8px;
        }

        .check-row {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            padding: 8px 10px;
            border-radius: 5px;
            border: 1px solid transparent;
            transition: border-color 0.2s;
        }
        .check-row.pass { border-color: #1a2f1e; background: #0d1f10; }
        .check-row.fail { border-color: #2f1a1a; background: #1f0d0d; }

        .check-icon {
            font-size: 12px;
            line-height: 1;
            margin-top: 1px;
            flex-shrink: 0;
        }

        .check-content { flex: 1; min-width: 0; }

        .check-label {
            font-weight: 600;
            color: var(--text);
            margin-bottom: 2px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .check-row.fail .check-label { color: var(--fail); }

        .check-detail {
            color: var(--dim);
            font-size: 10px;
            line-height: 1.5;
            word-break: break-all;
        }
        .check-row.pass .check-detail { color: #3d6b47; }

        .summary-box {
            padding: 12px 14px;
            border-radius: 6px;
            border: 1px solid;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .summary-box.pass { border-color: #1a3a20; background: #0d1f10; }
        .summary-box.fail { border-color: #3a1a1a; background: #1f0d0d; }

        .summary-icon { font-size: 22px; }

        .summary-text-main {
            font-family: 'Syne', sans-serif;
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }
        .summary-box.pass .summary-text-main { color: var(--pass); }
        .summary-box.fail .summary-text-main { color: var(--fail); }

        .summary-text-sub { color: var(--dim); font-size: 10px; margin-top: 2px; }

        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
        }

        .stat-cell {
            padding: 8px 10px;
            background: #0a0d12;
            border: 1px solid var(--border);
            border-radius: 4px;
        }

        .stat-key { color: var(--dim); font-size: 9px; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 3px; }
        .stat-val { color: var(--info); font-size: 11px; font-weight: 600; }

        .mat-row {
            display: flex; align-items: center; gap: 8px;
            padding: 7px 10px;
            background: #0a0d12;
            border: 1px solid var(--border);
            border-radius: 4px;
        }
        .mat-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
        .mat-label { color: var(--text); }
        .mat-value { color: var(--dim); margin-left: auto; font-size: 10px; }

        #webgpu-canvas {
            position: fixed;
            top: 0; left: 0;
            width: calc(100vw - 360px);
            height: 100vh;
        }
    `;
    document.head.appendChild(style);

    // resize canvas to not overlap panel
    const canvas = document.getElementById("webgpu-canvas") as HTMLCanvasElement;
    if (canvas) {
        canvas.style.cssText = "position:fixed;top:0;left:0;width:calc(100vw - 360px);height:100vh;";
    }

    // build panel
    const panel = document.createElement("div");
    panel.id = "diag-panel";
    panel.innerHTML = `
        <div id="diag-header">
            <h1>Geometry Inspector</h1>
            <div id="diag-subtitle">WebGPU · GLB Validation</div>
        </div>
        <div id="diag-status-bar">
            <div class="status-dot loading" id="status-dot"></div>
            <div id="diag-status-text">Initializing…</div>
        </div>
        <div id="diag-body"></div>
    `;
    document.body.appendChild(panel);

    const dot = panel.querySelector("#status-dot") as HTMLElement;
    const statusText = panel.querySelector("#diag-status-text") as HTMLElement;
    const body = panel.querySelector("#diag-body") as HTMLElement;

    function setStatus(msg: string, type: "loading" | "ok" | "warn" | "error") {
        dot.className = `status-dot ${type}`;
        statusText.textContent = msg;
    }

    function showReport(report: GeometryReport, hasTex: boolean) {
        body.innerHTML = "";

        // ── summary ───────────────────────────────────────────────────────────
        const summarySection = document.createElement("div");
        summarySection.innerHTML = `
            <div class="diag-section-title">Overall</div>
            <div class="summary-box ${report.overallPass ? "pass" : "fail"}">
                <div class="summary-icon">${report.overallPass ? "✓" : "✗"}</div>
                <div>
                    <div class="summary-text-main">${report.overallPass ? "All checks passed" : "Issues found"}</div>
                    <div class="summary-text-sub">${report.triangleCount.toLocaleString()} triangles · ${(report.triangleCount * 3).toLocaleString()} vertices</div>
                </div>
            </div>
        `;
        body.appendChild(summarySection);

        // ── checks ────────────────────────────────────────────────────────────
        const checksSection = document.createElement("div");
        checksSection.innerHTML = `<div class="diag-section-title">Validation Checks</div>`;
        for (const c of report.checks) {
            const row = document.createElement("div");
            row.className = `check-row ${c.pass ? "pass" : "fail"}`;
            row.innerHTML = `
                <div class="check-icon">${c.pass ? "✓" : "✗"}</div>
                <div class="check-content">
                    <div class="check-label">${c.label}</div>
                    <div class="check-detail">${c.details}</div>
                </div>
            `;
            checksSection.appendChild(row);
        }
        body.appendChild(checksSection);

        // ── stats ─────────────────────────────────────────────────────────────
        const s = report.stats;
        const statsSection = document.createElement("div");
        statsSection.innerHTML = `
            <div class="diag-section-title">Geometry Stats</div>
            <div class="stat-grid">
                <div class="stat-cell">
                    <div class="stat-key">Pos X range</div>
                    <div class="stat-val">[${s.posMin[0].toFixed(2)}, ${s.posMax[0].toFixed(2)}]</div>
                </div>
                <div class="stat-cell">
                    <div class="stat-key">Pos Y range</div>
                    <div class="stat-val">[${s.posMin[1].toFixed(2)}, ${s.posMax[1].toFixed(2)}]</div>
                </div>
                <div class="stat-cell">
                    <div class="stat-key">Pos Z range</div>
                    <div class="stat-val">[${s.posMin[2].toFixed(2)}, ${s.posMax[2].toFixed(2)}]</div>
                </div>
                <div class="stat-cell">
                    <div class="stat-key">Normal length</div>
                    <div class="stat-val">[${s.normalLengthMin.toFixed(4)}, ${s.normalLengthMax.toFixed(4)}]</div>
                </div>
                <div class="stat-cell">
                    <div class="stat-key">UV U range</div>
                    <div class="stat-val">[${s.uvMin[0].toFixed(3)}, ${s.uvMax[0].toFixed(3)}]</div>
                </div>
                <div class="stat-cell">
                    <div class="stat-key">UV V range</div>
                    <div class="stat-val">[${s.uvMin[1].toFixed(3)}, ${s.uvMax[1].toFixed(3)}]</div>
                </div>
            </div>
        `;
        body.appendChild(statsSection);

        // ── material ──────────────────────────────────────────────────────────
        const matSection = document.createElement("div");
        matSection.innerHTML = `
            <div class="diag-section-title">Material</div>
            <div class="mat-row">
                <div class="mat-dot" style="background:${hasTex ? "var(--pass)" : "var(--warn)"}"></div>
                <span class="mat-label">Base color texture</span>
                <span class="mat-value">${hasTex ? "present" : "missing"}</span>
            </div>
            <div class="mat-row" style="margin-top:6px">
                <div class="mat-dot" style="background:var(--info)"></div>
                <span class="mat-label">Pipeline mode</span>
                <span class="mat-value">${hasTex ? "textured" : "fallback"}</span>
            </div>
        `;
        body.appendChild(matSection);
    }

    return { panel, setStatus, showReport };
}

// ─── Buffer builder ────────────────────────────────────────────────────────────

function buildBuffers(device: GPUDevice, triangles: Triangle[]) {
    const FLOATS_PER_VERTEX = 12;
    const vertexData = new Float32Array(triangles.length * 3 * FLOATS_PER_VERTEX);
    const indexData = new Uint32Array(triangles.length * 3);

    for (let i = 0; i < triangles.length; i++) {
        const tri = triangles[i];
        for (let j = 0; j < 3; j++) {
            const base = (i * 3 + j) * FLOATS_PER_VERTEX;
            vertexData.set(tri.positions[j], base);
            vertexData[base + 3] = 0;
            vertexData.set(tri.normals[j], base + 4);
            vertexData[base + 7] = 0;
            vertexData.set(tri.uv[j], base + 8);
            vertexData[base + 10] = 0;
            vertexData[base + 11] = 0;
        }
        indexData[i * 3] = i * 3;
        indexData[i * 3 + 1] = i * 3 + 1;
        indexData[i * 3 + 2] = i * 3 + 2;
    }

    const vertexBuffer = device.createBuffer({
        size: vertexData.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(vertexBuffer, 0, vertexData);

    const indexBuffer = device.createBuffer({
        size: indexData.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(indexBuffer, 0, indexData);

    return { vertexBuffer, indexBuffer, indexCount: indexData.length };
}

// ─── Main ─────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    const { setStatus, showReport } = buildDiagnosticUI();

    (async () => {
        if (!navigator.gpu) {
            setStatus("WebGPU not supported in this browser", "error");
            return;
        }

        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter!.requestDevice();

        const canvas = document.getElementById("webgpu-canvas") as HTMLCanvasElement;
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;

        const context = canvas.getContext("webgpu")!;
        const FORMAT = "bgra8unorm" as GPUTextureFormat;
        context.configure({ device, format: FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT });

        // ── Load GLB ──────────────────────────────────────────────────────────
        setStatus("Loading GLB model…", "loading");
        const response = await fetch(
            "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Duck/glTF-Binary/Duck.glb"
        );
        const buffer = await response.arrayBuffer();
        const glbFile = await uploadGLBModel(buffer, device);
        const triangles: Triangle[] = glbFile.triangles;
        const materials = glbFile.materials;

        // ── Validate ──────────────────────────────────────────────────────────
        setStatus("Validating geometry…", "loading");
        const report = validateGeometry(triangles);

        const material = materials?.[0];
        const hasTexture = !!material?.baseColorTexture?.imageView;

        showReport(report, hasTexture);

        if (report.overallPass) {
            setStatus(`${triangles.length.toLocaleString()} triangles — all checks passed`, "ok");
        } else {
            const failCount = report.checks.filter(c => !c.pass).length;
            setStatus(`${failCount} check${failCount > 1 ? "s" : ""} failed — see details below`, "error");
        }

        // ── GPU resources ─────────────────────────────────────────────────────
        const { vertexBuffer, indexBuffer, indexCount } = buildBuffers(device, triangles);

        let depthTexture = device.createTexture({
            size: { width: canvas.width, height: canvas.height, depthOrArrayLayers: 1 },
            format: "depth24plus-stencil8",
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        const viewParamBuf = device.createBuffer({
            size: 4 * 4 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const sampler = device.createSampler({
            addressModeU: "repeat",
            addressModeV: "repeat",
            magFilter: "linear",
            minFilter: "linear",
        });

        const vsModule = device.createShaderModule({ code: VERTEX_SHADER });

        let pipeline: GPURenderPipeline;
        let bindGroup: GPUBindGroup;
        let bindGroupLayout: GPUBindGroupLayout;

        if (hasTexture) {
            const baseColorTextureView: GPUTextureView = material!.baseColorTexture!.imageView!;

            bindGroupLayout = device.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
                    { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
                    { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
                ],
            });
            bindGroup = device.createBindGroup({
                layout: bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: viewParamBuf } },
                    { binding: 1, resource: sampler },
                    { binding: 2, resource: baseColorTextureView },
                ],
            });
            pipeline = device.createRenderPipeline({
                layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
                vertex: {
                    module: vsModule, entryPoint: "vs_main",
                    buffers: [{ arrayStride: 12 * Float32Array.BYTES_PER_ELEMENT, attributes: [
                        { shaderLocation: 0, offset: 0,  format: "float32x3" },
                        { shaderLocation: 1, offset: 16, format: "float32x3" },
                        { shaderLocation: 2, offset: 32, format: "float32x2" },
                    ]}],
                },
                fragment: {
                    module: device.createShaderModule({ code: FRAGMENT_SHADER_TEXTURED }),
                    entryPoint: "fs_main", targets: [{ format: FORMAT }],
                },
                primitive: { topology: "triangle-list", cullMode: "back" },
                depthStencil: { format: "depth24plus-stencil8", depthWriteEnabled: true, depthCompare: "less" },
            });
        } else {
            bindGroupLayout = device.createBindGroupLayout({
                entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }],
            });
            bindGroup = device.createBindGroup({
                layout: bindGroupLayout,
                entries: [{ binding: 0, resource: { buffer: viewParamBuf } }],
            });
            pipeline = device.createRenderPipeline({
                layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
                vertex: {
                    module: vsModule, entryPoint: "vs_main",
                    buffers: [{ arrayStride: 12 * Float32Array.BYTES_PER_ELEMENT, attributes: [
                        { shaderLocation: 0, offset: 0,  format: "float32x3" },
                        { shaderLocation: 1, offset: 16, format: "float32x3" },
                        { shaderLocation: 2, offset: 32, format: "float32x2" },
                    ]}],
                },
                fragment: {
                    module: device.createShaderModule({ code: FRAGMENT_SHADER_FALLBACK }),
                    entryPoint: "fs_main", targets: [{ format: FORMAT }],
                },
                primitive: { topology: "triangle-list", cullMode: "back" },
                depthStencil: { format: "depth24plus-stencil8", depthWriteEnabled: true, depthCompare: "less" },
            });
        }

        // ── ArcballCamera + Controller ────────────────────────────────────────
        const defaultEye = vec3.set(vec3.create(), 0.0, 0.0, 3.5);
        const center = vec3.set(vec3.create(), 0.0, 0.0, 0.0);
        const up = vec3.set(vec3.create(), 0.0, 1.0, 0.0);
        let camera = new ArcballCamera(defaultEye, center, up, 2, [canvas.width, canvas.height]);
        let proj = mat4.perspective(mat4.create(), 50 * Math.PI / 180.0, canvas.width / canvas.height, 0.1, 1000);
        const projView = mat4.create();

        const controller = new Controller();
        controller.mousemove = (prev, cur, evt) => {
            if (evt.buttons === 1) camera.rotate(prev, cur);
            else if (evt.buttons === 2) camera.pan([cur[0] - prev[0], prev[1] - cur[1]]);
        };
        controller.wheel = (amt) => camera.zoom(amt * 0.5);
        controller.pinch = controller.wheel;
        controller.twoFingerDrag = (drag) => camera.pan(drag);
        controller.registerForCanvas(canvas);

        // ── Resize observer ───────────────────────────────────────────────────
        const resizeObserver = new ResizeObserver(() => {
            canvas.width = canvas.clientWidth;
            canvas.height = canvas.clientHeight;
            depthTexture.destroy();
            depthTexture = device.createTexture({
                size: { width: canvas.width, height: canvas.height, depthOrArrayLayers: 1 },
                format: "depth24plus-stencil8",
                usage: GPUTextureUsage.RENDER_ATTACHMENT,
            });
            proj = mat4.perspective(mat4.create(), 50 * Math.PI / 180.0, canvas.width / canvas.height, 0.1, 1000);
            camera = new ArcballCamera(defaultEye, center, up, 2, [canvas.width, canvas.height]);
            controller.registerForCanvas(canvas);
        });
        resizeObserver.observe(canvas);

        // ── Render loop ───────────────────────────────────────────────────────
        const render = () => {
            mat4.mul(projView, proj, camera.camera);
            device.queue.writeBuffer(viewParamBuf, 0, projView as Float32Array);

            const cmd = device.createCommandEncoder();
            const pass = cmd.beginRenderPass({
                colorAttachments: [{
                    view: context.getCurrentTexture().createView(),
                    loadOp: "clear",
                    clearValue: [0.08, 0.08, 0.10, 1],
                    storeOp: "store",
                }],
                depthStencilAttachment: {
                    view: depthTexture.createView(),
                    depthLoadOp: "clear",
                    depthClearValue: 1,
                    depthStoreOp: "store",
                    stencilLoadOp: "clear",
                    stencilClearValue: 0,
                    stencilStoreOp: "store",
                },
            });

            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.setVertexBuffer(0, vertexBuffer);
            pass.setIndexBuffer(indexBuffer, "uint32");
            pass.drawIndexed(indexCount);
            pass.end();

            device.queue.submit([cmd.finish()]);
            requestAnimationFrame(render);
        };

        requestAnimationFrame(render);
    })();
});