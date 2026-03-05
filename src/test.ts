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

// Fallback fragment shader — solid red tint — shown when material has NO texture
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
        // RED tint = no texture found — easy to spot
        return vec4<f32>(1.0, 0.0, 0.0, 1.0) * (0.3 + 0.7 * diff);
    }
`;

// ─── Buffer builder ────────────────────────────────────────────────────────────
// Layout per vertex: position(3) + pad(1) + normal(3) + pad(1) + uv(2) + pad(2) = 12 floats
function buildBuffers(device: GPUDevice, triangles: Triangle[]) {
    const FLOATS_PER_VERTEX = 12;
    const vertexData = new Float32Array(triangles.length * 3 * FLOATS_PER_VERTEX);
    const indexData = new Uint32Array(triangles.length * 3);

    for (let i = 0; i < triangles.length; i++) {
        const tri = triangles[i];
        for (let j = 0; j < 3; j++) {
            const base = (i * 3 + j) * FLOATS_PER_VERTEX;
            vertexData.set(tri.positions[j], base);      // offset 0  — position (3)
            vertexData[base + 3] = 0;                    // offset 3  — pad
            vertexData.set(tri.normals[j], base + 4);  // offset 4  — normal (3)
            vertexData[base + 7] = 0;                    // offset 7  — pad
            // uv is vec2 — tri.uv[j] is a Float32Array of length 2
            vertexData.set(tri.uv[j], base + 8);  // offset 8  — uv (2)
            vertexData[base + 10] = 0;                   // offset 10 — pad
            vertexData[base + 11] = 0;                   // offset 11 — pad
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
    // ── Status overlay ────────────────────────────────────────────────────────
    const overlay = document.createElement("div");
    overlay.style.cssText = `
        position: fixed; top: 16px; left: 50%; transform: translateX(-50%);
        background: #111; color: #e2e8f0; font-family: monospace; font-size: 13px;
        padding: 12px 20px; border-radius: 8px; border: 1px solid #334155;
        z-index: 999; white-space: pre-line; text-align: center; min-width: 360px;
        box-shadow: 0 4px 24px #0008;
    `;
    document.body.appendChild(overlay);

    const log = (msg: string, color = "#e2e8f0") => {
        console.log(msg);
        overlay.innerHTML += `<span style="color:${color}">${msg}</span>\n`;
    };

    (async () => {
        // ── WebGPU init ───────────────────────────────────────────────────────
        if (!navigator.gpu) { log("❌ WebGPU not supported", "#f87171"); return; }

        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter!.requestDevice();

        const canvas = document.getElementById("webgpu-canvas") as HTMLCanvasElement;
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        const context = canvas.getContext("webgpu")!;
        const FORMAT = "bgra8unorm" as GPUTextureFormat;
        context.configure({ device, format: FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT });

        // ── Load GLB ──────────────────────────────────────────────────────────
        log("⏳ Loading GLB...", "#94a3b8");
        const response = await fetch(
            "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Duck/glTF-Binary/Duck.glb"
        );
        const buffer = await response.arrayBuffer();
        const glbFile = await uploadGLBModel(buffer, device);

        const triangles: Triangle[] = glbFile.triangles;
        const materials = glbFile.materials;

        log(`✅ Loaded ${triangles.length} triangles`, "#4ade80");

        // ── Material check ────────────────────────────────────────────────────
        const material = materials?.[0];
        log(`\n── material[0] validation ──`, "#7dd3fc");

        if (!material) {
            log("❌ material[0] is undefined", "#f87171");
            return;
        }
        log("✅ material[0] exists", "#4ade80");

        // Decide texture vs fallback
        const hasTexture = !!material.baseColorTexture?.imageView;
        if (hasTexture) {
            log("✅ baseColorTexture present — rendering TEXTURED", "#4ade80");
        } else {
            log("⚠️  baseColorTexture missing — rendering FALLBACK (red tint)", "#fbbf24");
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

        // ── Bind group layout & pipeline ──────────────────────────────────────
        let pipeline: GPURenderPipeline;
        let bindGroup: GPUBindGroup;
        let bindGroupLayout: GPUBindGroupLayout;

        const vsModule = device.createShaderModule({ code: VERTEX_SHADER });

        if (hasTexture) {
            // ── Textured path ─────────────────────────────────────────────────
            const baseColorTextureView: GPUTextureView = material.baseColorTexture!.imageView!;

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
                    module: vsModule,
                    entryPoint: "vs_main",
                    buffers: [{
                        arrayStride: 12 * Float32Array.BYTES_PER_ELEMENT,
                        attributes: [
                            { shaderLocation: 0, offset: 0, format: "float32x3" }, // position
                            { shaderLocation: 1, offset: 16, format: "float32x3" }, // normal
                            { shaderLocation: 2, offset: 32, format: "float32x2" }, // uv
                        ],
                    }],
                },
                fragment: {
                    module: device.createShaderModule({ code: FRAGMENT_SHADER_TEXTURED }),
                    entryPoint: "fs_main",
                    targets: [{ format: FORMAT }],
                },
                primitive: { topology: "triangle-list", cullMode: "back" },
                depthStencil: { format: "depth24plus-stencil8", depthWriteEnabled: true, depthCompare: "less" },
            });

        } else {
            // ── Fallback path (no texture) ────────────────────────────────────
            bindGroupLayout = device.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
                ],
            });

            bindGroup = device.createBindGroup({
                layout: bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: viewParamBuf } },
                ],
            });

            pipeline = device.createRenderPipeline({
                layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
                vertex: {
                    module: vsModule,
                    entryPoint: "vs_main",
                    buffers: [{
                        arrayStride: 12 * Float32Array.BYTES_PER_ELEMENT,
                        attributes: [
                            { shaderLocation: 0, offset: 0, format: "float32x3" },
                            { shaderLocation: 1, offset: 16, format: "float32x3" },
                            { shaderLocation: 2, offset: 32, format: "float32x2" },
                        ],
                    }],
                },
                fragment: {
                    module: device.createShaderModule({ code: FRAGMENT_SHADER_FALLBACK }),
                    entryPoint: "fs_main",
                    targets: [{ format: FORMAT }],
                },
                primitive: { topology: "triangle-list", cullMode: "back" },
                depthStencil: { format: "depth24plus-stencil8", depthWriteEnabled: true, depthCompare: "less" },
            });
        }

        log(`\n── rendering ──`, "#7dd3fc");
        log(hasTexture ? "🟢 TEXTURED pipeline active" : "🔴 FALLBACK pipeline active (red = no texture)",
            hasTexture ? "#4ade80" : "#fbbf24");

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