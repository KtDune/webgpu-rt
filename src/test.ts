import { vec3, mat4 } from "gl-matrix";
import { ArcballCamera } from "./class/ArcballCamera";
import { Controller } from "./class/Controller";
import { Triangle, uploadGLBModel } from "./uploadGlb";

document.addEventListener("DOMContentLoaded", () => {

    // ─── WGSL Shader ─────────────────────────────────────────────────────
    const VALIDATION_SHADER = /* wgsl */`
        struct ViewParams {
            projView: mat4x4<f32>
        }
        struct VertexInput {
            @location(0) position: vec3<f32>,
            @location(1) normal:   vec3<f32>,
            @location(2) color:    vec3<f32>,
        }
        struct VertexOutput {
            @builtin(position) Position: vec4<f32>,
            @location(0) color:  vec3<f32>,
            @location(1) normal: vec3<f32>,
        }

        @group(0) @binding(0) var<uniform> viewParams: ViewParams;

        @vertex
        fn vs_main(in: VertexInput) -> VertexOutput {
            var out: VertexOutput;
            out.Position = viewParams.projView * vec4<f32>(in.position, 1.0);
            out.color    = in.color;
            out.normal   = in.normal;
            return out;
        }

        @fragment
        fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
            let lightDir = normalize(vec3<f32>(1.0, 1.0, 1.0));
            let diff     = max(dot(normalize(in.normal), lightDir), 0.0);
            let shaded   = in.color * (0.2 + 0.8 * diff);
            return vec4<f32>(shaded, 1.0);
        }
    `;

    // ─── Build GPU buffers from Triangle[] ───────────────────────────────
    function buildBuffersFromTriangles(device: GPUDevice, triangles: Triangle[]) {
        // Layout per vertex: position(3) + pad(1) + normal(3) + pad(1) + color(3) + pad(1) = 12 floats
        const FLOATS_PER_VERTEX = 12;
        const vertexData = new Float32Array(triangles.length * 3 * FLOATS_PER_VERTEX);
        const indexData  = new Uint32Array(triangles.length * 3);

        for (let i = 0; i < triangles.length; i++) {
            const tri = triangles[i];
            for (let j = 0; j < 3; j++) {
                const base = (i * 3 + j) * FLOATS_PER_VERTEX;
                vertexData.set(tri.positions[j], base);      // offset 0  — position
                vertexData[base + 3] = 0;                    // offset 3  — pad
                vertexData.set(tri.normals[j],   base + 4);  // offset 4  — normal
                vertexData[base + 7] = 0;                    // offset 7  — pad
                vertexData.set(tri.color,        base + 8);  // offset 8  — color
                vertexData[base + 11] = 0;                   // offset 11 — pad
            }
            indexData[i * 3]     = i * 3;
            indexData[i * 3 + 1] = i * 3 + 1;
            indexData[i * 3 + 2] = i * 3 + 2;
        }

        const vertexBuffer = device.createBuffer({
            size: vertexData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(vertexBuffer, 0, vertexData);

        const indexBuffer = device.createBuffer({
            size: indexData.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(indexBuffer, 0, indexData);

        return { vertexBuffer, indexBuffer, indexCount: indexData.length };
    }

    // ─── Main ─────────────────────────────────────────────────────────────
    (async () => {
        if (!navigator.gpu) {
            alert("WebGPU not supported");
            return;
        }

        const adapter = await navigator.gpu.requestAdapter();
        const device  = await adapter!.requestDevice();

        const canvas  = document.getElementById("webgpu-canvas") as HTMLCanvasElement;
        canvas.width  = window.innerWidth;
        canvas.height = window.innerHeight;

        const context        = canvas.getContext("webgpu")!;
        const swapChainFormat = "bgra8unorm";
        context.configure({ device, format: swapChainFormat, usage: GPUTextureUsage.RENDER_ATTACHMENT });

        // ── Load GLB ──────────────────────────────────────────────────────
        const response = await fetch("https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Duck/glTF-Binary/Duck.glb");
        const buffer   = await response.arrayBuffer();
        const glbFile  = await uploadGLBModel(buffer, device);   // <-- your existing function

        const triangles: Triangle[] = glbFile.triangles;
        //console.log(`Loaded ${triangles.length} triangles from GLB`);
        validateTriangleMesh(triangles);

        // ── GPU Buffers ───────────────────────────────────────────────────
        const { vertexBuffer, indexBuffer, indexCount } = buildBuffersFromTriangles(device, triangles);

        // ── Depth texture ─────────────────────────────────────────────────
        const depthTexture = device.createTexture({
            size: { width: canvas.width, height: canvas.height, depthOrArrayLayers: 1 },
            format: "depth24plus-stencil8",
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });

        // ── Camera uniform ────────────────────────────────────────────────
        const viewParamBuf = device.createBuffer({
            size: 4 * 4 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        const viewParamsLayout = device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }]
        });
        const viewParamsBindGroup = device.createBindGroup({
            layout: viewParamsLayout,
            entries: [{ binding: 0, resource: { buffer: viewParamBuf } }]
        });

        // ── Pipeline ──────────────────────────────────────────────────────
        const shaderModule = device.createShaderModule({ code: VALIDATION_SHADER });
        const pipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [viewParamsLayout] }),
            vertex: {
                module: shaderModule,
                entryPoint: "vs_main",
                buffers: [{
                    arrayStride: 12 * Float32Array.BYTES_PER_ELEMENT,
                    attributes: [
                        { shaderLocation: 0, offset: 0,  format: "float32x3" }, // position
                        { shaderLocation: 1, offset: 16, format: "float32x3" }, // normal
                        { shaderLocation: 2, offset: 32, format: "float32x3" }, // color
                    ]
                }]
            },
            fragment: {
                module: shaderModule,
                entryPoint: "fs_main",
                targets: [{ format: swapChainFormat }]
            },
            primitive: {
                topology: "triangle-list",
                cullMode: "none"  // show both sides so flipped normals are visible
            },
            depthStencil: {
                format: "depth24plus-stencil8",
                depthWriteEnabled: true,
                depthCompare: "less"
            }
        });

        // ── Camera & controller ───────────────────────────────────────────
        const defaultEye = vec3.set(vec3.create(), 0.0, 0.0, 5.0);
        const center     = vec3.set(vec3.create(), 0.0, 0.0, 0.0);
        const up         = vec3.set(vec3.create(), 0.0, 1.0, 0.0);
        let camera       = new ArcballCamera(defaultEye, center, up, 2, [canvas.width, canvas.height]);
        let proj         = mat4.perspective(mat4.create(), 50 * Math.PI / 180.0, canvas.width / canvas.height, 0.1, 1000);
        const projView   = mat4.create();

        const controller = new Controller();
        controller.mousemove = (prev, cur, evt) => {
            if (evt.buttons === 1) camera.rotate(prev, cur);
            else if (evt.buttons === 2) camera.pan([cur[0] - prev[0], prev[1] - cur[1]]);
        };
        controller.wheel         = (amt) => camera.zoom(amt * 0.5);
        controller.pinch         = controller.wheel;
        controller.twoFingerDrag = (drag) => camera.pan(drag);
        controller.registerForCanvas(canvas);

        // ── Render loop ───────────────────────────────────────────────────
        const render = async () => {
            mat4.mul(projView, proj, camera.camera);
            device.queue.writeBuffer(viewParamBuf, 0, projView as Float32Array);

            const commandEncoder = device.createCommandEncoder();
            const renderPass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: context.getCurrentTexture().createView(),
                    loadOp: "clear",
                    clearValue: [0.15, 0.15, 0.15, 1],
                    storeOp: "store"
                }],
                depthStencilAttachment: {
                    view: depthTexture.createView(),
                    depthLoadOp: "clear",
                    depthClearValue: 1,
                    depthStoreOp: "store",
                    stencilLoadOp: "clear",
                    stencilClearValue: 0,
                    stencilStoreOp: "store"
                }
            });

            renderPass.setPipeline(pipeline);
            renderPass.setBindGroup(0, viewParamsBindGroup);
            renderPass.setVertexBuffer(0, vertexBuffer);
            renderPass.setIndexBuffer(indexBuffer, "uint32");
            renderPass.drawIndexed(indexCount);
            renderPass.end();

            device.queue.submit([commandEncoder.finish()]);
            requestAnimationFrame(render);
        };

        window.addEventListener("resize", () => {
            canvas.width  = window.innerWidth;
            canvas.height = window.innerHeight;
            proj = mat4.perspective(mat4.create(), 50 * Math.PI / 180.0, canvas.width / canvas.height, 0.1, 1000);
            camera = new ArcballCamera(defaultEye, center, up, 2, [canvas.width, canvas.height]);
            controller.registerForCanvas(canvas);
        });

        requestAnimationFrame(render);
    })();
});

function validateTriangleMesh(triangles: Triangle[]): boolean {
    let allPassed = true;

    // ─── Helper Math Functions ───────────────────────────────────────────
    const subtract = (a: Float32Array, b: Float32Array): Float32Array =>
        new Float32Array([a[0] - b[0], a[1] - b[1], a[2] - b[2]]);

    const crossProduct = (a: Float32Array, b: Float32Array): Float32Array =>
        new Float32Array([
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]);

    const magnitude = (v: Float32Array): number =>
        Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2);

    const normalize = (v: Float32Array): Float32Array => {
        const mag = magnitude(v);
        return new Float32Array([v[0] / mag, v[1] / mag, v[2] / mag]);
    };

    const dot = (a: Float32Array, b: Float32Array): number =>
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

    const equals = (a: Float32Array, b: Float32Array): boolean =>
        a[0] === b[0] && a[1] === b[1] && a[2] === b[2];

    const edgeKey = (a: Float32Array, b: Float32Array): string => {
        const ka = Array.from(a).join(',');
        const kb = Array.from(b).join(',');
        return ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
    };

    const pass = (msg: string) => console.log(`  ✅ PASS: ${msg}`);
    const fail = (msg: string) => { console.log(`  ❌ FAIL: ${msg}`); allPassed = false; };

    console.log('='.repeat(60));
    console.log('         TRIANGLE MESH VALIDATION REPORT');
    console.log('='.repeat(60));
    console.log(`Total triangles: ${triangles.length}\n`);

    // ─── Check 1: Non-empty list ─────────────────────────────────────────
    console.log('[Check 1] Non-empty triangle list');
    if (triangles.length === 0) {
        fail('Triangle list is empty');
        return false; // No point continuing
    } else {
        pass(`Triangle list has ${triangles.length} triangles`);
    }

    // ─── Check 2: Per-triangle basic validity ────────────────────────────
    console.log('\n[Check 2] Per-triangle basic validity (positions, normals, degenerate)');
    let basicFailCount = 0;
    for (let i = 0; i < triangles.length; i++) {
        const tri = triangles[i];
        const [p0, p1, p2] = tri.positions;

        if (tri.positions.length !== 3) {
            fail(`Triangle ${i}: expected 3 positions, got ${tri.positions.length}`);
            basicFailCount++;
            continue;
        }
        if (tri.normals.length !== 3) {
            fail(`Triangle ${i}: expected 3 normals, got ${tri.normals.length}`);
            basicFailCount++;
            continue;
        }
        if (equals(p0, p1) || equals(p1, p2) || equals(p0, p2)) {
            fail(`Triangle ${i}: has duplicate vertices`);
            basicFailCount++;
            continue;
        }
        const cross = crossProduct(subtract(p1, p0), subtract(p2, p0));
        if (magnitude(cross) < 1e-10) {
            fail(`Triangle ${i}: vertices are collinear (zero area)`);
            basicFailCount++;
        }
    }
    if (basicFailCount === 0) pass(`All ${triangles.length} triangles have valid positions and normals`);

    // ─── Check 3: Manifold edges ─────────────────────────────────────────
    console.log('\n[Check 3] Manifold check (each edge shared by exactly 2 triangles)');
    const edgeCount = new Map<string, number>();
    for (const tri of triangles) {
        const [p0, p1, p2] = tri.positions;
        for (const edge of [edgeKey(p0, p1), edgeKey(p1, p2), edgeKey(p2, p0)]) {
            edgeCount.set(edge, (edgeCount.get(edge) ?? 0) + 1);
        }
    }
    const boundaryEdges = [...edgeCount.entries()].filter(([, c]) => c === 1);
    const nonManifoldEdges = [...edgeCount.entries()].filter(([, c]) => c > 2);

    if (boundaryEdges.length > 0) {
        fail(`Found ${boundaryEdges.length} boundary edge(s) — mesh has holes`);
    } else {
        pass('No boundary edges found');
    }
    if (nonManifoldEdges.length > 0) {
        fail(`Found ${nonManifoldEdges.length} non-manifold edge(s) — shared by 3+ triangles`);
    } else {
        pass('No non-manifold edges found');
    }

    // ─── Check 4: Normal consistency ─────────────────────────────────────
    console.log('\n[Check 4] Normal consistency (stored normals agree with face normal)');
    let normalFailCount = 0;
    for (let i = 0; i < triangles.length; i++) {
        const tri = triangles[i];
        const [p0, p1, p2] = tri.positions;
        const faceNormal = normalize(crossProduct(subtract(p1, p0), subtract(p2, p0)));
        for (const n of tri.normals) {
            if (dot(faceNormal, n) < 0) {
                fail(`Triangle ${i}: stored normal points inward (winding mismatch)`);
                normalFailCount++;
                break;
            }
        }
    }
    if (normalFailCount === 0) pass(`All ${triangles.length} triangles have consistent normals`);

    // ─── Check 5: Connectivity (single component) ─────────────────────────
    console.log('\n[Check 5] Connectivity (all triangles form one connected component)');
    const adjacency = new Map<number, Set<number>>();
    const edgeToTris = new Map<string, number[]>();
    for (let i = 0; i < triangles.length; i++) {
        const [p0, p1, p2] = triangles[i].positions;
        for (const edge of [edgeKey(p0, p1), edgeKey(p1, p2), edgeKey(p2, p0)]) {
            if (!edgeToTris.has(edge)) edgeToTris.set(edge, []);
            edgeToTris.get(edge)!.push(i);
        }
        adjacency.set(i, new Set());
    }
    for (const tris of edgeToTris.values()) {
        if (tris.length === 2) {
            adjacency.get(tris[0])!.add(tris[1]);
            adjacency.get(tris[1])!.add(tris[0]);
        }
    }
    const visited = new Set<number>();
    const queue = [0];
    while (queue.length > 0) {
        const curr = queue.pop()!;
        if (visited.has(curr)) continue;
        visited.add(curr);
        for (const neighbor of adjacency.get(curr)!) queue.push(neighbor);
    }
    if (visited.size !== triangles.length) {
        fail(`Mesh has ${triangles.length - visited.size} disconnected triangle(s)`);
    } else {
        pass('All triangles form a single connected component');
    }

    // ─── Final Result ─────────────────────────────────────────────────────
    console.log('\n' + '='.repeat(60));
    if (allPassed) {
        console.log('  🎉 RESULT: All checks passed — mesh is valid!');
    } else {
        console.log('  ⚠️  RESULT: Some checks failed — mesh may be invalid.');
    }
    console.log('='.repeat(60));

    return allPassed;
};