import { vec3, mat4 } from "gl-matrix";
import { ArcballCamera } from "./class/ArcballCamera";
import { Controller } from "./class/Controller";
import { GLBShaderCache } from "./class/GLBShaderCache";
import { uploadGLBModel } from "./uploadGlb";
import { WebUI } from "./class/WebUI";

document.addEventListener("DOMContentLoaded", () => {
    const rasterRadio = document.getElementById("mode-raster") as HTMLInputElement;
    const raytraceRadio = document.getElementById("mode-raytrace") as HTMLInputElement;
    var rtMode = false

    rasterRadio.addEventListener("change", () => {
        if (rasterRadio.checked) {
            rtMode = false
        }
    });

    raytraceRadio.addEventListener("change", () => {
        if (raytraceRadio.checked) {
            rtMode = true
        }
    });

    async function initRasterization(device: GPUDevice, glbFile: any, swapChainFormat: string, canvas: HTMLCanvasElement, shaderCache: GLBShaderCache) {
        const depthTexture = device.createTexture({
            size: { width: canvas.width, height: canvas.height, depthOrArrayLayers: 1 },
            format: "depth24plus-stencil8",
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });

        const renderPassDesc = {
            colorAttachments: [{ view: undefined, loadOp: "clear", clearValue: [0.3, 0.3, 0.3, 1], storeOp: "store" }],
            depthStencilAttachment: {
                view: depthTexture.createView(),
                depthLoadOp: "clear",
                depthClearValue: 1,
                depthStoreOp: "store",
                stencilLoadOp: "clear",
                stencilClearValue: 0,
                stencilStoreOp: "store"
            }
        };

        const viewParamsLayout = device.createBindGroupLayout({
            label: 'View Params Layout',
            entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }]
        });

        const viewParamBuf = device.createBuffer(
            { size: 4 * 4 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const viewParamsBindGroup = device.createBindGroup(
            { label: 'View Params Bind Group', layout: viewParamsLayout, entries: [{ binding: 0, resource: { buffer: viewParamBuf } }] });

        const utilsLayout = device.createBindGroupLayout({
            label: 'Utils Layout',
            entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }]
        });

        const utilsBuf = device.createBuffer(
            { size: (4 + 4) * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const utilsBindGroup = device.createBindGroup(
            { layout: utilsLayout, entries: [{ binding: 0, resource: { buffer: utilsBuf } }] });

        const renderBundles = glbFile.buildRenderBundles(
            device,
            shaderCache,
            viewParamsLayout,
            viewParamsBindGroup,
            utilsLayout,
            utilsBindGroup,
            swapChainFormat
        );

        return {
            depthTexture,
            renderPassDesc,
            viewParamsLayout,
            viewParamBuf,
            viewParamsBindGroup,
            utilsLayout,
            utilsBuf,
            utilsBindGroup,
            shaderCache,
            renderBundles
        };
    }

    async function initRaytrace(device: GPUDevice, glbFile: any, canvas: HTMLCanvasElement, shaderCache: GLBShaderCache) {
        const colorBuffer = device.createTexture({
            size: [canvas.width, canvas.height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
        });
        const colorBufferView = colorBuffer.createView();

        const sampler = device.createSampler({
            addressModeU: "repeat",
            addressModeV: "repeat",
            magFilter: "linear",
            minFilter: "nearest",
            mipmapFilter: "nearest",
            maxAnisotropy: 1
        });

        const sceneParamsBuffer = device.createBuffer({
            size: 16 * Float32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const triangles: Triangle[] = glbFile.triangles;
        validateTriangleMesh(triangles)

        const trianglesBuffer = device.createBuffer({
            size: 28 * Float32Array.BYTES_PER_ELEMENT * triangles.length,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const trianglesUploadData = new Float32Array(triangles.length * 28);
        for (let i = 0; i < triangles.length; i++) {
            trianglesUploadData.set(triangles[i].positions[0], i * 28);
            trianglesUploadData.set(triangles[i].normals[0], i * 28 + 4);
            trianglesUploadData.set(triangles[i].positions[1], i * 28 + 8);
            trianglesUploadData.set(triangles[i].normals[1], i * 28 + 12);
            trianglesUploadData.set(triangles[i].positions[2], i * 28 + 16);
            trianglesUploadData.set(triangles[i].normals[2], i * 28 + 20);
            trianglesUploadData.set(triangles[i].color, i * 28 + 24);
        }
        device.queue.writeBuffer(trianglesBuffer, 0, trianglesUploadData, 0);

        // RAY TRACING PIPELINE
        const rayTracingBindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        access: "write-only",
                        format: "rgba8unorm",
                        viewDimension: "2d"
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                }
            ],
        });

        const rayTracingBindGroup = device.createBindGroup({
            layout: rayTracingBindGroupLayout,
            entries: [
                { binding: 0, resource: colorBufferView },
                { binding: 1, resource: { buffer: sceneParamsBuffer } },
                { binding: 2, resource: { buffer: trianglesBuffer } },
            ]
        });

        const raytracer_kernel = shaderCache.getRayShader()

        const rayTracingPipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [rayTracingBindGroupLayout]
            }),
            compute: {
                module: device.createShaderModule({ label: 'Ray Tracing', code: raytracer_kernel }),
                entryPoint: "main"
            }
        });

        // SCREEN PIPELINE
        const screenBindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: {}
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: {}
                }
            ]
        });

        const screenBindGroup = device.createBindGroup({
            layout: screenBindGroupLayout,
            entries: [
                { binding: 0, resource: sampler },
                { binding: 1, resource: colorBufferView }
            ]
        });

        const screen_shader = shaderCache.getScreenShader()

        const screenPipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [screenBindGroupLayout]
            }),
            vertex: {
                module: device.createShaderModule({ code: screen_shader }),
                entryPoint: "vert_main"
            },
            fragment: {
                module: device.createShaderModule({ code: screen_shader }),
                entryPoint: "frag_main",
                targets: [{ format: "bgra8unorm" }]
            },
            primitive: {
                topology: "triangle-list"
            }
        });

        // UPLOAD INITIAL SCENE PARAMS
        const maxBounces = 2;
        const sceneParamsUploadData = new Float32Array(16);
        sceneParamsUploadData.set([0, 0, 5], 0);
        sceneParamsUploadData.set([0, 0, 1], 4);
        sceneParamsUploadData.set([-1, 0, 0], 8);
        sceneParamsUploadData.set([maxBounces], 11);
        sceneParamsUploadData.set([0, 1, 0], 12);
        sceneParamsUploadData.set([triangles.length], 15);
        device.queue.writeBuffer(sceneParamsBuffer, 0, sceneParamsUploadData, 0);

        return {
            colorBuffer,
            colorBufferView,
            sampler,
            sceneParamsBuffer,
            trianglesBuffer,
            rayTracingPipeline,
            rayTracingBindGroup,
            screenPipeline,
            screenBindGroup,
            maxBounces,
            triangleCount: triangles.length
        };
    }

    (async () => {
        if (navigator.gpu === undefined) {
            document.getElementById("webgpu-canvas").setAttribute("style", "display:none;");
            return;
        }

        var adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            document.getElementById("webgpu-canvas").setAttribute("style", "display:none;");
            return;
        }
        var device = await adapter.requestDevice();

        const response = await fetch("https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Duck/glTF-Binary/Duck.glb");
        const buffer = await response.arrayBuffer();
        let glbFile = await uploadGLBModel(buffer, device);

        var canvas = document.getElementById("webgpu-canvas") as HTMLCanvasElement;
        var context = canvas.getContext("webgpu");
        var swapChainFormat = "bgra8unorm";
        context.configure({ device: device, format: swapChainFormat, usage: GPUTextureUsage.RENDER_ATTACHMENT });

        const shaderCache = new GLBShaderCache(device);

        let raster = await initRasterization(device, glbFile, swapChainFormat, canvas, shaderCache);
        let raytrace = await initRaytrace(device, glbFile, canvas, shaderCache);

        const defaultEye = vec3.set(vec3.create(), 0.0, 0.0, 1.0);
        const center = vec3.set(vec3.create(), 0.0, 0.0, 0.0);
        const up = vec3.set(vec3.create(), 0.0, 1.0, 0.0);
        var camera = new ArcballCamera(defaultEye, center, up, 2, [canvas.width, canvas.height]);
        var proj = mat4.perspective(
            mat4.create(), 50 * Math.PI / 180.0, canvas.width / canvas.height, 0.1, 1000);
        var projView = mat4.create();

        var controller = new Controller();
        controller.mousemove = function (prev, cur, evt) {
            if (evt.buttons == 1) {
                camera.rotate(prev, cur);
            } else if (evt.buttons == 2) {
                camera.pan([cur[0] - prev[0], prev[1] - cur[1]]);
            }
        };
        controller.wheel = function (amt) {
            camera.zoom(amt * 0.5);
        };
        controller.pinch = controller.wheel;
        controller.twoFingerDrag = function (drag) {
            camera.pan(drag);
        };
        controller.registerForCanvas(canvas);

        var glbBuffer = null;
        document.getElementById("uploadGLB").onchange =
            function uploadGLB() {
                document.getElementById("loading-text").hidden = false;
                var reader = new FileReader();
                reader.onerror = function () {
                    alert("error reading GLB file");
                };
                reader.onload = function () {
                    glbBuffer = reader.result;
                };
                reader.readAsArrayBuffer(this.files[0]);
            }

        var fpsDisplay = document.getElementById("fps");
        var numFrames = 0;
        var totalTimeMS = 0;

        const webUI = new WebUI();

        const render = async () => {
            webUI.consumeUpdate();

            if (glbBuffer != null) {
                glbFile = await uploadGLBModel(glbBuffer, device);
                raster.renderBundles = glbFile.buildRenderBundles(
                    device,
                    raster.shaderCache,
                    raster.viewParamsLayout,
                    raster.viewParamsBindGroup,
                    raster.utilsLayout,
                    raster.utilsBindGroup,
                    swapChainFormat
                );
                camera = new ArcballCamera(defaultEye, center, up, 2, [canvas.width, canvas.height]);
                glbBuffer = null;
            }

            var start = performance.now();
            var commandEncoder = device.createCommandEncoder();

            var upload = null;
            var utilsUploadBuf = null;
            if (!rtMode) {
                raster.renderPassDesc.colorAttachments[0].view = context.getCurrentTexture().createView();

                projView = mat4.mul(projView, proj, camera.camera);
                upload = device.createBuffer({
                    size: 4 * 4 * 4,
                    usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
                    mappedAtCreation: true
                });
                new Float32Array(upload.getMappedRange()).set(projView);
                upload.unmap();

                const utilsData = new Float32Array(8);
                utilsData.set([camera.camera[12], camera.camera[13], camera.camera[14]], 0);
                utilsData.set(webUI.lightPosition, 4);
                utilsData.set([webUI.usePBR], 7);

                utilsUploadBuf = device.createBuffer({
                    size: utilsData.byteLength,
                    usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
                    mappedAtCreation: true
                });
                new Float32Array(utilsUploadBuf.getMappedRange()).set(utilsData);
                utilsUploadBuf.unmap();

                commandEncoder.copyBufferToBuffer(upload, 0, raster.viewParamBuf, 0, 4 * 4 * 4);
                commandEncoder.copyBufferToBuffer(utilsUploadBuf, 0, raster.utilsBuf, 0, utilsData.byteLength);

                var renderPass = commandEncoder.beginRenderPass(raster.renderPassDesc);
                renderPass.executeBundles(raster.renderBundles);
                renderPass.end();
            }
            else {
                const upVec3 = new Float32Array([camera.upDir()[0], camera.upDir()[1], camera.upDir()[2]]);
                const forwardVec3 = new Float32Array([camera.eyeDir()[0], camera.eyeDir()[1], camera.eyeDir()[2]]);
                const rightVec3 = vec3.create();
                vec3.cross(rightVec3, forwardVec3, upVec3);
                vec3.normalize(rightVec3, rightVec3);

                const sceneParamsUpdateData = new Float32Array(16);
                sceneParamsUpdateData.set([camera.eyePos()[0], camera.eyePos()[1], camera.eyePos()[2]], 0);
                sceneParamsUpdateData.set([camera.eyeDir()[0], camera.eyeDir()[1], camera.eyeDir()[2]], 4);
                sceneParamsUpdateData.set([rightVec3[0], rightVec3[1], rightVec3[2]], 8);
                sceneParamsUpdateData.set([raytrace.maxBounces], 11);
                sceneParamsUpdateData.set([camera.upDir()[0], camera.upDir()[1], camera.upDir()[2]], 12);
                sceneParamsUpdateData.set([raytrace.triangleCount], 15);

                const sceneParamsUpdateBuffer = device.createBuffer({
                    size: 16 * Float32Array.BYTES_PER_ELEMENT,
                    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
                });
                device.queue.writeBuffer(sceneParamsUpdateBuffer, 0, sceneParamsUpdateData, 0);
                commandEncoder.copyBufferToBuffer(sceneParamsUpdateBuffer, 0, raytrace.sceneParamsBuffer, 0, 16 * Float32Array.BYTES_PER_ELEMENT);

                // Compute pass
                const rayTracerPass = commandEncoder.beginComputePass();
                rayTracerPass.setPipeline(raytrace.rayTracingPipeline);
                rayTracerPass.setBindGroup(0, raytrace.rayTracingBindGroup);
                rayTracerPass.dispatchWorkgroups(canvas.width, canvas.height, 1);
                rayTracerPass.end();

                // Screen pass
                const textureView = context.getCurrentTexture().createView();
                const renderPass = commandEncoder.beginRenderPass({
                    colorAttachments: [{
                        view: textureView,
                        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                        loadOp: "clear",
                        storeOp: "store"
                    }]
                });
                renderPass.setPipeline(raytrace.screenPipeline);
                renderPass.setBindGroup(0, raytrace.screenBindGroup);
                renderPass.draw(6, 1, 0, 0);
                renderPass.end();
            }

            device.queue.submit([commandEncoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            if (!rtMode) {
                upload.destroy();
                utilsUploadBuf.destroy();
            }

            var end = performance.now();
            numFrames += 1;
            totalTimeMS += end - start;
            fpsDisplay.innerHTML = `Avg. FPS ${Math.round(1000.0 * numFrames / totalTimeMS)}`;
            requestAnimationFrame(render);
        };

        const resizeCanvas = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;

            raster.depthTexture.destroy();
            raster.depthTexture = device.createTexture({
                size: { width: canvas.width, height: canvas.height, depthOrArrayLayers: 1 },
                format: "depth24plus-stencil8",
                usage: GPUTextureUsage.RENDER_ATTACHMENT
            });
            raster.renderPassDesc.depthStencilAttachment.view = raster.depthTexture.createView();

            proj = mat4.perspective(mat4.create(), 50 * Math.PI / 180.0,
                canvas.width / canvas.height, 0.1, 1000);
            camera = new ArcballCamera(defaultEye, center, up, 2, [canvas.width, canvas.height]);
            controller.registerForCanvas(canvas);
        };

        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

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