import { vec3, mat4 } from "gl-matrix";
import { ArcballCamera } from "./class/ArcballCamera";
import { Controller } from "./class/Controller";
import { GLBShaderCache } from "./class/GLBShaderCache";
import { GLTFMaterial, uploadGLBModel } from "./uploadGlb";
import { WebUI } from "./class/WebUI";
import { BVHTree } from "./class/BVH";

document.addEventListener("DOMContentLoaded", () => {
    const rasterRadio = document.getElementById("mode-raster") as HTMLInputElement;
    const raytraceRadio = document.getElementById("mode-raytrace") as HTMLInputElement;
    var rtMode = false
    let defaultEye = vec3.set(vec3.create(), 0.0, 0.0, 1.0);
    const center = vec3.set(vec3.create(), 0.0, 0.0, 0.0);
    let up = vec3.set(vec3.create(), 0.0, 1.0, 0.0);
    var camera = null;

    rasterRadio.addEventListener("change", () => {
        if (rasterRadio.checked) {
            rtMode = false;
            up = vec3.set(vec3.create(), 0.0, 1.0, 0.0);
            defaultEye = vec3.set(vec3.create(), 0.0, 0.0, 1.0);
            camera?.reset(defaultEye, center, up);
        }
    });

    raytraceRadio.addEventListener("change", () => {
        if (raytraceRadio.checked) {
            rtMode = true;
            up = vec3.set(vec3.create(), 0.0, -1.0, 0.0);
            defaultEye = vec3.set(vec3.create(), 0.0, 0.0, -1.0);
            camera?.reset(defaultEye, center, up);
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
            minFilter: "linear",
            mipmapFilter: "nearest",
            maxAnisotropy: 1
        });

        const sceneParamsBuffer = device.createBuffer({
            size: 20 * Float32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const triangles: Triangle[] = glbFile.triangles;
        const bvhTree = new BVHTree(triangles);
        const materials: GLTFMaterial[] = glbFile.materials

        // For now assume only one material
        const material = materials[0];
        let baseColorTextureView: GPUTextureView;
        if (material.baseColorTexture) {
            baseColorTextureView = material.baseColorTexture.imageView!;
        }
        else {
            const solidColorTexture = createSolidColorTexture(device, 1, 0, 0, 1);
            baseColorTextureView = solidColorTexture.createView();
        }

        const trianglesBuffer = device?.createBuffer({
            size: 40 * Float32Array.BYTES_PER_ELEMENT * triangles.length,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const trianglesUploadData = new Float32Array(triangles.length * 40);
        for (let i = 0; i < triangles.length; i++) {
            const triangle = triangles[i];
            const defaultNormal = new Float32Array([0, 0, 1]);  // forward-facing normal, no lighting distortion
            const defaultUv = new Float32Array([1, 1, 1]);          // zero UVs, no texture sampling effect

            trianglesUploadData.set(triangle.positions[0], i * 40);
            trianglesUploadData.set(triangle.normals[0] ?? defaultNormal[0], i * 40 + 4);
            trianglesUploadData.set(triangle.positions[1], i * 40 + 8);
            trianglesUploadData.set(triangle.normals[1] ?? defaultNormal[1], i * 40 + 12);
            trianglesUploadData.set(triangle.positions[2], i * 40 + 16);
            trianglesUploadData.set(triangle.normals[2] ?? defaultNormal[2], i * 40 + 20);
            trianglesUploadData.set(triangle.uv[0] ?? defaultUv[0], i * 40 + 24);
            trianglesUploadData.set(triangle.uv[1] ?? defaultUv[1], i * 40 + 26);
            trianglesUploadData.set(triangle.uv[2] ?? defaultUv[2], i * 40 + 28);
            trianglesUploadData.set([triangle.ior], i * 40 + 32);
            trianglesUploadData.set([triangle.metalness], i * 40 + 36);
        }
        device?.queue.writeBuffer(trianglesBuffer, 0, trianglesUploadData, 0);

        const bvh_nodes_buffer_descriptor: GPUBufferDescriptor = {
            size: 8 * Float32Array.BYTES_PER_ELEMENT * bvhTree.nodes.length,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        };
        const bvh_nodes_buffer: GPUBuffer = device.createBuffer(bvh_nodes_buffer_descriptor);

        const triangle_indices_buffer_descriptor: GPUBufferDescriptor = {
            size: bvhTree.triangles.length * Float32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        };
        const triangle_indices_buffer: GPUBuffer = device.createBuffer(triangle_indices_buffer_descriptor);

        const rayTracingBindGroupLayout = device?.createBindGroupLayout({
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
                    buffer: { type: 'uniform' },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' },
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: {},
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    sampler: {},
                },
                {
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" }
                },
                {
                    binding: 6,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" }
                },
            ],
        });
        const rayTracingBindGroup = device?.createBindGroup({
            layout: rayTracingBindGroupLayout,
            entries: [
                { binding: 0, resource: colorBufferView },
                { binding: 1, resource: { buffer: sceneParamsBuffer } },
                { binding: 2, resource: { buffer: trianglesBuffer } },
                { binding: 3, resource: baseColorTextureView },
                { binding: 4, resource: sampler },
                { binding: 5, resource: { buffer: bvh_nodes_buffer } },
                { binding: 6, resource: { buffer: triangle_indices_buffer } },
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
        const maxBounces = 20;
        const sceneParamsUploadData = new Float32Array(18);
        sceneParamsUploadData.set([0, 0, 5], 0);
        sceneParamsUploadData.set([0, 0, 1], 4);
        sceneParamsUploadData.set([-1, 0, 0], 8);
        sceneParamsUploadData.set([maxBounces], 11);
        sceneParamsUploadData.set([0, 1, 0], 12);
        sceneParamsUploadData.set([triangles.length], 15);
        sceneParamsUploadData.set([canvas.width / canvas.height], 16);
        sceneParamsUploadData.set([50 * Math.PI / 180.0], 17);
        device.queue.writeBuffer(sceneParamsBuffer, 0, sceneParamsUploadData, 0);

        const triangleIndicesUploadData = new Float32Array(bvhTree.triangleIndices.length);
        for (let i = 0; i < bvhTree.triangleIndices.length; i++) {
            triangleIndicesUploadData.set([bvhTree.triangleIndices[i]], i);
        }
        device?.queue.writeBuffer(triangle_indices_buffer, 0, triangleIndicesUploadData, 0);

        // UPLOAD BVH NODES
        const bvhNodesUploadData = new Float32Array(bvhTree.nodesUsed * 8);
        for (let i = 0; i < bvhTree.nodesUsed; i++) {
            bvhNodesUploadData.set(bvhTree.nodes[i].minCorner, i * 8);
            bvhNodesUploadData.set([bvhTree.nodes[i].left], i * 8 + 3);
            bvhNodesUploadData.set(bvhTree.nodes[i].maxCorner, i * 8 + 4);
            bvhNodesUploadData.set([bvhTree.nodes[i].primitiveCount], i * 8 + 7);
        }
        device?.queue.writeBuffer(bvh_nodes_buffer, 0, bvhNodesUploadData, 0);

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

        camera = new ArcballCamera(defaultEye, center, up, 2, [canvas.width, canvas.height]);
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
                raytrace = await initRaytrace(device, glbFile, canvas, shaderCache);
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

                const sceneParamsUpdateData = new Float32Array(18);
                sceneParamsUpdateData.set([camera.eyePos()[0], camera.eyePos()[1], camera.eyePos()[2]], 0);
                sceneParamsUpdateData.set([camera.eyeDir()[0], camera.eyeDir()[1], camera.eyeDir()[2]], 4);
                sceneParamsUpdateData.set([rightVec3[0], rightVec3[1], rightVec3[2]], 8);
                sceneParamsUpdateData.set([raytrace.maxBounces], 11);
                sceneParamsUpdateData.set([camera.upDir()[0], camera.upDir()[1], camera.upDir()[2]], 12);
                sceneParamsUpdateData.set([raytrace.triangleCount], 15);
                sceneParamsUpdateData.set([canvas.width / canvas.height], 16);
                console.log(sceneParamsUpdateData[16])
                sceneParamsUpdateData.set([50 * Math.PI / 180.0], 17);

                const sceneParamsUpdateBuffer = device.createBuffer({
                    size: 18 * Float32Array.BYTES_PER_ELEMENT,
                    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
                });
                device.queue.writeBuffer(sceneParamsUpdateBuffer, 0, sceneParamsUpdateData, 0);
                commandEncoder.copyBufferToBuffer(sceneParamsUpdateBuffer, 0, raytrace.sceneParamsBuffer, 0, 18 * Float32Array.BYTES_PER_ELEMENT);
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
                upload?.destroy();
                utilsUploadBuf?.destroy();
            }

            var end = performance.now();
            numFrames += 1;
            totalTimeMS += end - start;
            fpsDisplay.innerHTML = `Avg. FPS ${Math.round(1000.0 * numFrames / totalTimeMS)}`;
            requestAnimationFrame(render);
        };

        const resizeCanvas = async () => {
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

            raytrace = await initRaytrace(device, glbFile, canvas, shaderCache);
        };

        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        requestAnimationFrame(render);
    })();
});

function createSolidColorTexture(device: GPUDevice, r: number, g: number, b: number, a: number) {
    const data = new Uint8Array([r * 255, g * 255, b * 255, a * 255]);
    const texture = device.createTexture({
        size: { width: 1, height: 1 },
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
    });
    device.queue.writeTexture({ texture }, data, {}, { width: 1, height: 1 });
    return texture;
}

function resetCamera(camera, defaultEye, center, up) {
    camera.reset(defaultEye, center, up)
}