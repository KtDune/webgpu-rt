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

    async function initRasterization(device: GPUDevice, glbFile: any, swapChainFormat: string, canvas: HTMLCanvasElement) {
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

        const shaderCache = new GLBShaderCache(device);

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

    async function initRaytrace(device: GPUDevice, glbFile: any, canvas: HTMLCanvasElement) {
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

        const trianglesBuffer = device.createBuffer({
            size: 28 * Float32Array.BYTES_PER_ELEMENT * triangles.length,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // UPLOAD TRIANGLES
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

    async function renderRaytrace(
        commandEncoder: GPUCommandEncoder,
        context: GPUCanvasContext,
        camera: ArcballCamera,
        canvas: HTMLCanvasElement,
        rt: Awaited<ReturnType<typeof initRaytrace>>
    ) {
        const upVec3 = new Float32Array([camera.upDir()[0], camera.upDir()[1], camera.upDir()[2]]);
        const forwardVec3 = new Float32Array([camera.eyeDir()[0], camera.eyeDir()[1], camera.eyeDir()[2]]);
        const rightVec3 = vec3.create();
        vec3.cross(rightVec3, forwardVec3, upVec3);
        vec3.normalize(rightVec3, rightVec3);

        const sceneParamsUpdateData = new Float32Array(16);
        sceneParamsUpdateData.set([camera.eyePos()[0], camera.eyePos()[1], camera.eyePos()[2]], 0);
        sceneParamsUpdateData.set([camera.eyeDir()[0], camera.eyeDir()[1], camera.eyeDir()[2]], 4);
        sceneParamsUpdateData.set([rightVec3[0], rightVec3[1], rightVec3[2]], 8);
        sceneParamsUpdateData.set([rt.maxBounces], 11);
        sceneParamsUpdateData.set([camera.upDir()[0], camera.upDir()[1], camera.upDir()[2]], 12);
        sceneParamsUpdateData.set([rt.triangleCount], 15);

        const sceneParamsUpdateBuffer = device.createBuffer({
            size: 16 * Float32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(sceneParamsUpdateBuffer, 0, sceneParamsUpdateData, 0);
        commandEncoder.copyBufferToBuffer(sceneParamsUpdateBuffer, 0, rt.sceneParamsBuffer, 0, 16 * Float32Array.BYTES_PER_ELEMENT);

        // Compute pass
        const rayTracerPass = commandEncoder.beginComputePass();
        rayTracerPass.setPipeline(rt.rayTracingPipeline);
        rayTracerPass.setBindGroup(0, rt.rayTracingBindGroup);
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
        renderPass.setPipeline(rt.screenPipeline);
        renderPass.setBindGroup(0, rt.screenBindGroup);
        renderPass.draw(6, 1, 0, 0);
        renderPass.end();
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

        let raster = await initRasterization(device, glbFile, swapChainFormat, canvas);
        let raytrace = await initRaytrace(device, glbFile, canvas);

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

            if (!rtMode) {
                raster.renderPassDesc.colorAttachments[0].view = context.getCurrentTexture().createView();

                projView = mat4.mul(projView, proj, camera.camera);
                var upload = device.createBuffer({
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

                var utilsUploadBuf = device.createBuffer({
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

                device.queue.submit([commandEncoder.finish()]);
                await device.queue.onSubmittedWorkDone();

                upload.destroy();
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

                device.queue.submit([commandEncoder.finish()]);
                await device.queue.onSubmittedWorkDone();
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

const screen_shader = `
@group(0) @binding(0)
var screen_sampler: sampler;
@group(0) @binding(1)
var screen_texture: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>
}

@vertex
fn vert_main(@builtin(vertex_index) VertexIndex: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2(1.0, 1.0),
        vec2(1.0, -1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(-1.0, 1.0)
    );

    var texCoords = array<vec2<f32>, 6>(
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 0.0)
    );

    var output: VertexOutput;
    output.position = vec4<f32>(positions[VertexIndex], 0.0, 1.0);
    output.uv = texCoords[VertexIndex];
    return output;
}

@fragment
fn frag_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(screen_texture, screen_sampler, input.uv);
}
`

const raytracer_kernel = `
@group(0) @binding(0)
var color_buffer: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(1)
var<uniform> scene: SceneData;

@group(0) @binding(2)
var<storage, read> primitives: PrimitiveData;

struct PrimitiveData {
    triangles: array<Triangle>
}

struct Triangle {
    corner_a: vec3<f32>,
    normal_a: vec3<f32>,
    corner_b: vec3<f32>,
    normal_b: vec3<f32>,
    corner_c: vec3<f32>,
    normal_c: vec3<f32>,
    color: vec3<f32>
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>
}

struct SceneData {
    cameraPos: vec3<f32>,
    cameraForward: vec3<f32>,
    cameraRight: vec3<f32>,
    maxBounces: f32,
    cameraUp: vec3<f32>,
    trianglesCount: f32
}

struct RenderState {
    t: f32,
    color: vec3<f32>,
    hit: bool,
    scatter_direction: vec3<f32>,
}

@compute @workgroup_size(1,1,1)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let screen_size: vec2<u32> = textureDimensions(color_buffer);
    let screen_pos: vec2<i32> = vec2(i32(GlobalInvocationID.x), i32(GlobalInvocationID.y));

    let horizontal_coefficient: f32 = (f32(screen_pos.x) - f32(screen_size.x) / 2.0) / f32(screen_size.x);
    let vertical_coefficient: f32 = (f32(screen_pos.y) - f32(screen_size.y) / 2.0) / f32(screen_size.y);
    
    let forwards: vec3<f32> = scene.cameraForward;
    let right: vec3<f32> = scene.cameraRight;
    let up: vec3<f32> = scene.cameraUp;

    var ray: Ray = Ray(scene.cameraPos, normalize(forwards + right * horizontal_coefficient + up * vertical_coefficient));

    var pixel_color: vec3<f32> = rayColor(ray);

    textureStore(color_buffer, screen_pos, vec4(pixel_color, 1.0));
}

fn rayColor(ray: Ray) -> vec3<f32> {
    var color: vec3<f32> = vec3(1.0, 1.0, 1.0);
    var result: RenderState;
   
    var worldRay: Ray;
    worldRay.origin = ray.origin;
    worldRay.direction = ray.direction;

    var bounce: u32 = u32(0);
    while (bounce < u32(scene.maxBounces)) {
        // we will bounce a certain number of times
        result.hit = false;
        result.t = 1.0e30;

        for (var t: u32 = u32(0); t < u32(scene.trianglesCount); t++) {
            // find the closest triangle
            result = hit_triangle(worldRay, primitives.triangles[t], 0.001, result.t, result);
        }

        
        if (!result.hit) {
            // if we didn't hit anything we will break
            break;
        } else {
            // if we hit something we will update the ray and accumulated color
            worldRay.origin = worldRay.origin + result.t * worldRay.direction;
            worldRay.direction = result.scatter_direction;
            color *= result.color;
        }
        bounce++;
    }

    if (bounce >= u32(scene.maxBounces)) {
        // if we were still hitting stuff when we are bouncing we will set the color to black (mean we are in a shadow)
        color = vec3(0.0, 0.0, 0.0);
    }

    return color;
}

fn hit_triangle(ray:Ray, triangle: Triangle, tMin: f32, tMax: f32, oldRenderState: RenderState) -> RenderState {
    // TODO: precompute surface normal and pass in with triangle
    var edgeAB: vec3<f32> = triangle.corner_b - triangle.corner_a;
    var edgeAC: vec3<f32> = triangle.corner_c - triangle.corner_a;
    var surface_normal: vec3<f32> = cross(edgeAB, edgeAC);

    var tri_normal_dot_ray_dir: f32 = dot(surface_normal, ray.direction);
    var front_face: bool = tri_normal_dot_ray_dir < 0.0;
    if (!front_face) {
        // flip normal if ray hits back face
        // surface_normal = -surface_normal;
        // tri_normal_dot_ray_dir = -tri_normal_dot_ray_dir;
        //TODO: if we ever need to send rays through objects (refraction) we cannot simply ignore back faces
        return oldRenderState;
    }

    if (tri_normal_dot_ray_dir > -0.00001) {
        // ray is parallel to triangle
        return oldRenderState;
    }

    var d = dot(surface_normal, triangle.corner_a); //TODO this could be in tri data
    var t = (d - dot(surface_normal, ray.origin)) / tri_normal_dot_ray_dir;
    if (t < tMin || t > tMax) {
        return oldRenderState;
    }

    // cramer's rule to solve for barycentric coordinates
    // TODO: see if I can make the barycentric coord code more clear
    var intersection_point: vec3<f32> = ray.origin + t * ray.direction;
    var plane_intersection_point: vec3<f32> = intersection_point - triangle.corner_a;
    var w = surface_normal / dot(surface_normal, surface_normal);

    var u = dot(w, cross(plane_intersection_point, edgeAC));
    if (u < 0.0 || u > 1.0) {
        return oldRenderState;
    }

    var v = dot(w, cross(edgeAB, plane_intersection_point));
    if (v < 0.0 || u + v > 1.0) {
        return oldRenderState;
    }

    let random_unit_vector: vec3<f32> = normalize(random_in_unit_sphere());
    var scatter_direction: vec3<f32> = random_unit_vector + vec3(0.0, 0.0, -1.0);
    if (length(scatter_direction) < 0.0001) {
     scatter_direction = vec3(0.0, 0.0, -1.0);
    }

    var renderState: RenderState;
    renderState.color = oldRenderState.color;
    renderState.scatter_direction = normalize(scatter_direction);
    renderState.t = t;
    renderState.hit = true;
    renderState.color = triangle.color;

    return renderState;
}

fn random_in_unit_sphere() -> vec3<f32> {
    var random_vector: vec3<f32> = vec3( 2.0 * random(vec2(0.0, 0.0)) - 1.0, 2.0 * random(vec2(1.0, 1.0)) - 1.0, 2.0 * random(vec2(2.0, 2.0)) - 1.0);
    while (dot(random_vector, random_vector) >= 1.0) {
        random_vector = vec3( 2.0 * random(vec2(0.0, 0.0)) - 1.0, 2.0 * random(vec2(1.0, 1.0)) - 1.0, 2.0 * random(vec2(2.0, 2.0)) - 1.0);
    }
    return random_vector;
}

fn random(uv: vec2<f32>) -> f32 {
    return fract(sin(dot(uv, vec2(12.9898, 78.233))) * 43758.5453123);
}
`
