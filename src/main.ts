import { vec3, mat4 } from "gl-matrix";
import { ArcballCamera } from "./class/ArcballCamera";
import { Controller } from "./class/Controller";
import { GLBShaderCache } from "./class/GLBShaderCache";
import { uploadGLBModel } from "./uploadGlb";
import { WebUI } from "./class/WebUI";

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

    var canvas = document.getElementById("webgpu-canvas");
    var context = canvas.getContext("webgpu");
    var swapChainFormat = "bgra8unorm";
    context.configure(
        { device: device, format: swapChainFormat, usage: GPUTextureUsage.RENDER_ATTACHMENT });

    var depthTexture = device.createTexture({
        size: { width: canvas.width, height: canvas.height, depthOrArrayLayers: 1 },
        format: "depth24plus-stencil8",
        usage: GPUTextureUsage.RENDER_ATTACHMENT
    });

    var renderPassDesc = {
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

    var viewParamsLayout = device.createBindGroupLayout({
        label: 'View Params Layout',
        entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }]
    });

    var viewParamBuf = device.createBuffer(
        { size: 4 * 4 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    var viewParamsBindGroup = device.createBindGroup(
        { label: 'View Params Bind Group', layout: viewParamsLayout, entries: [{ binding: 0, resource: { buffer: viewParamBuf } }] });

    var utilsLayout = device.createBindGroupLayout({
        label: 'Utils Layout',
        entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }]
    });

    // camera_matrix: 16 bytes
    // light position: 16 bytes
    // pbr: 4 bytes
    var utilsBuf = device.createBuffer(
        { size: (4 + 4) * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    var utilsBindGroup = device.createBindGroup(
        { layout: utilsLayout, entries: [{ binding: 0, resource: { buffer: utilsBuf } }] });

    var shaderCache = new GLBShaderCache(device);

    var renderBundles = glbFile.buildRenderBundles(
        device,
        shaderCache,
        viewParamsLayout,
        viewParamsBindGroup,
        utilsLayout,
        utilsBindGroup,
        swapChainFormat
    );

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

    // Setup onchange listener for file uploads
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
            renderBundles = glbFile.buildRenderBundles(
                device,
                shaderCache,
                viewParamsLayout,
                viewParamsBindGroup,
                utilsLayout,
                utilsBindGroup,
                swapChainFormat
            );
            camera =
                new ArcballCamera(defaultEye, center, up, 2, [canvas.width, canvas.height]);
            glbBuffer = null;
        }

        var start = performance.now();
        renderPassDesc.colorAttachments[0].view = context.getCurrentTexture().createView();

        var commandEncoder = device.createCommandEncoder();

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

        commandEncoder.copyBufferToBuffer(upload, 0, viewParamBuf, 0, 4 * 4 * 4);
        commandEncoder.copyBufferToBuffer(utilsUploadBuf, 0, utilsBuf, 0, utilsData.byteLength);

        var renderPass = commandEncoder.beginRenderPass(renderPassDesc);

        renderPass.executeBundles(renderBundles);

        renderPass.end();
        device.queue.submit([commandEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        var end = performance.now();
        numFrames += 1;
        totalTimeMS += end - start;
        fpsDisplay.innerHTML = `Avg. FPS ${Math.round(1000.0 * numFrames / totalTimeMS)}`;
        requestAnimationFrame(render);
        upload.destroy();
    };

    const resizeCanvas = () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        depthTexture.destroy();
        depthTexture = device.createTexture({
            size: { width: canvas.width, height: canvas.height, depthOrArrayLayers: 1 },
            format: "depth24plus-stencil8",
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });
        renderPassDesc.depthStencilAttachment.view = depthTexture.createView();

        proj = mat4.perspective(mat4.create(), 50 * Math.PI / 180.0,
            canvas.width / canvas.height, 0.1, 1000);
        camera = new ArcballCamera(defaultEye, center, up, 2, [canvas.width, canvas.height]);
        controller.registerForCanvas(canvas);  // Re-register!
    };

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    requestAnimationFrame(render);
})();
