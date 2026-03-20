import { mat4, vec3, vec4 } from "gl-matrix";

const GLTFRenderMode = {
    POINTS: 0,
    LINE: 1,
    LINE_LOOP: 2,
    LINE_STRIP: 3,
    TRIANGLES: 4,
    TRIANGLE_STRIP: 5,
    // Note: fans are not supported in WebGPU, use should be
    // an error or converted into a list/strip
    TRIANGLE_FAN: 6,
};

const GLTFComponentType = {
    BYTE: 5120,
    UNSIGNED_BYTE: 5121,
    SHORT: 5122,
    UNSIGNED_SHORT: 5123,
    INT: 5124,
    UNSIGNED_INT: 5125,
    FLOAT: 5126,
    DOUBLE: 5130,
};

const GLTFTextureFilter = {
    NEAREST: 9728,
    LINEAR: 9729,
    NEAREST_MIPMAP_NEAREST: 9984,
    LINEAR_MIPMAP_NEAREST: 9985,
    NEAREST_MIPMAP_LINEAR: 9986,
    LINEAR_MIPMAP_LINEAR: 9987,
};

const GLTFTextureWrap = {
    REPEAT: 10497,
    CLAMP_TO_EDGE: 33071,
    MIRRORED_REPEAT: 33648,
};

function alignTo(val, align) {
    return Math.floor((val + align - 1) / align) * align;
}

function gltfTypeNumComponents(type) {
    switch (type) {
        case 'SCALAR':
            return 1;
        case 'VEC2':
            return 2;
        case 'VEC3':
            return 3;
        case 'VEC4':
            return 4;
        default:
            alert('Unhandled glTF Type ' + type);
            return null;
    }
}

function gltfTypeSize(componentType, type) {
    var typeSize = 0;
    switch (componentType) {
        case GLTFComponentType.BYTE:
            typeSize = 1;
            break;
        case GLTFComponentType.UNSIGNED_BYTE:
            typeSize = 1;
            break;
        case GLTFComponentType.SHORT:
            typeSize = 2;
            break;
        case GLTFComponentType.UNSIGNED_SHORT:
            typeSize = 2;
            break;
        case GLTFComponentType.INT:
            typeSize = 4;
            break;
        case GLTFComponentType.UNSIGNED_INT:
            typeSize = 4;
            break;
        case GLTFComponentType.FLOAT:
            typeSize = 4;
            break;
        case GLTFComponentType.DOUBLE:
            typeSize = 4;
            break;
        default:
            alert('Unrecognized GLTF Component Type?');
    }
    return gltfTypeNumComponents(type) * typeSize;
}

export class Triangle {
    private _positions: Float32Array[];
    private _normals: Float32Array[];
    private _uv: Float32Array[];
    private _centroid: Float32Array;
    private _ior: number;
    private _metalness: number;
    private _roughness: number;
    private _surface_normal: Float32Array;
    private _surface_normal_d: number;
    private _barycentric_w: Float32Array;

    constructor(
        positions: Float32Array[],
        normals: Float32Array[],
        uv: Float32Array[],
        ior?: number,
        metalness?: number,
        roughness?: number
    ) {
        this._positions = positions;
        this._normals = normals;
        this._uv = uv;
        this._ior = ior ?? 1.5;
        this._metalness = metalness ?? 0.0;
        this._roughness = roughness ?? 0.5;

        // Centroid
        this._centroid = new Float32Array([0, 0, 0]);
        const weight = 1 / 3;
        for (const position of positions) {
            this._centroid[0] += position[0] * weight;
            this._centroid[1] += position[1] * weight;
            this._centroid[2] += position[2] * weight;
        }

        const edgeAB = vec3.sub(vec3.create(), positions[1], positions[0]);
        const edgeAC = vec3.sub(vec3.create(), positions[2], positions[0]);
        this._surface_normal = vec3.cross(vec3.create(), edgeAB, edgeAC);

        this._surface_normal_d = vec3.dot(this._surface_normal, positions[0]);

        const lenSq = vec3.dot(this._surface_normal, this._surface_normal);
        this._barycentric_w = vec3.scale(vec3.create(), this._surface_normal, 1.0 / lenSq);
    }

    get positions(): Float32Array[] { return this._positions; }
    get normals(): Float32Array[] { return this._normals; }
    get uv(): Float32Array[] { return this._uv; }
    get centroid(): Float32Array { return this._centroid; }
    get ior(): number { return this._ior; }
    get metalness(): number { return this._metalness; }
    get roughness(): number { return this._roughness; }
    get surfaceNormal(): Float32Array { return this._surface_normal; }
    get surfaceNormalD(): number { return this._surface_normal_d; }
    get barycentricW(): Float32Array { return this._barycentric_w; }
}

export class GLTFBuffer {
    constructor(buffer, size, offset) {
        this.arrayBuffer = buffer;
        this.size = size;
        this.byteOffset = offset;
    }
}

export class GLTFBufferView {
    constructor(buffer, view) {
        this.length = view['byteLength'];
        this.byteOffset = buffer.byteOffset;
        if (view['byteOffset'] !== undefined) {
            this.byteOffset += view['byteOffset'];
        }
        this.byteStride = 0;
        if (view['byteStride'] !== undefined) {
            this.byteStride = view['byteStride'];
        }
        this.buffer = new Uint8Array(buffer.arrayBuffer, this.byteOffset, this.length);

        this.needsUpload = false;
        this.gpuBuffer = null;
        this.usage = 0;
    }

    get elements(): Uint8Array {
        return this.buffer;
    }

    addUsage(usage) {
        this.usage = this.usage | usage;
    }

    upload(device) {
        // Note: must align to 4 byte size when mapped at creation is true
        var buf = device.createBuffer({
            size: alignTo(this.buffer.byteLength, 4),
            usage: this.usage,
            mappedAtCreation: true
        });
        new (this.buffer.constructor)(buf.getMappedRange()).set(this.buffer);
        buf.unmap();
        this.gpuBuffer = buf;
        this.needsUpload = false;
    }
}

export class GLTFAccessor {
    constructor(view, accessor) {
        this.count = accessor['count'];
        this.componentType = accessor['componentType'];
        this.gltfType = accessor['type'];
        this.numComponents = gltfTypeNumComponents(accessor['type']);
        this.numScalars = this.count * this.numComponents;
        this.view = view;
        this.byteOffset = 0;
        this.byteLength = this.count * this.byteStride
        if (accessor['byteOffset'] !== undefined) {
            this.byteOffset = accessor['byteOffset'];
        }
    }

    get byteStride() {
        var elementSize = gltfTypeSize(this.componentType, this.gltfType);
        return Math.max(elementSize, this.view.byteStride);
    }

    get elements(): Uint8Array {
        return this.view.elements.slice(this.byteOffset, this.byteOffset + this.byteLength);
    }
}

export class GLTFPrimitive {
    constructor(indices, positions, normals, texcoords, material, topology, triangles) {
        this.indices = indices;
        this.positions = positions;
        this.normals = normals;
        this.texcoords = texcoords;
        this.material = material;
        this.triangles = triangles;
        this.topology = topology;
        this.doubleSided = material['doubleSided'];
        this.alphaMode = material['alphaMode'];
    }

    // Build the primitive render commands into the bundle
    async buildRenderBundle(
        device, shaderCache, bindGroupLayouts, bundleEncoder, swapChainFormat, depthFormat) {
        var shaderModule = shaderCache.getShader(
            this.normals, this.texcoords.length > 0, this.material.baseColorTexture, this.material);

        var vertexBuffers = [{
            arrayStride: this.positions.byteStride,
            attributes: [{ format: 'float32x3', offset: 0, shaderLocation: 0 }]
        }];

        if (this.normals) {
            vertexBuffers.push({
                arrayStride: this.normals.byteStride,
                attributes: [{ format: 'float32x3', offset: 0, shaderLocation: 1 }]
            });
        }

        // TODO: Multi-texturing
        if (this.texcoords.length > 0) {
            vertexBuffers.push({
                arrayStride: this.texcoords[0].byteStride,
                attributes: [{ format: 'float32x2', offset: 0, shaderLocation: 2 }]
            });
        }

        var layout = device.createPipelineLayout({
            label: 'Pipeline Layout',
            bindGroupLayouts:
                [bindGroupLayouts[0], bindGroupLayouts[1], this.material.bindGroupLayout, bindGroupLayouts[2]],
        });

        var vertexStage = {
            module: shaderModule,
            entryPoint: 'vertex_main',
            buffers: vertexBuffers
        };
        var fragmentStage = {
            module: shaderModule,
            entryPoint: 'fragment_main',
            targets: [{
                format: swapChainFormat,
                blend: this.alphaMode === 'BLEND' ? {
                    color: { operation: 'add', srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
                    alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' }
                } : undefined,
                writeMask: GPUColorWrite.ALL
            }]
        };

        var primitive = {
            topology: 'triangle-list',
            cullingMode: this.doubleSided ? 'none' : 'back',
        };
        if (this.topology == GLTFRenderMode.TRIANGLE_STRIP) {
            primitive.topology = 'triangle-strip';
            primitive.stripIndexFormat =
                this.indices.componentType == GLTFComponentType.UNSIGNED_SHORT ? 'uint16'
                    : 'uint32';
        }

        var pipelineDescriptor = {
            layout: layout,
            vertex: vertexStage,
            fragment: fragmentStage,
            primitive: primitive,
            depthStencil: { format: depthFormat, depthWriteEnabled: true, depthCompare: 'less' }
        };

        var renderPipeline = await device.createRenderPipelineAsync(pipelineDescriptor);

        bundleEncoder.setPipeline(renderPipeline);
        bundleEncoder.setBindGroup(2, this.material.bindGroup);

        bundleEncoder.setVertexBuffer(0,
            this.positions.view.gpuBuffer,
            this.positions.byteOffset,
            this.positions.length);
        if (this.normals) {
            bundleEncoder.setVertexBuffer(
                1, this.normals.view.gpuBuffer, this.normals.byteOffset, this.normals.length);
        }
        if (this.texcoords.length > 0) {
            bundleEncoder.setVertexBuffer(2,
                this.texcoords[0].view.gpuBuffer,
                this.texcoords[0].byteOffset,
                this.texcoords[0].length);
        }
        if (this.indices) {
            var indexFormat = this.indices.componentType == GLTFComponentType.UNSIGNED_SHORT
                ? 'uint16'
                : 'uint32';
            bundleEncoder.setIndexBuffer(
                this.indices.view.gpuBuffer,
                indexFormat,
                this.indices.byteOffset,
                gltfTypeSize(this.indices.componentType, this.indices.gltfType) * this.indices.count
            );

            bundleEncoder.drawIndexed(this.indices.count);
        } else {
            bundleEncoder.draw(this.positions.count);
        }
    }
}

export class GLTFMesh {
    constructor(name, primitives) {
        this.name = name;
        this.primitives = primitives;
    }

    get triangles(): Triangle[] {
        return this.primitives.flatMap(primitive => primitive.triangles);
    }

    get materials(): GLTFMaterial[] {
        return this.primitives.flatMap(primitive => primitive.material);
    }
}

export class GLTFNode {
    constructor(name, mesh, transform) {
        this.name = name;
        this.mesh = mesh;
        this.transform = transform;

        this.gpuUniforms = null;
        this.bindGroup = null;
    }

    get triangles(): Triangle[] {
        const transformedTriangles: Triangle[] = [];

        // REMOVE the flipMatrix and combinedTransform — just use this.transform directly
        const normalMatrix = mat4.create();
        mat4.invert(normalMatrix, this.transform);
        mat4.transpose(normalMatrix, normalMatrix);

        for (const triangle of this.mesh.triangles) {
            const transformedPositions = triangle.positions
                .map(p => new Float32Array(vec3.transformMat4(vec3.create(), p, this.transform)));

            const transformedNormals = triangle.normals
                .map(n => {
                    const t = vec4.transformMat4(vec4.create(), [n[0], n[1], n[2], 0.0], normalMatrix);
                    return new Float32Array([t[0], t[1], t[2]]);
                });

            transformedTriangles.push(new Triangle(
                transformedPositions,
                transformedNormals,
                triangle.uv,
                triangle.ior,
                triangle.metalness,
                triangle.roughness
            ));
        }
        return transformedTriangles;
    }

    get materials(): GLTFMaterial[] {
        return this.mesh.materials;
    }

    upload(device) {
        var buf = device.createBuffer(
            { size: 4 * 4 * 4, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
        new Float32Array(buf.getMappedRange()).set(this.transform);
        buf.unmap();
        this.gpuUniforms = buf;
    }

    async buildRenderBundle(
        device,
        shaderCache,
        viewParamsLayout,
        viewParamsBindGroup,
        utilsLayout,
        utilsBindGroup,
        swapChainFormat,
        depthFormat) {
        var nodeParamsLayout = device.createBindGroupLayout({
            label: 'Node Params Layout',
            entries:
                [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }]
        });

        this.bindGroup = device.createBindGroup({
            label: 'Node Params Bind Group',
            layout: nodeParamsLayout,
            entries: [{ binding: 0, resource: { buffer: this.gpuUniforms } }]
        });

        var bindGroupLayouts = [viewParamsLayout, nodeParamsLayout, utilsLayout];

        var bundleEncoder = device.createRenderBundleEncoder({
            colorFormats: [swapChainFormat],
            depthStencilFormat: depthFormat,
        });

        bundleEncoder.setBindGroup(0, viewParamsBindGroup);
        bundleEncoder.setBindGroup(1, this.bindGroup);
        bundleEncoder.setBindGroup(3, utilsBindGroup);

        for (var i = 0; i < this.mesh.primitives.length; ++i) {
            await this.mesh.primitives[i].buildRenderBundle(device,
                shaderCache,
                bindGroupLayouts,
                bundleEncoder,
                swapChainFormat,
                depthFormat);
        }

        this.renderBundle = bundleEncoder.finish();
        return this.renderBundle;
    }
}

function readNodeTransform(node) {
    if (node['matrix']) {
        var m = node['matrix'];
        // Both glTF and gl matrix are column major
        return mat4.fromValues(m[0],
            m[1],
            m[2],
            m[3],
            m[4],
            m[5],
            m[6],
            m[7],
            m[8],
            m[9],
            m[10],
            m[11],
            m[12],
            m[13],
            m[14],
            m[15]);
    } else {
        var scale = [1, 1, 1];
        var rotation = [0, 0, 0, 1];
        var translation = [0, 0, 0];
        if (node['scale']) {
            scale = node['scale'];
        }
        if (node['rotation']) {
            rotation = node['rotation'];
        }
        if (node['translation']) {
            translation = node['translation'];
        }
        var m = mat4.create();
        return mat4.fromRotationTranslationScale(m, rotation, translation, scale);
    }
}

function flattenGLTFChildren(nodes, node, parent_transform) {
    var tfm = readNodeTransform(node);
    var tfm = mat4.mul(tfm, parent_transform, tfm);
    node['matrix'] = tfm;
    node['scale'] = undefined;
    node['rotation'] = undefined;
    node['translation'] = undefined;
    if (node['children']) {
        for (var i = 0; i < node['children'].length; ++i) {
            flattenGLTFChildren(nodes, nodes[node['children'][i]], tfm);
        }
        node['children'] = [];
    }
}

function makeGLTFSingleLevel(nodes) {
    var rootTfm = mat4.create();
    for (var i = 0; i < nodes.length; ++i) {
        flattenGLTFChildren(nodes, nodes[i], rootTfm);
    }
    return nodes;
}

export class GLTFMaterial {
    constructor(material, textures) {
        // In GLTFMaterial constructor
        this.alphaMode = material['alphaMode'] ?? 'OPAQUE';
        this.alphaCutoff = null;
        this.baseColorFactor = [1, 1, 1, 1];
        this.baseColorTexture = null;
        this.emissiveFactor = [0, 0, 0, 1];
        this.emissiveTexture = null
        this.emissiveSampler = null
        this.occlusionTexture = null
        this.occlusionSampler = null
        this.metallicFactor = 1.0;
        this.roughnessFactor = 1.0;
        this.metallicRoughnessTexture = null
        this.metallicRoughnessSampler = null
        this.normalTexture = null
        this.normalSampler = null
        this.doubleSided = material['doubleSided'] || false;

        if (material['pbrMetallicRoughness'] !== undefined) {
            var pbr = material['pbrMetallicRoughness'];
            if (pbr['baseColorFactor'] !== undefined) {
                this.baseColorFactor = pbr['baseColorFactor'];
            }
            if (pbr['baseColorTexture'] !== undefined) {
                // TODO multiple texcoords
                this.baseColorTexture = textures[pbr['baseColorTexture']['index']];
            }
            if (pbr['metallicFactor'] !== undefined) {
                this.metallicFactor = pbr['metallicFactor'];
            }
            if (pbr['roughnessFactor'] !== undefined) {
                this.roughnessFactor = pbr['roughnessFactor'];
            }
            if (pbr['metallicRoughnessTexture'] !== undefined) {
                this.metallicRoughnessTexture = textures[pbr['metallicRoughnessTexture']['index']];
                this.metallicRoughnessSampler = this.metallicRoughnessTexture.sampler;
            }
        }
        if (material['emissiveFactor'] !== undefined) {
            this.emissiveFactor[0] = material['emissiveFactor'][0];
            this.emissiveFactor[1] = material['emissiveFactor'][1];
            this.emissiveFactor[2] = material['emissiveFactor'][2];
        }
        if (material['emissiveTexture'] !== undefined) {
            this.emissiveTexture = textures[material['emissiveTexture']['index']];
            this.emissiveSampler = this.emissiveTexture.sampler;
        }
        if (material['occlusionTexture'] !== undefined) {
            this.occlusionTexture = textures[material['occlusionTexture']['index']];
            this.occlusionSampler = this.occlusionTexture.sampler;
        }
        if (material['normalTexture'] !== undefined) {
            this.normalTexture = textures[material['normalTexture']['index']];
            this.normalSampler = this.normalTexture.sampler;
        }
        if (material['alphaMode'] === 'MASK') {
            this.alphaCutoff = material['alphaCutoff'];
        }

        this.gpuBuffer = null;
        this.bindGroupLayout = null;
        this.bindGroup = null;
    }

    upload(device) {
        var buf = device.createBuffer(
            { size: 3 * 4 * 4, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
        var mappingView = new Float32Array(buf.getMappedRange());
        mappingView.set(this.baseColorFactor);
        mappingView.set(this.emissiveFactor, 4);
        mappingView.set([this.metallicFactor, this.roughnessFactor], 8);
        buf.unmap();
        this.gpuBuffer = buf;

        var layoutEntries =
            [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }];
        var bindGroupEntries = [{
            binding: 0,
            resource: {
                buffer: this.gpuBuffer,
            }
        }];

        if (this.baseColorTexture) {
            // Defaults for sampler and texture are fine, just make the objects
            // exist to pick them up
            layoutEntries.push({ binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} });
            layoutEntries.push({ binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} });

            bindGroupEntries.push({
                binding: 1,
                resource: this.baseColorTexture.sampler,
            });
            bindGroupEntries.push({
                binding: 2,
                resource: this.baseColorTexture.imageView,
            });
        }

        if (this.emissiveTexture) {
            layoutEntries.push({ binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} });
            layoutEntries.push({ binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: {} });

            bindGroupEntries.push({
                binding: 3,
                resource: this.emissiveSampler,
            });
            bindGroupEntries.push({
                binding: 4,
                resource: this.emissiveTexture.imageView,
            });
        }

        if (this.occlusionTexture) {
            layoutEntries.push({ binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: {} });
            layoutEntries.push({ binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: {} });

            bindGroupEntries.push({
                binding: 5,
                resource: this.occlusionSampler,
            });
            bindGroupEntries.push({
                binding: 6,
                resource: this.occlusionTexture.imageView,
            });
        }

        if (this.normalTexture) {
            layoutEntries.push({ binding: 7, visibility: GPUShaderStage.FRAGMENT, sampler: {} });
            layoutEntries.push({ binding: 8, visibility: GPUShaderStage.FRAGMENT, texture: {} });

            bindGroupEntries.push({
                binding: 7,
                resource: this.normalSampler,
            });
            bindGroupEntries.push({
                binding: 8,
                resource: this.normalTexture.imageView,
            });
        }

        if (this.metallicRoughnessTexture) {
            layoutEntries.push({ binding: 9, visibility: GPUShaderStage.FRAGMENT, sampler: {} });
            layoutEntries.push({ binding: 10, visibility: GPUShaderStage.FRAGMENT, texture: {} });

            bindGroupEntries.push({
                binding: 9,
                resource: this.metallicRoughnessSampler,
            });
            bindGroupEntries.push({
                binding: 10,
                resource: this.metallicRoughnessTexture.imageView,
            });
        }
        if (this.alphaMode === 'MASK') {
            layoutEntries.push({ binding: 11, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } });
            bindGroupEntries.push({
                binding: 11,
                resource: {
                    buffer: device.createBuffer({
                        size: 4,
                        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                        mappedAtCreation: true,
                    }),
                },
            });
            new Float32Array(bindGroupEntries[bindGroupEntries.length - 1].resource.buffer.getMappedRange()).set(
                [this.alphaCutoff]);
            bindGroupEntries[bindGroupEntries.length - 1].resource.buffer.unmap();
        }

        this.bindGroupLayout = device.createBindGroupLayout({ label: 'Material Bind Group Layout', entries: layoutEntries });

        this.bindGroup = device.createBindGroup({
            label: 'Material Bind Group',
            layout: this.bindGroupLayout,
            entries: bindGroupEntries,
        });
    }
}

export class GLTFSampler {
    constructor(sampler, device) {
        var magFilter = sampler['magFilter'] === undefined ||
            sampler['magFilter'] == GLTFTextureFilter.LINEAR
            ? 'linear'
            : 'nearest';
        var minFilter = sampler['minFilter'] === undefined ||
            sampler['minFilter'] == GLTFTextureFilter.LINEAR
            ? 'linear'
            : 'nearest';

        var wrapS = 'repeat';
        if (sampler['wrapS'] !== undefined) {
            if (sampler['wrapS'] == GLTFTextureFilter.REPEAT) {
                wrapS = 'repeat';
            } else if (sampler['wrapS'] == GLTFTextureFilter.CLAMP_TO_EDGE) {
                wrapS = 'clamp-to-edge';
            } else {
                wrapS = 'mirror-repeat';
            }
        }

        var wrapT = 'repeat';
        if (sampler['wrapT'] !== undefined) {
            if (sampler['wrapT'] == GLTFTextureFilter.REPEAT) {
                wrapT = 'repeat';
            } else if (sampler['wrapT'] == GLTFTextureFilter.CLAMP_TO_EDGE) {
                wrapT = 'clamp-to-edge';
            } else {
                wrapT = 'mirror-repeat';
            }
        }

        this.sampler = device.createSampler({
            magFilter: magFilter,
            minFilter: minFilter,
            addressModeU: wrapS,
            addressModeV: wrapT,
        });
    }
}

export class GLTFTexture {
    constructor(sampler, image) {
        this.gltfsampler = sampler;
        this.sampler = sampler.sampler;
        this.image = image;
        this.imageView = image.createView();
    }
}

export class GLBModel {
    constructor(nodes) {
        this.nodes = nodes;
    }

    get triangles(): Triangle[] {
        return this.nodes.flatMap(node => node.triangles);
    }

    get materials(): GLTFMaterial[] {
        return this.nodes.flatMap(node => node.materials);
    }

    async buildRenderBundles(
        device,
        shaderCache,
        viewParamsLayout,
        viewParamsBindGroup,
        utilsLayout,
        utilsBindGroup,
        swapChainFormat
    ) {
        var renderBundles = [];
        for (var i = 0; i < this.nodes.length; ++i) {
            var n = this.nodes[i];
            var bundle = await n.buildRenderBundle(
                device,
                shaderCache,
                viewParamsLayout,
                viewParamsBindGroup,
                utilsLayout,
                utilsBindGroup,
                swapChainFormat,
                'depth24plus-stencil8');
            renderBundles.push(bundle);
        }
        return renderBundles;
    }
};

// Upload a GLB model and return it
export async function uploadGLBModel(buffer, device) {
    document.getElementById("loading-text").hidden = false;
    // The file header and chunk 0 header
    // TODO: It sounds like the spec does allow for multiple binary chunks,
    // so then how do you know which chunk a buffer exists in? Maybe the buffer
    // id corresponds to the binary chunk ID? Would have to find info in the
    // spec or an example file to check this
    var header = new Uint32Array(buffer, 0, 5);
    if (header[0] != 0x46546C67) {
        alert('This does not appear to be a glb file?');
        return;
    }
    var glbJsonData =
        JSON.parse(new TextDecoder('utf-8').decode(new Uint8Array(buffer, 20, header[3])));

    var binaryHeader = new Uint32Array(buffer, 20 + header[3], 2);
    var glbBuffer = new GLTFBuffer(buffer, binaryHeader[0], 28 + header[3]);

    if (28 + header[3] + binaryHeader[0] != buffer.byteLength) {
        console.log('TODO: Multiple binary chunks in file');
    }

    // TODO: Later could look at merging buffers and actually using the starting
    // offsets, but want to avoid uploading the entire buffer since it may
    // contain packed images
    var bufferViews = [];
    for (var i = 0; i < glbJsonData.bufferViews.length; ++i) {
        bufferViews.push(new GLTFBufferView(glbBuffer, glbJsonData.bufferViews[i]));
    }

    var images = [];
    if (glbJsonData['images'] !== undefined) {
        for (var i = 0; i < glbJsonData['images'].length; ++i) {
            var imgJson = glbJsonData['images'][i];
            var imageView = new GLTFBufferView(
                glbBuffer, glbJsonData['bufferViews'][imgJson['bufferView']]);
            var imgBlob = new Blob([imageView.buffer], { type: imgJson['mime/type'] });
            var img = await createImageBitmap(imgBlob);

            // TODO: For glTF we need to look at where an image is used to know
            // if it should be srgb or not. We basically need to pass through
            // the material list and find if the texture which uses this image
            // is used by a metallic/roughness param
            var gpuImg = device.createTexture({
                size: [img.width, img.height, 1],
                format: 'rgba8unorm-srgb',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST |
                    GPUTextureUsage.RENDER_ATTACHMENT,
            });

            var src = { source: img };
            var dst = { texture: gpuImg };
            device.queue.copyExternalImageToTexture(src, dst, [img.width, img.height, 1]);

            images.push(gpuImg);
        }
    }

    var defaultSampler = new GLTFSampler({}, device);
    var samplers = [];
    if (glbJsonData['samplers'] !== undefined) {
        for (var i = 0; i < glbJsonData['samplers'].length; ++i) {
            samplers.push(new GLTFSampler(glbJsonData['samplers'][i], device));
        }
    }

    var textures = [];
    if (glbJsonData['textures'] !== undefined) {
        for (var i = 0; i < glbJsonData['textures'].length; ++i) {
            var tex = glbJsonData['textures'][i];
            var sampler =
                tex['sampler'] !== undefined ? samplers[tex['sampler']] : defaultSampler;
            textures.push(new GLTFTexture(sampler, images[tex['source']]));
        }
    }

    var defaultMaterial = new GLTFMaterial({});
    var materials = [];
    for (var i = 0; i < glbJsonData['materials'].length; ++i) {
        materials.push(new GLTFMaterial(glbJsonData['materials'][i], textures));
    }

    if (glbJsonData['extensionsUsed'] !== undefined) {
        alert('This model uses glTF extensions, which are not currently supported');
    }

    var meshes = [];
    for (var i = 0; i < glbJsonData.meshes.length; ++i) {
        var mesh = glbJsonData.meshes[i];

        var primitives = [];
        for (var j = 0; j < mesh.primitives.length; ++j) {
            var prim = mesh.primitives[j];
            var topology = prim['mode'];
            // Default is triangles if mode specified
            if (topology === undefined) {
                topology = GLTFRenderMode.TRIANGLES;
            }
            if (topology != GLTFRenderMode.TRIANGLES &&
                topology != GLTFRenderMode.TRIANGLE_STRIP) {
                alert('Ignoring primitive with unsupported mode ' + prim['mode']);
                continue;
            }

            var indices = null;
            if (glbJsonData['accessors'][prim['indices']] !== undefined) {
                var accessor = glbJsonData['accessors'][prim['indices']];
                var viewID = accessor['bufferView'];
                bufferViews[viewID].needsUpload = true;
                bufferViews[viewID].addUsage(GPUBufferUsage.INDEX);
                indices = new GLTFAccessor(bufferViews[viewID], accessor);
            }

            var positions = null;
            var normals = null;
            var texcoords = [];
            for (var attr in prim['attributes']) {
                var accessor = glbJsonData['accessors'][prim['attributes'][attr]];
                var viewID = accessor['bufferView'];
                bufferViews[viewID].needsUpload = true;
                bufferViews[viewID].addUsage(GPUBufferUsage.VERTEX);
                if (attr == 'POSITION') {
                    positions = new GLTFAccessor(bufferViews[viewID], accessor);
                } else if (attr == 'NORMAL') {
                    normals = new GLTFAccessor(bufferViews[viewID], accessor);
                } else if (attr.startsWith('TEXCOORD')) {
                    texcoords.push(new GLTFAccessor(bufferViews[viewID], accessor));
                }
            }

            var material = null;
            if (prim['material'] !== undefined) {
                material = materials[prim['material']];
            } else {
                material = defaultMaterial;
            }

            var triangles = []
            if (indices) {
                const vertexPositions = new Float32Array(
                    positions.elements.buffer,
                    positions.elements.byteOffset,
                    positions.elements.length / 4 // length in floats, not bytes
                );
                const vertexNormals = normals ? new Float32Array(
                    normals.elements.buffer,
                    normals.elements.byteOffset,
                    normals.elements.length / 4
                ) : null;
                const indicesArray = new Uint16Array(indices.elements.buffer);
                const vertexUvs = texcoords?.[0] ? new Float32Array(texcoords[0].elements.buffer) : null;

                if (texcoords && texcoords.length > 1) {
                    alert('This model has more than 1 TEXCOORD. Only the first TEXCOORD will be handled for ray tracing.')
                }

                for (let i = 0; i < indicesArray.length; i += 3) {
                    const indexOne = indicesArray[i] * 3;
                    const indexTwo = indicesArray[i + 1] * 3;
                    const indexThree = indicesArray[i + 2] * 3;

                    const positionOne = [vertexPositions[indexOne], vertexPositions[indexOne + 1], vertexPositions[indexOne + 2]];
                    const positionTwo = [vertexPositions[indexTwo], vertexPositions[indexTwo + 1], vertexPositions[indexTwo + 2]];
                    const positionThree = [vertexPositions[indexThree], vertexPositions[indexThree + 1], vertexPositions[indexThree + 2]];

                    let normalArrays: [Float32Array, Float32Array, Float32Array] | [] = [];
                    if (vertexNormals) {
                        const raw = [
                            vertexNormals[indexOne], vertexNormals[indexOne + 1], vertexNormals[indexOne + 2], // n1
                            vertexNormals[indexTwo], vertexNormals[indexTwo + 1], vertexNormals[indexTwo + 2], // n2
                            vertexNormals[indexThree], vertexNormals[indexThree + 1], vertexNormals[indexThree + 2] // n3
                        ];

                        // Make sure all normals are unit vector
                        const n1 = new Float32Array(3);
                        const n2 = new Float32Array(3);
                        const n3 = new Float32Array(3);
                        const results = [n1, n2, n3];

                        for (let j = 0; j < 3; j++) {
                            const offset = j * 3;
                            let x = raw[offset];
                            let y = raw[offset + 1];
                            let z = raw[offset + 2];

                            const lenSq = x * x + y * y + z * z;

                            if (lenSq > 0) {
                                const invLen = 1.0 / Math.sqrt(lenSq);
                                results[j][0] = x * invLen;
                                results[j][1] = y * invLen;
                                results[j][2] = z * invLen;
                            } else {
                                results[j][0] = 0;
                                results[j][1] = 1;
                                results[j][2] = 0;
                            }
                        }
                        normalArrays = [n1, n2, n3];
                    }

                    let uvArrays: [Float32Array, Float32Array, Float32Array] | [] = [];
                    if (vertexUvs) {
                        const vec2IndexOne = indicesArray[i] * 2;
                        const vec2IndexTwo = indicesArray[i + 1] * 2;
                        const vec2IndexThree = indicesArray[i + 2] * 2;

                        const uvOne = [vertexUvs[vec2IndexOne], vertexUvs[vec2IndexOne + 1]];
                        const uvTwo = [vertexUvs[vec2IndexTwo], vertexUvs[vec2IndexTwo + 1]];
                        const uvThree = [vertexUvs[vec2IndexThree], vertexUvs[vec2IndexThree + 1]];
                        uvArrays = [new Float32Array(uvOne), new Float32Array(uvTwo), new Float32Array(uvThree)];
                    }

                    const triangle = new Triangle(
                        [new Float32Array(positionOne), new Float32Array(positionTwo), new Float32Array(positionThree)],
                        normalArrays,
                        uvArrays,
                        -1,
                        material.metallicFactor,
                        material.roughnessFactor
                    );
                    triangles.push(triangle);
                }
            }

            var gltfPrim = new GLTFPrimitive(indices, positions, normals, texcoords, material, topology, triangles);
            primitives.push(gltfPrim);
        }
        meshes.push(new GLTFMesh(mesh['name'], primitives));
    }

    // Upload the different views used by meshes
    for (var i = 0; i < bufferViews.length; ++i) {
        if (bufferViews[i].needsUpload) {
            bufferViews[i].upload(device);
        }
    }

    defaultMaterial.upload(device);
    for (var i = 0; i < materials.length; ++i) {
        materials[i].upload(device);
    }

    var nodes = [];
    var gltfNodes = makeGLTFSingleLevel(glbJsonData['nodes']);
    for (var i = 0; i < gltfNodes.length; ++i) {
        var n = gltfNodes[i];
        if (n['mesh'] !== undefined) {
            var node = new GLTFNode(n['name'], meshes[n['mesh']], readNodeTransform(n));
            node.upload(device);
            nodes.push(node);
        }
    }
    document.getElementById("loading-text").hidden = true;
    return new GLBModel(nodes);
}