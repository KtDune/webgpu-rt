export class GLBShaderCache {
    constructor(device) {
        this.device = device;
        this.shaderCache = {};
    }

    getShader(hasNormals, hasUVs, hasColorTexture, material) {
        var shaderID = "glb";
        if (hasNormals) {
            shaderID += "n";
        }
        if (hasUVs) {
            shaderID += "uv";
        }
        if (hasColorTexture) {
            shaderID += "colortex";
        }
        if (material['emissiveTexture']) {
            shaderID += "emissivetex";
        }
        if (material['occlusionTexture']) {
            shaderID += "occlusiontex";
        }
        if (material['normalTexture']) {
            shaderID += "normaltex";
        }
        if (material.metallicRoughnessTexture) {
            shaderID += "mrtex";
        }
        if (material.alphaMode === 'MASK') {
            shaderID += "alphamask";
        } else if (material.alphaMode === 'BLEND') {
            shaderID += "alphablend";
        }
        if (!(shaderID in this.shaderCache)) {
            var shaderSource = generateGLTFShader(hasNormals, hasUVs, hasColorTexture, material);
            this.shaderCache[shaderID] = this.device.createShaderModule({ code: shaderSource });
        }
        return this.shaderCache[shaderID];
    }

    getScreenShader() {
        return generateScreenshader()
    }

    getRayShader() {
        return generateRayShader()
    }
}

function generateGLTFShader(hasNormals, hasUVs, hasColorTexture, material) {
    var typeDefs = `
        alias float2 = vec2<f32>;
        alias float3 = vec3<f32>;
        alias float4 = vec4<f32>;
    `;

    var vertexInputStruct = `
        struct VertexInput {
            @location(0) position: float3,
    `;

    var vertexOutputStruct = `
        struct VertexOutput {
            @builtin(position) position: float4,
            @location(1) normal_world: float3,  // World-space normal
            @location(2) position_world: float3, // World-space position
            ${hasUVs ? `@location(3) uv: float2,` : ""}
        };`;

    if (hasNormals) {
        vertexInputStruct += `
            @location(1) normal: float3,`;
    }
    if (hasUVs) {
        vertexInputStruct += `
            @location(2) uv: float2,`;
    }
    vertexInputStruct += '};';

    var vertexUniformParams = `
        struct Mat4Uniform {
            m: mat4x4<f32>,
        };

        struct Uniform {
            camera_matrix: vec3<f32>,
            light_pos: vec3<f32>,
            pbr: f32
        };

        @group(0) @binding(0) var<uniform> view_proj: Mat4Uniform;
        @group(1) @binding(0) var<uniform> node_transform: Mat4Uniform;
        @group(3) @binding(0) var<uniform> uni: Uniform;
    `;

    var vertexStage = vertexInputStruct + vertexOutputStruct + vertexUniformParams + `
        @vertex
        fn vertex_main(vin: VertexInput) -> VertexOutput {
            var vout: VertexOutput;
            
            // Transform position to clip space
            vout.position = view_proj.m * node_transform.m * float4(vin.position, 1.0);
            
            // World-space position (for lighting)
            vout.position_world = (node_transform.m * float4(vin.position, 1.0)).xyz;
            
            ${hasNormals ? `
                // World-space normal (no normalization here, do in fragment)
                vout.normal_world = (node_transform.m * float4(vin.normal, 0.0)).xyz;
            ` : `
                vout.normal_world = float3(0.0, 1.0, 0.0);
            `}
            
            ${hasUVs ? `vout.uv = vin.uv;` : ""}
            return vout;
        }`;

    var fragmentParams = `
        struct MaterialParams {
            base_color_factor: float4,
            emissive_factor: float4,
            metallic_factor: f32,
            roughness_factor: f32,
        };
        @group(2) @binding(0) var<uniform> material: MaterialParams;
    `;

    if (hasColorTexture) {
        fragmentParams += `
            @group(2) @binding(1) var base_color_sampler: sampler;
            @group(2) @binding(2) var base_color_texture: texture_2d<f32>;
        `;
    }

    if (material['emissiveTexture']) {
        fragmentParams += `
        @group(2) @binding(3) var emissive_sampler: sampler;
        @group(2) @binding(4) var emissive_texture: texture_2d<f32>;
        `
    }

    if (material['occlusionTexture']) {
        fragmentParams += `
        @group(2) @binding(5) var occlusion_sampler: sampler;
        @group(2) @binding(6) var occlusion_texture: texture_2d<f32>;
        `
    }

    if (material['normalTexture']) {
        fragmentParams += `
        @group(2) @binding(7) var normal_sampler: sampler;
        @group(2) @binding(8) var normal_texture: texture_2d<f32>;
        `
    }

    if (material.metallicRoughnessTexture) {
        fragmentParams += `
        
        @group(2) @binding(9) var metallic_roughness_sampler: sampler;
        @group(2) @binding(10) var metallic_roughness_texture: texture_2d<f32>;
        `
    }

    if (material.alphaMode === 'MASK') {
        fragmentParams += `
        @group(2) @binding(11) var<uniform> alpha_cutoff: f32;
        `
    }

    // PBR FUNCTIONS (complete Cook-Torrance)
    var pbrFunctions = `
        fn distributionGGX(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
            let square_roughness = roughness * roughness;  // nom = square of roughness
            let NdotH = max(dot(N, H), 0.0);
            let square_NdotH = NdotH * NdotH;              // square of dot product
            
            let denom_inner = square_NdotH * (square_roughness - 1.0) + 1.0;
            let denom = 3.1415926 * denom_inner * denom_inner;    // PI * (inner)^2
            
            return square_roughness / denom;
        }

        fn geometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
            let r = roughness + 1.0;
            let k = (1 + (r * r)) / 8.0;
            let nom = NdotV;
            let denom = NdotV * (1.0 - k) + k;
            return nom / denom;
        }

        fn geometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
            let NdotV = max(dot(N, V), 0.0);
            let NdotL = max(dot(N, L), 0.0);
            let ggx2 = geometrySchlickGGX(NdotV, roughness);
            let ggx1 = geometrySchlickGGX(NdotL, roughness);
            return ggx1 * ggx2;
        }

        fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
            return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
        }

        fn fresnelSchlickRoughness(cosTheta: f32, f0: vec3f, roughness: f32, albedo: vec3f, metallic: f32) -> vec3f {
            // For metals: F0 = albedo (material color)
            // For dielectrics: F0 = dielectric specular (0.04) or passed f0
            let F0 = mix(f0, albedo, metallic);
            
            return F0 + (max(vec3f(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
        }

        fn cotangent_frame(N: vec3<f32>, p: vec3<f32>, uv: vec2<f32>) -> mat3x3<f32> {
            let dp1  = dpdx(p);
            let dp2  = dpdy(p);
            let duv1 = dpdx(uv);
            let duv2 = dpdy(uv);

            let dp2perp = cross(dp2, N);
            let dp1perp = cross(N, dp1);
            let T = dp2perp * duv1.x + dp1perp * duv2.x;
            let B = dp2perp * duv1.y + dp1perp * duv2.y;

            let invmax = inverseSqrt(max(dot(T, T), dot(B, B)));
            return mat3x3<f32>(T * invmax, B * invmax, N);
        }
    `;

    var fragmentStage = fragmentParams + pbrFunctions + `
    @fragment
    fn fragment_main(fin: VertexOutput) -> @location(0) float4 {
        let PI = 3.1415926;
        
        let lightPos_world = uni.light_pos;
        let lightColor = vec3f(1.0, 1.0, 1.0);
        let lightIntensity = 50.0;
        
        // Material properties
        var albedo = material.base_color_factor.rgb;
        
        var alpha = material.base_color_factor.a;
        ${hasUVs && hasColorTexture ? `
            let texColor = textureSample(base_color_texture, base_color_sampler, fin.uv);
            alpha = texColor.a * material.base_color_factor.a;
            albedo *= texColor.rgb;

            ${material['alphaMode'] === 'MASK' ? `
            var alphaCutoff = 0.5;
            alphaCutoff = alpha_cutoff;
            if (alpha < alphaCutoff) { discard; }
            ` : ''}
        ` : ""}

        var ao = 1.0;
        ${material['occlusionTexture']
            ?
            `
            ao = textureSample(occlusion_texture, occlusion_sampler, fin.uv).r;
            `
            : ''
        }

        var emissive = vec3f(0);
        ${material['emissiveTexture']
            ? ` emissive = textureSample(emissive_texture, emissive_sampler, fin.uv).rgb;
                emissive *= material.emissive_factor.rgb;`
            : ''
        }

        if (uni.pbr == 0.0) {
            let base = albedo * ao + emissive;

            let final_color = vec3f(
                linear_to_srgb(base.x),
                linear_to_srgb(base.y),
                linear_to_srgb(base.z)
            );
            
            return vec4f(final_color, material.base_color_factor.a);
        }

        var metallic = material.metallic_factor;
        var roughness = material.roughness_factor;

        ${hasUVs && material.metallicRoughnessTexture ? `
            var mrSample = textureSample(metallic_roughness_texture, metallic_roughness_sampler, fin.uv);
            metallic *= mrSample.b;  // glTF: B=metallic
            roughness *= mrSample.g; // glTF: G=roughness
        ` : ''}

        // Normals & View
        var N = normalize(fin.normal_world);
        ${material['normalTexture']
            ? `
            let normal_sample = textureSample(normal_texture, normal_sampler, fin.uv).rgb;
            let tangent_normal = normal_sample * 2.0 - 1.0;
            let TBN = cotangent_frame(N, fin.position_world, fin.uv);
            N = normalize(TBN * tangent_normal);
            `
            : ''
        }

        let V = normalize(uni.camera_matrix - fin.position_world);
        
        // Light vector
        let L_world = lightPos_world - fin.position_world;
        var L_distance = length(L_world);
        let L = normalize(L_world);

        // Clamp maximum distance to 5 units
        if (L_distance > 5.0) {
            L_distance = 5.0;
        }
        
        // Light falloff (irradiance)
        let attenuation = 1.0 / (L_distance * L_distance + 1.0);
        let irradiance = lightColor * lightIntensity * attenuation;
        
        // Cook-Torrance BRDF - FIXED fresnelSchlickRoughness
        let H = normalize(V + L);
        let NDF = distributionGGX(N, H, roughness);
        let G = geometrySmith(N, V, L, roughness);
        
        let F0 = mix(vec3f(0.04), albedo, metallic);  // Already correct!
        let F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness, albedo, metallic);
        
        let numerator = NDF * G * F;
        let denominator = 4.0 * max(dot(N, V), 0.0001) * max(dot(N, L), 0.0) + 0.0001;
        let specular = numerator / denominator;
        
        var kD = vec3f(1.0) - F;
        kD *= 1.0 - metallic;
        let diffuse = kD * albedo / PI;
        
        let NdotL = max(dot(N, L), 0.0);
        let Lo = (diffuse + specular) * irradiance * NdotL;
        
        // Ambient (IBL approximation)
        let ambient = vec3f(0.03) * albedo * ao;
        
        var color = ambient + Lo + emissive;
        
        // Tone mapping + gamma
        color = color / (color + vec3f(1.0));
        color = vec3f(
            linear_to_srgb(color.x),
            linear_to_srgb(color.y),
            linear_to_srgb(color.z)
        );

        ${material['alphaMode'] === 'BLEND' ? `
            return vec4f(color, alpha);
            ` : `
            return vec4f(color, material.base_color_factor.a);
    `}
    }
    
    fn linear_to_srgb(x: f32) -> f32 {
        if (x <= 0.0031308) {
            return 12.92 * x;
        }
        return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
    }
    `;


    // console.log(typeDefs + vertexStage + fragmentStage)
    return typeDefs + vertexStage + fragmentStage;
}

function generateScreenshader() {
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
    return screen_shader
}

function generateRayShader() {
    const raytracer_kernel = `
        @group(0) @binding(0)
        var color_buffer: texture_storage_2d<rgba8unorm, write>;

        @group(0) @binding(1)
        var<uniform> scene: SceneData;

        @group(0) @binding(2)
        var<storage, read> primitives: PrimitiveData;

        @group(0) @binding(3)
        var base_color_map: texture_2d<f32>;
        @group(0) @binding(4)
        var base_color_sampler: sampler;
        @group(0) @binding(5)
        var<storage, read> bvh_nodes: BVHTree;
        @group(0) @binding(6)
        var<storage, read> triangle_indices: TriangleIndices;

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
            uv_a: vec2<f32>,
            uv_b: vec2<f32>,
            uv_c: vec2<f32>,
            ior: vec4<f32>,
            metalness: vec4<f32>
        }

        struct BVHTree {
            nodes: array<BVHNode>
        }

        struct BVHNode {
            minCorner: vec3<f32>,
            leftChild: f32,
            maxCorner: vec3<f32>,
            objectCount: f32
        }

        struct TriangleIndices {
            indices: array<f32>
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
            trianglesCount: f32,
            aspectRatio: f32,
            fov: f32
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

        
        let forwards: vec3<f32> = scene.cameraForward;
        let right: vec3<f32> = scene.cameraRight;
        let up: vec3<f32> = scene.cameraUp;

        let samples_per_pixel: u32 = u32(2);
        var color: vec3<f32> = vec3(0.0, 0.0, 0.0);
        for (var i: u32 = u32(0); i < samples_per_pixel; i++) {
            var x_offset: f32 = random(vec2(f32(i) + f32(GlobalInvocationID.x), f32(i) + f32(GlobalInvocationID.y)));
            var y_offset: f32 = random(vec2(f32(i) + f32(GlobalInvocationID.y), f32(i) + f32(GlobalInvocationID.x)));
            let horizontal_coefficient: f32 = (f32(screen_pos.x) + x_offset - f32(screen_size.x) / 2.0) / f32(screen_size.x) * scene.aspectRatio;
            let vertical_coefficient: f32 = (f32(screen_pos.y) + y_offset - f32(screen_size.y) / 2.0) / f32(screen_size.y);            
            var ray: Ray = Ray(scene.cameraPos, normalize(forwards + right * horizontal_coefficient + up * vertical_coefficient));
            color += rayColor(ray, vec2(f32(GlobalInvocationID.x) + f32(i), f32(GlobalInvocationID.y) + f32(i)), i);
        }
        color /= f32(samples_per_pixel);
        color = linear_to_gamma(color);
        textureStore(color_buffer, screen_pos, vec4(color, 1.0));
    }

    fn rayColor(ray: Ray, random_seed: vec2<f32>, sample_number: u32) -> vec3<f32> {
        var result: RenderState;
        var color: vec3<f32> = vec3(1.0, 1.0, 1.0);
    
        var worldRay: Ray;
        worldRay.origin = ray.origin;
        worldRay.direction = ray.direction;
        
        var bounce: u32 = u32(0);
        while (bounce < u32(scene.maxBounces)) {
            // we will bounce a certain number of times
            result = trace(worldRay, random_seed, sample_number);

            
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

    fn trace(ray: Ray, random_seed: vec2<f32>, sample_number: u32) -> RenderState {
        var nearest_hit: f32 = 1.0e30;
        var renderState: RenderState;
        renderState.hit = false;
        renderState.t = 1.0e30;

        var node: BVHNode = bvh_nodes.nodes[0];
        var stack: array<BVHNode, 15>;
        var stackLocation: u32 = u32(0);

        // NAIVE BRUTE FORCE TRIANGLES
        // for (var t: u32 = u32(0); t < u32(scene.trianglesCount); t++) {
        //     // find the closest triangle
        //     renderState = hit_triangle(ray, primitives.triangles[t], 0.001, renderState.t, renderState, random_seed);
        // }

        while (true) {
            var objectCount: u32 = u32(node.objectCount);
            var contents: u32 = u32(node.leftChild);

            // internal node goes and checks children
            if (objectCount == u32(0) && node.leftChild > 0.0) {
                var child_one: BVHNode = bvh_nodes.nodes[contents];
                var child_two: BVHNode = bvh_nodes.nodes[contents + u32(1)];

                var distance_one: f32 = hit_aabb(ray, child_one);
                var distance_two: f32 = hit_aabb(ray, child_two);

                var closest_child: BVHNode;
                var farthest_child: BVHNode;

                if (distance_one < distance_two) {
                    closest_child = child_one;
                    farthest_child = child_two;
                } else {
                    closest_child = child_two;
                    farthest_child = child_one;
                }

                
                var closest_distance: f32 = min(distance_one, distance_two);
                if (closest_distance > nearest_hit) {
                    // ray misses both children grab next BVH node to explore (or break if stack is empty)
                    if (stackLocation == u32(0)) {
                        break;
                    } else {
                        stackLocation--;
                        node = stack[stackLocation];
                        continue;
                    }
                } else {
                    // ray hits at least one child, assign exploring child to closest one
                    node = closest_child;
                    var farthest_distance: f32 = max(distance_one, distance_two);
                    if (farthest_distance < nearest_hit) {
                        // if other node is closer than nearest hit, push it on the stack so we can explore it if needed
                        stack[stackLocation] = farthest_child;
                        stackLocation++;
                    }
                }
            } else {
                // actual data node, test triangles
                for (var i: u32 = u32(0); i < objectCount; i++) {
                    var newRenderState: RenderState = hit_triangle(ray, primitives.triangles[u32(triangle_indices.indices[contents + i])], 0.001, nearest_hit, renderState, random_seed, sample_number);

                    if (newRenderState.hit) {
                        nearest_hit = newRenderState.t;
                        renderState = newRenderState;
                    }
                }

                if (stackLocation == u32(0) ) {
                    break;
                } else {
                    stackLocation--;
                    node = stack[stackLocation];
                    continue;
                }
            }
        }

        return renderState;
    }

    fn hit_aabb(ray: Ray, aabb: BVHNode) -> f32 {
        var t1: vec3<f32> = (aabb.minCorner - ray.origin) / ray.direction;
        var t2: vec3<f32> = (aabb.maxCorner - ray.origin) / ray.direction;

        var tMin: vec3<f32> = min(t1, t2);
        var tMax: vec3<f32> = max(t1, t2);

        var t_min: f32 = max(max(tMin.x, tMin.y), tMin.z);
        var t_max: f32 = min(min(tMax.x, tMax.y), tMax.z);

        if (t_min > t_max || t_max < 0.0) {
            return 1.0e30;
        } else {
            return t_min;
        }
    }

    fn hit_triangle(ray:Ray, triangle: Triangle, tMin: f32, tMax: f32, oldRenderState: RenderState, random_seed: vec2<f32>, sample_number: u32) -> RenderState {
        var ior: f32 = triangle.ior.x;
        var metalness: f32 = triangle.metalness.x;
        
        // TODO: precompute surface normal and pass in with triangle
        var edgeAB: vec3<f32> = triangle.corner_b - triangle.corner_a;
        var edgeAC: vec3<f32> = triangle.corner_c - triangle.corner_a;
        var surface_normal: vec3<f32> = cross(edgeAB, edgeAC);

        var use_ior: bool = ior > 0.0 && (sample_number % 2u == 0u);

        var tri_normal_dot_ray_dir: f32 = dot(surface_normal, ray.direction);
        var front_face: bool = tri_normal_dot_ray_dir < 0.0;
        if (!front_face) {
            if (!use_ior) {
                return oldRenderState;
            } else {
                surface_normal = -surface_normal;
            }
        }

        if (tri_normal_dot_ray_dir > -0.00000001) {
            // ray is parallel to triangle
            return oldRenderState;
        }

        var d = dot(surface_normal, triangle.corner_a); //TODO this could be in tri data
        var t = (d - dot(surface_normal, ray.origin)) / tri_normal_dot_ray_dir;
        if (t < tMin || t > tMax) {
            return oldRenderState;
        }

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

        
        var normal = normalize((1.0 - u - v) * triangle.normal_a + u * triangle.normal_b + v * triangle.normal_c);

        var scatter_direction: vec3<f32>;
        var base_color: vec3<f32>;

        if (use_ior) {
            scatter_direction = dielectric_scattering(ray, normal, ior, front_face, random_seed);
            base_color = vec3(1.0, 1.0, 1.0);
        } else {
            if (random(random_seed) < metalness) {
                scatter_direction = metal_scattering(ray, normal);
            } else {
                scatter_direction = lambertian_scattering(normal, random_seed);
            }
            var uv = (1.0 - u - v) * triangle.uv_a + u * triangle.uv_b + v * triangle.uv_c;
            base_color = textureSampleLevel(base_color_map, base_color_sampler, uv, 0.0).rgb;
        }

        var renderState: RenderState;
        renderState.scatter_direction = normalize(scatter_direction);
        renderState.t = t;
        renderState.hit = true;
        renderState.color = base_color;

        return renderState;
    }

    fn lambertian_scattering(normal: vec3<f32>, random_seed: vec2<f32>) -> vec3<f32> {
    let random_unit_vector: vec3<f32> = random_in_unit_sphere(vec2(normal.x + random_seed.x, normal.y + random_seed.y));
    var scatter_direction: vec3<f32> = random_unit_vector + normal;
    return scatter_direction;
    }

    fn metal_scattering(ray: Ray, normal: vec3<f32>) -> vec3<f32> {
        var scatter_direction = reflect(normalize(ray.direction), normal);
        return scatter_direction;
    }

    fn dielectric_scattering(ray: Ray, normal: vec3<f32>, ior: f32, front_face: bool, random_seed: vec2<f32>) -> vec3<f32> {
        var refrecation_ratio = 1.0 / ior;
        if (!front_face) {
            refrecation_ratio = ior;
        }
        var unit_direction = normalize(ray.direction);
        var cos_theta = min(dot(-unit_direction, normal), 1.0);
        var sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        var cannot_refract = refrecation_ratio * sin_theta > 1.0;
        
        var scatter_direction: vec3<f32>;
        if (cannot_refract) {
            var r0 = (1.0 - refrecation_ratio) / (1.0 + refrecation_ratio);
            r0 = r0 * r0;
            var reflectance = r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);
            if (random(random_seed) < reflectance) {
                scatter_direction = reflect(unit_direction, normal);
            } else {
                scatter_direction = refract(unit_direction, normal, refrecation_ratio);
            }
        } else {
            scatter_direction = refract(unit_direction, normal, refrecation_ratio);
        }
        return scatter_direction;
    }

    fn random_in_unit_sphere(random_seed: vec2<f32>) -> vec3<f32> {
        var random_vector: vec3<f32> = vec3( 2.0 * random(random_seed) - 1.0, 2.0 * random(random_seed) - 1.0, 2.0 * random(random_seed) - 1.0);
        var nonce: f32 = 0.0;
        while (dot(random_vector, random_vector) >= 1.0 && nonce < 100.0) {
            random_vector = vec3( 2.0 * random(vec2(random_seed.x + nonce, random_seed.y)) - 1.0, 2.0 * random(vec2(random_seed.y + nonce, random_seed.x)) - 1.0, 2.0 * random(vec2(random_seed.x + random_seed.y, nonce)) - 1.0);
            nonce += 1.0;
        }
        return random_vector;
    }

    fn random(uv: vec2<f32>) -> f32 {
        return fract(sin(dot(uv, vec2(12.9898, 78.233))) * 43758.5453123);
    }

    fn linear_to_gamma(color: vec3<f32>) -> vec3<f32> {
        var color_gamma: vec3<f32> = vec3(0.0, 0.0, 0.0);
        if (color.r > 0.0) {
            color_gamma.r = sqrt(color.r);
        }
        if (color.g > 0.0) {
            color_gamma.g = sqrt(color.g);
        }
        if (color.b > 0.0) {
            color_gamma.b = sqrt(color.b);
        }
        return color_gamma;
    }
    `
    return raytracer_kernel
}
