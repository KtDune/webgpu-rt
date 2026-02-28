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
        
        ${hasUVs && hasColorTexture ? `
            let texColor = textureSample(base_color_texture, base_color_sampler, fin.uv);
            let alpha = texColor.a * material.base_color_factor.a;
            albedo *= texColor.rgb;

            ${material['alphaMode'] === 'MASK' ? `
                if (alpha < alpha_cutoff) { discard; }
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
                return vec4f(color, 1.0);
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
