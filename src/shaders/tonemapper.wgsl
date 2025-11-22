@group(0) @binding(0)
// FIX 1: Access the rgba32float texture (requires unfilterable-float sample type in JS)
var hdrTexture: texture_2d<f32>;

// FIX 2: Removed unused sampler binding, as we use textureLoad

struct VSOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vmain(@builtin(vertex_index) vertexIndex: u32) -> VSOutput {
    // Generate a full-screen triangle
    var pos = array<vec2<f32>, 3>(vec2<f32>(- 1.0, - 1.0), vec2<f32>(3.0, - 1.0), vec2<f32>(- 1.0, 3.0));

    var out: VSOutput;
    out.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
    // Convert NDC position to UV coordinates (0 to 1)
    out.uv = (out.position.xy * 0.5 + vec2<f32>(0.5));
    return out;
}

@fragment
fn fmain(in: VSOutput) -> @location(0) vec4<f32> {
    // FIX 3: Use textureLoad with integer coordinates derived from normalized UV
    let size = textureDimensions(hdrTexture);
    let coord = vec2<i32>(in.uv * vec2<f32>(size));
    let hdrColor = textureLoad(hdrTexture, coord, 0); 
    
    // Reinhard Tone Mapping: L = L / (L + 1)
    let mapped = hdrColor.rgb / (hdrColor.rgb + vec3<f32>(1.0));

    // sRGB Gamma Correction: L = L^(1/2.2)
    let gamma = 1.0 / 2.2;
    // FIX 4: Correctly apply tone mapping and gamma correction
    let color = pow(mapped, vec3<f32>(gamma));

    // Output final color to the canvas
    return vec4<f32>(color, 1.0);
}