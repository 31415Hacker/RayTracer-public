@group(0) @binding(0)
var hdrTexture: texture_2d<f32>;
@group(0) @binding(1)
var linearSampler: sampler;

struct VSOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vmain(@builtin(vertex_index) vertexIndex: u32) -> VSOutput {
    var pos = array<vec2<f32>, 3>(vec2<f32>(- 1.0, - 1.0), vec2<f32>(3.0, - 1.0), vec2<f32>(- 1.0, 3.0));

    var out: VSOutput;
    out.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
    out.uv = (out.position.xy * 0.5 + vec2<f32>(0.5));
    // normalized UVs
    return out;
}

@fragment
fn fmain(in: VSOutput) -> @location(0) vec4<f32> {
    let hdrColor = textureSample(hdrTexture, linearSampler, in.uv);
    let mapped = hdrColor.rgb / (hdrColor.rgb + vec3<f32>(1.0));
    let gamma = 1.0 / 2.2;
    let color = hdrColor.rgb;//pow(mapped, vec3<f32>(gamma));
    return vec4<f32>(color, 1.0);
}