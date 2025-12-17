#version 430 core

#define SH_C0 0.28209479177387814f
#define SH_C1 0.4886025119029199f

#define SH_C2_0 1.0925484305920792f
#define SH_C2_1 -1.0925484305920792f
#define SH_C2_2 0.31539156525252005f
#define SH_C2_3 -1.0925484305920792f
#define SH_C2_4 0.5462742152960396f

#define SH_C3_0 -0.5900435899266435f
#define SH_C3_1 2.890611442640554f
#define SH_C3_2 -0.4570457994644658f
#define SH_C3_3 0.3731763325901154f
#define SH_C3_4 -0.4570457994644658f
#define SH_C3_5 1.445305721320277f
#define SH_C3_6 -0.5900435899266435f

layout(location = 0) in vec2 position;

struct Gaussian {
    vec4 pos_opacity; // [x, y, z, opacity]
    vec4 sigma_part1; // [sxx, syy, szz, pad]
    vec4 sigma_part2; // [sxy, sxz, syz, pad]
    vec4 sh_dc;       // [sh0.r, sh0.g, sh0.b, pad]
    vec4 sh_rest[3]; // 1次(3), 2次(5), 3次(7) の計15個のvec4
};

layout (std430, binding=0) buffer gaussian_data {
    Gaussian g_data[];
};

layout (std430, binding=1) buffer gaussian_order {
    int gi[];
};
uniform mat4 view_matrix;
uniform mat3 W_u;
uniform mat4 projection_matrix;
uniform vec3 hfovxy_focal;
uniform vec3 cam_pos;
uniform int sh_dim;
uniform int render_mod;  // > 0 render 0-ith SH dim, -1 depth, -2 bill board, -3 gaussian

out vec3 color;
out float alpha;
out vec3 conic;
out vec2 coordxy;  // local coordinate in quad, unit in pixel

vec3 computeCov2D(vec4 mean_view, float focal_x, float focal_y, float tan_fovx, float tan_fovy, mat3 cov3D, mat3 W)
{
    vec4 t = mean_view;
    float limx = 1.3f * tan_fovx;
    float limy = 1.3f * tan_fovy;
    float txtz = t.x / t.z;
    float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    mat3 J = mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
        0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0.0, 0.0, 0.0
    );
    mat3 T = W * J;

    mat3 cov = transpose(T) * transpose(cov3D) * T;
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;
    return vec3(cov[0][0], cov[0][1], cov[1][1]);
}

void main()
{
    int boxid = gi[gl_InstanceID];
    Gaussian g = g_data[boxid];

    vec4 g_pos = vec4(g.pos_opacity.xyz, 1.0f);
    vec4 g_pos_view = view_matrix * g_pos;

    if (g_pos_view.z >= 0.0) {
        gl_Position = vec4(-100, -100, -100, 1);
        return;
    }

    vec4 g_pos_screen = projection_matrix * g_pos_view;
    g_pos_screen.xyz /= g_pos_screen.w;
	g_pos_screen.w = 1.f;

    if (any(greaterThan(abs(g_pos_screen.xyz), vec3(1.3)))) {
        gl_Position = vec4(-100, -100, -100, 1);
        return;
    }

    mat3 cov3d = mat3(
        g.sigma_part1.x, g.sigma_part2.x, g.sigma_part2.y, // 列1: sxx, sxy, sxz
        g.sigma_part2.x, g.sigma_part1.y, g.sigma_part2.z, // 列2: sxy, syy, syz
        g.sigma_part2.y, g.sigma_part2.z, g.sigma_part1.z  // 列3: sxz, syz, szz
    );

    vec3 cov2d = computeCov2D(
							g_pos_view,
                            hfovxy_focal.z,
							hfovxy_focal.z,
                            hfovxy_focal.x,
							hfovxy_focal.y,
                            cov3d,
							W_u
							);

    if (max(cov2d.x, cov2d.z) < 0.25f) {
        gl_Position = vec4(-100, -100, -100, 1);
        return;
    }
    // Invert covariance (EWA algorithm)
    float det = (cov2d.x * cov2d.z - cov2d.y * cov2d.y);
    if (det == 0.0f) { gl_Position = vec4(0.0); return; }

    float det_inv = 1.0f / det;
    conic = vec3(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);

    vec2 wh = 2.0 * hfovxy_focal.xy * hfovxy_focal.z;
    vec2 quadwh_scr = vec2(3.0 * sqrt(cov2d.x), 3.0 * sqrt(cov2d.z));  // screen space half quad height and width
    vec2 quadwh_ndc = quadwh_scr / wh * 2.0;  // in ndc space
	g_pos_screen.xy += position * quadwh_ndc;

    coordxy = position * quadwh_scr;
    gl_Position = g_pos_screen;

    alpha = g.pos_opacity.w;

	if (render_mod == -1)
	{
		float depth = -g_pos_view.z;
		depth = depth < 0.05 ? 1 : depth;
		depth = 1 / depth;
		color = vec3(depth, depth, depth);
		return;
	}

    // --- SH Color 計算 (球面調和関数) ---
    vec3 dir = normalize(g_pos.xyz - cam_pos);
    color = SH_C0 * g.sh_dc.xyz; // 0次 (DC)

    if (sh_dim > 3 && render_mod >= 1) {
        float x = dir.x, y = dir.y, z = dir.z;
        // 1次: sh_rest[0..2]
        color += SH_C1 * (-y * g.sh_rest[0].xyz + z * g.sh_rest[1].xyz - x * g.sh_rest[2].xyz);

        // if (sh_dim > 12 && render_mod >= 2) {
        //     float xx = x*x, yy = y*y, zz = z*z, xy = x*y, yz = y*z, xz = x*z;
        //     // 2次: sh_rest[3..7]
        //     color += SH_C2_0 * xy * g.sh_rest[3].xyz +
        //              SH_C2_1 * yz * g.sh_rest[4].xyz +
        //              SH_C2_2 * (2.0*zz - xx - yy) * g.sh_rest[5].xyz +
        //              SH_C2_3 * xz * g.sh_rest[6].xyz +
        //              SH_C2_4 * (xx - yy) * g.sh_rest[7].xyz;

        //     if (sh_dim > 27 && render_mod >= 3) {
        //         // 3次: sh_rest[8..14]
        //         color += SH_C3_0 * y * (3.0*xx - yy) * g.sh_rest[8].xyz +
        //                  SH_C3_1 * xy * z * g.sh_rest[9].xyz +
        //                  SH_C3_2 * y * (4.0*zz - xx - yy) * g.sh_rest[10].xyz +
        //                  SH_C3_3 * z * (2.0*zz - 3.0*xx - 3.0*yy) * g.sh_rest[11].xyz +
        //                  SH_C3_4 * x * (4.0*zz - xx - yy) * g.sh_rest[12].xyz +
        //                  SH_C3_5 * z * (xx - yy) * g.sh_rest[13].xyz +
        //                  SH_C3_6 * x * (xx - 3.0*yy) * g.sh_rest[14].xyz;
        //     }
        // }
    }
    color += 0.5f;
}