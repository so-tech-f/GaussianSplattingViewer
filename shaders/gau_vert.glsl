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

#define SIGMA_ELEMENTS 6
#define OPACITY_ELEMENTS 1
#define SIGMA_OPACITY_ELEMENTS (SIGMA_ELEMENTS + OPACITY_ELEMENTS)

layout (std430, binding=0) buffer PosBuffer {
	float g_pos_data[];
};
layout (std430, binding=1) buffer SigmaBuffer {
	float g_sigma_and_opacity[];
};
layout (std430, binding=2) buffer SHBuffer {
	float g_sh_data[];
};
layout (std430, binding=3) buffer gaussian_order {
	int gi[];
};

uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform vec3 hfovxy_focal;
uniform vec3 cam_pos;
uniform int sh_dim;
uniform float scale_modifier;
uniform int render_mod;  // > 0 render 0-ith SH dim, -1 depth, -2 bill board, -3 gaussian

out vec3 color;
out float alpha;
out vec3 conic;
out vec2 coordxy;  // local coordinate in quad, unit in pixel

vec3 computeCov2D(vec4 mean_view, float focal_x, float focal_y, float tan_fovx, float tan_fovy, mat3 cov3D, mat4 viewmatrix)
{
    vec4 t = mean_view;
    // why need this? Try remove this later
    float limx = 1.3f * tan_fovx;
    float limy = 1.3f * tan_fovy;
    float txtz = t.x / t.z;
    float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    mat3 J = mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0
    );
    mat3 W = transpose(mat3(viewmatrix));
    mat3 T = W * J;

    mat3 cov = transpose(T) * transpose(cov3D) * T;
    // Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
    return vec3(cov[0][0], cov[0][1], cov[1][1]);
}

vec3 get_sh_vec3(int offset)
{
	return vec3(g_sh_data[offset], g_sh_data[offset + 1], g_sh_data[offset + 2]);
}

void main()
{
	int boxid = gi[gl_InstanceID];
	vec4 g_pos_w = vec4(
    g_pos_data[boxid * 3 + 0],
    g_pos_data[boxid * 3 + 1],
    g_pos_data[boxid * 3 + 2],
    1.f
	);
	vec4 g_pos_view = view_matrix * g_pos_w;

	// back face culling
	if (g_pos_view.z >= 0.0) {
		gl_Position = vec4(-100, -100, -100, 1);
		return;
		}
    vec4 g_pos_screen = projection_matrix * g_pos_view;
	g_pos_screen.xyz = g_pos_screen.xyz / g_pos_screen.w;
    g_pos_screen.w = 1.f;
	// early culling
	if (any(greaterThan(abs(g_pos_screen.xyz), vec3(1.3))))
	{
		gl_Position = vec4(-100, -100, -100, 1);
		return;
	}

	int sigma_start = boxid * SIGMA_OPACITY_ELEMENTS;

	float sxx = g_sigma_and_opacity[sigma_start + 0];
	float syy = g_sigma_and_opacity[sigma_start + 1];
	float szz = g_sigma_and_opacity[sigma_start + 2];
	float sxy = g_sigma_and_opacity[sigma_start + 3];
	float sxz = g_sigma_and_opacity[sigma_start + 4];
	float syz = g_sigma_and_opacity[sigma_start + 5];
	float g_opacity = g_sigma_and_opacity[sigma_start + 6];

	mat3 cov3d = mat3(
		sxx, sxy, sxz,
		sxy, syy, syz,
		sxz, syz, szz
	);

	vec2 wh = 2 * hfovxy_focal.xy * hfovxy_focal.z;
	vec3 cov2d = computeCov2D(g_pos_view,
								hfovxy_focal.z,
								hfovxy_focal.z,
								hfovxy_focal.x,
								hfovxy_focal.y,
								cov3d,
								view_matrix);

	float sigma_x = sqrt(max(cov2d.x, 0.0));
	float sigma_y = sqrt(max(cov2d.z, 0.0));

	if (max(sigma_x, sigma_y) <= 0.5) {
		gl_Position = vec4(-100, -100, -100, 1);
		return;
	}

	float det = (cov2d.x * cov2d.z - cov2d.y * cov2d.y);
	if (det == 0.0f)
		gl_Position = vec4(0.f, 0.f, 0.f, 0.f);

	float det_inv = 1.f / det;
	conic = vec3(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);

	vec2 quadwh_scr = vec2(3.f * sqrt(cov2d.x), 3.f * sqrt(cov2d.z));  // screen space half quad height and width
	vec2 quadwh_ndc = quadwh_scr / wh * 2;  // in ndc space
	g_pos_screen.xy = g_pos_screen.xy + position * quadwh_ndc;
	coordxy = position * quadwh_scr;
	gl_Position = g_pos_screen;

	alpha = g_opacity;

	if (render_mod == -1)
	{
		float depth = -g_pos_view.z;
		depth = depth < 0.05 ? 1 : depth;
		depth = 1 / depth;
		color = vec3(depth, depth, depth);
		return;
	}

	int sh_elements_per_gaussian = sh_dim;
	int sh_start = boxid * sh_elements_per_gaussian;

	vec3 dir = g_pos_w.xyz - cam_pos;
	dir = normalize(dir);

	// L0 (バンド0)
	color = SH_C0 * get_sh_vec3(sh_start);

	// L1 (バンド1)
	if (sh_dim > 3 && render_mod >= 1)  // 1 * 3
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;

		color = color - SH_C1 * y * get_sh_vec3(sh_start + 1 * 3)
				+ SH_C1 * z * get_sh_vec3(sh_start + 2 * 3)
				- SH_C1 * x * get_sh_vec3(sh_start + 3 * 3);

		// L2 (バンド2)
		if (sh_dim > 12 && render_mod >= 2)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			color = color +
				SH_C2_0 * xy * get_sh_vec3(sh_start + 4 * 3) +
				SH_C2_1 * yz * get_sh_vec3(sh_start + 5 * 3) +
				SH_C2_2 * (2.0f * zz - xx - yy) * get_sh_vec3(sh_start + 6 * 3) +
				SH_C2_3 * xz * get_sh_vec3(sh_start + 7 * 3) +
				SH_C2_4 * (xx - yy) * get_sh_vec3(sh_start + 8 * 3);

			// L3 (バンド3)
			if (sh_dim > 27 && render_mod >= 3)
			{
				color = color +
					SH_C3_0 * y * (3.0f * xx - yy) * get_sh_vec3(sh_start + 9 * 3) +
					SH_C3_1 * xy * z * get_sh_vec3(sh_start + 10 * 3) +
					SH_C3_2 * y * (4.0f * zz - xx - yy) * get_sh_vec3(sh_start + 11 * 3) +
					SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * get_sh_vec3(sh_start + 12 * 3) +
					SH_C3_4 * x * (4.0f * zz - xx - yy) * get_sh_vec3(sh_start + 13 * 3) +
					SH_C3_5 * z * (xx - yy) * get_sh_vec3(sh_start + 14 * 3) +
					SH_C3_6 * x * (xx - 3.0f * yy) * get_sh_vec3(sh_start + 15 * 3);
			}
		}
	}
	color += 0.5f;
}
