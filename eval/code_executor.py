"""Execute CadQuery code, render views, and compute Chamfer Distance."""
import os
import traceback
import numpy as np
import cadquery as cq
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from gen_view import analyze_view, plot_view


def execute_code(code_str):
    """Execute CadQuery code and return (success, message, cq_object)."""
    # Sharing globals and locals keeps top-level imports visible to nested
    # functions through their __globals__ namespace.
    namespace = {}
    try:
        exec(code_str, namespace, namespace)
        if 'r' not in namespace:
            return False, "Variable 'r' not found. Please ensure your code defines a variable 'r' (e.g., r = cq.Workplane...)", None
        return True, "Success", namespace['r']
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}", None


def _view_axis_size(edges, has_dims=True):
    """Estimate plot_view's axis ranges directly from projected edges."""
    all_x, all_y = [], []
    for e in edges:
        if e['type'] == 'LINE':
            all_x += [e['start'][0], e['end'][0]]
            all_y += [e['start'][1], e['end'][1]]
        elif e['type'] == 'POLY':
            all_x += [p[0] for p in e['points']]
            all_y += [p[1] for p in e['points']]
    if not all_x:
        return 1.0, 1.0
    rx = max(all_x) - min(all_x)
    ry = max(all_y) - min(all_y)
    if not has_dims:
        mx = rx * 0.3 if rx > 0 else 0.1
        my = ry * 0.3 if ry > 0 else 0.1
        return rx + 2 * mx, ry + 2 * my
    # Mirror plot_view's spacing, conservatively allowing three label tracks.
    mx  = rx * 0.15 if rx > 0 else 0.1
    my  = ry * 0.15 if ry > 0 else 0.1
    ext_top   = 4 * (ry * 0.15 if ry > 0 else 0.05)   # extra_margin_top
    ext_right = 4 * (rx * 0.20 if rx > 0 else 0.05)   # extra_margin_right
    return rx + 2 * mx + ext_right, ry + 2 * my + ext_top


def render_views(cq_object, save_path):
    """Render orthographic and isometric views in a proportional 2x2 layout.

    Estimated axis ranges keep FRONT/RIGHT equally tall and TOP/FRONT equally
    wide without temporary rendering.
    """
    try:
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        edges = {
            'TOP':   analyze_view(cq_object, 'TOP'),
            'FRONT': analyze_view(cq_object, 'FRONT'),
            'RIGHT': analyze_view(cq_object, 'RIGHT'),
            'ISO':   analyze_view(cq_object, 'ISO'),
        }

        w_front, h_front = _view_axis_size(edges['FRONT'], has_dims=True)
        w_right, h_right = _view_axis_size(edges['RIGHT'], has_dims=True)
        w_top,   h_top   = _view_axis_size(edges['TOP'],   has_dims=True)
        w_iso,   h_iso   = _view_axis_size(edges['ISO'],   has_dims=False)

        width_ratios  = [w_front, max(w_right, w_iso)]
        height_ratios = [max(h_top, h_iso), h_front]

        total_w = sum(width_ratios)
        total_h = sum(height_ratios)
        fig_w = 14.0
        fig_h = max(4.0, min(24.0, fig_w * total_h / total_w))

        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = fig.add_gridspec(2, 2,
                              width_ratios=width_ratios,
                              height_ratios=height_ratios,
                              hspace=0.05, wspace=0.05)
        ax_top   = fig.add_subplot(gs[0, 0])
        ax_iso   = fig.add_subplot(gs[0, 1])
        ax_front = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1])

        plot_view(ax_top,   edges['TOP'],   "", dim_mode='overall', add_radius_dims=True)
        plot_view(ax_iso,   edges['ISO'],   "", dim_mode='none',    add_radius_dims=False)
        plot_view(ax_front, edges['FRONT'], "", dim_mode='overall', add_radius_dims=True)
        plot_view(ax_right, edges['RIGHT'], "", dim_mode='overall', add_radius_dims=True)

        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as e:
        traceback.print_exc()
        plt.close('all')
        return False


# Point-cloud and Chamfer Distance utilities

def normalize_pc(pc):
    """Normalize a point cloud by its bounding-box center and diagonal."""
    if pc is None or len(pc) == 0:
        return None
    mn = pc.min(axis=0)
    mx = pc.max(axis=0)
    center = (mn + mx) / 2
    scale = np.linalg.norm(mx - mn)
    return (pc - center) / (scale + 1e-12)


def get_pcd_from_cq(cq_object, num_points, seed=123):
    """Sample a point cloud from a CadQuery object."""
    import tempfile
    tmp = None
    try:
        # The .stl suffix lets trimesh infer the temporary export format.
        fd, tmp = tempfile.mkstemp(suffix='.stl')
        os.close(fd)  # CadQuery writes after the descriptor is closed.
        cq.exporters.export(cq_object, tmp)
        if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
            raise RuntimeError(f"STL export produced empty or missing file: {tmp}")
        mesh = trimesh.load(tmp)
        np.random.seed(seed)
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        return points
    except Exception:
        traceback.print_exc()
        return None
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def get_pcd_from_stl(stl_path, num_points, seed=123):
    """Sample a point cloud from an STL file."""
    try:
        mesh = trimesh.load(stl_path)
        np.random.seed(seed)
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        return points
    except Exception:
        traceback.print_exc()
        return None


def _rot_x(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

def _rot_y(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

def _rot_z(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def chamfer_dist(pts_a, pts_b):
    """Compute symmetric Chamfer Distance using squared L2 distances."""
    kd_b = cKDTree(pts_b)
    d_a2b, _ = kd_b.query(pts_a)

    kd_a = cKDTree(pts_a)
    d_b2a, _ = kd_a.query(pts_b)

    return float(np.mean(d_a2b ** 2) + np.mean(d_b2a ** 2))


# Map SolidWorks Top-Plane sketches extruded along +Y to CadQuery XY
# sketches extruded along +Z via a fixed -90-degree X rotation.
_R_SW_TO_CQ = _rot_x(-90)


def compute_cd(cq_object, gt_stl_path, sample_points=10000):
    """Compute Chamfer Distance between generated geometry and a ground-truth STL.

    Both point clouds are bounding-box normalized to remove translation and
    scale differences. The minimum distance is taken before and after the
    fixed SolidWorks-to-CadQuery coordinate rotation.
    """
    if not gt_stl_path or not os.path.exists(gt_stl_path):
        return None
    try:
        pts_gt_raw = get_pcd_from_stl(gt_stl_path, sample_points)
        pts_gen_raw = get_pcd_from_cq(cq_object, sample_points)
        if pts_gt_raw is None or pts_gen_raw is None:
            return None

        # Normalize both clouds to remove origin and scale differences.
        pts_gt  = normalize_pc(pts_gt_raw)
        pts_gen = normalize_pc(pts_gen_raw)

        cd_direct = chamfer_dist(pts_gt, pts_gen)

        # Also evaluate the fixed SolidWorks-to-CadQuery rotation.
        cd_rotated = chamfer_dist(pts_gt, pts_gen @ _R_SW_TO_CQ.T)

        return min(cd_direct, cd_rotated)

    except Exception:
        traceback.print_exc()
        return None