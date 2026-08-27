"""Analyze and render orthographic CAD views."""
from OCP.BRepAdaptor import BRepAdaptor_Surface
import numpy as np
import math

# -----------------------------------------------------------------------------
# Geometry Projection & Analysis
# -----------------------------------------------------------------------------
def project_edge(edge, view_plane_normal, view_plane_x_dir):
    """Project a 3D edge onto the 2D plane defined by its normal and x-axis.

    Degenerate or corrupt dataset edges can raise OCCT geometry errors. Such
    edges are skipped so batch rendering can continue.
    """

    # Calculate Y direction of the view plane
    z_axis = np.array(view_plane_normal)
    x_axis = np.array(view_plane_x_dir)
    y_axis = np.cross(z_axis, x_axis)

    def to_2d(vec):
        v = np.array([vec.x, vec.y, vec.z])
        return (np.dot(v, x_axis), np.dot(v, y_axis))

    def sample_edge_points(e, n=120):
        """Uniformly sample an edge via positionAt for CadQuery 2.6."""
        ts = np.linspace(0.0, 1.0, int(n))
        pts = [e.positionAt(float(t)) for t in ts]
        # Remove duplicate samples that can destabilize rendering.
        if len(pts) >= 2:
            dedup = [pts[0]]
            for p in pts[1:]:
                lp = dedup[-1]
                if (abs(p.x - lp.x) > 1e-12) or (abs(p.y - lp.y) > 1e-12) or (abs(p.z - lp.z) > 1e-12):
                    dedup.append(p)
            pts = dedup
        return pts

    try:
        geom_type = edge.geomType()
    except Exception:
        return None

    if geom_type == 'LINE':
        try:
            p0 = edge.startPoint()
            p1 = edge.endPoint()
        except Exception:
            return None
        p0_2d = to_2d(p0)
        p1_2d = to_2d(p1)
        return {
            'type': 'LINE',
            'start': p0_2d,
            'end': p1_2d
        }
    elif geom_type in ['CIRCLE', 'ARC']:
        # CadQuery 2.6 edges lack discretize(), so sample with positionAt().
        try:
            pts = sample_edge_points(edge, n=120)
        except Exception:
            return None
        if not pts or len(pts) < 2:
            return None
        pts_2d = [to_2d(p) for p in pts]

        # Preserve arc geometry for radius annotations.
        center_2d = None
        radius = None
        mid_2d = None
        try:
            if hasattr(edge, "arcCenter"):
                c3 = edge.arcCenter()
                center_2d = to_2d(c3)
            if hasattr(edge, "radius"):
                radius = float(edge.radius())
            mid_2d = to_2d(edge.positionAt(0.5))
        except Exception:
            # The sampled polyline remains drawable without arc metadata.
            pass

        # An edge-on circle projects to a near-collinear polyline.
        xs = [p[0] for p in pts_2d]
        ys = [p[1] for p in pts_2d]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        is_visual_circle = True
        if width < 1e-5 or height < 1e-5:
             is_visual_circle = False

        return {
            'type': 'POLY', # Generic drawable polyline
            'points': pts_2d,
            'start': pts_2d[0],
            'end': pts_2d[-1],
            'is_circle': is_visual_circle, # True only for a circle-like 2D projection
            'center': center_2d,
            'radius': radius,
            'mid': mid_2d,
        }
    else:
        try:
            pts = sample_edge_points(edge, n=120)
        except Exception:
            return None
        if not pts or len(pts) < 2:
            return None
        pts_2d = [to_2d(p) for p in pts]
        return {
            'type': 'POLY',
            'points': pts_2d,
            'start': pts_2d[0],
            'end': pts_2d[-1]
        }

def analyze_view(solid, view_name):
    """Return projected 2D edges for the dataset's view convention.

    - FRONT: XY plane (Look -Z) -> Large face
    - TOP: XZ plane (Look -Y) -> Flat face
    - RIGHT: YZ plane (Look -X, Z horizontal) -> Side face
    """
    if view_name == 'FRONT':
        # Dataset "Front" is XY plane (0.75 x 0.42)
        normal = (0, 0, 1)
        x_dir = (1, 0, 0)
    elif view_name == 'TOP':
        # Dataset "Top" is XZ plane (0.75 x 0.17)
        normal = (0, -1, 0)
        x_dir = (1, 0, 0)
    elif view_name == 'RIGHT':
        # Looking from +X with -Z horizontal makes +Y vertical.
        normal = (1, 0, 0)
        x_dir = (0, 0, -1)
    elif view_name == 'ISO': # Isometric View (Front-Right-Top)
        n = np.array([1, 1, 1])
        normal = tuple(n / np.linalg.norm(n))
        x = np.array([1, 0, -1])
        x_dir = tuple(x / np.linalg.norm(x))

    # Unwrap Workplanes while accepting Solid objects directly.
    try:
        if hasattr(solid, 'val') and callable(getattr(solid, 'val', None)):
            actual_solid = solid.val()
        else:
            actual_solid = solid
    except (AttributeError, TypeError):
        actual_solid = solid

    edges = actual_solid.Edges()

    projected_edges = []
    seen_lines = set() # To deduplicate identical projections (e.g. top and bottom of a cube)

    # Edges() omits cylindrical side outlines, so derive their silhouettes.

    faces = actual_solid.Faces()
    for face in faces:
        if face.geomType() == 'CYLINDER':
            surf = BRepAdaptor_Surface(face.wrapped) # Adaptor3d_Surface
            cyl = surf.Cylinder() # gp_Cylinder

            radius = cyl.Radius()
            axis = cyl.Axis() # gp_Ax1
            location = axis.Location() # gp_Pnt
            direction = axis.Direction() # gp_Dir

            axis_vec = np.array([direction.X(), direction.Y(), direction.Z()])
            axis_loc = np.array([location.X(), location.Y(), location.Z()])

            view_n = np.array(normal)

            # Side views expose two silhouette generators parallel to the axis.
            if abs(np.dot(axis_vec, view_n)) < 0.05: # Perpendicular enough
                side_vec = np.cross(axis_vec, view_n)
                norm = np.linalg.norm(side_vec)
                if norm < 1e-6: continue
                side_vec = side_vec / norm

                # Bound the infinite cylinder by projecting face-edge endpoints.

                face_edges = face.Edges()
                min_k, max_k = float('inf'), float('-inf')

                valid_face = False
                for fe in face_edges:
                    pts = [fe.positionAt(t) for t in [0.0, 1.0]]
                    for p in pts:
                        p_vec = np.array([p.x, p.y, p.z])
                        k = np.dot(p_vec - axis_loc, axis_vec)
                        if k < min_k: min_k = k
                        if k > max_k: max_k = k
                        valid_face = True

                if not valid_face or min_k > max_k:
                    continue

                p_center_bottom = axis_loc + axis_vec * min_k
                p_center_top    = axis_loc + axis_vec * max_k

                shift = side_vec * radius

                silhouettes = [
                    (p_center_bottom + shift, p_center_top + shift),
                    (p_center_bottom - shift, p_center_top - shift)
                ]

                for p_start, p_end in silhouettes:
                    # project_edge expects a CadQuery Edge, so project raw points here.
                    z_axis_v = np.array(normal)
                    x_axis_v = np.array(x_dir)
                    y_axis_v = np.cross(z_axis_v, x_axis_v)

                    def vec_to_2d(v_3d):
                        return (np.dot(v_3d, x_axis_v), np.dot(v_3d, y_axis_v))

                    s_2d = vec_to_2d(p_start)
                    e_2d = vec_to_2d(p_end)

                    proj = {
                        'type': 'LINE',
                        'start': s_2d,
                        'end': e_2d
                    }

                    x1, y1 = s_2d
                    x2, y2 = e_2d
                    if (x1 > x2) or (x1 == x2 and y1 > y2):
                        x1, y1, x2, y2 = x2, y2, x1, y1

                    key = ('LINE', round(x1, 5), round(y1, 5), round(x2, 5), round(y2, 5))
                    if key in seen_lines:
                        continue
                    seen_lines.add(key)

                    projected_edges.append(proj)

    for edge in edges:
        try:
            proj = project_edge(edge, normal, x_dir)
        except Exception:
            continue
        if not proj:
            continue

        # Rounded canonical keys merge coincident projected edges.
        if proj['type'] == 'LINE':
            x1, y1 = proj['start']
            x2, y2 = proj['end']
            if (x1 > x2) or (x1 == x2 and y1 > y2):
                x1, y1, x2, y2 = x2, y2, x1, y1

            key = ('LINE', round(x1, 5), round(y1, 5), round(x2, 5), round(y2, 5))
            if key in seen_lines:
                continue
            seen_lines.add(key)
        elif proj['type'] == 'POLY':
            pts = proj['points']
            if len(pts) < 2: continue
            p_s = pts[0]
            p_e = pts[-1]
            p_m = pts[len(pts)//2]

            k1 = ('POLY', round(p_s[0],5), round(p_s[1],5),
                  round(p_m[0],5), round(p_m[1],5),
                  round(p_e[0],5), round(p_e[1],5))
            k2 = ('POLY', round(p_e[0],5), round(p_e[1],5),
                  round(p_m[0],5), round(p_m[1],5),
                  round(p_s[0],5), round(p_s[1],5))

            if (k1 in seen_lines) or (k2 in seen_lines):
                continue
            seen_lines.add(k1)

        projected_edges.append(proj)

    return projected_edges

# -----------------------------------------------------------------------------
# 3. Plotting & Dimensioning
# -----------------------------------------------------------------------------
def plot_view(ax, edges, title, dim_mode='overall', add_radius_dims=False):
    ax.set_aspect('equal')
    ax.axis('off')

    all_x = []
    all_y = []

    # Track dimension intervals by level to prevent overlap.
    h_dims = []
    v_dims = []

    for edge in edges:
        if edge['type'] == 'LINE':
            xs = [edge['start'][0], edge['end'][0]]
            ys = [edge['start'][1], edge['end'][1]]
            if math.hypot(xs[1]-xs[0], ys[1]-ys[0]) < 1e-4:
                continue
            ax.plot(xs, ys, 'k-', lw=1.5)
            all_x.extend(xs)
            all_y.extend(ys)
        elif edge['type'] == 'POLY':
            xs = [p[0] for p in edge['points']]
            ys = [p[1] for p in edge['points']]
            if not xs or (max(xs)-min(xs) < 1e-4 and max(ys)-min(ys) < 1e-4):
                continue
            ax.plot(xs, ys, 'k-', lw=1.5)
            all_x.extend(xs)
            all_y.extend(ys)

    if not all_x:
        return

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    margin_x = (max_x - min_x) * 0.3
    margin_y = (max_y - min_y) * 0.3
    if margin_x == 0: margin_x = 0.1
    if margin_y == 0: margin_y = 0.1

    ax.set_xlim(min_x - margin_x, max_x + margin_x)
    ax.set_ylim(min_y - margin_y, max_y + margin_y)

    if dim_mode == 'none':
        return

    def get_dim_levels(segments, pad_ratio=0.05):
        """Assign non-overlapping tracks to dimension intervals.

        Short dimensions are placed first and kept closest to the geometry.
        """
        indexed = []
        for i, seg in enumerate(segments):
            s, e = seg['s'], seg['e']
            if s > e: s, e = e, s
            indexed.append({'id': i, 's': s, 'e': e, 'len': e - s})

        indexed.sort(key=lambda x: x['len'])

        levels = [] # Occupied intervals per track
        results = {}

        for item in indexed:
            s, e = item['s'], item['e']
            placed = False
            for lvl_idx, occupied in enumerate(levels):
                collision = False
                for occ_s, occ_e in occupied:
                    margin = (item['len'] + (occ_e - occ_s)) * pad_ratio / 2.0
                    if not (e < occ_s - margin or s > occ_e + margin):
                        collision = True
                        break
                if not collision:
                    occupied.append((s, e))
                    results[item['id']] = lvl_idx
                    placed = True
                    break

            if not placed:
                levels.append([(s, e)])
                results[item['id']] = len(levels) - 1

        return results

    h_candidates = {} # (x1, x2) -> y_ref (closest to top, for dims above)

    v_candidates = {} # (y1, y2) -> x_ref (closest to right, for dims on right)

    for edge in edges:
        if edge['type'] == 'LINE':
            x1, y1 = edge['start']
            x2, y2 = edge['end']

            if abs(y1 - y2) < 1e-5 and abs(x1 - x2) > 1e-4:
                sx, ex = min(x1, x2), max(x1, x2)
                k = (round(sx, 4), round(ex, 4))
                if k not in h_candidates or y1 > h_candidates[k]:
                    h_candidates[k] = y1

            elif abs(x1 - x2) < 1e-5 and abs(y1 - y2) > 1e-4:
                sy, ey = min(y1, y2), max(y1, y2)
                k = (round(sy, 4), round(ey, 4))
                if k not in v_candidates or x1 > v_candidates[k]:
                    v_candidates[k] = x1

    h_dims = []
    for (sx, ex), y_ref in h_candidates.items():
        h_dims.append({'s': sx, 'e': ex, 'ref': y_ref, 'val': abs(ex-sx)})

    v_dims = []
    for (sy, ey), x_ref in v_candidates.items():
        v_dims.append({'s': sy, 'e': ey, 'ref': x_ref, 'val': abs(ey-sy)})

    h_levels = get_dim_levels(h_dims)
    v_levels = get_dim_levels(v_dims)

    rx = max_x - min_x
    ry = max_y - min_y

    max_h_level = max(h_levels.values()) if h_levels else 0
    max_v_level = max(v_levels.values()) if v_levels else 0

    margin_x_base = rx * 0.15 if rx > 0 else 0.1
    margin_y_base = ry * 0.15 if ry > 0 else 0.1

    step_y = ry * 0.15 if ry > 0 else 0.05
    step_x = rx * 0.20 if rx > 0 else 0.05

    # Reserve space above and right for stacked dimension tracks.
    extra_margin_top   = (max_h_level + 2) * step_y
    extra_margin_right = (max_v_level + 2) * step_x

    ax.set_xlim(min_x - margin_x_base, max_x + margin_x_base + extra_margin_right)
    ax.set_ylim(min_y - margin_y_base, max_y + margin_y_base + extra_margin_top)

    dim_color = '#1f77b4' # Muted Blue
    text_color = '#d62728' # Muted Red
    ext_color = 'gray' # Gray for extension lines

    h_base_y = max_y + margin_y_base
    v_base_x = max_x + margin_x_base

    for i, dim in enumerate(h_dims):
        lvl = h_levels[i]
        y_pos = h_base_y + (lvl * step_y)
        sx, ex = dim['s'], dim['e']

        ax.plot([sx, ex], [y_pos, y_pos], '-', lw=0.8, color=dim_color)
        ax.annotate('', xy=(sx, y_pos), xytext=(sx + (ex-sx)*0.1, y_pos),
                   arrowprops=dict(arrowstyle='<|-,head_length=0.4,head_width=0.2', lw=0.8, color=dim_color))
        ax.annotate('', xy=(ex, y_pos), xytext=(ex - (ex-sx)*0.1, y_pos),
                   arrowprops=dict(arrowstyle='<|-,head_length=0.4,head_width=0.2', lw=0.8, color=dim_color))

        txt = f"{dim['val']:.4f}"
        ax.text((sx+ex)/2, y_pos, txt, ha='center', va='center',
               fontsize=11, color=text_color, fontweight='bold',
               bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.0))

        gap = ry * 0.02
        ax.plot([sx, sx], [dim['ref'] + gap, y_pos], '-', lw=0.5, color=ext_color, alpha=0.6)
        ax.plot([ex, ex], [dim['ref'] + gap, y_pos], '-', lw=0.5, color=ext_color, alpha=0.6)

    for i, dim in enumerate(v_dims):
        lvl = v_levels[i]
        x_pos = v_base_x + (lvl * step_x)
        sy, ey = dim['s'], dim['e']

        ax.plot([x_pos, x_pos], [sy, ey], '-', lw=0.8, color=dim_color)
        ax.annotate('', xy=(x_pos, sy), xytext=(x_pos, sy + (ey-sy)*0.1),
                   arrowprops=dict(arrowstyle='<|-,head_length=0.4,head_width=0.2', lw=0.8, color=dim_color))
        ax.annotate('', xy=(x_pos, ey), xytext=(x_pos, ey - (ey-sy)*0.1),
                   arrowprops=dict(arrowstyle='<|-,head_length=0.4,head_width=0.2', lw=0.8, color=dim_color))

        txt = f"{dim['val']:.4f}"
        ax.text(x_pos, (sy+ey)/2, txt, ha='center', va='center',
               rotation=90, fontsize=11, color=text_color, fontweight='bold',
               bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.0))

        gap = rx * 0.02
        ax.plot([dim['ref'] + gap, x_pos], [sy, sy], '-', lw=0.5, color=ext_color, alpha=0.6)
        ax.plot([dim['ref'] + gap, x_pos], [ey, ey], '-', lw=0.5, color=ext_color, alpha=0.6)


    if add_radius_dims:
        add_radius_annotations(ax, edges, rx, ry, bounds=(min_x, max_x, min_y, max_y))

def add_radius_annotations(ax, edges, rx, ry, bounds=None):
    """Add deduplicated radius labels with approximate collision avoidance."""
    circles = []
    for e in edges:
        if not e.get('is_circle'): continue
        if e.get('radius') is None or e.get('center') is None: continue
        circles.append(e)

    if not circles:
        return

    if bounds:
        min_x, max_x, min_y, max_y = bounds
        part_cx = (min_x + max_x) / 2
        part_cy = (min_y + max_y) / 2
    else:
        all_pts = []
        for e in edges:
            if 'points' in e: all_pts.extend(e['points'])
            elif 'start' in e:
                all_pts.append(e['start'])
                all_pts.append(e['end'])
        if all_pts:
            xs = [p[0] for p in all_pts]
            ys = [p[1] for p in all_pts]
            part_cx = (min(xs) + max(xs)) / 2
            part_cy = (min(ys) + max(ys)) / 2
        else:
            part_cx, part_cy = 0, 0

    circles.sort(key=lambda x: (x['center'][0], x['center'][1]))

    seen_keys = set()

    # Approximate occupied label centers for collision checks.
    occupied_positions = []

    base_length = max(rx, ry) * 0.3

    for e in circles:
        r = e['radius']
        cx, cy = e['center']

        key = (round(cx, 6), round(cy, 6), round(r, 6))
        if key in seen_keys:
            continue
        seen_keys.add(key)

        # Bias leaders away from the part center to reduce interior clutter.
        dir_x = cx - part_cx
        dir_y = cy - part_cy

        base_angle = math.atan2(dir_y, dir_x)
        if base_angle < 0: base_angle += 2*math.pi

        angle_deg = round(math.degrees(base_angle) / 45) * 45

        angle_rad = math.radians(angle_deg)
        ux = math.cos(angle_rad)
        uy = math.sin(angle_rad)

        start_x = cx + r * ux
        start_y = cy + r * uy

        current_len = base_length
        max_attempts = 12

        label = f"R{r:.3f}"

        final_tx, final_ty = 0, 0
        final_align_h, final_align_v = 'left', 'bottom'

        for attempt in range(max_attempts):
            tx = cx + (r + current_len) * ux
            ty = cy + (r + current_len) * uy

            collision = False
            for (ox, oy) in occupied_positions:
                dist = math.hypot(tx - ox, ty - oy)
                if dist < max(rx, ry) * 0.20:
                    collision = True
                    break

            if not collision:
                final_tx, final_ty = tx, ty
                occupied_positions.append((tx, ty))
                break
            else:
                # Rotate the leader before extending it.
                angle_rad += math.radians(30)
                ux = math.cos(angle_rad)
                uy = math.sin(angle_rad)

                start_x = cx + r * ux
                start_y = cy + r * uy

                if attempt > 0 and attempt % 12 == 0:
                     current_len += base_length * 0.2

        if final_tx == 0:
             final_tx = cx + (r + base_length) * ux
             final_ty = cy + (r + base_length) * uy

        ha = 'left' if ux >= 0 else 'right'
        va = 'bottom' if uy >= 0 else 'top'

        ax.annotate(
            label,
            xy=(start_x, start_y),
            xytext=(final_tx, final_ty),
            textcoords='data',
            ha=ha,
            va=va,
            fontsize=9,
            color='darkgreen',
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
            arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=0", color='darkgreen', lw=1.2)
        )

        ax.plot(cx, cy, 'g+', ms=5)
