"""
360 GIF generator for dental plaque models.
Each model rotates 180° (left side → front → right side), then switches to next.
"""
import trimesh, numpy as np, imageio, time
from scipy.spatial import KDTree
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Config ──────────────────────────────────────────────────────────────────
BASE = Path("/home/Zhen/projects/SegmentAnyTooth")

MODEL_PATHS = [
    BASE / "data/2/plaque_output/plaque_by_fdi.obj",
    BASE / "data/1/plaque_output/plaque_by_fdi.obj",
    BASE / "data/4/plaque_output/plaque_by_fdi.obj",
    BASE / "plaque_output/plaque_by_fdi.obj",
]
MODEL_PATHS = [p for p in MODEL_PATHS if p.exists()]

OUTPUT_GIF   = BASE / "teeth_360.gif"
FACE_COUNT   = 40000    # decimation target
N_FRAMES     = 24       # frames per 180° rotation
PAUSE_FRAMES = 4        # freeze frames between models
ELEV         = 15        # camera elevation (degrees)
AZIM_START   = 0        # left side
AZIM_END     = -180     # right side (passes through front at -90)
FPS          = 10
IMG_SIZE     = (800, 600)
DPI          = 100
LIGHT_DIR    = np.array([0.4, 0.6, 1.0])   # light direction (xyz)
LIGHT_DIR    = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
AMBIENT      = 0.35     # ambient light strength (0~1)
BG_COLOR     = 'black'
CAM_DIST     = 4.0      # smaller = closer / bigger model (was 6.5)
# ────────────────────────────────────────────────────────────────────────────


def load_and_prep(obj_path: Path):
    """Load OBJ, decimate, transfer vertex colors, return face data."""
    print(f"  Loading {obj_path.parent.parent.name}/{obj_path.parent.name}...")
    scene = trimesh.load(str(obj_path), force='scene')
    geo   = list(scene.geometry.values())[0]

    orig_verts  = np.array(geo.vertices)
    orig_colors = np.array(geo.visual.vertex_colors)[:, :3] / 255.0

    print(f"  Decimating {len(geo.faces):,} → {FACE_COUNT:,} faces...")
    dec       = geo.simplify_quadric_decimation(face_count=FACE_COUNT)
    dec_verts = np.array(dec.vertices)
    dec_faces = np.array(dec.faces)

    # Nearest-neighbor color transfer
    _, idx     = KDTree(orig_verts).query(dec_verts)
    dec_colors = orig_colors[idx].copy()

    # Remap tooth base color → near-white (match browser rendering)
    # Plaque = red channel dominant; everything else → white
    is_plaque = (dec_colors[:, 0] > 0.5) & (dec_colors[:, 1] < 0.4)
    tooth_white = np.array([0.94, 0.93, 0.91])
    dec_colors[~is_plaque] = tooth_white
    # Keep plaque bright red
    dec_colors[is_plaque] = [0.88, 0.12, 0.12]

    # Face vertices and base colors (mean of 3 verts)
    face_verts  = dec_verts[dec_faces]          # (F, 3, 3)
    face_colors = dec_colors[dec_faces].mean(1)  # (F, 3)

    # Per-face normals for Lambertian shading
    v0, v1, v2  = face_verts[:,0], face_verts[:,1], face_verts[:,2]
    normals     = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    norms_mag   = np.linalg.norm(normals, axis=1, keepdims=True)
    norms_mag   = np.where(norms_mag < 1e-8, 1.0, norms_mag)
    normals    /= norms_mag

    return face_verts, face_colors, normals, geo.bounds


def shade_colors(face_colors, normals, azim_deg):
    """Apply Lambertian shading that rotates with the view."""
    # Rotate light direction opposite to model rotation (keeps light consistent)
    a = np.radians(-azim_deg)
    rot = np.array([[np.cos(a), -np.sin(a), 0],
                    [np.sin(a),  np.cos(a), 0],
                    [0,          0,         1]])
    light = rot @ LIGHT_DIR
    diffuse   = np.clip(normals @ light, 0, 1)
    intensity = AMBIENT + (1 - AMBIENT) * diffuse[:, None]
    return np.clip(face_colors * intensity, 0, 1)


def render_frame(face_verts, face_colors, normals, bounds, elev, azim):
    w_px = int(IMG_SIZE[0])
    h_px = int(IMG_SIZE[1])
    fig  = plt.figure(figsize=(w_px/DPI, h_px/DPI), facecolor=BG_COLOR)
    ax   = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(BG_COLOR)

    shaded = shade_colors(face_colors, normals, azim)

    poly = Poly3DCollection(face_verts, alpha=1.0, linewidth=0, antialiased=False)
    poly.set_facecolor(shaded)
    ax.add_collection3d(poly)

    b = bounds
    ax.set_xlim(b[0,0], b[1,0])
    ax.set_ylim(b[0,1], b[1,1])
    ax.set_zlim(b[0,2], b[1,2])
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    ax._dist = CAM_DIST

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    return buf


def main():
    print(f"Models to render: {len(MODEL_PATHS)}")
    all_frames = []

    for m_idx, obj_path in enumerate(MODEL_PATHS):
        print(f"\n[{m_idx+1}/{len(MODEL_PATHS)}] {obj_path.name}")
        t0 = time.time()

        face_verts, face_colors, normals, bounds = load_and_prep(obj_path)

        azims = np.linspace(AZIM_START, AZIM_END, N_FRAMES)

        for i, azim in enumerate(azims):
            print(f"  frame {i+1:2d}/{N_FRAMES}  azim={azim:7.1f}°", end='\r')
            frame = render_frame(face_verts, face_colors, normals, bounds, ELEV, azim)
            all_frames.append(frame)

        # Pause on last frame before switching model
        for _ in range(PAUSE_FRAMES):
            all_frames.append(frame)

        print(f"\n  Done in {time.time()-t0:.1f}s")

    print(f"\nSaving GIF → {OUTPUT_GIF}")
    print(f"Total frames: {len(all_frames)}, size: {all_frames[0].shape}")
    imageio.mimsave(str(OUTPUT_GIF), all_frames, fps=FPS, loop=0)
    size_mb = OUTPUT_GIF.stat().st_size / 1024 / 1024
    print(f"Done! {size_mb:.1f} MB → {OUTPUT_GIF}")


if __name__ == '__main__':
    main()
