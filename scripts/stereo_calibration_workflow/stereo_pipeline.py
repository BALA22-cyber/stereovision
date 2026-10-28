#!/usr/bin/env python3
"""Reusable stereo calibration and reconstruction for Basler or other camera pairs.

Requirements: Python >= 3.9; numpy; opencv-contrib-python (GUI build for --debug).
All physical distances are in meters. Input PNGs must be already demosaiced if Bayer.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# Set to False for unattended/headless processing, or pass --no-debug per command.
DEBUG = True
PAIR_RE = re.compile(r"^(?P<stem>.+)_(?P<side>left|right)_(?P<serial>\d+)\.(png|jpg|jpeg|tif|tiff)$", re.I)


def camera_pairs(folder: Path) -> list[tuple[str, Path, Path]]:
    found = {}
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        m = PAIR_RE.match(p.name)
        if m:
            found.setdefault(m.group('stem'), {})[m.group('side').lower()] = p
    pairs = [(name, sides['left'], sides['right'])
             for name, sides in sorted(found.items())
             if 'left' in sides and 'right' in sides]
    print(f"[pairs] {folder}: {len(pairs)} matched left/right pairs")
    if not pairs:
        raise ValueError(f'No complete stereo pairs under {folder}. Expected NAME_left_SERIAL.png and NAME_right_SERIAL.png.')
    return pairs


def read_pair(left_path: Path, right_path: Path, expected_size=None):
    left, right = cv2.imread(str(left_path), cv2.IMREAD_COLOR), cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise ValueError(f'Cannot read pair: {left_path}, {right_path}')
    size = (left.shape[1], left.shape[0])
    if size != (right.shape[1], right.shape[0]):
        raise ValueError(f'Left and right resolutions differ: {left_path}, {right_path}')
    if expected_size is not None and tuple(size) != tuple(expected_size):
        raise ValueError(f'Image {left_path.name}: got {size}; calibration expects {tuple(expected_size)}. '
                         'Do not resize images without appropriately rescaling intrinsics.')
    return left, right


class Debugger:
    def __init__(self, enabled: bool, out: Path):
        self.enabled = enabled
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.gui = bool(enabled) and (os.name == 'nt' or sys.platform == 'darwin' or
                                      bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')))
        self.delay_ms = 180
        self._warned = False

    def show(self, name: str, img: np.ndarray, file: str | None = None):
        if not self.enabled:
            return
        if img.dtype != np.uint8:
            raise ValueError('Debugger.show expects uint8 visualization')
        if file:
            p = self.out / file
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), img)
        if not self.gui:
            if not self._warned:
                print('[debug] No graphical display found. Saving previews to disk instead of cv2.imshow.')
                self._warned = True
            return
        h, w = img.shape[:2]
        scale = min(1.0, 1500.0 / w, 900.0 / h)
        display = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
        cv2.imshow(name, display)
        key = cv2.waitKey(self.delay_ms) & 0xFF
        if key in (ord('p'), ord(' ')):
            cv2.waitKey(0)
        if key == ord('q'):
            self.gui = False
            cv2.destroyAllWindows()
            print('[debug] Preview windows disabled (processing continues).')

    def close(self):
        if self.gui:
            cv2.destroyAllWindows()


def side_by_side(left: np.ndarray, right: np.ndarray, lines=False) -> np.ndarray:
    a, b = left.copy(), right.copy()
    if lines:
        for y in range(0, a.shape[0], max(80, a.shape[0] // 14)):
            cv2.line(a, (0, y), (a.shape[1], y), (20, 220, 255), 2)
            cv2.line(b, (0, y), (b.shape[1], y), (20, 220, 255), 2)
    return np.hstack([a, b])


def chess_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    # cols, rows are the number of INNER corners, not number of black/white squares.
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[:cols, :rows].T.reshape(-1, 2) * square_size
    return obj


def detect_checkerboard(image: np.ndarray, cols: int, rows: int):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pattern = (cols, rows)
    if hasattr(cv2, 'findChessboardCornersSB'):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
        ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
    else:
        ok, corners = cv2.findChessboardCorners(gray, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
    return corners.astype(np.float32) if ok else None


def detect_checkerboard_dataset(folder, cols, rows, square_size, dbg):
    obj_template = chess_object_points(cols, rows, square_size)
    by_side = {'left': ([], []), 'right': ([], [])}
    stereo_obj, stereo_l, stereo_r = [], [], []
    size = None
    for i, (name, lp, rp) in enumerate(camera_pairs(folder), 1):
        l, r = read_pair(lp, rp, size)
        size = (l.shape[1], l.shape[0])
        cl, cr = detect_checkerboard(l, cols, rows), detect_checkerboard(r, cols, rows)
        for side, c in [('left', cl), ('right', cr)]:
            if c is not None:
                by_side[side][0].append(obj_template.copy())
                by_side[side][1].append(c)
        if cl is not None and cr is not None:
            stereo_obj.append(obj_template.copy())
            stereo_l.append(cl)
            stereo_r.append(cr)
        print(f'  [{i:02d}] {name}: left={cl is not None} right={cr is not None}')
        if dbg.enabled:
            if cl is not None: cv2.drawChessboardCorners(l, (cols, rows), cl, True)
            if cr is not None: cv2.drawChessboardCorners(r, (cols, rows), cr, True)
            dbg.show('checkerboard detection', side_by_side(l, r),
                     f'checkerboard/{name}.jpg' if i <= 8 else None)
    if size is None:
        raise RuntimeError('No images found.')
    return by_side, (stereo_obj, stereo_l, stereo_r), size


def output_npz(path: Path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **values)
    print(f'[save] {path}')


def as_dict(path: Path):
    with np.load(str(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False))
    print(f'[save] {path}')


def calibrate_intrinsics(args, dbg):
    sides, stereo, size = detect_checkerboard_dataset(args.images, args.cols, args.rows, args.square_size, dbg)
    result = {}
    for label, suffix in [('left', '1'), ('right', '2')]:
        obj, pts = sides[label]
        if len(obj) < 8:
            raise RuntimeError(f'Only {len(obj)} checkerboards detected in {label}; need at least 8 diverse poses.')
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(obj, pts, size, None, None)
        result[f'K{suffix}'] = K
        result[f'D{suffix}'] = D
        print(f'[{label}] RMS={rms:.3f} px | detected={len(obj)} | K=\n{K}\nD={D.ravel()}')
        result[f'rms_{label}'] = np.array([rms])
        result[f'n_{label}'] = np.array([len(obj)])
    result['size'] = np.array(size)
    output_npz(args.out, **result)
    save_json(args.out.with_suffix('.json'),
              {'image_size_px': size, 'checkerboard_inner_corners': [args.cols, args.rows],
               'square_size_m': args.square_size,
               'left_detected': len(sides['left'][0]), 'right_detected': len(sides['right'][0]),
               'jointly_detected': len(stereo[0]),
               'left_rms_px': float(result['rms_left'][0]),
               'right_rms_px': float(result['rms_right'][0])})


def rectification_matrices(calib: dict, R: np.ndarray, T: np.ndarray, alpha: float):
    size = tuple(int(x) for x in calib['size'])
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        calib['K1'], calib['D1'], calib['K2'], calib['D2'], size, R, T.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
    return dict(R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
                roi1=np.asarray(roi1), roi2=np.asarray(roi2))


def validate_rectification(rig):
    """Reject unusable rectification before exporting misleading depth products."""
    width, height = (int(v) for v in rig['size'])
    P = rig['P1']
    if not np.isfinite(P).all() or P[0, 0] <= 0 or P[1, 1] <= 0:
        raise ValueError('Rectification has an invalid focal length; recalibrate the rig and intrinsics.')
    for key in ('roi1', 'roi2'):
        if key in rig and (rig[key][2] <= 0 or rig[key][3] <= 0):
            raise ValueError(f'Rectification {key} is empty. The saved camera model or stereo geometry is '
                             'unusable; check calibration images and intrinsics, then recalibrate the rig.')
    if not (0 <= P[0, 2] < width and 0 <= P[1, 2] < height):
        raise ValueError(f'Rectified principal point ({P[0, 2]:.1f}, {P[1, 2]:.1f}) lies outside '
                         f'the {width}x{height} image. Check the camera model and recalibrate the rig.')


def rectified_points(c, K, D, R, P):
    return cv2.undistortPoints(np.asarray(c, np.float32).reshape(-1, 1, 2), K, D,
                               R=R, P=P).reshape(-1, 2)


def epi_metrics(corners_l, corners_r, calib):
    errors = []
    for l, r in zip(corners_l, corners_r):
        rl = rectified_points(l, calib['K1'], calib['D1'], calib['R1'], calib['P1'])
        rr = rectified_points(r, calib['K2'], calib['D2'], calib['R2'], calib['P2'])
        errors.extend(np.abs(rl[:, 1] - rr[:, 1]).tolist())
    if not errors:
        return {}
    return {'median_vertical_epipolar_error_px': float(np.median(errors)),
            'mean_vertical_epipolar_error_px': float(np.mean(errors)),
            'p95_vertical_epipolar_error_px': float(np.percentile(errors, 95)),
            'num_correspondences': len(errors)}


def save_rig(out_dir: Path, intrinsics: dict, R, T, source, diagnostics, alpha):
    rect = rectification_matrices(intrinsics, R, T, alpha)
    validate_rectification({**intrinsics, **rect})
    saved = {k: intrinsics[k] for k in ('K1', 'D1', 'K2', 'D2', 'size')}
    saved.update(R=R, T=T.reshape(3, 1), **rect)
    baseline = float(np.linalg.norm(T))
    if abs(rect['P2'][1, 3]) > abs(rect['P2'][0, 3]):
        raise ValueError('This arrangement produces VERTICAL stereo epipolar lines. Standard disparity models '
                         'expect a horizontal baseline. Reorient cameras or rotate both images with matching '
                         'camera-matrix transforms before stereo inference.')
    rect_baseline = float(-rect['P2'][0, 3] / rect['P2'][0, 0])
    output_npz(out_dir / 'rig.npz', **saved)
    save_json(out_dir / 'rig_report.json', {'source': source, 'baseline_m': baseline,
              'rectified_signed_baseline_m': rect_baseline, 'R_right_from_left': R.tolist(),
              'T_right_from_left_m': T.reshape(3).tolist(),
              'image_size_px': intrinsics['size'].astype(int).tolist(),
              'alpha': float(alpha), 'diagnostics': diagnostics})
    # Generic JSON export for depth-model integrations (NPZ remains authoritative).
    save_json(out_dir / 'camera_geometry.json', {
        'units': 'meters', 'image_size_px': intrinsics['size'].astype(int).tolist(),
        'left': {'K': intrinsics['K1'].tolist(), 'distortion': intrinsics['D1'].reshape(-1).tolist(),
                 'R_rectify': rect['R1'].tolist(), 'P_rectified': rect['P1'].tolist()},
        'right': {'K': intrinsics['K2'].tolist(), 'distortion': intrinsics['D2'].reshape(-1).tolist(),
                  'R_rectify': rect['R2'].tolist(), 'P_rectified': rect['P2'].tolist()},
        'T_right_from_left': T.reshape(3).tolist(), 'R_right_from_left': R.tolist(),
        'Q_disparity_to_3D': rect['Q'].tolist(),
        'baseline_m': baseline,
        'rectified_K': rect['P1'][:3,:3].tolist(),
        'rectification_ROI_left': rect['roi1'].tolist(),
        'rectification_ROI_right': rect['roi2'].tolist(),
    })
    Krect = rect['P1'][:3, :3]
    # FoundationStereo uses a flattened 3x3 rectified intrinsic followed by a positive baseline in meters.
    (out_dir / 'foundationstereo_intrinsics.txt').write_text(
        ' '.join(f'{v:.10f}' for v in Krect.reshape(-1)) + f'\n{baseline:.10f}\n')
    print(f'[rig] baseline={baseline:.6f} m; rectified signed baseline={rect_baseline:.6f} m')
    if rect_baseline <= 0:
        print('[WARN] Right camera may be physically on the wrong side of the designated left camera. '
              'Most left-to-right disparity algorithms expect positive left-minus-right disparities.')
    print(f'[save] {out_dir / "foundationstereo_intrinsics.txt"}')
    return saved


def calibrate_rig_board(args, dbg):
    intr = as_dict(args.intrinsics)
    sides, (obj, left, right), size = detect_checkerboard_dataset(
        args.images, args.cols, args.rows, args.square_size, dbg)
    if tuple(intr['size']) != size:
        raise ValueError(f'Calibration image size {size} differs from saved intrinsics {intr["size"]}')
    if len(obj) < 6:
        raise RuntimeError(f'Only {len(obj)} valid stereo checkerboard pairs; need more diverse paired views.')
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 150, 1e-7)
    rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj, left, right, intr['K1'].copy(), intr['D1'].copy(),
        intr['K2'].copy(), intr['D2'].copy(), size,
        criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)
    rig = rectification_matrices(intr, R, T, args.alpha)
    metrics = epi_metrics(left, right, {**intr, **rig})
    metrics.update(stereo_rms_px=float(rms), jointly_detected_pairs=len(obj))
    saved = save_rig(args.out, intr, R, T, 'checkerboard', metrics, args.alpha)
    print('[rig-board] ' + json.dumps(metrics, indent=2))
    if dbg.enabled:
        _, lp, rp = camera_pairs(args.images)[0]
        a, b = read_pair(lp, rp, intr['size'])
        ra, rb = rectify_pair(a, b, saved)
        dbg.show('before rectification', side_by_side(a, b, True), 'raw_example.jpg')
        dbg.show('after rectification', side_by_side(ra, rb, True), 'rectified_example.jpg')


def april_dictionary(family: str):
    if not hasattr(cv2, 'aruco'):
        raise RuntimeError('Missing cv2.aruco. Install opencv-contrib-python (avoid conflicting OpenCV pip wheels).')
    fam = family.upper()
    if not hasattr(cv2.aruco, fam):
        raise ValueError(f'Unsupported --tag-family {family}. Example: DICT_APRILTAG_36h11')
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, fam))
    params = cv2.aruco.DetectorParameters()
    if hasattr(params, 'cornerRefinementMethod'):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return d, params


def detect_tags(frame, dictionary, parameters):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        corners, ids, rejected = cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    result = {} if ids is None else {int(mid): corner.reshape(4, 2).astype(np.float32)
                                     for mid, corner in zip(ids.flatten(), corners)}
    return result


def tag_object_corners(size_m):
    half = size_m / 2.0
    # Coordinate convention expected by OpenCV's aruco marker corner ordering.
    return np.array([[-half, +half, 0], [+half, +half, 0],
                     [+half, -half, 0], [-half, -half, 0]], dtype=np.float32)


def tag_pose(corners, K, D, size_m):
    obj = tag_object_corners(size_m)
    # IPPE_SQUARE explicitly supports planar square tags with this corner ordering.
    ok, rvec, tvec = cv2.solvePnP(obj, corners.reshape(4, 1, 2), K, D,
                                   flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok or tvec[2, 0] <= 0:
        raise ValueError('AprilTag pose estimate failed or is behind the camera')
    R, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    error = np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - corners) ** 2, axis=1)))
    return R, tvec.reshape(3, 1), float(error)


def rotation_distance_deg(a, b):
    delta = a @ b.T
    c = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def average_rotation(rotations):
    acc = np.sum(rotations, axis=0)
    U, S, Vt = np.linalg.svd(acc)
    diag = np.diag([1, 1, np.linalg.det(U @ Vt)])
    return U @ diag @ Vt


def tag_samples(images, intr, tag_size, tag_family, dbg):
    dictionary, params = april_dictionary(tag_family)
    samples = []
    observations = []
    for n, lp, rp in camera_pairs(images):
        left, right = read_pair(lp, rp, intr['size'])
        dl, dr = detect_tags(left, dictionary, params), detect_tags(right, dictionary, params)
        shared = sorted(dl.keys() & dr.keys())
        print(f'  [tags] {n}: left={sorted(dl)} right={sorted(dr)} shared={shared}')
        for tagid in shared:
            cl, cr = dl[tagid], dr[tagid]
            try:
                Rl, tl, err_l = tag_pose(cl, intr['K1'], intr['D1'], tag_size)
                Rr, tr, err_r = tag_pose(cr, intr['K2'], intr['D2'], tag_size)
            except (cv2.error, ValueError) as exc:
                print(f'    [skip] tag {tagid}: {exc}')
                continue
            # X_right = R @ X_left + T; tag-to-camera transforms compose as below.
            R = Rr @ Rl.T
            T = tr - R @ tl
            samples.append({'pair': n, 'tag_id': tagid, 'R': R, 'T': T.reshape(3),
                            'left_pnp_error_px': err_l, 'right_pnp_error_px': err_r})
            observations.append((cl, cr))
        if dbg.enabled:
            for frame, det in ((left, dl), (right, dr)):
                for ident, pts in det.items():
                    cv2.polylines(frame, [pts.astype(np.int32)], True, (0, 255, 0), 4)
                    cv2.putText(frame, f'ID:{ident}', tuple(pts[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                                1.1, (0, 255, 255), 3)
            dbg.show('AprilTag detections', side_by_side(left, right),
                     f'tags/{n}.jpg' if len(samples) < 10 else None)
    return samples, observations


def calibrate_rig_tags(args, dbg):
    intr = as_dict(args.intrinsics)
    samples, obs = tag_samples(args.images, intr, args.tag_size, args.tag_family, dbg)
    if not samples:
        raise RuntimeError('No shared AprilTags detected. Confirm tag family, size, image visibility and IDs.')
    # Select medoid (most consistent measured transform), then discard inconsistent poses.
    # Translation threshold is in meters, rotation threshold in degrees.
    scores = []
    for a in samples:
        scores.append(sum(min(rotation_distance_deg(a['R'], b['R']) / args.max_rotation_deg, 10) +
                          min(np.linalg.norm(a['T'] - b['T']) / args.max_translation_m, 10)
                          for b in samples))
    ref = samples[int(np.argmin(scores))]
    inliers = [s for s in samples if rotation_distance_deg(s['R'], ref['R']) <= args.max_rotation_deg and
               np.linalg.norm(s['T'] - ref['T']) <= args.max_translation_m]
    if len(samples) > 1 and len(inliers) < 2:
        raise RuntimeError('Tag poses are inconsistent: only one inlier. Check tag dimensions/corner order, '
                           'tag visibility, and camera motion/synchronization.')
    R = average_rotation([s['R'] for s in inliers])
    T = np.median(np.array([s['T'] for s in inliers]), axis=0).reshape(3, 1)
    rt = rectification_matrices(intr, R, T, args.alpha)
    # Evaluation on tag observations used to estimate the rig is IN-SAMPLE only.
    # Each sample and observation share positional order; compare identities to avoid ndarray equality.
    kept_obs = [o for i, o in enumerate(obs) if any(samples[i] is j for j in inliers)]
    metrics = epi_metrics([p[0] for p in kept_obs], [p[1] for p in kept_obs], {**intr, **rt})
    metrics.update(candidate_tag_observations=len(samples), inlier_observations=len(inliers),
                   in_sample_only=True,
                   median_tag_pose_error_px=float(np.median(
                       [(s['left_pnp_error_px'] + s['right_pnp_error_px']) / 2 for s in inliers])))
    rig = save_rig(args.out, intr, R, T, 'apriltag', metrics, args.alpha)
    print('[rig-tags] ' + json.dumps(metrics, indent=2))
    if dbg.enabled:
        _, lp, rp = camera_pairs(args.images)[0]
        a, b = read_pair(lp, rp, intr['size'])
        ra, rb = rectify_pair(a, b, rig)
        dbg.show('before rectification', side_by_side(a, b, True), 'raw_example.jpg')
        dbg.show('after rectification', side_by_side(ra, rb, True), 'rectified_example.jpg')


def rectify_pair(left, right, rig, maps=None):
    if maps is None:
        maps = make_maps(rig)
    lmx, lmy, rmx, rmy = maps
    a = cv2.remap(left, lmx, lmy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    b = cv2.remap(right, rmx, rmy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return a, b


def make_maps(rig):
    size = tuple(int(x) for x in rig['size'])
    lmx, lmy = cv2.initUndistortRectifyMap(rig['K1'], rig['D1'], rig['R1'], rig['P1'],
                                          size, cv2.CV_32FC1)
    rmx, rmy = cv2.initUndistortRectifyMap(rig['K2'], rig['D2'], rig['R2'], rig['P2'],
                                          size, cv2.CV_32FC1)
    return lmx, lmy, rmx, rmy


def color_preview(values, valid, invert=False):
    positive = np.asarray(values)[valid & np.isfinite(values)]
    vis = np.zeros(values.shape, dtype=np.uint8)
    if len(positive):
        lo, hi = np.percentile(positive, [3, 97])
        if hi > lo:
            normalized = np.clip((values - lo) / (hi - lo), 0, 1)
            if invert:
                normalized = 1 - normalized
            vis[valid] = (normalized[valid] * 255).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def save_ply(path, xyz, bgr, mask, max_points):
    coords = xyz[mask].reshape(-1, 3)
    colors = bgr[mask].reshape(-1, 3)[:, ::-1]  # BGR -> RGB
    if len(coords) > max_points:
        choose = np.linspace(0, len(coords)-1, max_points).astype(int)
        coords, colors = coords[choose], colors[choose]
    points = np.empty(len(coords), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                          ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for i, k in enumerate(['x', 'y', 'z']): points[k] = coords[:, i]
    for i, k in enumerate(['red', 'green', 'blue']): points[k] = colors[:, i]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\n'
                 f'element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n'
                 'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode('ascii'))
        points.tofile(f)
    print(f'[save] {path} ({len(points)} points)')


def raw_sgbm_preview(l, r, args, path, dbg):
    """Unrectified SGBM is NOT valid depth; saved only to illustrate failure before rectification."""
    scale = args.sgbm_scale
    if scale != 1:
        l = cv2.resize(l, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        r = cv2.resize(r, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    l = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
    r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    b = args.block_size
    model = cv2.StereoSGBM_create(minDisparity=args.min_disparity,
        numDisparities=args.num_disparities, blockSize=b, P1=8*b*b, P2=32*b*b,
        disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100,
        speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    d = model.compute(l, r).astype(np.float32) / 16.0
    valid = np.isfinite(d) & (d > max(0, args.min_disparity))
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / 'raw_UNRECTIFIED_disparity_NOT_METRIC.npy', d)
    preview = color_preview(d, valid)
    cv2.putText(preview, 'UNRECTIFIED - INVALID AS METRIC DEPTH', (20,45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)
    cv2.imwrite(str(path / 'raw_UNRECTIFIED_disparity_preview.png'), preview)
    dbg.show('uncalibrated raw disparity (qualitative only)', preview,
             f'validation/{path.name}_raw_disparity.png')


def sgbm_depth(l, r, rig, args, path, dbg):
    scale = args.sgbm_scale
    if not (0 < scale <= 1):
        raise ValueError('Use --sgbm-scale in (0, 1].')
    if args.num_disparities <= 0 or args.num_disparities % 16:
        raise ValueError('--num-disparities must be positive and divisible by 16')
    if not (0 < args.min_depth < args.max_depth):
        raise ValueError('Depth limits must satisfy 0 < --min-depth < --max-depth.')
    if args.block_size < 3 or args.block_size % 2 != 1:
        raise ValueError('--block-size must be an odd integer >= 3')
    if scale != 1:
        a = cv2.resize(l, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        b = cv2.resize(r, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        a, b = l, r
    sx, sy = a.shape[1] / l.shape[1], a.shape[0] / l.shape[0]
    baseline = abs(float(-rig['P2'][0, 3] / rig['P2'][0, 0]))
    disparity_offset = sx * (rig['P1'][0, 2] - rig['P2'][0, 2])
    disparity_far = disparity_offset + sx * rig['P1'][0, 0] * baseline / args.max_depth
    disparity_near = disparity_offset + sx * rig['P1'][0, 0] * baseline / args.min_depth
    max_search = args.min_disparity + args.num_disparities - 1
    if disparity_far > max_search or disparity_near < args.min_disparity:
        raise ValueError(f'SGBM search [{args.min_disparity}, {max_search}] px does not overlap '
                         f'the requested depth range, which needs approximately '
                         f'{disparity_far:.1f} to {disparity_near:.1f} px at this scale.')
    if disparity_near > max_search:
        print(f'[WARN] The requested minimum depth needs approximately {disparity_near:.1f} px '
              f'disparity, beyond the SGBM maximum {max_search}; close objects will be missing.')
    g1, g2 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    block = args.block_size
    sgbm = cv2.StereoSGBM_create(minDisparity=args.min_disparity,
                                 numDisparities=args.num_disparities, blockSize=block,
                                 P1=8 * block * block, P2=32 * block * block,
                                 disp12MaxDiff=1, uniquenessRatio=10,
                                 speckleWindowSize=100, speckleRange=2,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disparity = sgbm.compute(g1, g2).astype(np.float32) / 16.0
    valid_disp = np.isfinite(disparity) & (disparity > max(0, args.min_disparity))
    # cv2.reprojectImageTo3D uses Q in the original rectified pixel coordinate system.
    # Map (u_scaled, v_scaled, d_scaled, 1) into the original coordinates.
    scaled_to_full = np.diag([1.0/sx, 1.0/sy, 1.0/sx, 1.0])
    Qscaled = rig['Q'] @ scaled_to_full
    xyz = cv2.reprojectImageTo3D(disparity, Qscaled)
    z = xyz[:, :, 2]
    valid = valid_disp & np.isfinite(xyz).all(axis=2) & (z >= args.min_depth) & (z <= args.max_depth)
    depth = np.where(valid, z, np.nan).astype(np.float32)
    disparity_out = np.where(valid, disparity, np.nan).astype(np.float32)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / 'disparity_px.npy', disparity_out)
    np.save(path / 'depth_m.npy', depth)
    cv2.imwrite(str(path / 'disparity_preview.png'), color_preview(disparity, valid))
    cv2.imwrite(str(path / 'depth_preview.png'), color_preview(z, valid, invert=True))
    save_ply(path / 'pointcloud_m.ply', xyz, a, valid, args.max_points)
    dbg.show('SGBM disparity (illustrative baseline)', color_preview(disparity, valid),
             f'validation/{path.name}_disparity.png')
    dbg.show('SGBM depth (meters; preview)', color_preview(z, valid, invert=True),
             f'validation/{path.name}_depth.png')
    print(f'[SGBM] {path.name}: valid={100 * np.mean(valid):.1f}% at scale={scale}; '
          f'depth median={float(np.median(z[valid])) if valid.any() else float("nan"):.3f} m')
    return {'scale': scale, 'valid_coverage_percent': float(100 * np.mean(valid)),
            'valid_depth_median_m': float(np.median(z[valid])) if valid.any() else None,
            'num_disparities': args.num_disparities, 'min_disparity': args.min_disparity}


def rectify(args, dbg):
    rig = as_dict(args.rig)
    validate_rectification(rig)
    maps = make_maps(rig)
    images = camera_pairs(args.images)
    if args.max_pairs is not None:
        images = images[:args.max_pairs]
    (args.out / 'rectified').mkdir(parents=True, exist_ok=True)
    report = {'rig_file': str(args.rig), 'pairs': {}}
    for i, (name, lp, rp) in enumerate(images, 1):
        l, r = read_pair(lp, rp, rig['size'])
        ra, rb = rectify_pair(l, r, rig, maps)
        cv2.imwrite(str(args.out / 'rectified' / f'{name}_left.png'), ra)
        cv2.imwrite(str(args.out / 'rectified' / f'{name}_right.png'), rb)
        if args.save_undistorted:
            (args.out / 'undistorted').mkdir(parents=True, exist_ok=True)
            # These images are corrected for lens distortion only; NOT epipolar-rectified.
            und_l = cv2.undistort(l, rig['K1'], rig['D1'])
            und_r = cv2.undistort(r, rig['K2'], rig['D2'])
            cv2.imwrite(str(args.out / 'undistorted' / f'{name}_left.png'), und_l)
            cv2.imwrite(str(args.out / 'undistorted' / f'{name}_right.png'), und_r)
            if dbg.enabled:
                dbg.show('undistorted-only (not rectified)', side_by_side(und_l, und_r, True),
                         f'undistorted/{name}.jpg' if i <= 6 else None)
        print(f'[rectify] {i}/{len(images)} {name}')
        if dbg.enabled:
            dbg.show('original stereo (lines)', side_by_side(l, r, True),
                     f'original/{name}.jpg' if i <= 6 else None)
            dbg.show('rectified stereo (lines)', side_by_side(ra, rb, True),
                     f'rectified/{name}.jpg' if i <= 6 else None)
        if args.compare_raw:
            raw_sgbm_preview(l, r, args, args.out / 'raw_sgbm' / name, dbg)
        report['pairs'][name] = (sgbm_depth(ra, rb, rig, args, args.out / 'sgbm' / name, dbg)
                                 if args.sgbm else {'rectified': True})
    # Copy calibration information alongside rectified outputs for consumers.
    output_npz(args.out / 'geometry.npz', **rig)
    (args.out / 'foundationstereo_intrinsics.txt').write_text(
        ' '.join(f'{v:.10f}' for v in rig['P1'][:3, :3].reshape(-1)) +
        f'\n{np.linalg.norm(rig["T"]):.10f}\n')
    # Copy model-agnostic geometry for downstream depth, matching or SLAM pipelines.
    if (args.rig.parent / 'camera_geometry.json').exists():
        (args.out / 'camera_geometry.json').write_text((args.rig.parent / 'camera_geometry.json').read_text())
    save_json(args.out / 'rectification_report.json', report)
    print(f'[done] Rectified PNG pairs under {args.out / "rectified"}')


def validate_tags(args, dbg):
    rig = as_dict(args.rig)
    d, params = april_dictionary(args.tag_family)
    reports = []
    for name, lp, rp in camera_pairs(args.images):
        l, r = read_pair(lp, rp, rig['size'])
        dl, dr = detect_tags(l, d, params), detect_tags(r, d, params)
        for tagid in sorted(dl.keys() & dr.keys()):
            p1 = rectified_points(dl[tagid], rig['K1'], rig['D1'], rig['R1'], rig['P1'])
            p2 = rectified_points(dr[tagid], rig['K2'], rig['D2'], rig['R2'], rig['P2'])
            epi = np.abs(p1[:, 1] - p2[:, 1])
            X4 = cv2.triangulatePoints(rig['P1'], rig['P2'], p1.T, p2.T)
            X = (X4[:3] / X4[3:4]).T
            # All four edges share the known physical tag length.
            lengths = np.linalg.norm(X - np.roll(X, 1, axis=0), axis=1)
            good = np.isfinite(lengths).all() and (X[:, 2] > 0).all()
            median_length = float(np.median(lengths)) if good else None
            raw_y_offset = np.abs(dl[tagid][:, 1] - dr[tagid][:, 1])
            reports.append({'pair': name, 'tag_id': tagid,
                            'raw_vertical_coordinate_offset_px': float(np.median(raw_y_offset)),
                            'epi_median_px': float(np.median(epi)),
                            'epi_p95_px': float(np.percentile(epi, 95)),
                            'tag_edge_length_m': median_length,
                            'tag_edge_relative_error_percent': float(100 * (median_length - args.tag_size)
                                                                    / args.tag_size) if good else None,
                            'median_tag_depth_m': float(np.median(X[:, 2])) if good else None})
    if not reports:
        print('[WARN] No common AprilTags to validate (check --tag-family, images, visibility).')
        return
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'tag_validation.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    median_epi = float(np.median([x['epi_median_px'] for x in reports]))
    size_errors = [abs(x['tag_edge_relative_error_percent']) for x in reports
                   if x['tag_edge_relative_error_percent'] is not None]
    summary = {'num_tag_observations': len(reports), 'median_vertical_epipolar_error_px': median_epi,
               'median_raw_vertical_offset_px': float(np.median(
                   [x['raw_vertical_coordinate_offset_px'] for x in reports])),
               'median_absolute_tag_size_error_percent': float(np.median(size_errors)) if size_errors else None,
               'note': 'If the same tags/images estimated R,T, these errors are in-sample and not independent.'}
    save_json(args.out / 'tag_validation_summary.json', summary)
    print('[validation] ' + json.dumps(summary, indent=2))
    if dbg.enabled:
        _, lp, rp = camera_pairs(args.images)[0]
        l, r = read_pair(lp, rp, rig['size'])
        rl, rr = rectify_pair(l, r, rig)
        dbg.show('AprilTag validation - rectified', side_by_side(rl, rr, True), 'tag_validation.jpg')



def depth_from_disparity(args, dbg):
    """Convert any model's REAL pixel-disparity array to metric depth and a point cloud.

    Input disparity is in pixels at its own array resolution (not normalized/inverse-depth).
    The calibrated rectified LEFT image must correspond to the same frame.
    """
    rig = as_dict(args.rig)
    validate_rectification(rig)
    if args.disparity.suffix.lower() == '.npz':
        with np.load(args.disparity) as f:
            if args.disparity_key not in f:
                raise KeyError(f'No key {args.disparity_key!r} in {args.disparity}; keys={list(f.files)}')
            disparity = f[args.disparity_key].astype(np.float32)
    else:
        disparity = np.load(args.disparity).astype(np.float32)
    disparity = np.squeeze(disparity)
    if disparity.ndim != 2:
        raise ValueError(f'Expected HxW disparity array, got {disparity.shape}')
    left = cv2.imread(str(args.rectified_left), cv2.IMREAD_COLOR)
    if left is None:
        raise ValueError(f'Cannot read rectified left image {args.rectified_left}')
    if (left.shape[1], left.shape[0]) != tuple(int(x) for x in rig['size']):
        raise ValueError('Rectified left image must be full calibrated resolution; use original rectified PNG.')
    h, w = disparity.shape
    sx, sy = w / left.shape[1], h / left.shape[0]
    image = cv2.resize(left, (w, h), interpolation=cv2.INTER_AREA) if (w, h) != (left.shape[1], left.shape[0]) else left
    if args.disparity_units == 'full-resolution':
        # Model may resize input internally then upsample disparity without adjusting its pixel units.
        disparity = disparity * sx
    elif args.disparity_units != 'array-resolution':
        raise ValueError(f'Unknown disparity units: {args.disparity_units}')
    transform = np.diag([1.0 / sx, 1.0 / sy, 1.0 / sx, 1.0])
    Qscaled = rig['Q'] @ transform
    xyz = cv2.reprojectImageTo3D(disparity, Qscaled)
    z = xyz[:, :, 2]
    valid = (np.isfinite(disparity) & (disparity > args.min_disparity) &
             np.isfinite(xyz).all(axis=2) & (z >= args.min_depth) & (z <= args.max_depth))
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / 'depth_m.npy', np.where(valid, z, np.nan).astype(np.float32))
    np.save(args.out / 'disparity_px.npy', np.where(valid, disparity, np.nan).astype(np.float32))
    cv2.imwrite(str(args.out / 'depth_preview.png'), color_preview(z, valid, invert=True))
    cv2.imwrite(str(args.out / 'disparity_preview.png'), color_preview(disparity, valid))
    save_ply(args.out / 'pointcloud_m.ply', xyz, image, valid, args.max_points)
    save_json(args.out / 'depth_report.json', {
        'disparity_array_width': w, 'disparity_array_height': h,
        'input_disparity_units': args.disparity_units,
        'depth_unit': 'meters', 'valid_percent': float(100 * valid.mean()),
        'median_depth_m': float(np.median(z[valid])) if valid.any() else None,
        'min_depth_m': args.min_depth, 'max_depth_m': args.max_depth,
        'baseline_m': float(np.linalg.norm(rig['T']))})
    dbg.show('model disparity', color_preview(disparity, valid), 'model_disparity.png')
    dbg.show('model depth', color_preview(z, valid, invert=True), 'model_depth.png')
    print(f'[depth-from-disparity] Valid coverage {100 * valid.mean():.2f}% -> {args.out}')


def parse_args():
    parser = argparse.ArgumentParser(description='Repeated stereo calibration, rectification, and depth tests')
    sub = parser.add_subparsers(dest='cmd', required=True)
    def common(subp):
        subp.add_argument('--debug', action=argparse.BooleanOptionalAction, default=DEBUG,
                          help='Use OpenCV windows and save previews (default from DEBUG global).')
        subp.add_argument('--debug-dir', type=Path, default=Path('output/debug_previews'))
    def board(subp):
        subp.add_argument('--images', type=Path, required=True)
        subp.add_argument('--cols', type=int, required=True, help='checkerboard INNER corners across width')
        subp.add_argument('--rows', type=int, required=True, help='checkerboard INNER corners across height')
        subp.add_argument('--square-size', type=float, required=True, help='checkerboard square edge in METERS')
    def tags(subp):
        subp.add_argument('--images', type=Path, required=True)
        subp.add_argument('--tag-size', type=float, required=True, help='physical detected AprilTag edge length in METERS')
        subp.add_argument('--tag-family', type=str, default='DICT_APRILTAG_36h11')
    p = sub.add_parser('intrinsics', help='Calibrate K1, D1, K2, D2 once')
    board(p); p.add_argument('--out', type=Path, required=True); common(p)
    p = sub.add_parser('rig-board', help='Calibrate stereo R, T from synchronized checkerboard pairs')
    board(p); p.add_argument('--intrinsics', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--alpha', type=float, default=0, help='0 crop invalid borders; 1 retain all pixels')
    common(p)
    p = sub.add_parser('rig-tags', help='Re-estimate R, T from one or more shared AprilTags after rig movement')
    tags(p); p.add_argument('--intrinsics', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--alpha', type=float, default=0)
    p.add_argument('--max-rotation-deg', type=float, default=7.0)
    p.add_argument('--max-translation-m', type=float, default=0.08)
    common(p)
    p = sub.add_parser('rectify', help='Rectify all paired PNGs, optionally generate disparity/depth/PLY')
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--rig', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-pairs', type=int)
    p.add_argument('--sgbm', action='store_true', help='Run OpenCV SGBM reference baseline (slow at full HD+)')
    p.add_argument('--compare-raw', action='store_true', help='Also visualize SGBM on raw unrectified images (not valid metric depth)')
    p.add_argument('--save-undistorted', action='store_true', help='Also export distortion-corrected-only images for monocular processing')
    p.add_argument('--sgbm-scale', type=float, default=0.5)
    p.add_argument('--num-disparities', type=int, default=256)
    p.add_argument('--min-disparity', type=int, default=0)
    p.add_argument('--block-size', type=int, default=5)
    p.add_argument('--min-depth', type=float, default=0.1)
    p.add_argument('--max-depth', type=float, default=10.0)
    p.add_argument('--max-points', type=int, default=250_000)
    common(p)
    p = sub.add_parser('validate-tags', help='Independently check epipolar alignment and metric tag size')
    tags(p); p.add_argument('--rig', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    common(p)
    p = sub.add_parser('depth-from-disparity', help='Convert model pixel disparities into metric depth / PLY')
    p.add_argument('--disparity', type=Path, required=True, help='npy or npz HxW pixel-disparity array')
    p.add_argument('--disparity-key', type=str, default='disparity', help='npz array name')
    p.add_argument('--disparity-units', choices=['array-resolution', 'full-resolution'],
                   default='array-resolution',
                   help='Pixel disparity units; full-resolution rescales when output is downsampled')
    p.add_argument('--rectified-left', type=Path, required=True)
    p.add_argument('--rig', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--min-disparity', type=float, default=0.0)
    p.add_argument('--min-depth', type=float, default=0.1)
    p.add_argument('--max-depth', type=float, default=10.0)
    p.add_argument('--max-points', type=int, default=250_000)
    common(p)
    return parser.parse_args()


def main():
    args = parse_args()
    dbg = Debugger(args.debug, args.debug_dir)
    try:
        if args.cmd == 'intrinsics': calibrate_intrinsics(args, dbg)
        elif args.cmd == 'rig-board': calibrate_rig_board(args, dbg)
        elif args.cmd == 'rig-tags': calibrate_rig_tags(args, dbg)
        elif args.cmd == 'rectify': rectify(args, dbg)
        elif args.cmd == 'validate-tags': validate_tags(args, dbg)
        elif args.cmd == 'depth-from-disparity': depth_from_disparity(args, dbg)
    finally:
        dbg.close()


if __name__ == '__main__':
    main()
