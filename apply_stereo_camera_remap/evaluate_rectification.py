"""Rectification 품질 평가 모듈.

Dot grid 이미지에 remap 적용 후, ideal grid 대비 오차를 측정·시각화한다.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_maps(map_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """cam_map_x.tiff / cam_map_y.tiff 로드."""
    mx = map_dir / "cam_map_x.tiff"
    my = map_dir / "cam_map_y.tiff"
    if not mx.exists() or not my.exists():
        raise FileNotFoundError(f"cam_map_x.tiff / cam_map_y.tiff not found in {map_dir}")
    map_x = tiff.imread(str(mx)).astype(np.float32, copy=False)
    map_y = tiff.imread(str(my)).astype(np.float32, copy=False)
    return map_x, map_y


def _read_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        return tiff.imread(str(path))
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Failed to read image: {path}")
    return img


# ---------------------------------------------------------------------------
# Dot detection
# ---------------------------------------------------------------------------

def _detect_dots(gray: np.ndarray, dot_diameter_px: Optional[float] = None) -> np.ndarray:
    """SimpleBlobDetector 로 dot centroid 추출. (N, 2) — (x, y) 순서."""
    if gray.dtype != np.uint8:
        # 16bit 등일 경우 8bit 로 변환
        mn, mx = gray.min(), gray.max()
        if mx > mn:
            gray = ((gray - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            gray = np.zeros_like(gray, dtype=np.uint8)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0  # dark dots on bright background (일반적)

    params.filterByArea = True
    if dot_diameter_px is not None:
        r = dot_diameter_px / 2.0
        params.minArea = max(1.0, np.pi * (r * 0.5) ** 2)
        params.maxArea = np.pi * (r * 2.0) ** 2
    else:
        params.minArea = 10
        params.maxArea = 100000

    params.filterByCircularity = True
    params.minCircularity = 0.90

    params.filterByConvexity = True
    params.minConvexity = 0.90

    params.filterByInertia = True
    params.minInertiaRatio = 0.90

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)

    if len(keypoints) == 0:
        # dark-on-bright 실패 시 bright-on-dark 시도
        params.blobColor = 255
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)

    pts = np.array([kp.pt for kp in keypoints], dtype=np.float64)  # (N, 2) x,y
    return pts


def _filter_outlier_dots(pts: np.ndarray, tolerance: float = 2.0) -> np.ndarray:
    """인접 dot 거리가 median 대비 너무 멀거나 가까운 dot 제거.

    tolerance: median * (1/tolerance) ~ median * tolerance 범위 밖이면 제거
    """
    from scipy.spatial import KDTree
    if len(pts) < 4:
        return pts
    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=2)
    nn_dists = dists[:, 1]
    med = np.median(nn_dists)
    lo = med / tolerance
    hi = med * tolerance
    mask = (nn_dists >= lo) & (nn_dists <= hi)
    removed = int(np.sum(~mask))
    if removed > 0:
        print(f"[filter] {removed} outlier dots removed "
              f"(nn_dist outside [{lo:.1f}, {hi:.1f}] px, median={med:.1f})")
    return pts[mask]


# ---------------------------------------------------------------------------
# Ideal grid fitting
# ---------------------------------------------------------------------------

def _estimate_pixel_pitch(pts: np.ndarray) -> Tuple[float, np.ndarray]:
    """검출점에서 수평/수직 pitch를 직접 추정.

    인접점 벡터를 수평(|dx|>|dy|) / 수직(|dy|>|dx|) 으로 분류하여
    각 축의 평균 벡터를 기저로 사용한다.

    Returns:
        pitch_px: 평균 pitch (두 축 평균)
        basis: (2, 2) — basis[0]= 수평 축 벡터, basis[1]= 수직 축 벡터
    """
    from scipy.spatial import KDTree
    tree = KDTree(pts)
    dists, idx = tree.query(pts, k=5)
    nn_dists = dists[:, 1]
    median_d = float(np.median(nn_dists))

    # 인접 벡터 수집 — 길이 필터: [0.7, 1.3] * median (대각선 ~1.41x 제외)
    lo_d = median_d * 0.7
    hi_d = median_d * 1.3
    vectors = []
    for i in range(len(pts)):
        for j in range(1, 5):
            d = dists[i, j]
            if lo_d <= d <= hi_d:
                v = pts[idx[i, j]] - pts[i]
                vectors.append(v)
    vectors = np.array(vectors)

    # 수평 vs 수직 분류: |dx| > |dy| → 수평, else → 수직
    abs_dx = np.abs(vectors[:, 0])
    abs_dy = np.abs(vectors[:, 1])
    h_mask = abs_dx > abs_dy
    v_mask = ~h_mask

    # 방향 통일: 수평은 dx>0, 수직은 dy>0 으로
    h_vecs = vectors[h_mask].copy()
    h_vecs[h_vecs[:, 0] < 0] *= -1
    v_vecs = vectors[v_mask].copy()
    v_vecs[v_vecs[:, 1] < 0] *= -1

    basis = np.zeros((2, 2))
    basis[0] = h_vecs.mean(axis=0)  # 수평 기저
    basis[1] = v_vecs.mean(axis=0)  # 수직 기저

    pitch_px = float(np.mean(np.linalg.norm(basis, axis=1)))
    print(f"[pitch] median_nn={median_d:.1f}, "
          f"h_vectors={len(h_vecs)}, v_vectors={len(v_vecs)}, "
          f"|basis_h|={np.linalg.norm(basis[0]):.1f}, |basis_v|={np.linalg.norm(basis[1]):.1f}")
    return pitch_px, basis


def _assign_grid_indices(pts: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """기저 벡터 기반으로 각 점에 (col, row) 정수 grid index 부여.

    basis 의 역행렬을 이용해서 연속 좌표를 구한 후 round.
    """
    center = pts.mean(axis=0)
    rel = pts - center  # (N, 2)

    # basis 역행렬: rel = indices @ basis  =>  indices = rel @ inv(basis^T)
    B = basis.T  # (2, 2) — 열이 기저벡터
    B_inv = np.linalg.inv(B)
    continuous = rel @ B_inv  # (N, 2) — 연속 grid 좌표
    indices = np.round(continuous).astype(int)
    return indices


def _check_duplicate_indices(indices: np.ndarray) -> int:
    """중복 grid index 개수 반환."""
    _, counts = np.unique(indices, axis=0, return_counts=True)
    return int(np.sum(counts > 1))


def _plot_debug_detection(gray: np.ndarray, pts: np.ndarray, save_path: Path, title: str = ""):
    """디버그: 검출된 dot 위치만 이미지 위에 표시."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(gray, cmap="gray", interpolation="nearest")
    ax.scatter(pts[:, 0], pts[:, 1], s=8, c="lime", marker="o", linewidths=0.5)
    for i, (x, y) in enumerate(pts):
        ax.text(x + 3, y - 3, str(i), fontsize=3, color="yellow")
    ax.set_title(f"{title}  (N={len(pts)})", fontsize=10)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=200)
    plt.close(fig)


def _plot_debug_grid_index(gray: np.ndarray, pts: np.ndarray, indices: np.ndarray,
                           save_path: Path, title: str = ""):
    """디버그: grid index 부여 결과. 같은 row는 같은 색."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(gray, cmap="gray", interpolation="nearest")
    rows = indices[:, 1]
    scatter = ax.scatter(pts[:, 0], pts[:, 1], s=10, c=rows, cmap="tab20", linewidths=0.5)
    for i, (x, y) in enumerate(pts):
        ax.text(x + 3, y - 3, f"{indices[i, 0]},{indices[i, 1]}", fontsize=2.5, color="white")
    fig.colorbar(scatter, ax=ax, label="row index")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=200)
    plt.close(fig)


def _fit_affine_grid(pts: np.ndarray, grid_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Affine 변환으로 ideal grid 좌표 fitting.

    Returns:
        ideal_pts: (N, 2) affine-fitted ideal 좌표
        A: (2, 3) affine matrix
    """
    N = len(pts)
    G = np.column_stack([grid_idx, np.ones(N)])  # (N, 3)
    A_T, _, _, _ = np.linalg.lstsq(G, pts, rcond=None)  # (3, 2)
    ideal_pts = G @ A_T  # (N, 2)
    A = A_T.T  # (2, 3)
    return ideal_pts, A


# ---------------------------------------------------------------------------
# Error computation
# ---------------------------------------------------------------------------

def _compute_errors(detected: np.ndarray, ideal: np.ndarray, mm_per_px: Optional[float]):
    """오차 벡터 및 통계 계산 (distortion % 포함)."""
    err_vec = detected - ideal  # (N, 2)
    err_mag = np.linalg.norm(err_vec, axis=1)  # (N,)

    # --- Distortion % ---
    # center = ideal grid 의 무게중심
    center = ideal.mean(axis=0)
    r_ideal = np.linalg.norm(ideal - center, axis=1)
    r_actual = np.linalg.norm(detected - center, axis=1)
    r_max = r_ideal.max()

    # 중심 근처는 분모가 작아 distortion % 가 과장되므로,
    # 외곽 영역(r_ideal > 50% of max radius)만 사용하여 리포트
    valid = r_ideal > (r_max * 0.5)
    dist_pct = np.zeros(len(detected))
    dist_pct[valid] = (r_actual[valid] - r_ideal[valid]) / r_ideal[valid] * 100.0

    # 최외곽 dot (상위 10%) 에서의 distortion — 표준 리포트 값
    outer = r_ideal > (r_max * 0.9)
    dist_pct_outer = np.zeros(len(detected))
    dist_pct_outer[outer] = (r_actual[outer] - r_ideal[outer]) / r_ideal[outer] * 100.0

    stats = {
        "n_dots": len(detected),
        "error_px": {
            "mean": float(np.mean(err_mag)),
            "max": float(np.max(err_mag)),
            "rms": float(np.sqrt(np.mean(err_mag ** 2))),
            "std": float(np.std(err_mag)),
        },
        "distortion_pct": {
            "note": "outer 50%+ of field",
            "mean": float(np.mean(dist_pct[valid])),
            "max": float(np.max(dist_pct[valid])),
            "min": float(np.min(dist_pct[valid])),
            "max_abs": float(np.max(np.abs(dist_pct[valid]))),
        },
        "distortion_pct_edge": {
            "note": "outer 90%+ of field (standard report value)",
            "mean": float(np.mean(dist_pct_outer[outer])) if outer.any() else 0.0,
            "max_abs": float(np.max(np.abs(dist_pct_outer[outer]))) if outer.any() else 0.0,
        },
    }
    if mm_per_px is not None:
        err_mag_mm = err_mag * mm_per_px
        stats["error_mm"] = {
            "mean": float(np.mean(err_mag_mm)),
            "max": float(np.max(err_mag_mm)),
            "rms": float(np.sqrt(np.mean(err_mag_mm ** 2))),
            "std": float(np.std(err_mag_mm)),
        }
        stats["mm_per_px"] = mm_per_px
    return err_vec, err_mag, dist_pct, stats


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _plot_overlay(
    img_gray: np.ndarray,
    detected: np.ndarray,
    ideal: np.ndarray,
    err_vec: np.ndarray,
    err_mag: np.ndarray,
    arrow_scale: float,
    save_path: Path,
    title: str = "",
):
    """Rectified image 위에 ideal(red) + detected(blue) + error arrow overlay."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(img_gray, cmap="gray", interpolation="nearest")

    # ideal dots
    ax.scatter(ideal[:, 0], ideal[:, 1], s=12, c="red", marker="+", linewidths=0.6, label="ideal")
    # detected dots
    ax.scatter(detected[:, 0], detected[:, 1], s=12, c="cyan", marker="x", linewidths=0.6, label="detected")

    # error arrows (확대)
    ax.quiver(
        ideal[:, 0], ideal[:, 1],
        err_vec[:, 0], err_vec[:, 1],
        err_mag,
        angles="xy", scale_units="xy", scale=1.0 / arrow_scale,
        cmap="hot", width=0.002,
    )

    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=200)
    plt.close(fig)


def _plot_quiver(
    detected: np.ndarray,
    err_vec: np.ndarray,
    err_mag: np.ndarray,
    arrow_scale: float,
    save_path: Path,
    title: str = "",
):
    """Error vector field (quiver plot) — 배경 없이 오차만."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    q = ax.quiver(
        detected[:, 0], detected[:, 1],
        err_vec[:, 0], err_vec[:, 1],
        err_mag,
        angles="xy", scale_units="xy", scale=1.0 / arrow_scale,
        cmap="hot", width=0.003,
    )
    fig.colorbar(q, ax=ax, label="error [px]")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=200)
    plt.close(fig)


def _plot_error_hist(err_mag: np.ndarray, save_path: Path, title: str = ""):
    """Error magnitude histogram."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.hist(err_mag, bins=40, edgecolor="black", alpha=0.75)
    ax.axvline(np.mean(err_mag), color="red", ls="--", label=f"mean={np.mean(err_mag):.3f}")
    ax.axvline(np.max(err_mag), color="orange", ls="--", label=f"max={np.max(err_mag):.3f}")
    ax.set_xlabel("error [px]")
    ax.set_ylabel("count")
    ax.set_title(title, fontsize=11)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def evaluate_rectification(
    image_path: str,
    map_dir: str,
    output_dir: str,
    dot_diameter_mm: float = 0.5,
    dot_pitch_mm: float = 1.0,
    pixel_pitch_mm: Optional[float] = None,
    arrow_scale: float = 10.0,
    label: str = "result",
    filter_outlier: bool = False,
    outlier_tolerance: float = 2.0,
) -> dict:
    """Rectification 적용 후 dot grid 정밀도 평가.

    Parameters
    ----------
    image_path : 원본 dot grid 이미지 경로
    map_dir : cam_map_x.tiff, cam_map_y.tiff 가 있는 폴더
    output_dir : 결과 저장 폴더
    dot_diameter_mm : dot 직경 (mm)
    dot_pitch_mm : dot 간격 (mm), center-to-center
    pixel_pitch_mm : mm/pixel. None이면 dot 간격에서 자동 추정
    arrow_scale : error arrow 확대 배율
    label : 결과 파일 prefix
    filter_outlier : 인접 거리 기반 outlier dot 제거 여부
    outlier_tolerance : outlier 판정 배수 (median 대비)

    Returns
    -------
    dict : 오차 통계
    """
    img_path = Path(image_path)
    map_dir_p = Path(map_dir)

    # 날짜시간 서브폴더: output_dir/{label}_YYYYMMDD_HHMMSS/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir) / f"{label}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load & remap
    map_x, map_y = _load_maps(map_dir_p)
    src = _read_image(img_path)
    rectified = cv2.remap(src, map_x, map_y,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # grayscale 변환
    if rectified.ndim == 3:
        gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    else:
        gray = rectified.copy()

    # 저장: rectified image
    cv2.imwrite(str(out_dir / f"{label}_rectified.png"), rectified)

    # 2) Detect dots
    # pixel pitch 결정
    if pixel_pitch_mm is not None:
        mm_per_px = pixel_pitch_mm
        dot_diameter_px = dot_diameter_mm / mm_per_px
    else:
        dot_diameter_px = None
        mm_per_px = None

    detected = _detect_dots(gray, dot_diameter_px)
    print(f"[{label}] {len(detected)} dots detected (before filter)")

    if filter_outlier:
        detected = _filter_outlier_dots(detected, tolerance=outlier_tolerance)
    if len(detected) < 9:
        raise RuntimeError(f"Dot 검출 부족: {len(detected)}개. 이미지/파라미터를 확인하세요.")

    print(f"[{label}] {len(detected)} dots after filtering")

    # DEBUG: 검출 결과 시각화
    _plot_debug_detection(gray, detected,
                          save_path=out_dir / f"{label}_debug_detection.png",
                          title=f"[{label}] Detected Dots (filtered)")

    # 3) Pixel pitch 자동 추정 (user 입력 없을 때)
    pitch_px, basis = _estimate_pixel_pitch(detected)
    if mm_per_px is None:
        mm_per_px = dot_pitch_mm / pitch_px
        print(f"[{label}] pixel pitch auto-estimated: {mm_per_px:.6f} mm/px  "
              f"(dot pitch = {pitch_px:.1f} px)")
    print(f"[{label}] basis vectors: {basis[0]} , {basis[1]}")

    dot_diameter_px_est = dot_diameter_mm / mm_per_px

    # 4) Grid index 부여 & affine fitting (기저 벡터 기반)
    grid_idx = _assign_grid_indices(detected, basis)

    n_dup = _check_duplicate_indices(grid_idx)
    if n_dup > 0:
        print(f"[{label}] WARNING: {n_dup} dots have duplicate grid indices!")

    # DEBUG: grid index 시각화
    _plot_debug_grid_index(gray, detected, grid_idx,
                           save_path=out_dir / f"{label}_debug_grid_index.png",
                           title=f"[{label}] Grid Index Assignment (dup={n_dup})")

    ideal, affine_mat = _fit_affine_grid(detected, grid_idx)

    # 5) Error 계산
    err_vec, err_mag, dist_pct, stats = _compute_errors(detected, ideal, mm_per_px)
    stats["label"] = label
    stats["image"] = str(img_path.name)
    stats["map_dir"] = str(map_dir_p)
    stats["pixel_pitch_px"] = float(pitch_px)

    # 6) 시각화
    _plot_overlay(
        gray, detected, ideal, err_vec, err_mag, arrow_scale,
        save_path=out_dir / f"{label}_overlay.png",
        title=f"[{label}] Rectification Error Overlay  |  "
              f"RMS={stats['error_px']['rms']:.3f} px, Max={stats['error_px']['max']:.3f} px, "
              f"Dist(edge)={stats['distortion_pct_edge']['max_abs']:.4f}%",
    )
    _plot_quiver(
        detected, err_vec, err_mag, arrow_scale,
        save_path=out_dir / f"{label}_quiver.png",
        title=f"[{label}] Error Vector Field  |  "
              f"RMS={stats['error_px']['rms']:.3f} px, Dist={stats['distortion_pct']['max_abs']:.4f}%",
    )
    _plot_error_hist(
        err_mag,
        save_path=out_dir / f"{label}_error_hist.png",
        title=f"[{label}] Error Distribution",
    )

    # 7) 통계 저장
    with open(out_dir / f"{label}_stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"[{label}] RMS error: {stats['error_px']['rms']:.4f} px "
          f"({stats.get('error_mm', {}).get('rms', 0):.4f} mm)")
    print(f"[{label}] Max error: {stats['error_px']['max']:.4f} px")
    print(f"[{label}] Distortion (outer>50%): max_abs={stats['distortion_pct']['max_abs']:.4f}%")
    print(f"[{label}] Distortion (edge>90%): max_abs={stats['distortion_pct_edge']['max_abs']:.4f}%")
    print(f"[{label}] Results saved to {out_dir}")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Rectification 품질 평가: dot grid 기반 오차 분석")
    ap.add_argument("image", help="dot grid 이미지 경로")
    ap.add_argument("map_dir", help="cam_map_x.tiff, cam_map_y.tiff 폴더")
    ap.add_argument("-o", "--output", default="output", help="결과 저장 폴더")
    ap.add_argument("--label", default="result", help="결과 파일 prefix")
    ap.add_argument("--dot_diameter", type=float, default=0.5, help="dot diameter [mm]")
    ap.add_argument("--dot_pitch", type=float, default=1.0, help="dot pitch [mm]")
    ap.add_argument("--pixel_pitch", type=float, default=None,
                     help="mm/pixel (미입력시 자동 추정)")
    ap.add_argument("--arrow_scale", type=float, default=10.0,
                     help="error arrow 확대 배율")
    ap.add_argument("--filter_outlier", action="store_true", default=False,
                     help="인접 거리 기반 outlier dot 제거 활성화")
    ap.add_argument("--outlier_tolerance", type=float, default=2.0,
                     help="outlier 판정 배수 (median 대비)")
    args = ap.parse_args()

    evaluate_rectification(
        image_path=args.image,
        map_dir=args.map_dir,
        output_dir=args.output,
        dot_diameter_mm=args.dot_diameter,
        dot_pitch_mm=args.dot_pitch,
        pixel_pitch_mm=args.pixel_pitch,
        arrow_scale=args.arrow_scale,
        label=args.label,
        filter_outlier=args.filter_outlier,
        outlier_tolerance=args.outlier_tolerance,
    )


if __name__ == "__main__":
    main()
