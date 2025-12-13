from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np
from ..utils.config import DebugConfig
DEBUG = DebugConfig()


def save_grid_report(
    image_path: Path,
    gray: np.ndarray,
    pts_xy: np.ndarray,
    grid_uv_to_idx: Dict[Tuple[int, int], int],
    Tc: Optional[np.ndarray],
    out_path: Path,
    triplet: Optional[Tuple[int, int, int]] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.stem
    if Tc is not None and len(pts_xy) > 0:
        P = np.hstack([pts_xy, np.ones((len(pts_xy), 1), np.float32)]) @ Tc.T
    else:
        P = None
    keys: List[Tuple[int, int]] = list(grid_uv_to_idx.keys())
    if keys:
        radii = [max(abs(u), abs(v)) for (u, v) in keys]
        max_radius = int(max(radii))
    else:
        max_radius = 0
    if P is not None and keys:
        errs = []
        for uv, idx in grid_uv_to_idx.items():
            uv_arr = np.array(uv, np.float32)
            e = float(np.linalg.norm(P[idx] - uv_arr))
            errs.append(e)
        mean_err = float(np.mean(errs)) if errs else 0.0
        std_err = float(np.std(errs)) if errs else 0.0
    else:
        mean_err = 0.0
        std_err = 0.0
    '''
    txt = out_dir / f"{stem}_grid_report.txt"
    with txt.open("w", encoding="utf-8") as f:
        f.write(f"image: {image_path}\n")
        f.write(f"num_assigned: {len(grid_uv_to_idx)}\n")
        f.write(f"max_radius: {max_radius}\n")
        f.write(f"mean_residual: {mean_err:.4f}\n")
        f.write(f"std_residual: {std_err:.4f}\n")
    '''
    if gray is not None and len(pts_xy) > 0 and grid_uv_to_idx:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Triplet 강조 (save_detection_overlay와 동일한 방식)
        if triplet is not None:
            c, p, m = triplet
            def draw_triplet(idx: int, color: Tuple[int, int, int], label: str) -> None:
                x, y = pts_xy[idx]
                cv2.circle(
                    vis,
                    (int(round(x)), int(round(y))),
                    DEBUG.point_radius + 2,  # 더 큰 원으로 강조
                    color,
                    DEBUG.point_thickness + 1,  # 더 두꺼운 선
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    vis,
                    label,
                    (int(round(x)) + 6, int(round(y)) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    DEBUG.label_font_scale + 0.2,  # 더 큰 폰트
                    color,
                    DEBUG.label_thickness + 1,  # 더 두꺼운 폰트
                    cv2.LINE_AA,
                )
            draw_triplet(p, DEBUG.triplet_iplus_color, "i+")
            draw_triplet(c, DEBUG.triplet_i_color, "i")
            draw_triplet(m, DEBUG.triplet_iminus_color, "i-")
        
        # Grid 점들 그리기
        for (u, v), idx in grid_uv_to_idx.items():
            x, y = pts_xy[idx]
            cv2.circle(vis, (int(round(x)), int(round(y))), DEBUG.grid_circle_radius, DEBUG.grid_color, 1, cv2.LINE_AA)
            cv2.putText(vis, f"({u},{v})", (int(round(x))+3, int(round(y))-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, DEBUG.grid_color, 1, cv2.LINE_AA)
            if P is not None:
                px, py = P[idx]
                rx, ry = (px - u), (py - v)
                tip = (int(round(x + rx*DEBUG.arrow_scale)), int(round(y + ry*DEBUG.arrow_scale)))
                err_norm = float(np.hypot(rx, ry))
                # 임계 초과 시 강조
                if err_norm > float(DEBUG.grid_error_thr):
                    cv2.arrowedLine(
                        vis,
                        (int(round(x)), int(round(y))),
                        tip,
                        DEBUG.grid_error_color,
                        max(1, int(DEBUG.grid_error_thickness)),
                        cv2.LINE_AA,
                        tipLength=0.3,
                    )
                    # 에러 값 라벨
                    cv2.putText(
                        vis,
                        f"{err_norm:.2f}",
                        (int(round(x))+6, int(round(y))+16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        DEBUG.grid_error_color,
                        1,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.arrowedLine(vis, (int(round(x)), int(round(y))), tip, (0, 0, 255), 1, cv2.LINE_AA, tipLength=0.3)
        cv2.imwrite(str(out_path), vis)



def save_detection_overlay(
    gray: np.ndarray,
    pts_xy: np.ndarray,
    triplet: Optional[Tuple[int, int, int]],
    out_path: Path,
    P_grid: Optional[np.ndarray] = None,
    nn24: Optional[np.ndarray] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if pts_xy is None:
        if gray is not None:
            cv2.imwrite(str(out_path), gray)
        return

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for (x, y) in pts_xy:
        cv2.circle(
            vis,
            (int(round(x)), int(round(y))),
            DEBUG.point_radius,
            DEBUG.point_color,
            DEBUG.point_thickness,
            lineType=cv2.LINE_AA,
        )
    if triplet is None:
        if gray is not None:
            cv2.imwrite(str(out_path), vis)
        return


    c, p, m = triplet
    def draw(idx: int, color: Tuple[int, int, int], label: str) -> None:
        x, y = pts_xy[idx]
        cv2.circle(
            vis,
            (int(round(x)), int(round(y))),
            DEBUG.point_radius,
            color,
            DEBUG.point_thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            label,
            (int(round(x)) + 6, int(round(y)) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            DEBUG.label_font_scale,
            color,
            DEBUG.label_thickness,
            cv2.LINE_AA,
        )
    draw(p, DEBUG.triplet_iplus_color, "i+")
    draw(c, DEBUG.triplet_i_color, "i")
    draw(m, DEBUG.triplet_iminus_color, "i-")
    if P_grid is not None and nn24 is not None and len(nn24) >= 3:
        targets = [
            np.array([-2.0, -2.0], np.float32),
            np.array([-2.0,  2.0], np.float32),
            np.array([ 2.0,  2.0], np.float32),
        ]
        def nearest_idx(tg: np.ndarray) -> int:
            cand = P_grid[nn24]
            d = np.linalg.norm(cand - tg[None, :], axis=1)
            return int(nn24[int(np.argmin(d))])
        try:
            i_tl = nearest_idx(targets[0])
            i_bl = nearest_idx(targets[1])
            i_br = nearest_idx(targets[2])
            p_tl = (int(round(pts_xy[i_tl][0])), int(round(pts_xy[i_tl][1])))
            p_bl = (int(round(pts_xy[i_bl][0])), int(round(pts_xy[i_bl][1])))
            p_br = (int(round(pts_xy[i_br][0])), int(round(pts_xy[i_br][1])))
            cv2.line(vis, p_tl, p_bl, (0, 0, 255), DEBUG.line_thickness, cv2.LINE_AA)
            cv2.line(vis, p_bl, p_br, (0, 0, 255), DEBUG.line_thickness, cv2.LINE_AA)
        except Exception:
            pass
    cv2.imwrite(str(out_path), vis)
    return


def save_reprojection_report(
    image_path: Path,
    gray: np.ndarray,
    image_points: np.ndarray,
    object_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    out_path: Path,
    save_txt: bool = False,
) -> None:
    """재투영 에러 리포트 생성 및 저장"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 재투영 계산
    proj, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)

    # 시각화 이미지 생성
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for p, q in zip(image_points, proj):
        p1 = (int(round(p[0])), int(round(p[1])))
        p2 = (int(round(q[0])), int(round(q[1])))
        cv2.circle(vis, p1, 2, DEBUG.reproj_actual_color, -1, cv2.LINE_AA)  # 실제 점
        cv2.circle(vis, p2, 2, DEBUG.reproj_projected_color, -1, cv2.LINE_AA)  # 재투영 점
        cv2.line(vis, p1, p2, DEBUG.reproj_error_color, 1, cv2.LINE_AA)    # 에러 벡터
    cv2.imwrite(str(out_path), vis)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # 에러 계산
        errs = image_points - proj
        err_norm = np.linalg.norm(errs, axis=1)
        xs = list(range(len(err_norm)))
        
        # 플롯 생성
        plt.figure(figsize=DEBUG.plot_figsize)
        plt.plot(xs, err_norm, '-o', markersize=3)
        plt.title(f'reprojection error: {image_path.stem}')
        plt.xlabel('Point Index')
        plt.ylabel('Error (pixels)')
        plt.grid(True, alpha=0.3)
        plot_path = out_path.parent / f"{out_path.stem}_plot.png"
        plt.savefig(plot_path, bbox_inches='tight', dpi=DEBUG.plot_dpi)
        plt.close()
        
        # 텍스트 리포트 저장
        if save_txt:
            txt_path = out_path.parent / f"{out_path.stem}.txt"
            with txt_path.open("w", encoding="utf-8") as f:
                f.write(f"image: {image_path}\n")
                f.write(f"num_points: {len(image_points)}\n")
                f.write(f"mean_err_px: {float(np.mean(err_norm)):.4f}\n")
                f.write(f"std_err_px: {float(np.std(err_norm)):.4f}\n")
                f.write(f"max_err_px: {float(np.max(err_norm)):.4f}\n")
                f.write(f"rms_err_px: {float(np.sqrt(np.mean(err_norm**2))):.4f}\n")
                
    except Exception as e:
        print(f"Error saving reprojection plot: {e}")


