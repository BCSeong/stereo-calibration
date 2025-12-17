from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np
from ..utils.config import DebugConfig
DEBUG = DebugConfig()

def _spiral_order(keys: Dict[Tuple[int, int], int]) -> List[Tuple[int, int]]:
    """(0,0)에서 시작해 위→오른쪽→아래→왼쪽으로 감아 올라가는 나선 순서 생성."""
    if not keys:
        return []
    key_set = set(keys)
    max_r = max(max(abs(u), abs(v)) for (u, v) in key_set)
    order: List[Tuple[int, int]] = []
    if (0, 0) in key_set:
        order.append((0, 0))
    for r in range(1, max_r + 1):
        x, y = -(r - 1), r  # 시작: 윗변의 좌측(예: r=1 → (0,1), r=2 → (-1,2))
        # 오른쪽으로 이동
        while x <= r:
            if (x, y) in key_set:
                order.append((x, y))
            x += 1
        x -= 1  # 마지막 valid 위치로 보정
        # 아래로 이동
        y -= 1
        while y >= -r:
            if (x, y) in key_set:
                order.append((x, y))
            y -= 1
        y += 1
        # 왼쪽으로 이동
        x -= 1
        while x >= -r:
            if (x, y) in key_set:
                order.append((x, y))
            x -= 1
        x += 1
        # 위로 이동
        y += 1
        while y <= r:
            if (x, y) in key_set:
                order.append((x, y))
            y += 1
    return order


def _draw_grid_mesh(vis: np.ndarray, pts_xy: np.ndarray, grid_uv_to_idx: Dict[Tuple[int, int], int]) -> None:
    """격자 이웃 간 mesh를 그려 점 누락/왜곡을 시각적으로 강조."""
    key_set = set(grid_uv_to_idx.keys())
    edges: List[Tuple[Tuple[int, int], Tuple[int, int], float]] = []
    lengths: List[float] = []
    for (u, v), idx in grid_uv_to_idx.items():
        for du, dv in ((1, 0), (0, 1)):  # 오른쪽, 아래쪽 이웃만 그려 중복 방지
            nb = (u + du, v + dv)
            if nb not in key_set:
                continue
            j = grid_uv_to_idx[nb]
            x0, y0 = pts_xy[idx]
            x1, y1 = pts_xy[j]
            length = float(np.hypot(x1 - x0, y1 - y0))
            edges.append(((int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), length))
            lengths.append(length)
    if not edges:
        return
    median_len = float(np.median(lengths)) if lengths else 0.0
    median_len = median_len if median_len > 0 else 1.0
    for (p0, p1, length) in edges:
        thick = max(1, int(round(min(6.0, length / median_len * 2.0))))  # 길이에 비례해 두께
        color = DEBUG.grid_color if length <= 1.2 * median_len else DEBUG.grid_error_color
        cv2.line(vis, p0, p1, color, thick, cv2.LINE_AA)


def save_grid_path_report(
    image_path: Path,
    gray: np.ndarray,
    pts_xy: np.ndarray,
    grid_uv_to_idx: Dict[Tuple[int, int], int],
    Tc: Optional[np.ndarray],
    out_path: Path,
    triplet: Optional[Tuple[int, int, int]] = None,
) -> None:
    '''
    2D grid 공간에서 나선형으로 (0,0) → (0,1) → (1,1) → (1,0) → (1,-1) → ...
    순서로 인접 dot 사이의 arrow를 그림. 길이가 길수록 두께/색상으로 강조.
    '''
    if gray is None or pts_xy is None or len(pts_xy) == 0 or not grid_uv_to_idx:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # u/v 방향 화살표용 anchor 설정
    anchor = (-2, -2)
    u_tip = (3, -2)
    v_tip = (-2, 3)

    # 점 및 ID 라벨
    for (u, v), idx in grid_uv_to_idx.items():
        x, y = pts_xy[idx]
        cv2.circle(vis, (int(round(x)), int(round(y))), DEBUG.grid_circle_radius, DEBUG.grid_color, 1, cv2.LINE_AA)
        cv2.putText(vis, f"({u},{v})", (int(round(x))+3, int(round(y))-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, DEBUG.grid_color, 1, cv2.LINE_AA)


    # 나선 경로 생성
    order_uv = _spiral_order(grid_uv_to_idx)
    order_idx = [grid_uv_to_idx[uv] for uv in order_uv if uv in grid_uv_to_idx]
    if len(order_idx) >= 2:
        lengths: List[float] = []
        segments: List[Tuple[Tuple[int, int], Tuple[int, int], float]] = []
        for i in range(len(order_idx) - 1):
            a = order_idx[i]
            b = order_idx[i + 1]
            # grid 인접(맨해튼 거리 1) 쌍만 연결하고, 그 외의 jump는 건너뜀
            du = abs(order_uv[i][0] - order_uv[i + 1][0])
            dv = abs(order_uv[i][1] - order_uv[i + 1][1])
            if (du + dv) != 1:
                continue
            x0, y0 = pts_xy[a]
            x1, y1 = pts_xy[b]
            length = float(np.hypot(x1 - x0, y1 - y0))
            lengths.append(length)
            segments.append(((int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), length))
        median_len = float(np.median(lengths)) if lengths else 1.0
        median_len = median_len if median_len > 0 else 1.0
        for (p0, p1, length) in segments:
            thick = max(1, int(round(min(8.0, 1.0 + length / median_len * 2.0))))
            color = DEBUG.grid_color if length <= 1.2 * median_len else DEBUG.grid_error_color
            cv2.arrowedLine(vis, p0, p1, color, thick, cv2.LINE_AA, tipLength=0.2)

    # u/v 화살표 그리기 (img 좌표계)
    if anchor in grid_uv_to_idx:
        p0 = pts_xy[grid_uv_to_idx[anchor]]
        if u_tip in grid_uv_to_idx:
            pU = pts_xy[grid_uv_to_idx[u_tip]]
            cv2.arrowedLine(vis, (int(round(p0[0])), int(round(p0[1]))), (int(round(pU[0])), int(round(pU[1]))),
                            (255, 0, 0), 4, cv2.LINE_AA, tipLength=0.2)  # blue: u
            cv2.putText(vis, "u", (int(round((p0[0]+pU[0])*0.5)), int(round((p0[1]+pU[1])*0.5))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        if v_tip in grid_uv_to_idx:
            pV = pts_xy[grid_uv_to_idx[v_tip]]
            cv2.arrowedLine(vis, (int(round(p0[0])), int(round(p0[1]))), (int(round(pV[0])), int(round(pV[1]))),
                            (0, 0, 255), 4, cv2.LINE_AA, tipLength=0.2)  # red: v
            cv2.putText(vis, "v", (int(round((p0[0]+pV[0])*0.5)), int(round((p0[1]+pV[1])*0.5))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), vis)

    # 나선 경로에서 실제로 그린 grid 인접(맨해튼 1) 구간의 길이 통계 반환
    dist_stats = {
        "mean": float(np.mean(lengths)) if lengths else 0.0,
        "std": float(np.std(lengths)) if lengths else 0.0,
        "max": float(np.max(lengths)) if lengths else 0.0,
        "min": float(np.min(lengths)) if lengths else 0.0,
        "count": len(lengths)
    }
    return dist_stats

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

        # 격자 mesh를 먼저 그려 점 누락/왜곡을 한눈에 확인
        _draw_grid_mesh(vis, pts_xy, grid_uv_to_idx)
        
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


def _remap_image(src_img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, interpolation: int = cv2.INTER_LINEAR, border_mode: int = cv2.BORDER_CONSTANT, border_value: float = 0.0) -> np.ndarray:
    """cv2.remap을 사용해 src_img를 재배치. 채널 수/비트심도 보존.
    
    Args:
        src_img: 원본 이미지
        map_x: X 방향 remap 맵
        map_y: Y 방향 remap 맵
        interpolation: 보간 방법 (기본값: cv2.INTER_LINEAR)
        border_mode: 경계 처리 방법 (기본값: cv2.BORDER_CONSTANT)
        border_value: 경계 값 (기본값: 0.0)
    
    Returns:
        remap된 이미지
    """
    dst = cv2.remap(src_img, map_x, map_y, interpolation=interpolation, borderMode=border_mode, borderValue=border_value)
    return dst


def save_remap_report(
    image_path: Path,
    gray: np.ndarray,
    image_points: np.ndarray,
    object_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    out_path: Path,
    map_x: Optional[np.ndarray] = None,
    map_y: Optional[np.ndarray] = None,
) -> None:
    """remap LUT 적용 결과 이미지 저장.
    
    Args:
        image_path: 원본 이미지 경로
        gray: 원본 그레이스케일 이미지
        image_points: 2D 이미지 포인트 (사용하지 않음, 호환성 유지)
        object_points: 3D 객체 포인트 (사용하지 않음, 호환성 유지)
        K: 카메라 내부 파라미터 행렬 (사용하지 않음, 호환성 유지)
        dist: 왜곡 계수 (사용하지 않음, 호환성 유지)
        rvec: 회전 벡터 (사용하지 않음, 호환성 유지)
        tvec: 이동 벡터 (사용하지 않음, 호환성 유지)
        out_path: 출력 경로
        map_x: X 방향 remap 맵 (None이면 remap 생략)
        map_y: Y 방향 remap 맵 (None이면 remap 생략)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # remap 적용
    if map_x is not None and map_y is not None:
        remapped = _remap_image(gray, map_x, map_y, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, border_value=0.0)
        cv2.imwrite(str(out_path), remapped)
    else:
        # map_x 또는 map_y가 None이면 원본 이미지만 저장
        cv2.imwrite(str(out_path), gray)


