from __future__ import annotations

from typing import Tuple
import numpy as np
import cv2
import tifffile as tiff
from pathlib import Path
from ..utils.config import TRANSPORT


def compute_inscribed_bbox(K: np.ndarray, dist: np.ndarray, w: int, h: int, safety: float) -> Tuple[np.ndarray, np.ndarray]:
    """왜곡 보정 후 유효한 영역의 경계 상자를 계산"""
    R_eye = np.eye(3, dtype=np.float64)
    step = 1  # step size in pixels
    xs = np.arange(0, w, step, dtype=np.float64)
    ys = np.arange(0, h, step, dtype=np.float64)
    if xs[-1] != (w - 1):
        xs = np.append(xs, float(w - 1))
    if ys[-1] != (h - 1):
        ys = np.append(ys, float(h - 1))
    
    # 경계 점들 생성
    top = np.stack([xs, np.zeros_like(xs)], axis=1)
    bottom = np.stack([xs, np.full_like(xs, h - 1.0)], axis=1)
    left = np.stack([np.zeros_like(ys), ys], axis=1)
    right = np.stack([np.full_like(ys, w - 1.0), ys], axis=1)
    rings = [(top, bottom, left, right)]
    
    if w > 2 and h > 2:
        top2 = np.stack([xs, np.full_like(xs, 1.0)], axis=1)
        bottom2 = np.stack([xs, np.full_like(xs, h - 2.0)], axis=1)
        left2 = np.stack([np.full_like(ys, 1.0), ys], axis=1)
        right2 = np.stack([np.full_like(ys, w - 2.0), ys], axis=1)
        rings.append((top2, bottom2, left2, right2))
    
    und_rings = []
    for (t, b, l, r) in rings:
        border_pts = np.concatenate([t, b, l, r], axis=0).reshape(-1, 1, 2)
        und_all = cv2.undistortPoints(
            border_pts, K.astype(np.float64), dist.astype(np.float64), R=R_eye, P=K.astype(np.float64)
        ).reshape(-1, 2)
        n_t, n_b, n_l = t.shape[0], b.shape[0], l.shape[0]
        und_t = und_all[:n_t]
        und_b = und_all[n_t:n_t + n_b]
        und_l = und_all[n_t + n_b:n_t + n_b + n_l]
        und_r = und_all[n_t + n_b + n_l:]
        und_rings.append((und_t, und_b, und_l, und_r))
    
    top_max = max(np.max(ut[:, 1]) for (ut, _, _, _) in und_rings)
    bottom_min = min(np.min(ub[:, 1]) for (_, ub, _, _) in und_rings)
    left_max = max(np.max(ul[:, 0]) for (_, _, ul, _) in und_rings)
    right_min = min(np.min(ur[:, 0]) for (_, _, _, ur) in und_rings)
    
    y_min = np.ceil(top_max + safety)
    y_max = np.floor(bottom_min - safety)
    x_min = np.ceil(left_max + safety)
    x_max = np.floor(right_min - safety)
    
    return np.array([x_min, y_min], np.float64), np.array([x_max, y_max], np.float64)


def build_rectify_map(K: np.ndarray, dist: np.ndarray, img_size: Tuple[int, int], min_xy: np.ndarray, max_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """왜곡 보정 맵 생성"""
    R_eye = np.eye(3, dtype=np.float64)
    newW = int(max(1, int(max_xy[0] - min_xy[0] + 1)))
    newH = int(max(1, int(max_xy[1] - min_xy[1] + 1)))
    P_shift = K.astype(np.float64).copy()
    P_shift[0, 2] -= float(min_xy[0])
    P_shift[1, 2] -= float(min_xy[1])
    map_x, map_y = cv2.initUndistortRectifyMap(K.astype(np.float64), dist.astype(np.float64), R_eye, P_shift, (newW, newH), cv2.CV_32FC1)
    return map_x, map_y, P_shift, newW, newH


def apply_hflip_if_needed(map_x: np.ndarray, map_y: np.ndarray, trel_x: float) -> Tuple[np.ndarray, np.ndarray, bool]:
    """수평 플립 정책 적용 (TransportConfig 기준)
    
    Args:
        map_x, map_y: LUT 맵
        trel_x: transport vector의 X 성분
    
    Returns:
        (map_x, map_y, did_flip): 플립된 맵과 플립 여부
    """
    from ..utils.config import TRANSPORT
    
    # TransportConfig를 이용한 flip 결정
    axis_x_sign = TRANSPORT.axis_sign[0]  # TransportConfig의 X축 부호
    
    # hflip_on_negative_mean_trel_x 조건이 활성화되어 있고,
    # 산출된 X 위치가 axis_sign과 부호가 다르면 flip 수행
    should_flip = False
    if TRANSPORT.hflip_on_negative_mean_trel_x:
        # trel_x와 axis_x_sign의 부호가 다르면 flip
        if (trel_x > 0) != (axis_x_sign > 0):
            should_flip = True
    
    # 디버그 로그 (verbose 모드에서만)
    print(f'[LUT] Flip decision: trel_x={trel_x:.6f}, axis_x_sign={axis_x_sign}, should_flip={should_flip}')
    
    if should_flip:
        return map_x[:, ::-1], map_y[:, ::-1], True
    return map_x, map_y, False


def largest_all_valid_rect(map_x: np.ndarray, map_y: np.ndarray, w: int, h: int, margin: float = 0.0) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """유효한 영역에서 가장 큰 직사각형 찾기"""
    Hm, Wm = map_x.shape
    valid = (map_x >= margin) & (map_x <= (w - 1.0 - margin)) & (map_y >= margin) & (map_y <= (h - 1.0 - margin))
    
    heights = np.zeros(Wm, dtype=np.int32)
    best_area = 0
    best_bbox = (0, Hm-1, 0, Wm-1)
    
    for y in range(Hm):
        row = valid[y]
        heights[row] += 1
        heights[~row] = 0
        stack: list[tuple[int, int]] = []
        
        for x in range(Wm + 1):
            hgt = int(heights[x]) if x < Wm else 0
            start_x = x
            while stack and stack[-1][1] > hgt:
                sx, sh = stack.pop()
                area = sh * (x - sx)
                if area > best_area:
                    best_area = area
                    best_bbox = (y - sh + 1, y, sx, x - 1)
                start_x = sx
            stack.append((start_x, hgt))
    
    y0, y1, x0, x1 = best_bbox
    return map_x[y0:y1+1, x0:x1+1], map_y[y0:y1+1, x0:x1+1], best_bbox


def save_maps(lut_dir: Path, map_x: np.ndarray, map_y: np.ndarray) -> None:
    """LUT 맵을 다양한 형태로 저장"""
    lut_dir.mkdir(parents=True, exist_ok=True)
    
    # TIFF 형태로 저장
    tiff.imwrite(str(lut_dir / 'map_x.tiff'), map_x.astype(np.float32))
    tiff.imwrite(str(lut_dir / 'map_y.tiff'), map_y.astype(np.float32))
    
    # 바이너리 형태로 저장
    map_x.astype(np.float32).ravel(order='C').tofile(str(lut_dir / 'map_x.bin'))
    map_y.astype(np.float32).ravel(order='C').tofile(str(lut_dir / 'map_y.bin'))
    


def generate_lut(
    K: np.ndarray, 
    dist: np.ndarray, 
    img_size: Tuple[int, int],
    lut_policy: str = 'crop',
    lut_crop_margin: float = 0.0,
    trel_x_sign: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """LUT 생성 및 저장
    
    Args:
        K: 카메라 내부 파라미터
        dist: 왜곡 계수
        img_size: 이미지 크기 (width, height)
        lut_policy: LUT 정책 ('crop' 또는 'expand')
        lut_crop_margin: 크롭 마진
        trel_x_sign: 상대 이동량 X 부호
    
    Returns:
        (map_x, map_y, lut_info): LUT 맵과 정보
    """
    w, h = img_size
    
    # 안전 마진 계산
    safety = max(1.0, float(lut_crop_margin)) if lut_policy == 'crop' else 0.0
    
    # 내접 경계 상자 계산
    min_xy, max_xy = compute_inscribed_bbox(K, dist, w, h, safety)
    
    # 정규화 맵 생성
    map_x, map_y, P_shift, newW, newH = build_rectify_map(K, dist, img_size, min_xy, max_xy)
    
    # 수평 플립 적용 (TransportConfig 정책 사용)
    map_x, map_y, did_flip = apply_hflip_if_needed(map_x, map_y, trel_x_sign)
    
    # 크롭 정책 적용
    if lut_policy == 'crop':
        map_x, map_y, crop_bbox = largest_all_valid_rect(map_x, map_y, w, h, margin=float(max(0.0, lut_crop_margin)))
    else:
        crop_bbox = (0, map_x.shape[0]-1, 0, map_x.shape[1]-1)
    
    # LUT 정보
    lut_info = {
        'map_shape': (int(map_x.shape[0]), int(map_x.shape[1])),
        'original_size': img_size,        
        'crop_bbox': crop_bbox,
        'did_flip': did_flip,
        'policy': lut_policy,
        'margin': lut_crop_margin
    }
    
    return map_x, map_y, lut_info
