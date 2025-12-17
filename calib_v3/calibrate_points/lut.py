from __future__ import annotations

from typing import Tuple
import numpy as np
import cv2
import tifffile as tiff
from pathlib import Path
from ..utils.config import TransportConfig

def _compute_inscribed_bbox(K: np.ndarray, dist: np.ndarray, w: int, h: int, safety: float) -> Tuple[np.ndarray, np.ndarray]:
    """왜곡 보정 후 유효한 영역의 경계 상자 계산
    
    Returns:
        (min_width_index, min_height_index), (max_width_index, max_height_index)
        각 인덱스는 width→x, height→y 순서로 반환된다.
    """
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
    
    min_height_index = np.ceil(top_max + safety)
    max_height_index = np.floor(bottom_min - safety)
    min_width_index = np.ceil(left_max + safety)
    max_width_index = np.floor(right_min - safety)
    
    return (
        np.array([min_height_index, max_height_index], np.float64),
        np.array([min_width_index, max_width_index], np.float64),
    )


def _build_rectify_map(K: np.ndarray, dist: np.ndarray, img_size: Tuple[int, int], height_idx_minmax: np.ndarray, width_idx_minmax: np.ndarray, max_component_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """왜곡 보정 맵 생성"""
    R_eye = np.eye(3, dtype=np.float64)
    newW = int(max(1, int(width_idx_minmax[1] - width_idx_minmax[0] + 1)))
    newH = int(max(1, int(height_idx_minmax[1] - height_idx_minmax[0] + 1)))

    size = (newW, newH) if max_component_idx == 0 else (newH, newW)

    P_shift = K.astype(np.float64).copy()
    P_shift[0, 2] -= float(width_idx_minmax[0])
    P_shift[1, 2] -= float(height_idx_minmax[0])
    map_x, map_y = cv2.initUndistortRectifyMap(K.astype(np.float64), dist.astype(np.float64), R_eye, P_shift, size, cv2.CV_32FC1)
    return map_x, map_y, P_shift, newH, newW


def _apply_flip_if_needed(map_x: np.ndarray, map_y: np.ndarray, transport: Tuple[float, float, float], hflip_on_negative_mean_trel_x: bool) -> Tuple[np.ndarray, np.ndarray, bool]:
    """수평 플립 정책 적용 (TransportConfig 기준)
    
    Args:
        map_x, map_y: LUT 맵
        transport: transport vector x,y,z components
    
    Returns:
        (map_x, map_y, did_flip, did_swap): 플립된 맵과 플립 여부
    """

    '''
    option 1 : max_component == 0 and transport[max_component] > 0 (x-axis positive); → object moves along x-axis positive direction in the next frame on image coordinate        
        --> do nothing
    
    option 2 : max_component == 0 and transport[max_component] < 0 (x-axis negative); ← object moves along x-axis negative direction in the next frame on image coordinate
        --> flip horizontal and vertical axis
    
    option 3 : max_component == 1 and transport[max_component] > 0 (y-axis positive); ↓ object moves along y-axis positive direction in the next frame on image coordinate
        --> do not flip
        --> swap map_x and map_y
    
    option 4 : max_component == 1 and transport[max_component] < 0 (y-axis negative); ↑ object moves along y-axis negative direction in the next frame on image coordinate        
        --> flip horizontal and vertical axis
        --> swap map_x and map_y
    
    '''

    # transport vector 의 max component 찾기
    max_component_idx = np.argmax(np.abs(transport))
    
    # estimated_transport_sign 의 부호를 판단하여 option 1, 2, 3, 4 중 하나를 선택
    estimated_transport_sign = transport[max_component_idx]
    
    # option 1
    if max_component_idx == 0 and estimated_transport_sign > 0:
        return map_x, map_y, False, False # do nothing
    # option 2
    elif max_component_idx == 0 and estimated_transport_sign < 0:
        return map_x[:, ::-1], map_y[:, ::-1], True, False # flip horizontal and vertical axis
    # option 3
    elif max_component_idx == 1 and estimated_transport_sign > 0:
        map_x_swapped, map_y_swapped = map_y, map_x
        return map_x_swapped, map_y_swapped, False, True # swap map_x and map_y        
    # option 4
    elif max_component_idx == 1 and estimated_transport_sign < 0:
        map_x_flipped, map_y_flipped = map_x[::-1, ::-1], map_y[::-1, ::-1] # flip horizontal and vertical axis
        map_x_flipped_swapped, map_y_flipped_swapped = map_y_flipped, map_x_flipped # swap map_x and map_y
        return map_x_flipped_swapped, map_y_flipped_swapped, True, True 




def _largest_all_valid_rect(map_x: np.ndarray, map_y: np.ndarray, max_index_for_map_x: int, max_index_for_map_y: int, margin: float = 0.0) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """유효한 영역에서 가장 큰 직사각형 찾기"""
    Hm, Wm = map_x.shape
    valid = (map_x >= margin) & (map_x <= (max_index_for_map_x - 1.0 - margin)) & (map_y >= margin) & (map_y <= (max_index_for_map_y - 1.0 - margin))
    
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
    transport: Tuple[float, float, float],
    TRANSPORT_CONFIG: TransportConfig = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """LUT 생성 및 저장
    
    Args:
        K: 카메라 내부 파라미터
        dist: 왜곡 계수
        img_size: 이미지 크기 (height, width)
        lut_policy: LUT 정책 ('crop' 또는 'expand')
        lut_crop_margin: 크롭 마진
        trel_x_sign: 상대 이동량 X 부호
    
    Returns:
        (map_x, map_y, lut_info): LUT 맵과 정보
    """
    h, w = img_size

    # transport vector 의 max component 찾기
    max_component_idx = np.argmax(np.abs(transport))
    
    # 안전 마진 계산
    safety = max(1.0, float(TRANSPORT_CONFIG.lut_crop_margin)) if TRANSPORT_CONFIG.lut_policy == 'crop' else 0.0
    
    # 내접 경계 상자 계산
    height_idx_minmax, width_idx_minmax = _compute_inscribed_bbox(K, dist, w, h, safety)
    
    # 정규화 맵 생성 # max_component_idx == 0 이면 map_xy 의 aspect ratio 는 image_size 와 동일. max_component_idx == 1 이면 map_xy 의 aspect ratio 는 image_size 와 반대.
    map_x, map_y, P_shift, newW, newH = _build_rectify_map(K, dist, img_size, height_idx_minmax, width_idx_minmax, max_component_idx)
    # from matplotlib import pyplot as plt
    
    # trasnport vector 의 max component 의 부호에 따라 X-axis <-> Y-axis     
    # 수평/수직 플립 적용 (TransportConfig 정책 사용)
    map_x, map_y, did_flip, did_swap = _apply_flip_if_needed(map_x, map_y, transport, TRANSPORT_CONFIG.hflip_on_negative_mean_trel_x)

    # did_swap = True 이면 remap 이후 이미지가 90deg 회전되어 map_x 가 가지는 최대 index 는 height 가 되고, map_y 가 가지는 최대 index 는 width 가 된다.
    max_index_for_map_x = h if not did_swap else w
    max_index_for_map_y = w if not did_swap else h

    # 크롭 정책 적용
    if TRANSPORT_CONFIG.lut_policy == 'crop':
        map_x, map_y, crop_bbox = _largest_all_valid_rect(map_x, map_y, max_index_for_map_x, max_index_for_map_y, margin=float(max(0.0, TRANSPORT_CONFIG.lut_crop_margin)))
    else:
        crop_bbox = (0, map_x.shape[0]-1, 0, map_x.shape[1]-1)
    
    cx = K[0, 2]
    cy = K[1, 2]
    # cx, cy는 float이므로 interpolation을 사용하여 float 위치의 값을 추정해야 함
    # order=1은 bilinear interpolation을 의미, mode는 boundary 처리 방식
    from scipy.ndimage import map_coordinates
    # 좌표는 (ndim, npoints) 형태: [[y좌표들], [x좌표들]]
    # Transport 가 [1,0,0] 이거나 [-1,0,0] 인 경우만 해당
    # Transport 가 [0,1,0] 이거나 [0,-1,0] 인 경우 별도 처리 필요
    coords = np.array([[cy], [cx]], dtype=np.float64)
    cam_center_y = map_coordinates(map_x, coords, order=1, mode='constant', cval=0.0)[0] # AIT-ICI convention
    cam_center_x = map_coordinates(map_y, coords, order=1, mode='constant', cval=0.0)[0] # AIT-ICI convention
    
    # LUT 정보
    lut_info = {
        'map_shape': (int(map_x.shape[0]), int(map_x.shape[1])),
        'cam_center_x': cam_center_x,
        'cam_center_y': cam_center_y,
        'original_size': img_size,        
        'crop_bbox': crop_bbox,
        'did_flip': did_flip,
        'did_swap': did_swap,
        'policy': TRANSPORT_CONFIG.lut_policy,
        'margin': TRANSPORT_CONFIG.lut_crop_margin
    }
    
    return map_x, map_y, lut_info
