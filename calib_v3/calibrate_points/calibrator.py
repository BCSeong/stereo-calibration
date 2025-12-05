from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import cv2
from ..utils.logger import get_logger


class CalibResult:
    def __init__(self,
                 K: np.ndarray,
                 dist: np.ndarray,
                 rvecs: List[np.ndarray],
                 tvecs: List[np.ndarray],
                 std_intr: Optional[np.ndarray],
                 std_extr: Optional[np.ndarray],
                 per_view_errs: Optional[np.ndarray],
                 kept_indices: Optional[List[int]] = None):
        self.K = K
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.std_intr = std_intr
        self.std_extr = std_extr
        self.per_view_errs = per_view_errs
        self.kept_indices = kept_indices or []


def calibrate_shared(object_points_list: List[np.ndarray],
                     image_points_list: List[np.ndarray],
                     image_size: Tuple[int, int],
                     K_guess: Optional[np.ndarray] = None,
                     use_guess: bool = False,
                     remove_outliers: bool = True,
                     outlier_threshold: float = 2.0) -> CalibResult:
    flags = 0
    K0 = None
    dist0 = None
    if use_guess and K_guess is not None:
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        K0 = K_guess.astype(np.float64)

    # Outlier 제거 (iterative approach)
    kept_indices = list(range(len(object_points_list)))
    if remove_outliers:
        object_points_list, image_points_list, kept_indices = _remove_outlier_frames(
            object_points_list, image_points_list, image_size, 
            outlier_threshold, K_guess, use_guess
        )
    
    # PnP 경로 제거됨
    
    try:
        ret, K, dist, rvecs, tvecs, std_intr, std_extr, per_view_errs = cv2.calibrateCameraExtended(
            object_points_list, image_points_list, image_size, K0, dist0, flags=flags
        )
    except Exception:
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points_list, image_points_list, image_size, K0, dist0, flags=flags
        )
        std_intr, std_extr, per_view_errs = None, None, None
    
    # PnP refine LM 제거됨
    
    return CalibResult(K=K, dist=dist, rvecs=rvecs, tvecs=tvecs, std_intr=std_intr, std_extr=std_extr, per_view_errs=per_view_errs, kept_indices=kept_indices)


def _remove_outlier_frames(object_points_list: List[np.ndarray],
                          image_points_list: List[np.ndarray], 
                          image_size: Tuple[int, int],
                          outlier_threshold: float,
                          K_guess: Optional[np.ndarray] = None,
                          use_guess: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """Iterative outlier removal for calibration frames"""
    
    # 1단계: 초기 캘리브레이션
    try:
        ret, K, dist, rvecs, tvecs, std_intr, std_extr, per_view_errs = cv2.calibrateCameraExtended(
            object_points_list, image_points_list, image_size, None, None, flags=0
        )
    except Exception:
        # Extended 버전 실패 시 기본 버전 사용
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points_list, image_points_list, image_size, None, None, flags=0
        )
        per_view_errs = None
    
    if per_view_errs is None:
        # per_view_errs가 없으면 outlier 제거 불가
        return object_points_list, image_points_list, list(range(len(object_points_list)))
    
    # 2단계: Outlier 프레임 식별 및 제거  
    threshold =  outlier_threshold 
    
    # 좋은 프레임만 유지
    good_frames = per_view_errs <= threshold
    
    if np.sum(good_frames) < 3:  # 최소 3개 프레임은 유지
        get_logger().warning('[WARN] Too many outliers detected. Keeping all frames.')
        return object_points_list, image_points_list, list(range(len(object_points_list)))
    
    # Outlier 제거된 데이터 반환
    filtered_object_points = [obj_pts for i, obj_pts in enumerate(object_points_list) if good_frames[i]]
    filtered_image_points = [img_pts for i, img_pts in enumerate(image_points_list) if good_frames[i]]
    kept_indices = [i for i in range(len(object_points_list)) if good_frames[i]]

    # 어느 프레임이 outlier 인지 로그 출력    
    for i in range(len(good_frames)):
        if not good_frames[i]:
            get_logger().info('[INFO] Frame %d is outlier (threshold: %.3f, per_view_errs: %.3f)', i, threshold, float(per_view_errs[i]))
    
    removed_count = len(object_points_list) - len(filtered_object_points)
    if removed_count > 0:
        get_logger().info('[INFO] Removed %d outlier frames (threshold: %.3f)', removed_count, threshold)
    
    return filtered_object_points, filtered_image_points, kept_indices


