from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import cv2
from ..utils.logger import get_logger
from ..utils.types import CalibResult, RuntimeState


def calibrate_shared(RuntimeState: RuntimeState,
                     K_guess: Optional[np.ndarray] = None,
                     use_guess: bool = False,
                     remove_outliers: bool = False,
                     outlier_threshold: float = 2.0) -> CalibResult:
    flags = 0
    K0 = None
    dist0 = None
    if use_guess and K_guess is not None:
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        K0 = K_guess.astype(np.float64)
        get_logger().info('[INFO] Using guess K: %s', str(K0))
    else:
        get_logger().info('[INFO] No guess K provided')

    object_points_list = [f.object_points for f in RuntimeState.FRAME_DATA_LIST]
    image_points_list = [f.image_points for f in RuntimeState.FRAME_DATA_LIST]
    image_size = RuntimeState.image_size

    
    kept_indices = list(range(len(object_points_list)))
    
    # Outlier 제거 (iterative approach)
    # Outlier 제거 시 object_points_list, image_points_list, kept_indices 이 업데이트됨
    if remove_outliers:
        object_points_list, image_points_list, kept_indices = _remove_outlier_frames(
            object_points_list, image_points_list, image_size, 
            outlier_threshold, K_guess, use_guess
        )
        
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points_list, image_points_list, image_size, K0, dist0, flags=flags
    )        
    std_intr, std_extr = None, None
    per_view_errs = _compute_per_view_errors(object_points_list, image_points_list, rvecs, tvecs, K, dist)    

    # deprecated it was used in comparison test btw calibrateCameraExtended and calibrateCamera, but not used now
    '''
    try:
        ret, K, dist, rvecs, tvecs, std_intr, std_extr, per_view_errs = cv2.calibrateCameraExtended(
            object_points_list, image_points_list, image_size, K0, dist0, flags=flags
        )

        get_logger().info('[TEST] calibrateCameraExtended VS calibrateCamera comparision')
        ret_t, K_t, dist_t, rvecs_t, tvecs_t = cv2.calibrateCamera(
            object_points_list, image_points_list, image_size, K0, dist0, flags=flags
        )        
        per_view_errs_t = _compute_per_view_errors(object_points_list, image_points_list, rvecs_t, tvecs_t, K_t, dist_t)
        get_logger().info('[TEST] calibrateCameraExtended reprojection error %s', str(per_view_errs_t))
        get_logger().info('[TEST] calibrateCamera reprojection error %s', str(per_view_errs))
        get_logger().info('[TEST] reprojection error difference %s', str(np.abs(per_view_errs_t -  per_view_errs)))
        get_logger().info('[TEST] reprojection error difference percentage %s', str(100 * np.mean(np.abs(per_view_errs_t -  per_view_errs) / per_view_errs)))
        get_logger().info('[TEST] ret difference %s', str(np.abs(ret -  ret_t)))
        get_logger().info('[TEST] K difference %s', str(np.abs(K -  K_t)))
        get_logger().info('[TEST] dist difference %s', str(np.abs(dist -  dist_t)))    
        get_logger().info('[TEST] rvecs difference %s', str(np.abs(rvecs -  rvecs_t)))
        get_logger().info('[TEST] tvecs difference %s', str(np.abs(tvecs -  tvecs_t)))      
        # Test 결과 calibrateCamera 사용하여도 무관. 오차 6% 이내.


    except Exception:
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points_list, image_points_list, image_size, K0, dist0, flags=flags
        )
        std_intr, std_extr = None, None
        per_view_errs = _compute_per_view_errors(object_points_list, image_points_list, rvecs, tvecs, K, dist)
    '''
    
    return CalibResult(camera_matrix=K, distortion=dist, rvecs=rvecs, tvecs=tvecs, std_intrinsic=std_intr, std_extrinsic=std_extr, reprojected=ret, per_view_errors=per_view_errs, kept_indices=kept_indices)


def _remove_outlier_frames(object_points_list: List[np.ndarray],
                          image_points_list: List[np.ndarray], 
                          image_size: Tuple[int, int],
                          outlier_threshold: float,
                          K_guess: Optional[np.ndarray] = None,
                          use_guess: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """Iterative outlier removal for calibration frames"""
    
    # 1단계: 초기 캘리브레이션
    # See opencv documentation for more details: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
    # flags=0 is default, no special flags are used
    # Useful flags examples:
    # flags=cv2.CALIB_USE_INTRINSIC_GUESS is used when K_guess is provided
    # flags=cv2.CALIB_FIX_ASPECT_RATIO is used when aspect ratio is fixed
    flags = 0
    K0 = None
    dist0 = None
    if use_guess and K_guess is not None:
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        K0 = K_guess.astype(np.float64)

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=object_points_list, imagePoints=image_points_list, imageSize=image_size, cameraMatrix=K0, distCoeffs=dist0, flags=flags
    )
    per_view_errs = _compute_per_view_errors(object_points_list, image_points_list, rvecs, tvecs, K, dist)        
    
    # deprecated it was used in comparison test btw calibrateCameraExtended and calibrateCamera, but not used now
    '''
    try:
        ret, K, dist, rvecs, tvecs, std_intr, std_extr, per_view_errs = cv2.calibrateCameraExtended(
            objectPoints=object_points_list, imagePoints=image_points_list, imageSize=image_size, cameraMatrix=K0, distCoeffs=dist0, flags=flags
        ) 
    except Exception:
        # Extended 버전 실패 시 기본 버전 사용
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=object_points_list, imagePoints=image_points_list, imageSize=image_size, cameraMatrix=K0, distCoeffs=dist0, flags=flags
        )
        per_view_errs = _compute_per_view_errors(object_points_list, image_points_list, rvecs, tvecs, K, dist)
    '''
    
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


def _compute_per_view_errors(object_points_list: List[np.ndarray],
                             image_points_list: List[np.ndarray],
                             rvecs: List[np.ndarray],
                             tvecs: List[np.ndarray],
                             K: np.ndarray,
                             dist: np.ndarray) -> Optional[np.ndarray]:
    """calibrateCameraExtended가 없을 때 per-view reprojection error를 직접 계산."""
    try:
        errs = []
        for obj_pts, img_pts, rvec, tvec in zip(object_points_list, image_points_list, rvecs, tvecs):
            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
            proj = proj.reshape(-1, 2)
            img = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
            diff = img - proj
            err = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
            errs.append(float(err))
        return np.asarray(errs, dtype=np.float64)
    except Exception as e:
        get_logger().warning('[WARN] per_view_errs computation failed: %s', e)
        return None

