from __future__ import annotations
import traceback
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from .analysis import (
    compute_transport_vector,
    compute_resolution_um_per_px,
    compute_relative_transforms,
    compute_relative_transforms_world,
    compute_relative_transforms_without_rotation,
    plot_series,
    plot_series_with_prefix,
    compute_trel_stats,
)
from ..calibrate_points.lut import generate_lut, save_maps
from .types import RuntimeState
from .logger import get_logger
from .config import TransportConfig


def _flatten_intrinsics_std(std_intr: Optional[np.ndarray]) -> Dict[str, float | str]:
    """calib_v2와 동일한 내부 파라미터 표준편차 처리"""
    if std_intr is None or np.size(std_intr) < 4:
        return {
            'f_x': 'N/A', 'f_y': 'N/A', 'c_x': 'N/A', 'c_y': 'N/A',
            'k_1': 'N/A', 'k_2': 'N/A', 'p_1': 'N/A', 'p_2': 'N/A', 'k_3': 'N/A',
        }
    flat = np.asarray(std_intr, dtype=np.float64).reshape(-1)
    get = lambda i: float(flat[i]) if flat.size > i else 0.0
    return {
        'f_x': get(0), 'f_y': get(1), 'c_x': get(2), 'c_y': get(3),
        'k_1': get(4), 'k_2': get(5), 'p_1': get(6), 'p_2': get(7), 'k_3': get(8),
    }


def _summarize_extrinsics_std(std_extr: Optional[np.ndarray]) -> Tuple[List[float], List[float]]:
    """calib_v2와 동일한 외부 파라미터 표준편차 처리"""
    rot_mean = [0.0, 0.0, 0.0]
    trans_mean = [0.0, 0.0, 0.0]
    if std_extr is None:
        return rot_mean, trans_mean
    try:
        flat = np.asarray(std_extr, dtype=np.float64).reshape(-1)
        n_views = flat.size // 6
        if n_views <= 0:
            return rot_mean, trans_mean
        ex = flat[:n_views * 6].reshape(n_views, 6)
        r_std = ex[:, 0:3].mean(axis=0)
        t_std = ex[:, 3:6].mean(axis=0)
        rot_mean = [float(r_std[0]), float(r_std[1]), float(r_std[2])]
        trans_mean = [float(t_std[0]), float(t_std[1]), float(t_std[2])]
        return rot_mean, trans_mean
    except Exception:
        print(f"[ERR] _summarize_extrinsics_std failed")
        traceback.print_exc()


def _per_view_translation_std(std_extr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """뷰별 translation 표준편차 (N_views, 3) 반환. 실패 시 None.
    std_extr layout: [rvec(3), tvec(3)] per view.
    """
    if std_extr is None:
        return None
    try:
        flat = np.asarray(std_extr, dtype=np.float64).reshape(-1)
        n_views = flat.size // 6
        if n_views <= 0:
            return None
        ex = flat[:n_views * 6].reshape(n_views, 6)
        # columns 3:6 are translation stds
        t_std = ex[:, 3:6]
        return t_std
    except Exception:
        traceback.print_exc()
        return None


def make_calibration_json(
    K: np.ndarray,
    dist: np.ndarray,
    img_size: Tuple[int, int],
    transport: List[float],
    resolution_um_per_px: float,
    map_shape: Tuple[int, int],  # (H, W)
    std_intr: Optional[np.ndarray],
    std_extr: Optional[np.ndarray],
    rms_reproj: float,
) -> Dict:
    """calib_v2와 동일한 JSON 구조로 캘리브레이션 결과 생성"""
    cam_w, cam_h = int(img_size[0]), int(img_size[1])
    cam_cx, cam_cy = float(K[0, 2]), float(K[1, 2])
    cam_f = float((K[0, 0] + K[1, 1]) * 0.5)
    map_h, map_w = int(map_shape[0]), int(map_shape[1])

    intr_err = _flatten_intrinsics_std(std_intr)
    rot_mean, trans_mean = _summarize_extrinsics_std(std_extr)

    out = {
        'camera_matrix': K.tolist(),
        'distortion': [dist.ravel().tolist()],
        'size': [cam_w, cam_h],
        'transport': transport,
        'cam_width': cam_w,
        'cam_height': cam_h,
        'cam_center_x': cam_cx,
        'cam_center_y': cam_cy,
        'cam_focal': cam_f,
        'resolution': float(resolution_um_per_px),
        'map_width': map_w,
        'map_height': map_h,
        'error': {
            'reprojected': float(rms_reproj),
            'intrinsics': intr_err,
            'rotation': rot_mean,
            'translation': trans_mean,
        },
        'version': '0.0.0-1, b.seong@kohyoung.com',
    }
    return out


def make_err_dots_json(
    rms_reproj: float,
    names_detected: List[str],
    obj_lists: List[List[float]],
    img_lists: List[List[float]],
    r_lists: List[List[float]],
    t_lists: List[List[float]],
    per_view_errs: Optional[np.ndarray],
) -> Dict:
    """calib_v2와 동일한 에러 디버그 JSON 생성"""
    per_view_list: List[float] = []
    if per_view_errs is not None:
        per_view_list = [float(x) for x in np.asarray(per_view_errs).reshape(-1)]
    out = {
        'error': {
            'reprojected': float(rms_reproj),
            'per_view_reprojection': per_view_list,
        },
        'version': '0.0.0-1, b.seong@kohyoung.com',
        'debug': {
            'names_detected': names_detected,
            'object_points': obj_lists,
            'image_points': img_lists,
            'rvecs': r_lists,
            'tvecs': t_lists,
        },
    }
    return out


def save_json(path: Path, data: Dict) -> None:
    """JSON 파일로 저장"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_calibration_results(
    output_dir: Path,
    calib_result,
    frame_names: List[str],
    object_points_list: List[np.ndarray],
    image_points_list: List[np.ndarray],
    img_size: Tuple[int, int],
    by_folder: Dict[str, List[int]] = None,
    transport: List[float] = None,
    resolution_um_per_px: float = None,
    map_shape: Tuple[int, int] = None,
    verbose: bool = True
) -> bool:
    """calib_v2와 동일한 구조로 캘리브레이션 결과 저장 및 분석"""
    
    # map_shape가 전달되면 그대로 사용, 없으면 이미지 크기로 대체
    map_shape = map_shape if map_shape is not None else (None, None)  # (H, W)

    # RMS 재투영 에러 계산
    rms_reproj = 0.0
    if calib_result.per_view_errs is not None:
        rms_reproj = float(np.sqrt(np.mean(calib_result.per_view_errs ** 2)))
    
    
    # 2. JSON 파일로 저장 (calib_v2와 동일한 구조)
    calib_json = make_calibration_json(
        calib_result.K, calib_result.dist, img_size,
        transport, resolution_um_per_px, map_shape, 
        calib_result.std_intr, calib_result.std_extr, rms_reproj
    )
    save_json(Path(output_dir) / 'calibration.json', calib_json)
    
    # 3. 에러 디버그 JSON 저장 (calib_v2와 동일한 구조)
    obj_lists = [arr.tolist() for arr in object_points_list]
    img_lists = [arr.tolist() for arr in image_points_list]
    r_lists = [arr.tolist() for arr in calib_result.rvecs]
    t_lists = [arr.tolist() for arr in calib_result.tvecs]
    
    err_dots_json = make_err_dots_json(
        rms_reproj, frame_names, obj_lists, img_lists, r_lists, t_lists, calib_result.per_view_errs
    )
    save_json(Path(output_dir) / 'calibration_err_dots.json', err_dots_json)
    
    
    if verbose:
        logger = get_logger()
        logger.info('[REPORTING] Calibration results saved to %s', str(output_dir))
        logger.info('[REPORTING] - NPZ: calibration_result.npz')
        logger.info('[REPORTING] - JSON: calibration.json')
        logger.info('[REPORTING] - Debug: calibration_err_dots.json')
        if save_error_plots:
            logger.info('[REPORTING] - Error plots: debug/')

        logger.info('[REPORTING] Camera calibration result:')
        logger.info('\t Camera matrix K:\n%s', str(calib_result.K))
        logger.info('\t Distortion coefficients (k1, k2, p1, p2, k3): %s', str(calib_result.dist.flatten()))
        logger.info('\t RMS reprojection error (≤ 0.5 px): %.6f', rms_reproj)
        logger.info('\t Transport vector (x, y, z): %s', str(transport))
        logger.info('\t Resolution on center fiducial (um/px): %.6f', resolution_um_per_px)
    
    # Reprojection error 체크 및 디버그 가이드
    if rms_reproj > 0.5:
        logger = get_logger()
        logger.error('[FAIL] RMS reprojection error (%.6f) exceeds threshold (0.5 px)', rms_reproj)
        logger.error('This indicates potential issues with point matching or grid assignment.')
        logger.error('Please check the debug results and adjust parameters accordingly.')
        return False
    else:
        logger = get_logger()
        logger.info('[SUCCESS] RMS reprojection error (%.6f) is within acceptable range (≤ 0.5 px)', rms_reproj)
        return True


def save_error_plots(
    output_dir: Path,
    calib_result,
    by_folder: Dict[str, List[int]],
    verbose: bool = True
) -> None:
    """Error plots 생성 (translation/rotation 시리즈)"""
    try:
        debug_dir = Path(output_dir) / 'debug'
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # 상대 변환 계산
        by_folder_series = compute_relative_transforms(
            calib_result.rvecs, calib_result.tvecs, by_folder
        )
        
        # 시리즈 플롯 생성
        plot_series(by_folder_series, debug_dir)
        
        if verbose:
            logger = get_logger()
            logger.info('[REPORTING] Error plots saved to %s', str(debug_dir))
            
    except Exception as e:
        logger = get_logger()
        logger.exception('[ERR] Error plot generation failed: %s', e)


def save_lut_maps(
    output_dir: Path,
    calib_result,
    img_size: Tuple[int, int],
    transport: List[float],    
    TRANSPORT_CONFIG: TransportConfig = None,
    verbose: bool = True    
) -> dict:
    """LUT 맵 생성 및 저장"""
    try:
        if verbose:
            logger = get_logger()
            logger.info('[REPORTING] Generating LUT with policy: %s', TRANSPORT_CONFIG.lut_policy)
        
        # transport의 X 성분을 이용해 horizontal flip 결정
        trel_x_sign = transport[0]  # transport vector의 X 성분
        
        map_x, map_y, lut_info = generate_lut(
            calib_result.K, calib_result.dist, img_size,
            trel_x_sign=trel_x_sign,            
            TRANSPORT_CONFIG=TRANSPORT_CONFIG            
        )
        
        # LUT 저장
        lut_dir = Path(output_dir) / 'calibration_lut'
        save_maps(lut_dir, map_x, map_y)
        
        if verbose:
            logger = get_logger()
            logger.info('[REPORTING] LUT saved to %s', str(lut_dir))
            logger.info('[REPORTING] LUT info: %s', lut_info)
            
    except Exception as e:
        logger = get_logger()
        logger.exception('[ERR] LUT generation failed: %s', e)
    return lut_info




def save_all_results(
    output_dir: Path,
    calib_result,
    frame_names: List[str],
    object_points_list: List[np.ndarray],
    image_points_list: List[np.ndarray],
    img_size: Tuple[int, int],
    frames: List = None,
    by_folder: Dict[str, List[int]] = None,
    save_error_plots_flag: bool = False,
    verbose: bool = True,
    TRANSPORT_CONFIG: TransportConfig = None,
    state: Optional[RuntimeState] = None
) -> dict:
    """모든 결과 저장 및 분석을 통합 관리하는 메인 함수
    
    Args:
        output_dir: 출력 디렉터리
        calib_result: 캘리브레이션 결과
        frame_names: 프레임 이름 리스트
        object_points_list: 객체 포인트 리스트
        image_points_list: 이미지 포인트 리스트
        img_size: 이미지 크기 (width, height)
        frames: 프레임 데이터 리스트
        by_folder: 폴더별 프레임 인덱스 매핑
        save_error_plots_flag: 에러 플롯 저장 여부
        lut_policy: LUT 정책
        lut_crop_margin: LUT 크롭 마진
        verbose: 상세 출력 여부
        state: RuntimeState (제공 시 자동 업데이트)
        
    Returns:
        dict: LUT 정보 (map_shape, did_flip, crop_bbox 등)
    """
    
    # Transport 및 Resolution 계산
    transport = compute_transport_vector(calib_result.tvecs, by_folder, TRANSPORT_CONFIG.axis_sign)
    resolution_um_per_px = compute_resolution_um_per_px(calib_result.K, calib_result.tvecs)

    # RuntimeState 업데이트
    if state is not None:
        state.transport = tuple(transport)
        state.resolution_um_per_px = resolution_um_per_px

    # LUT 생성 및 저장
    lut_info = save_lut_maps(output_dir, calib_result, img_size, transport, TRANSPORT_CONFIG, verbose)
    map_shape = lut_info['map_shape']

    # RuntimeState 업데이트 (LUT 정보)
    if state is not None:
        state.map_shape = lut_info.get('map_shape')
        state.did_flip = lut_info.get('did_flip', False)
        state.crop_bbox = lut_info.get('crop_bbox')
        state.cx_rect = lut_info.get('cx_rect')
        state.cy_rect = lut_info.get('cy_rect')
        state.min_xy = lut_info.get('min_xy')
        state.max_xy = lut_info.get('max_xy')

    # 기본 캘리브레이션 결과 저장
    calibration_success = save_calibration_results(
        output_dir, calib_result, frame_names,
        object_points_list, image_points_list, img_size,
        by_folder=by_folder,
        transport=transport,
        resolution_um_per_px=resolution_um_per_px,
        map_shape=map_shape,
        verbose=verbose
    )
    
    # RuntimeState 업데이트 (RMS reprojection error)
    if state is not None:
        if calib_result.per_view_errs is not None:
            state.rms_reproj = float(np.sqrt(np.mean(calib_result.per_view_errs ** 2)))
        else:
            state.rms_reproj = 0.0
    
    if save_error_plots_flag and by_folder is not None:        
        rel_series_naive = compute_relative_transforms_without_rotation(calib_result.rvecs, calib_result.tvecs, by_folder)
        plot_series_with_prefix(rel_series_naive, Path(output_dir) / 'debug', prefix='')
    
    # Consistency report: translation_series 표준편차 vs extrinsics std 비교
    # deprecated, instead use compute_relative_transforms_without_rotation
    '''
    try:
        # 1) 카메라 j좌표계 기준 (기존)
        rel_series_cam = compute_relative_transforms(calib_result.rvecs, calib_result.tvecs, by_folder)
        trel_mean, trel_std, trel_count = compute_trel_stats(rel_series_cam)
        # 2) 월드(타겟) 좌표계 기준 (참고용)
        rel_series_w = compute_relative_transforms_world(calib_result.rvecs, calib_result.tvecs, by_folder)
        trel_mean_w, trel_std_w, trel_count_w = compute_trel_stats(rel_series_w)

        # 방향 투영 기반 비교: 카메라 기준 Δt를 하나의 대표 방향 v로 투영하여 std 측정
        # pairs 및 trel 목록 구성 (카메라 기준 식으로 직접 계산해 방향 일치 보장)
        pairs = []
        trels = []
        for key, idxs in by_folder.items():
            for a, b in zip(idxs[:-1], idxs[1:]):
                Ri, _ = cv2.Rodrigues(calib_result.rvecs[a])
                Rj, _ = cv2.Rodrigues(calib_result.rvecs[b])
                ti = calib_result.tvecs[a].reshape(3, 1)
                tj = calib_result.tvecs[b].reshape(3, 1)
                trel = (tj - Rj @ Ri.T @ ti).ravel()
                pairs.append((a, b))
                trels.append(trel)
        if trels:
            T = np.vstack(trels)
            v = np.mean(T, axis=0)
            nv = float(np.linalg.norm(v))
            if nv == 0.0:
                # fallback: 주성분 방향
                u, s, vh = np.linalg.svd(T - T.mean(axis=0, keepdims=True), full_matrices=False)
                v = vh[0, :]
            else:
                v = v / nv
            # 평행/수직 성분 분해
            proj = (T @ v.reshape(3, 1)).reshape(-1)
            parallel_std = float(np.std(proj, ddof=1)) if proj.size > 1 else 0.0
            perp = T - np.outer(proj, v)
            perp_norm = np.linalg.norm(perp, axis=1)
            perp_std = float(np.std(perp_norm, ddof=1)) if perp_norm.size > 1 else 0.0
        else:
            parallel_std = 0.0
            perp_std = 0.0

        # per-view translation std에서 방향 v에 대한 예측 std (근사)
        t_std_views = _per_view_translation_std(calib_result.std_extr)
        sigma_pred_parallel = None
        if t_std_views is not None and trels and pairs:
            sigmas = []
            for (a, b) in pairs:
                # 독립축 가정: var_along_v = sum_k (v_k^2 * sigma_k^2)
                sa = float(np.sqrt(np.sum((t_std_views[a] * v)**2)))
                sb = float(np.sqrt(np.sum((t_std_views[b] * v)**2)))
                sigmas.append(np.sqrt(max(0.0, sa * sa + sb * sb)))
            sigma_pred_parallel = float(np.mean(sigmas)) if sigmas else 0.0

        # extrinsics std 평균값 요약 
        rot_std_mean, trans_std_mean = _summarize_extrinsics_std(calib_result.std_extr)
        # 회전 무시 버전(참고): 순수 tvec 차분 기반 std
        rel_series_naive = compute_relative_transforms_without_rotation(calib_result.rvecs, calib_result.tvecs, by_folder)
        trel_mean_naive, trel_std_naive, trel_count_naive = compute_trel_stats(rel_series_naive)
        consistency = {
            'trel_mean_cam': [float(x) for x in trel_mean.reshape(-1)],
            'trel_std_cam': [float(x) for x in trel_std.reshape(-1)],
            'trel_count_cam': int(trel_count),
            'trel_mean_world': [float(x) for x in trel_mean_w.reshape(-1)],
            'trel_std_world': [float(x) for x in trel_std_w.reshape(-1)],
            'trel_count_world': int(trel_count_w),
            'parallel_std_cam': parallel_std,
            'perp_std_cam': perp_std,
            'sigma_pred_parallel_from_std_extr': (sigma_pred_parallel if sigma_pred_parallel is not None else 'N/A'),
            'extrinsics_translation_std_mean': [float(x) for x in trans_std_mean],
            'trel_std_naive_tonly': [float(x) for x in trel_std_naive.reshape(-1)],
        }
        # 저장
        debug_dir = Path(output_dir) / 'debug'
        debug_dir.mkdir(parents=True, exist_ok=True)
        save_json(debug_dir / 'consistency.json', consistency)
        logger = get_logger()
        logger.info('[CONSISTENCY] cam trel_std=%s, world trel_std=%s, parallel_std_cam=%.6f, sigma_pred_parallel=%.6f, extrinsics_t_std_mean=%s, trel_std_naive_tonly=%s', str(consistency['trel_std_cam']), str(consistency['trel_std_world']), float(consistency['parallel_std_cam']), float(consistency['sigma_pred_parallel_from_std_extr'] if isinstance(consistency['sigma_pred_parallel_from_std_extr'], (int, float)) else 0.0), str(consistency['extrinsics_translation_std_mean']), str(consistency['trel_std_naive_tonly']))
    except Exception as e:
        logger = get_logger()
        logger.exception('[ERR] Consistency report failed: %s', e)
    # Error plots 생성
    if save_error_plots_flag and by_folder is not None:
        # 기존 카메라좌표계 시리즈        
        rel_series_cam = compute_relative_transforms(calib_result.rvecs, calib_result.tvecs, by_folder)
        plot_series_with_prefix(rel_series_cam, Path(output_dir) / 'debug', prefix='camera')
        
        # 월드좌표계 시리즈        
        rel_series_w = compute_relative_transforms_world(calib_result.rvecs, calib_result.tvecs, by_folder)
        plot_series_with_prefix(rel_series_w, Path(output_dir) / 'debug', prefix='world', apply_abs=True)

        # 회전 무시 버전(참고): 순수 tvec 차분 기반 std
        rel_series_naive = compute_relative_transforms_without_rotation(calib_result.rvecs, calib_result.tvecs, by_folder)
        plot_series_with_prefix(rel_series_naive, Path(output_dir) / 'debug', prefix='naive_tonly')
    '''
    

    return lut_info
