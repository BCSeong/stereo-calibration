from __future__ import annotations
import traceback
import json
from pathlib import Path
from dataclasses import replace
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from .analysis import (
    compute_transport_vector,
    compute_resolution_from_tvecs,
    compute_relative_transforms,
    # compute_relative_transforms_world,
    # compute_relative_transforms_without_rotation,
    # compute_trel_stats,
    plot_series,
    plot_series_with_prefix,    
    compute_mean_disparity
)
from ..calibrate_points.lut import generate_lut, save_maps
from .types import RuntimeState, CalibResult, VERSION
from .logger import get_logger
from .config import TransportConfig

# deprecated it was used in Consistency report, but not used now
def _flatten_intrinsics_std(std_intr: Optional[np.ndarray]) -> Dict[str, float | str]:
    """AIT_ICI와 동일한 내부 파라미터 표준편차 처리"""
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

# deprecated it was used in Consistency report, but not used now
def _summarize_extrinsics_std(std_extr: Optional[np.ndarray]) -> Tuple[List[float], List[float]]:
    """AIT_ICI와 동일한 내부 파라미터 표준편차 처리"""
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

# deprecated it was used in Consistency report, but not used now
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


def _make_calibration_json(RuntimeState: RuntimeState, backward_RuntimeState: Optional[RuntimeState] = None) -> Dict:

    out = {
        'camera_matrix': RuntimeState.camera_matrix.tolist(),
        'distortion': RuntimeState.distortion.ravel().tolist(),
        'resolution': float(RuntimeState.resolution),
        'reprojected': float(RuntimeState.reprojected),
        'size': [RuntimeState.cam_width, RuntimeState.cam_height],
        'cam_width': RuntimeState.cam_width,
        'cam_height': RuntimeState.cam_height,
        'cam_focal': RuntimeState.cam_focal,
        'transport': RuntimeState.transport,        
        'cam_center_x': RuntimeState.cam_center_x,
        'cam_center_y': RuntimeState.cam_center_y,        
        'map_width': RuntimeState.map_width,
        'map_height': RuntimeState.map_height,
        'reprojection_prior': RuntimeState.reprojection_prior
    }
    if backward_RuntimeState is not None:
        out['transport_backward'] = backward_RuntimeState.transport
        out['cam_center_x_backward'] = backward_RuntimeState.cam_center_x
        out['cam_center_y_backward'] = backward_RuntimeState.cam_center_y
        out['version'] = VERSION + '_with_backwardInfo'

    out['version'] = VERSION
    return out


def _make_err_dots_json(RuntimeState: RuntimeState) -> Dict:
    per_view_list = [float(x) for x in np.asarray(RuntimeState.CALIB_RESULT.per_view_errors).reshape(-1)]
    reprojected = float(RuntimeState.reprojected)
    names_detected = RuntimeState.frame_names_list
    object_points = [arr.tolist() for arr in RuntimeState.object_points_list]
    image_points = [arr.tolist() for arr in RuntimeState.image_points_list]
    rvecs = [arr.tolist() for arr in RuntimeState.CALIB_RESULT.rvecs]
    tvecs = [arr.tolist() for arr in RuntimeState.CALIB_RESULT.tvecs]
    out = {
        'error': {
            'reprojected': float(reprojected),
            'per_view_reprojection': per_view_list,
        },
        'version': VERSION,
        'debug': {
            'names_detected': names_detected,
            'object_points': object_points,
            'image_points': image_points,
            'rvecs': rvecs,
            'tvecs': tvecs,
        },
    }
    return out


def _save_json(path: Path, data: Dict) -> None:
    """JSON 파일로 저장"""
    path.parent.mkdir(parents=True, exist_ok=True)
    def _json_default(obj):
        # numpy 스칼라/배열을 JSON 직렬화 가능 형태로 변환
        import numpy as _np
        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        return str(obj)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


def save_calibration_results_from_runtime_state(
    RuntimeState: RuntimeState,
    output_dir: Path,
    debug_dir: Path,
    save_error_plots_flag: bool = True,
    verbose: bool = True,
    generate_pseudo_backward: bool = True
) -> None:

    if generate_pseudo_backward:
        # dataclass를 복사한 뒤 필요한 필드만 교체
        # map_x/map_y는 뒤집힌 새 배열을 사용해 원본을 보호
        flipped_map_x = np.ascontiguousarray(RuntimeState.map_x[::-1, ::-1])
        flipped_map_y = np.ascontiguousarray(RuntimeState.map_y[::-1, ::-1])
        backward_RuntimeState = replace(
            RuntimeState,
            map_x=flipped_map_x,
            map_y=flipped_map_y,
            transport=(
                -RuntimeState.transport[0],
                -RuntimeState.transport[1],
                -RuntimeState.transport[2],
            ),
            cam_center_x=RuntimeState.map_width - RuntimeState.cam_center_x,
            cam_center_y=RuntimeState.map_height - RuntimeState.cam_center_y,
        )
        logger = get_logger()
        logger.info('[REPORTING] Pseudo backward RuntimeState created')

    # 1. ------------------------------------------------------------
    # LUT 저장 (forward)
    lut_dir = Path(output_dir) 
    save_maps(lut_dir, RuntimeState.map_x, RuntimeState.map_y)

    # LUT 저장 (backward)    
    if generate_pseudo_backward:
        backward_lut_dir = Path(output_dir) / 'calibration_lut_backward'
        save_maps(backward_lut_dir, backward_RuntimeState.map_x, backward_RuntimeState.map_y)  

    if verbose:
        logger = get_logger()
        logger.info('[REPORTING] LUT saved to %s', str(lut_dir))
        logger.info('[REPORTING] LUT info: %s', str(RuntimeState.lut_info))
  
    # 2. ------------------------------------------------------------
    # LotaCalibrationResult.json 저장
    if generate_pseudo_backward:
        calib_json = _make_calibration_json(RuntimeState, backward_RuntimeState)
        _save_json(Path(output_dir) / 'LotaCalibrationResult.json', calib_json)
    else:
        calib_json = _make_calibration_json(RuntimeState)
        _save_json(Path(output_dir) / 'LotaCalibrationResult.json', calib_json)


    if verbose:
        logger = get_logger()
        logger.info('[REPORTING] Calibration JSON saved to %s', str(Path(output_dir) / 'LotaCalibrationResult.json'))
    
    # 3. ------------------------------------------------------------
    # ErrorDotsReport.json 저장   
    err_dots_json = _make_err_dots_json(RuntimeState)
    _save_json(debug_dir / 'ErrorDotsReport.json', err_dots_json)
    if verbose:
        logger = get_logger()
        logger.info('[REPORTING] Calibration error dots JSON saved to %s', str(debug_dir / 'ErrorDotsReport.json'))


    if save_error_plots_flag and RuntimeState.folder_index_list is not None:
        # 기존 카메라좌표계 시리즈        
        rel_series_cam = compute_relative_transforms(RuntimeState.CALIB_RESULT.rvecs, RuntimeState.CALIB_RESULT.tvecs, RuntimeState.folder_index_list)
        plot_series_with_prefix(rel_series_cam, debug_dir, prefix='translation_series')

    '''
    # Consistency report: translation_series 표준편차 vs extrinsics std 비교
    # deprecated, instead use compute_relative_transforms_without_rotation
    try:
        # 1) 카메라 j좌표계 기준 (기존)
        rel_series_cam = compute_relative_transforms(RuntimeState.CALIB_RESULT.rvecs, RuntimeState.CALIB_RESULT.tvecs, RuntimeState.folder_index_list)
        trel_mean, trel_std, trel_count = compute_trel_stats(rel_series_cam)
        # 2) 월드(타겟) 좌표계 기준 (참고용)
        rel_series_w = compute_relative_transforms_world(RuntimeState.CALIB_RESULT.rvecs, RuntimeState.CALIB_RESULT.tvecs, RuntimeState.folder_index_list)
        trel_mean_w, trel_std_w, trel_count_w = compute_trel_stats(rel_series_w)

        # 방향 투영 기반 비교: 카메라 기준 Δt를 하나의 대표 방향 v로 투영하여 std 측정
        # pairs 및 trel 목록 구성 (카메라 기준 식으로 직접 계산해 방향 일치 보장)
        pairs = []
        trels = []
        for key, idxs in RuntimeState.folder_index_list.items():
            for a, b in zip(idxs[:-1], idxs[1:]):
                Ri, _ = cv2.Rodrigues(RuntimeState.CALIB_RESULT.rvecs[a])
                Rj, _ = cv2.Rodrigues(RuntimeState.CALIB_RESULT.rvecs[b])
                ti = RuntimeState.CALIB_RESULT.tvecs[a].reshape(3, 1)
                tj = RuntimeState.CALIB_RESULT.tvecs[b].reshape(3, 1)
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
        t_std_views = _per_view_translation_std(RuntimeState.CALIB_RESULT.std_extr)
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
        rot_std_mean, trans_std_mean = _summarize_extrinsics_std(RuntimeState.CALIB_RESULT.std_extr)
        # 회전 무시 버전(참고): 순수 tvec 차분 기반 std
        rel_series_naive = compute_relative_transforms_without_rotation(RuntimeState.CALIB_RESULT.rvecs, RuntimeState.CALIB_RESULT.tvecs, RuntimeState.folder_index_list)
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
        debug_dir = Path(output_dir) / 'report'
        debug_dir.mkdir(parents=True, exist_ok=True)
        _save_json(debug_dir / 'consistency.json', consistency)
        logger = get_logger()
        logger.info('[CONSISTENCY] cam trel_std=%s, world trel_std=%s, parallel_std_cam=%.6f, sigma_pred_parallel=%.6f, extrinsics_t_std_mean=%s, trel_std_naive_tonly=%s', str(consistency['trel_std_cam']), str(consistency['trel_std_world']), float(consistency['parallel_std_cam']), float(consistency['sigma_pred_parallel_from_std_extr'] if isinstance(consistency['sigma_pred_parallel_from_std_extr'], (int, float)) else 0.0), str(consistency['extrinsics_translation_std_mean']), str(consistency['trel_std_naive_tonly']))
    except Exception as e:
        logger = get_logger()
        logger.exception('[ERR] Consistency report failed: %s', e)
    # Error plots 생성
    if save_error_plots_flag and RuntimeState.folder_index_list is not None:
        # 기존 카메라좌표계 시리즈        
        rel_series_cam = compute_relative_transforms(RuntimeState.CALIB_RESULT.rvecs, RuntimeState.CALIB_RESULT.tvecs, RuntimeState.folder_index_list)
        plot_series_with_prefix(rel_series_cam, Path(output_dir) / 'report', prefix='camera')
        
        # 월드좌표계 시리즈        
        rel_series_w = compute_relative_transforms_world(RuntimeState.CALIB_RESULT.rvecs, RuntimeState.CALIB_RESULT.tvecs, RuntimeState.folder_index_list)
        plot_series_with_prefix(rel_series_w, Path(output_dir) / 'report', prefix='world', apply_abs=True)

        # 회전 무시 버전(참고): 순수 tvec 차분 기반 std
        rel_series_naive = compute_relative_transforms_without_rotation(RuntimeState.CALIB_RESULT.rvecs, RuntimeState.CALIB_RESULT.tvecs, RuntimeState.folder_index_list)
        plot_series_with_prefix(rel_series_naive, Path(output_dir) / 'report', prefix='naive_tonly')
    '''

    if verbose:
        logger = get_logger()
        logger.info('[REPORTING] Calibration results saved to %s', str(output_dir))

        logger.info('[REPORTING] Camera calibration result:')
        logger.info('\t Camera matrix K:\n%s', str(RuntimeState.camera_matrix))
        logger.info('\t Distortion coefficients (k1, k2, p1, p2, k3): %s', str(RuntimeState.distortion.flatten()))
        logger.info('\t RMS reprojection error (≤ 0.5 px): %.6f', RuntimeState.reprojected)
        logger.info('\t Transport vector (x, y, z): %s', str(RuntimeState.transport))
        logger.info('\t Resolution on center fiducial (um/px): %.6f', RuntimeState.resolution)
    
    # Reprojection error 체크 및 디버그 가이드
    if RuntimeState.reprojected > 0.5:
        logger = get_logger()
        logger.error('[FAIL] RMS reprojection error (%.6f) exceeds threshold (0.5 px)', RuntimeState.reprojected)
        logger.error('This indicates potential issues with point matching or grid assignment.')
        logger.error('Please check the debug results and adjust parameters accordingly.')
        return False
    else:
        logger = get_logger()
        logger.info('[SUCCESS] RMS reprojection error (%.6f) is within acceptable range (≤ 0.5 px)', RuntimeState.reprojected)
        return True


def save_error_plots(
    output_dir: Path,
    calib_result,
    by_folder: Dict[str, List[int]],
    verbose: bool = True
) -> None:
    """Error plots 생성 (translation/rotation 시리즈)"""
    try:
        debug_dir = Path(output_dir) / 'report'
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


def _get_lut_maps(
    calib_result: CalibResult,
    img_size: Tuple[int, int],
    transport: List[float],    
    TRANSPORT_CONFIG: TransportConfig,
    verbose: bool = True    
) -> dict:
    """LUT 맵 생성 및 저장"""
    try:
        if verbose:
            logger = get_logger()
            logger.info('[REPORTING] Generating LUT with policy: %s', TRANSPORT_CONFIG.lut_policy)
                
        map_x, map_y, lut_info = generate_lut(
            K = calib_result.camera_matrix,
            dist = calib_result.distortion,
            img_size = img_size,
            transport = transport,            
            TRANSPORT_CONFIG=TRANSPORT_CONFIG            
        )

        # TODO : logger info 로 LUT 정책과 transport 결과 비교하여 출력.swap, flip 여부 등 출력.
            
    except Exception as e:
        logger = get_logger()
        logger.exception('[ERR] LUT generation failed: %s', e)

    return map_x, map_y, lut_info


def convert_and_update_runtime_state(
    RuntimeState: RuntimeState,
    TRANSPORT_CONFIG: TransportConfig,
    verbose: bool = True    
) -> RuntimeState:
    """모든 결과 저장 및 분석을 통합 관리하는 메인 함수
    
    Args:
        RuntimeState: RuntimeState with calibration results
        TRANSPORT_CONFIG: TransportConfig
        save_error_plots_flag: 에러 플롯 저장 여부
        verbose: 상세 출력 여부
        
    Returns:        
        RuntimeState: RuntimeState with updated LUT info and calibration results        
    """

    CALIB_RESULT = RuntimeState.CALIB_RESULT
    folder_index_list = RuntimeState.folder_index_list
    image_size = RuntimeState.image_size
    
    # mean camera focal length 계산
    cam_focal: float = (CALIB_RESULT.camera_matrix[0, 0] + CALIB_RESULT.camera_matrix[1, 1]) * 0.5
    
    # Transport 및 Resolution 계산
    transport = compute_transport_vector(CALIB_RESULT.tvecs, folder_index_list)
    
    # 정사투영 목표 Z 위치 제공 (stereo target을 촬영한 Z 위치임)
    target_Z_um, resolution_um_per_px_at_target_Z_um = compute_resolution_from_tvecs(CALIB_RESULT.camera_matrix, CALIB_RESULT.tvecs)

    # Mean disparity 계산
    disparity_at_target_Z_um_with_predefined_baseline: dict = compute_mean_disparity(focal_length_px=cam_focal, target_Z_um=target_Z_um, baseline_um=TRANSPORT_CONFIG.predefined_baseline_um)

    # LUT 생성 및 저장
    map_x, map_y, lut_info = _get_lut_maps(CALIB_RESULT, image_size, transport, TRANSPORT_CONFIG, verbose)

    # Calibration.json 에서 정사투영 목표 Z 위치 제공 (stereo target을 촬영한 Z 위치임)
    reprojection_prior: dict = {
        'target_Z_um': target_Z_um,
        'predefined_baseline_um': TRANSPORT_CONFIG.predefined_baseline_um,
        'resolution_um_per_px_at_target_Z_um': resolution_um_per_px_at_target_Z_um,
        'disparity_at_target_Z_um_with_predefined_baseline': disparity_at_target_Z_um_with_predefined_baseline        
    }

    '''
    !!! reprojection_prior 은 stereo matching 후 생성된 depth map 을 이용해서 3D point cloud 를 생성 뒤
    orthographic projection 을 통해 생성된 2D image 를 이용해서 생성된 것임.
    
    필요 파라미터와 용도:
    target_Z_um: 정사투영할 평면 위치.
    resolution_at_target_Z: um_per_px = target_Z_um / fx_rect (y방향은 fy_rect). 정사영상의 물리 해상도.
    disparity_at_target_Z: d_target = fx_rect * baseline_um / target_Z_um. 이론 시차(검증/스케일 참고용).
    camera_matrix_rect (rectified fx, fy, cx, cy) + map_x/map_y (undistort/rectify LUT).    
    map_x/map_y로 원본 이미지를 리매핑 → rectified 좌표계에서 stereo 매칭.
    camera_matrix의 fx, fy, cx, cy를 깊이/좌표 변환에 사용. 

    유저가 사용하는 절차(정사투영용)
    1) map_x/map_y로 left/right 영상을 undistort+rectify. (remap 으로 인해 camera_matrix 의 fx, fy 는 변경되진 않으나 주점(cx, cy) 는 변경될 수 있음.)
    2) rectified 좌표계에서 disparity map 계산.
    3) 깊이 복원: Z_px = f * baseline_um / disparity_px (단위 일관: B,Z um, f,d px).
        - f 는 camera_matrix에서 제공
        - baseline_um 은 left/right 카메라 간 거리로 제공됨
        - disparity_px 는 rectified 좌표계에서 stereo 매칭 결과로 제공됨
    4) 3D 좌표 변환:
        - X = (x - cx) * Z_px / f,
        - Y = (y - cy) * Z_px / f,
        - Z = Z_px.
    5) 정사투영: 원하는 target_Z_um 평면에 투영/보간하여 2D 맵 생성. 이때 um/px는 resolution_at_target_Z_um를 사용.

    '''
    

    # !!!!!!!!!!!!!!!!!!!!!!
    # RuntimeState 업데이트
    RuntimeState.camera_matrix: np.ndarray = CALIB_RESULT.camera_matrix
    RuntimeState.distortion: np.ndarray = CALIB_RESULT.distortion
    RuntimeState.resolution: float = resolution_um_per_px_at_target_Z_um
    RuntimeState.reprojection_prior: Dict = reprojection_prior    

    RuntimeState.reprojected: float = CALIB_RESULT.reprojected # or include intrinsic, rotation, translation error
    RuntimeState.size: Tuple[int, int] = image_size    
    RuntimeState.cam_height: int = image_size[0]
    RuntimeState.cam_width: int = image_size[1]    
    RuntimeState.cam_focal: float = cam_focal
    RuntimeState.transport: Tuple[float, float, float] = transport    

    # RuntimeState 업데이트 (LUT 정보)    
    RuntimeState.map_height: int = lut_info['map_shape'][0]
    RuntimeState.map_width: int = lut_info['map_shape'][1]    
    RuntimeState.cam_center_x: float = lut_info['cam_center_x']
    RuntimeState.cam_center_y: float = lut_info['cam_center_y']
    RuntimeState.map_x: np.ndarray = map_x
    RuntimeState.map_y: np.ndarray = map_y
    RuntimeState.lut_info: dict = lut_info

    return RuntimeState
