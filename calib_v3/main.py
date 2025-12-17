from __future__ import annotations

import traceback
from dataclasses import asdict, replace
from pathlib import Path
from pprint import pprint
import numpy as np
import cv2
from typing import List
import math

from .cli import build_argparser, update_dataclass_from_namespace
from .get_points.three_dots import ThreeDotDetector
from .utils.logger import init_logger, get_logger
from .calibrate_points.calibrator import calibrate_shared
from .utils.reporting import convert_and_update_runtime_state, save_calibration_results_from_runtime_state
from .utils.types import FrameData, AppConfig, RuntimeState, CalibResult, update_runtime_state_by_kept_indices
from .debug.visuals import save_detection_overlay, save_grid_report, save_reprojection_report, save_grid_path_report
from .utils.config import BlobDetectorConfig, GridConfig, TransportConfig, AffineCandidateConfig, ScoringConfig


def run(argv=None) -> RuntimeState:
    ap = build_argparser()
    args = ap.parse_args(argv)
    
    # AppConfig 생성 및 CLI 인자로 업데이트
    config = AppConfig(
        src=Path(args.src),
        dst=Path(args.dst),
        debug_dir=Path(args.dst) / 'report'
    )
    if args.blob_dia_in_px is not None and args.min_area is None and args.max_area is None:
        config.max_area = 4.00 * 3.14* (args.blob_dia_in_px/2)**2
        config.min_area = 0.60 * 3.14* (args.blob_dia_in_px/2)**2
    
    # CLI 인자로 AppConfig 업데이트
    update_dataclass_from_namespace(config, args)

    print(f"config.blob_dia_in_px: {config.blob_dia_in_px}")
    print(f"config.dot_pitch_um: {config.dot_pitch_um}")
    
    
    # 로거 초기화 (파일: dst/calibration.log, 콘솔: INFO 이상)
    log_path = Path(config.debug_dir) / 'calibration.log'
    init_logger(log_path)
    logger = get_logger()
    logger.info('[ENTER] main.run')

    # 디렉터리 생성
    config.debug_dir.mkdir(parents=True, exist_ok=True)


    # 검출 파라미터 병합 생성 (AppConfig에서 덮어씌우기)
    from .utils.types import merge_config_from_app
    BLOB_CONFIG: BlobDetectorConfig = merge_config_from_app(config, BlobDetectorConfig())
    GRID_CONFIG: GridConfig = merge_config_from_app(config, GridConfig())
    AFFINE_CANDIDATE_CONFIG: AffineCandidateConfig = merge_config_from_app(config, AffineCandidateConfig())
    SCORE_CONFIG: ScoringConfig = merge_config_from_app(config, ScoringConfig())
    TRANSPORT_CONFIG: TransportConfig = merge_config_from_app(config, TransportConfig())

    # 이미지 재귀 수집, 모든 이미지 처리를 하지 않고 일부 이미지만 처리하여 속도를 빠르게 할 수도 있음.
    # 예) 매 100번째 이미지만 처리하려면 step=100:
    def iter_images_recursive(root, step=1):
        root_path = Path(root)
        # 이미지 파일만 먼저 필터링
        image_files = [p for p in sorted(root_path.rglob('*')) 
                    if p.is_file() and (p.suffix.lower() in ('.bmp', '.png'))]
        # 이미지 파일만 enumerate
        for i, p in enumerate(image_files):
            if i % step == 0:
                yield p

    if config.verbose:
        logger.info('[CONFIG] %s', {'src': str(config.src),
         'dst': str(config.dst),
         'blob_detector': {
            'dot_pitch_um': float(GRID_CONFIG.dot_pitch_um),   
            'max_grid_size': int(GRID_CONFIG.max_grid_size),
            'blob_dia_in_px': float(BLOB_CONFIG.blob_dia_in_px),            
            'retrieval': str(BLOB_CONFIG.retrieval),
            'bin_threshold': BLOB_CONFIG.bin_threshold, # None or float
            'min_area': float(BLOB_CONFIG.min_area),
            'max_area': float(BLOB_CONFIG.max_area),
            'min_fill': float(BLOB_CONFIG.min_fill),
            'max_eccentricity': float(BLOB_CONFIG.max_eccentricity)}})

    det = ThreeDotDetector(BLOB_CONFIG, GRID_CONFIG, AFFINE_CANDIDATE_CONFIG, SCORE_CONFIG, config.debug_dir)

    # 디버그 및 산출물 저장 위치 정의
    out_spool = config.debug_dir / 'spool'           # Points 저장용 (항상 생성)
    
    # 디버그 관련 디렉터리들 (save_debug 활성화 시에만 생성)
    out_bin = config.debug_dir / 'bin'               # Binarized 이미지용
    out_success = config.debug_dir / 'Blob_Success'  # 성공한 blob 검출 결과
    out_fails = config.debug_dir / 'Blob_Fails'      # 실패한 blob 검출 결과
    out_reproj = config.debug_dir / 'reprojection_reports'  # 재투영 리포트
    
    # 필수 디렉터리 생성 (항상)
    out_spool.mkdir(parents=True, exist_ok=True)
    
    # 디버그 디렉터리 생성 (save_debug 활성화 시에만)
    if config.save_debug:
        out_bin.mkdir(parents=True, exist_ok=True)
        out_success.mkdir(parents=True, exist_ok=True)
        out_fails.mkdir(parents=True, exist_ok=True)
        out_reproj.mkdir(parents=True, exist_ok=True)

    
    processed = 0
    failed = 0
    # RuntimeState 초기화
    STATE = RuntimeState()
    frame_stats: List[dict] = []

    def compute_grid_orientation(grid_uv_to_idx, pts_xy):
        if not grid_uv_to_idx or pts_xy is None:
            return {"u_deg": 0.0, "v_deg": 0.0}
        u_vecs = []
        v_vecs = []
        key_set = set(grid_uv_to_idx.keys())
        for (u, v), idx in grid_uv_to_idx.items():
            if (u + 1, v) in key_set:
                j = grid_uv_to_idx[(u + 1, v)]
                u_vecs.append(pts_xy[j] - pts_xy[idx])
            if (u, v + 1) in key_set:
                j = grid_uv_to_idx[(u, v + 1)]
                v_vecs.append(pts_xy[j] - pts_xy[idx])

        def mean_angle(vecs):
            if not vecs:
                return 0.0, np.zeros(2, dtype=np.float64)
            arr = np.array(vecs, dtype=np.float64)
            ang = np.arctan2(arr[:, 1], arr[:, 0])
            mean_ang = math.degrees(math.atan2(np.mean(np.sin(ang)), np.mean(np.cos(ang))))
            mean_vec = np.mean(arr, axis=0)
            return float(mean_ang), mean_vec

        u_deg, u_mean_vec = mean_angle(u_vecs)
        v_deg, v_mean_vec = mean_angle(v_vecs)
        return {
            "u_deg": u_deg,
            "v_deg": v_deg
        }
    
    # 캘리브레이션용 points 수집 (types.py 구조 사용)
    STATE.FRAME_DATA_LIST: List[FrameData] = []
    import gc    
    
    # ------------------------------------------------------------------------------------------
    # Blob scan
    # ------------------------------------------------------------------------------------------
    total_frames = len(list(iter_images_recursive(config.src)))
    target_frames = len(list(iter_images_recursive(config.src, step = config.skip)))
    logger.info('[INFO] total frames=%d (skip=%d)', total_frames, config.skip)
    logger.info('[INFO] target frames=%d', target_frames)
    logger.info('[ENTER] Blob scan start with %d frames', target_frames)
    for img_path in iter_images_recursive(config.src, step = config.skip):
        gray = None
        pts = None
        diam = None
        try:
            # 이미지 로드 → 즉시 검출 → 저장 후 해제
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                failed += 1
                continue
            res = det.run_on_image(img_path, gray)
            pts = res.points_xy
            diam = res.diameters

            # 프레임 통계 누적 (dot/diameter, 인접 거리)
            dot_count = int(len(pts)) if pts is not None else 0
            if diam is not None and len(diam) > 0:
                dot_dia_mean = float(np.mean(diam))
                dot_dia_std = float(np.std(diam))
                dot_dia_max = float(np.max(diam))
                dot_dia_min = float(np.min(diam))
            else:
                dot_dia_mean = dot_dia_std = dot_dia_max = dot_dia_min = 0.0

            # grid 인접 거리/방향 통계 초기화
            dist_stats = {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0, "count": 0}
            orient_stats = {"u_deg": 0.0, "v_deg": 0.0}

            try:
                rel_path = img_path.relative_to(config.src)
                img_label = str(rel_path)
            except ValueError:
                img_label = str(img_path.name)

            # DEBUG 시각화 처리 
            if config.save_debug:
                # 이미지의 상대 경로 구조 생성 (src 기준)
                try:
                    # src 경로를 기준으로 상대 경로 계산
                    rel_path = img_path.relative_to(config.src)
                    # parent 경로 구조 유지
                    debug_subdir = rel_path.parent
                    debug_filename = rel_path.stem
                except ValueError:
                    # src 밖에 있는 경우 절대 경로 사용
                    debug_subdir = img_path.parent.name
                    debug_filename = img_path.stem
                
                # 디버그 서브디렉터리 생성
                debug_bin_dir = out_bin / debug_subdir
                debug_success_dir = out_success / debug_subdir  
                debug_fails_dir = out_fails / debug_subdir
                
                # 디렉터리 생성
                debug_bin_dir.mkdir(parents=True, exist_ok=True)
                debug_success_dir.mkdir(parents=True, exist_ok=True)
                debug_fails_dir.mkdir(parents=True, exist_ok=True)
                
                # binarized image 저장 (contour 모드에서만 의미 있음)
                if res.binarized_image is not None:
                    cv2.imwrite(str(debug_bin_dir / f"{debug_filename}_bin.png"), res.binarized_image)
                
                if res.grid_assign and len(res.grid_assign) > 0:
                    # 성공한 경우
                    out_png = debug_success_dir / f"{debug_filename}_blobs.png"
                    save_detection_overlay(gray, pts, res.chosen_triplet, out_png, None, res.nn24_indices)
                    
                    '''
                    # Grid report 저장 (triplet 포함)                    
                    grid_report_path = debug_success_dir / f"{debug_filename}_grid.png"
                    save_grid_report(img_path, gray, pts, res.grid_assign, res.Tc, grid_report_path, res.chosen_triplet)
                    '''

                    # (2025-12-15) 추가 : Grid path 저장
                    grid_path_path = debug_success_dir / f"{debug_filename}_grid_path.png"
                    dist_stats = save_grid_path_report(img_path, gray, pts, res.grid_assign, res.Tc, grid_path_path, res.chosen_triplet)
                    orient_stats = compute_grid_orientation(res.grid_assign, pts)
                else:
                    # 실패한 경우
                    out_png = debug_fails_dir / f"{debug_filename}_blobs.png"
                    save_detection_overlay(gray, pts, None, out_png, None, None)
                    dist_stats = {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0, "count": 0}
                    orient_stats = {"u_deg": 0.0, "v_deg": 0.0}
            else:
                if res.grid_assign and len(res.grid_assign) > 0:
                    dist_stats = {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0, "count": 0}
                    orient_stats = compute_grid_orientation(res.grid_assign, pts)
                else:
                    dist_stats = {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0, "count": 0}
                    orient_stats = {"u_deg": 0.0, "v_deg": 0.0}
            
            # RuntimeState 업데이트, 성공한 프레임의 points 수집 (grid_assign이 비어있지 않으면 성공)
            if res.grid_assign and len(res.grid_assign) > 0:
                # object points 생성 (실제 dot_pitch_um 사용, 회전/shift 적용 없음)
                grid_keys = list(res.grid_assign.keys())
                if grid_keys:
                    uv = np.array(grid_keys, dtype=np.int32)
                    object_pts = np.zeros((len(uv), 3), dtype=np.float32)
                    object_pts[:, 0] = uv[:, 0] * float(GRID_CONFIG.dot_pitch_um)  # X: um
                    object_pts[:, 1] = uv[:, 1] * float(GRID_CONFIG.dot_pitch_um)  # Y: um
                    object_pts[:, 2] = 0.0  # Z: um (평면)

                    # FrameData 구조로 저장
                    frame_data = FrameData(
                        image_path=img_path,
                        image_points=pts[list(res.grid_assign.values())],
                        object_points=object_pts
                    )
                    STATE.FRAME_DATA_LIST.append(frame_data) # RuntimeState 업데이트
                    
                    # RuntimeState 업데이트, 이미지 크기 저장 (첫 번째 성공 프레임에서)
                    if STATE.image_size is None:
                        STATE.image_size = (gray.shape[0], gray.shape[1])  # (height, width)

            
            # blob & grid 통계 누적
            # 성공 여부는 grid_assign 유무로 판단
            success_flag = 1.0 if (res.grid_assign and len(res.grid_assign) > 0) else 0.0
            frame_stats.append({
                "image": img_label,
                "success": bool(success_flag),
                "dot_count": dot_count,
                "dot_dia_mean": dot_dia_mean,
                "dot_dia_std": dot_dia_std,
                "dot_dia_max": dot_dia_max,
                "dot_dia_min": dot_dia_min,
                "grid_dist_mean": dist_stats["mean"],
                "grid_dist_std": dist_stats["std"],
                "grid_dist_max": dist_stats["max"],
                "grid_dist_min": dist_stats["min"],
                "grid_dist_count": dist_stats["count"],
                "img_u_deg": orient_stats["u_deg"],
                "img_v_deg": orient_stats["v_deg"]
            })            
            
            processed += 1

        except Exception as e:
            logger.exception('[EXCEPTION] run_on_image failed: %s', e)
            failed += 1

        finally:
            # 메모리 해제
            del gray, pts, diam
            gc.collect()
            logger.info('[INFO] Blob and Grid assignment done: processed=%d/%d, exception=%d, grid_success=%s, img_path=%s', processed, target_frames, bool(failed), bool(success_flag), str(img_path))
            
        # save_points 옵션이 활성화된 경우 전체 points를 하나의 NPZ로 저장
        # 현재 저장한 뒤 debugging 하는 프로세스 없음 (삭제 무관)
        if config.save_points and STATE.FRAME_DATA_LIST:
            npz_path = out_spool / 'all_points.npz'
            import pickle
            data = {
                'frames': STATE.FRAME_DATA_LIST,
                'num_frames': len(STATE.FRAME_DATA_LIST)
            }
            with open(str(npz_path).replace('.npz', '.pkl'), 'wb') as f:
                pickle.dump(data, f)
            if config.verbose:
                logger.info('[INFO] Saved %d frames to %s', len(STATE.FRAME_DATA_LIST), str(npz_path))
        
        
    # ------------------------------------------------------------------------------------------
    # Blob & Grid Summary report (HTML) 저장
    # ------------------------------------------------------------------------------------------
    def dict_to_table(title: str, data: dict) -> str:
        rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in data.items())
        return f"<h3>{title}</h3><table border='1' cellspacing='0' cellpadding='4'>{rows}</table>"

    def list_of_dicts_to_table(title: str, rows: List[dict]) -> str:
        if not rows:
            return f"<h3>{title}</h3><p>no data</p>"
        headers = list(rows[0].keys())
        header_html = "".join(f"<th>{h}</th>" for h in headers)
        body_html = ""
        for row in rows:
            cells = []
            for h in headers:
                v = row.get(h, "")
                if isinstance(v, float):
                    cells.append(f"{v:.3f}")
                else:
                    cells.append(str(v))
            body_html += "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
        return f"<h3>{title}</h3><table border='1' cellspacing='0' cellpadding='4'><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>"

    config_tables = [
        dict_to_table("BlobDetectorConfig", asdict(BLOB_CONFIG)),
        dict_to_table("GridConfig", asdict(GRID_CONFIG)),
        dict_to_table("AffineCandidateConfig", asdict(AFFINE_CANDIDATE_CONFIG)),
        dict_to_table("ScoringConfig", asdict(SCORE_CONFIG)),
        dict_to_table("TransportConfig", asdict(TRANSPORT_CONFIG)),
    ]

    frame_table = list_of_dicts_to_table("Frame stats", frame_stats)

    summary_html = "<html><body>" + "".join(config_tables) + frame_table + "</body></html>"
    summary_path = config.debug_dir / "Blob_Grid_Assignment_Summary.html"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(summary_html)
    
    # ------------------------------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------------------------------
    # 캘리브레이션 수행
    STATE.num_frames = len(STATE.FRAME_DATA_LIST)
    logger.info('[ENTER] Calibration start with %d frames', STATE.num_frames)

    if STATE.FRAME_DATA_LIST and STATE.image_size:
        try:
            if args.verbose:
                logger.info('[INFO] Starting calibration with %d frames', len(STATE.FRAME_DATA_LIST))
            
            # CalibResult를 RuntimeState에 저장
            logger.info('[BRANCH] calibrate_shared called (remove_outliers=%s)', str(config.remove_outliers))
            STATE.CALIB_RESULT = calibrate_shared(
                RuntimeState=STATE,
                remove_outliers=config.remove_outliers,
                outlier_threshold=config.outlier_threshold
            )
                                    
            # Reprojection error 체크
            calibration_success = (float(STATE.CALIB_RESULT.reprojected) <= 0.5)
            logger.info('[INFO] Reprojection error: %s', str(STATE.CALIB_RESULT.reprojected))

            # Reprojection error 체크 실패 시 remove_outliers=True 재실행, 재실행해도 큰 개선 없을 확률 높음.
            if not calibration_success and not config.remove_outliers:
                if config.save_debug:
                    logger.info('[GUIDE] Check debug results:')
                    logger.info(' 1) Blob params: min_area, min_fill, max_eccentricity, bin_threshold')
                    logger.info(' 2) Grid params: grid_assign_thr, dilation_thr, affine_score_thr, max_affine_mean_error')
                    logger.info(' 3) Use --remove_outliers with --outlier_threshold 2.0')
                else:
                    print("\n[FAIL] Calibration quality is poor. Would you like to save debug results? (y/n): ", end="")
                    response = input().strip().lower()
                    if response == 'y' or response == 'yes':
                        logger.info('[BRANCH] Re-running with debug enabled...')

                        # remove_outliers 옵션 활성화하여 재실행
                        logger.info('[BRANCH] calibrate_shared called (remove_outliers=%s)', str(config.remove_outliers))
                        STATE.CALIB_RESULT = calibrate_shared(
                            RuntimeState=STATE,
                            remove_outliers=True,
                            outlier_threshold=config.outlier_threshold
                        )
                        
                        # kept_indices를 기반으로 FRAME_DATA_LIST를 정렬/필터링
                        # kept_indices 는 outlier 제거 후 남은 프레임 인덱스 리스트임.
                        STATE = update_runtime_state_by_kept_indices(
                            RuntimeState=STATE,
                            kept_indices=STATE.CALIB_RESULT.kept_indices
                        )
                        
                        calibration_success = (STATE.reprojected <= 0.5)
                        logger.info('[INFO] Reprojection error with outlier removal: %s', str(STATE.CALIB_RESULT.reprojected))
                        if calibration_success:
                            print("[SUCCESS] Re-calibration completed successfully with remove_outliers enabled\n")
                            
                        else:
                            print("[FAIL] Re-calibration still failed. Please check debug results and adjust parameters.")                            
                    else:
                        logger.info('[LEAVE] Exiting without re-calibration(remove_outliers=True).')
            
            # outlier 제거 후 남은 프레임 인덱스 리스트를 기반으로 FRAME_DATA_LIST를 정렬/필터링
            STATE = update_runtime_state_by_kept_indices(
                RuntimeState=STATE,
                kept_indices=STATE.CALIB_RESULT.kept_indices
            )
            # 결과 변환 및 저장
            STATE = convert_and_update_runtime_state(
                RuntimeState=STATE,
                TRANSPORT_CONFIG=TRANSPORT_CONFIG,                
                verbose=config.verbose                
            )
            # 결과 저장 (실패하더라도 저장 및 디버깅)
            save_calibration_results_from_runtime_state(
                RuntimeState=STATE,
                output_dir=config.dst,
                debug_dir=config.debug_dir,
                save_error_plots_flag=config.save_error,                
                verbose=config.verbose
            )
                
            # DEBUG reprojection_report 재투영 리포트 생성
            if config.save_reproj_png:
                if config.verbose:
                    logger.info('[INFO] Generating reprojection reports for %d kept frames', len(STATE.CALIB_RESULT.kept_indices))
                
                for new_i, old_i in enumerate(STATE.CALIB_RESULT.kept_indices):
                    frame = STATE.FRAME_DATA_LIST[old_i]
                    # 이미지의 상대 경로 구조 생성 (src 기준)
                    try:
                        rel_path = frame.image_path.relative_to(config.src)
                        debug_subdir = rel_path.parent
                        debug_filename = rel_path.stem
                    except ValueError:
                        debug_subdir = frame.image_path.parent.name
                        debug_filename = frame.image_path.stem
                    
                    # 재투영 리포트 서브디렉터리 생성
                    reproj_subdir = out_reproj / debug_subdir
                    reproj_subdir.mkdir(parents=True, exist_ok=True)
                    
                    # 이미지 로드 (재투영 리포트용)
                    if config.save_reproj_png:
                        gray = cv2.imread(str(frame.image_path), cv2.IMREAD_GRAYSCALE)
                        if gray is not None:
                            reproj_path = reproj_subdir / f"{debug_filename}_reproj.png"
                            save_reprojection_report(
                                frame.image_path, gray, STATE.image_points_list[new_i], STATE.object_points_list[new_i],
                                STATE.CALIB_RESULT.camera_matrix, STATE.CALIB_RESULT.distortion,
                                STATE.CALIB_RESULT.rvecs[new_i], STATE.CALIB_RESULT.tvecs[new_i],
                                reproj_path, save_txt=True
                            )
                
                if config.verbose:
                    logger.info('[INFO] Reprojection reports saved to %s', str(out_reproj))
                                


        except Exception as e:
            logger.exception('[EXCEPTION] Calibration failed: %s', e)
        finally:
            logger.info('[LEAVE] Calibration done: %s', str(config.dst))
    
    else:
        logger.error('[FAIL] No successful frames found for calibration')
    
    return STATE


if __name__ == '__main__':
    STATE1 = run()
    # GUI나 외부에서 사용할 수 있도록 RuntimeState 반환
    # CLI 실행 시에는 성공 여부에 따라 exit code 반환
    reproj = getattr(STATE1, 'reprojected', None)
    # reprojected가 없거나 숫자가 아니면 실패로 간주
    if reproj is None or not isinstance(reproj, (int, float, np.floating)):
        exit_code = 1
    else:
        exit_code = 0 if reproj <= 0.5 else 1   

