from __future__ import annotations

import traceback
from dataclasses import asdict
from pathlib import Path
from pprint import pprint
import numpy as np
import cv2
from typing import List

from .cli import build_argparser, update_dataclass_from_namespace
from .get_points.three_dots import ThreeDotDetector, GridPreset
from .utils.types import DetectorParams, build_detector_params
from .utils.logger import init_logger, get_logger
from .calibrate_points.calibrator import calibrate_shared
from .utils.reporting import save_all_results
from .utils.types import FrameData, AppConfig, RuntimeState
from .debug.visuals import save_detection_overlay, save_grid_report, save_reprojection_report
from .utils.config import DETECTOR


def _rerun_with_debug(config: AppConfig, frames: List[FrameData], state: RuntimeState, by_folder: dict) -> RuntimeState:
    """Debug 옵션으로 재실행
    
    Args:
        config: 앱 설정
        frames: 프레임 데이터 리스트
        state: 현재 실행 상태
        by_folder: 폴더별 프레임 인덱스 매핑
        
    Returns:
        RuntimeState: 업데이트된 실행 상태
    """
    try:
        if config.verbose:
            print("[INFO] Re-running calibration with debug enabled...")
        
        # FrameData에서 points 추출
        object_points_list = [f.object_points for f in frames]
        image_points_list = [f.image_points for f in frames]
        
        # Outlier 제거 권장 메시지
        if not config.remove_outliers:
            print("[RECOMMENDATION] Consider using --remove_outliers flag to automatically remove bad frames")
            print("Example: python -m calib_v3.main --remove_outliers --outlier_threshold 2.0")
        
        # 재캘리브레이션 (outlier 제거 포함)
        calib_result = calibrate_shared(
            object_points_list, image_points_list, state.image_size,
            remove_outliers=True,  # 강제로 outlier 제거 활성화
            outlier_threshold=config.outlier_threshold
        )
        
        # RuntimeState 업데이트
        state.K = calib_result.K
        state.dist = calib_result.dist
        state.rvecs = calib_result.rvecs
        state.tvecs = calib_result.tvecs
        
        # 결과 재저장 및 상태 업데이트
        frame_names = [f"{f.image_path.parent.name}/{f.image_path.name}" for f in frames]
        lut_info = save_all_results(
            config.dst, calib_result, frame_names, 
            object_points_list, image_points_list, state.image_size,
            frames=frames,
            by_folder=by_folder,
            save_error_plots_flag=config.save_error,
            lut_policy=config.lut_policy,
            lut_crop_margin=config.lut_crop_margin,
            verbose=config.verbose,
            state=state  # RuntimeState 전달
        )
        
        # LUT 정보로 RuntimeState 업데이트
        if lut_info:
            state.map_shape = lut_info.get('map_shape')
            state.did_flip = lut_info.get('did_flip', False)
            state.crop_bbox = lut_info.get('crop_bbox')
            state.cx_rect = lut_info.get('cx_rect')
            state.cy_rect = lut_info.get('cy_rect')
            state.min_xy = lut_info.get('min_xy')
            state.max_xy = lut_info.get('max_xy')
        
        if state.rms_reproj <= 0.5:
            print("[SUCCESS] Re-calibration completed successfully with debug enabled\n")
        else:
            print("[FAIL] Re-calibration still failed. Please check debug results and adjust parameters.")
        
        return state
            
    except Exception as e:
        print(f"[ERR] Re-run failed: {e}")
        traceback.print_exc()
        return state


def run(argv=None) -> RuntimeState:
    ap = build_argparser()
    args = ap.parse_args(argv)
    
    # AppConfig 생성 및 CLI 인자로 업데이트
    config = AppConfig(
        src=Path(args.src),
        dst=Path(args.dst),
        debug_dir=Path(args.dst) / 'debug'
    )
    
    # CLI 인자로 AppConfig 업데이트
    update_dataclass_from_namespace(config, args)
    
    # 로거 초기화 (파일: dst/calibration.log, 콘솔: INFO 이상)
    log_path = Path(config.dst) / 'calibration.log'
    logger = init_logger(log_path)
    logger.info('[ENTER] main.run')

    # 디렉터리 생성
    config.debug_dir.mkdir(parents=True, exist_ok=True)
    
    # 검출 파라미터 병합 생성
    det_params: DetectorParams = build_detector_params(config, DETECTOR)

    # 이미지 재귀 수집
    def iter_images_recursive(root):
        root_path = Path(root)  # str을 Path로 변환
        for p in sorted(root_path.rglob('*')):
            if p.is_file() and (p.suffix.lower() in ('.bmp', '.png')):
                yield p

    if config.verbose:
        logger.info('[CONFIG] %s', {'src': str(config.src), 'dst': str(config.dst), 'blob': {'min_area': float(det_params.min_area), 'min_fill': float(det_params.min_fill), 'max_ecc': float(det_params.max_eccentricity)}})

    # GridPreset with dot_pitch_mm
    grid_preset = GridPreset(dot_pitch_mm=config.dot_pitch_mm)
    det = ThreeDotDetector(det_params, grid_preset, config.debug_dir, debug_sample_rate=config.debug_sample_rate, debug_shard_size=config.debug_shard_size)

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
    state = RuntimeState()
    
    # 캘리브레이션용 points 수집 (types.py 구조 사용)
    frames: List[FrameData] = []
    debug_counter = 0  # main에서 통합 관리
    import gc    
    
    # ------------------------------------------------------------------------------------------
    # Blob scan
    # ------------------------------------------------------------------------------------------
    logger.info('[ENTER] Blob scan start, images=%d', len(list(iter_images_recursive(config.src))))
    for img_path in iter_images_recursive(config.src):
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
            
            # should_save 로직을 main에서 통합 관리
            debug_counter += 1
            should_save = config.save_debug # 필요시 N 개마다 디버그 저장 기능 추가
            
            # DEBUG 시각화 처리 
            if should_save:
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
                    
                    # Grid report 저장 (triplet 포함)
                    grid_report_path = debug_success_dir / f"{debug_filename}_grid.png"
                    save_grid_report(img_path, gray, pts, res.grid_assign, res.Tc, grid_report_path, res.chosen_triplet)
                else:
                    # 실패한 경우
                    out_png = debug_fails_dir / f"{debug_filename}_blobs.png"
                    save_detection_overlay(gray, pts, None, out_png, None, None)
            
            # 성공한 프레임의 points 수집 (grid_assign이 비어있지 않으면 성공)
            if res.grid_assign and len(res.grid_assign) > 0:
                # object points 생성 (실제 dot_pitch_mm 사용)
                grid_keys = list(res.grid_assign.keys())
                if grid_keys:
                    # 격자 좌표를 실제 월드 좌표로 변환 (dot_pitch_mm 사용)
                    uv = np.array(grid_keys, dtype=np.int32)
                    object_pts = np.zeros((len(uv), 3), dtype=np.float32)
                    object_pts[:, 0] = uv[:, 0] * float(grid_preset.dot_pitch_mm)  # X: mm
                    object_pts[:, 1] = uv[:, 1] * float(grid_preset.dot_pitch_mm)  # Y: mm
                    object_pts[:, 2] = 0.0  # Z: mm (평면)
                    
                    # FrameData 구조로 저장
                    frame_data = FrameData(
                        image_path=img_path,
                        image_points=pts[list(res.grid_assign.values())],
                        object_points=object_pts
                    )
                    frames.append(frame_data)
                    
                    # 이미지 크기 저장 (첫 번째 성공 프레임에서)
                    if state.image_size is None:
                        state.image_size = (gray.shape[1], gray.shape[0])  # (width, height)
            

            processed += 1
        except Exception as e:
            logger.exception('[EXCEPTION] run_on_image failed: %s', e)
            failed += 1
        finally:
            # 메모리 해제
            try:
                del gray, pts, diam
            except Exception:
                logger.warning('[WARN] del gray, pts, diam failed')
                pass
            gc.collect()

            # save_points 옵션이 활성화된 경우 전체 points를 하나의 NPZ로 저장
            if config.save_points and frames:
                npz_path = out_spool / 'all_points.npz'
                # inhomogeneous shape 문제 해결: pickle 방식으로 저장
                import pickle
                data = {
                    'frames': frames,
                    'num_frames': len(frames)
                }
                with open(str(npz_path).replace('.npz', '.pkl'), 'wb') as f:
                    pickle.dump(data, f)
                if config.verbose:
                    logger.info('[INFO] Saved %d frames to %s', len(frames), str(npz_path))
            
            logger.info('[LEAVE] Blob scan done: %s', {'processed': processed, 'failed': failed, 'dst': str(config.dst)})
    
    # ------------------------------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------------------------------
    # 캘리브레이션 수행
    state.num_frames = len(frames)
    logger.info('[ENTER] Calibration start with %d frames', state.num_frames)
    if frames and state.image_size:
        try:
            if args.verbose:
                logger.info('[INFO] Starting calibration with %d frames', len(frames))
            
            # FrameData에서 points 추출
            object_points_list = [f.object_points for f in frames]
            image_points_list = [f.image_points for f in frames]
            
            logger.info('[BRANCH] calibrate_shared called (remove_outliers=%s)', str(config.remove_outliers))
            calib_result = calibrate_shared(
                object_points_list, image_points_list, state.image_size,
                remove_outliers=config.remove_outliers,
                outlier_threshold=config.outlier_threshold
            )
            
            # CalibResult를 RuntimeState에 저장
            state.K = calib_result.K
            state.dist = calib_result.dist
            state.rvecs = calib_result.rvecs
            state.tvecs = calib_result.tvecs
            
            # 폴더별 프레임 인덱스 매핑 생성 (간단한 구현)
            by_folder = {}
            for i, frame in enumerate(frames):
                folder_name = frame.image_path.parent.name
                if folder_name not in by_folder:
                    by_folder[folder_name] = []
                by_folder[folder_name].append(i)
            
            # 통합된 결과 저장 및 분석 (reporting 모듈에서 모든 처리)
            # kept_indices를 기반으로 모든 리스트(by_folder, frame_names, points)를 정렬/필터
            kept = calib_result.kept_indices if getattr(calib_result, 'kept_indices', None) else list(range(len(frames)))
            frame_names_all = [f"{f.image_path.parent.name}/{f.image_path.name}" for f in frames]
            frame_names = [frame_names_all[i] for i in kept]
            object_points_list = [object_points_list[i] for i in kept]
            image_points_list = [image_points_list[i] for i in kept]
            # by_folder 필터링
            kept_set = set(kept)
            by_folder = {}
            for i, frame in enumerate(frames):
                if i in kept_set:
                    folder_name = frame.image_path.parent.name
                    by_folder.setdefault(folder_name, []).append(len(by_folder.get(folder_name, [])) + (0))
            # 위 by_folder는 새 인덱스가 아닌 누적 길이로 잘못될 수 있어, 실제 kept 인덱스의 상대 인덱스로 재구성
            by_folder = {}
            for new_idx, old_idx in enumerate(kept):
                folder_name = frames[old_idx].image_path.parent.name
                by_folder.setdefault(folder_name, []).append(new_idx)
            lut_info = save_all_results(
                config.dst, calib_result, frame_names, 
                object_points_list, image_points_list, state.image_size,
                frames=frames,
                by_folder=by_folder,
                save_error_plots_flag=config.save_error,
                lut_policy=config.lut_policy,
                lut_crop_margin=config.lut_crop_margin,
                verbose=config.verbose,
                state=state  # RuntimeState 전달
            )
            
            # LUT 정보로 RuntimeState 업데이트
            if lut_info:
                state.map_shape = lut_info.get('map_shape')
                state.did_flip = lut_info.get('did_flip', False)
                state.crop_bbox = lut_info.get('crop_bbox')
                state.cx_rect = lut_info.get('cx_rect')
                state.cy_rect = lut_info.get('cy_rect')
                state.min_xy = lut_info.get('min_xy')
                state.max_xy = lut_info.get('max_xy')
            
            # Reprojection error 체크 및 디버그 가이드
            calibration_success = (state.rms_reproj <= 0.5)
            if not calibration_success:
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
                        # Debug 옵션 활성화하여 재실행
                        config.save_debug = True
                        # 재실행 로직
                        state = _rerun_with_debug(config, frames, state, by_folder)
                        return state
                    else:
                        logger.info('[LEAVE] Exiting without debug information.')
                        return state
            # DEBUG 재투영 리포트 생성
            if should_save:
                try:
                    if config.verbose:
                        logger.info('[INFO] Generating reprojection reports for %d kept frames', len(kept))
                    
                    for new_i, old_i in enumerate(kept):
                        frame = frames[old_i]
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
                        gray = cv2.imread(str(frame.image_path), cv2.IMREAD_GRAYSCALE)
                        if gray is not None:
                            reproj_path = reproj_subdir / f"{debug_filename}_reproj.png"
                            save_reprojection_report(
                                frame.image_path, gray, image_points_list[new_i], object_points_list[new_i],
                                calib_result.K, calib_result.dist,
                                calib_result.rvecs[new_i], calib_result.tvecs[new_i],
                                reproj_path, save_txt=True
                            )
                    
                    if config.verbose:
                        logger.info('[INFO] Reprojection reports saved to %s', str(out_reproj))
                                
                except Exception as e:
                    logger.exception('[EXCEPTION] Reprojection report generation failed: %s', e)

        except Exception as e:
            logger.exception('[EXCEPTION] Calibration failed: %s', e)
        finally:
            logger.info('[LEAVE] Calibration done: %s', str(config.dst))
    
    else:
        logger.error('[FAIL] No successful frames found for calibration')
    
    return state


if __name__ == '__main__':
    state = run()
    # GUI나 외부에서 사용할 수 있도록 state 반환
    # CLI 실행 시에는 성공 여부에 따라 exit code 반환
    exit_code = 0 if state.rms_reproj <= 0.5 else 1
    raise SystemExit(exit_code)


