from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
from .utils.types import AppConfig


def build_argparser() -> argparse.ArgumentParser:
    """CLI 파서 생성 - 모든 default 값은 AppConfig에서 관리"""
    
    # AppConfig에서 default 값들 가져오기
    default_config = AppConfig(
        src=None, dst=None, debug_dir=None  # Required fields는 None으로 설정
    )
    
    ap = argparse.ArgumentParser(description='calib_v3 simple blob scan')
    
    # positional --------------------------------------------------------------
    ap.add_argument('src', type=str, help='source directory')
    ap.add_argument('dst', type=str, help='destination directory')
    
    # blob params -------------------------------------------------------------
    ap.add_argument('-blob_dia_in_px', dest='blob_dia_in_px', type=float, default=default_config.blob_dia_in_px, help='blob diameter in pixels, default is 37.5')
    ap.add_argument('-min_area', dest='min_area', type=float, default=default_config.min_area, help='minimum area of blob, if None, it is calculated from blob_dia_in_px')
    ap.add_argument('-max_area', dest='max_area', type=float, default=default_config.max_area, help='maximum area of blob, if None, it is calculated from blob_dia_in_px')
    ap.add_argument('-retrieval', dest='retrieval', type=str, choices=['SBD','external','list'], default=default_config.retrieval, help='retrieval method, default is "list"')
    ap.add_argument('-binarize_thd', dest='bin_threshold', type=float, default=default_config.binarize_thd, help='binarization threshold, default is "Otsu"')
    
    # pattern/camera scale ---------------------------------------------------------
    ap.add_argument('-s', '--spacing', dest='dot_pitch_um', type=float, default=default_config.dot_pitch_um, help='dot pitch in um')
    
    # lut -----------------------------------------------------------
    ap.add_argument('--lut_policy', dest='lut_policy', type=str, choices=['crop', 'expand'], default=default_config.lut_policy, help='lut policy, crop or expand')
    ap.add_argument('--lut_crop_margin', dest='lut_crop_margin', type=float, default=default_config.lut_crop_margin, help='lut crop margin, 0.0 means no additional crop')
    
    # runtime/control ---------------------------------------------------------
    ap.add_argument('--skip', dest='skip', type=int, default=default_config.skip, help='skip frames, 1 means all frames, 10 means every 10th frame')
    ap.add_argument('--save_debug', dest='save_debug', action='store_true', default=default_config.save_debug, help='save debug images')
    ap.add_argument('--save_points', dest='save_points', action='store_true', default=default_config.save_points, help='save points')    
    ap.add_argument('--save_error', dest='save_error', action='store_true', default=default_config.save_error, help='save error')    
    
    # outlier removal ---------------------------------------------------------
    ap.add_argument('--remove_outliers', dest='remove_outliers', action='store_true', default=default_config.remove_outliers, help='remove outliers, default is False')
    ap.add_argument('--outlier_threshold', dest='outlier_threshold', type=float, default=default_config.outlier_threshold, help='outlier threshold')
    
    ap.add_argument('--verbose', dest='verbose', action='store_true', default=default_config.verbose, help='verbose output')    
    return ap


def update_dataclass_from_namespace(dc_obj, ns) -> None:
    """Update dataclass instance from argparse Namespace.

    - Matches by exact field names
    - Only sets when value is not None
    - Leaves other fields untouched
    """
    if not is_dataclass(dc_obj):
        raise TypeError('dc_obj must be a dataclass instance')
    for f in fields(dc_obj):
        if hasattr(ns, f.name):
            v = getattr(ns, f.name)
            if v is not None:
                setattr(dc_obj, f.name, v)

