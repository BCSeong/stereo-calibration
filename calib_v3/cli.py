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
    ap.add_argument('src', type=str)
    ap.add_argument('dst', type=str)
    
    # blob params -------------------------------------------------------------
    ap.add_argument('-min_area', dest='min_area', type=float, default=default_config.min_area)
    ap.add_argument('-min_fill', dest='min_fill', type=float, default=default_config.min_fill)
    ap.add_argument('-max_ecc', dest='max_eccentricity', type=float, default=default_config.max_eccentricity)
    ap.add_argument('-max_area', dest='max_area', type=float, default=default_config.max_area)
    ap.add_argument('-retrieval', dest='retrieval', type=str, choices=['SBD','external','list'], default=default_config.retrieval)
    ap.add_argument('-binarize_thd', dest='bin_threshold', type=float, default=default_config.bin_threshold)
    
    # pattern/camera scale ---------------------------------------------------------
    ap.add_argument('-s', '--spacing', dest='dot_pitch_mm', type=float, default=default_config.dot_pitch_mm)
    
    # lut -----------------------------------------------------------
    ap.add_argument('--lut_policy', dest='lut_policy', type=str, choices=['crop', 'expand'], default=default_config.lut_policy)
    ap.add_argument('--lut_crop_margin', dest='lut_crop_margin', type=float, default=default_config.lut_crop_margin)
    
    # runtime/control ---------------------------------------------------------
    ap.add_argument('--save_debug', dest='save_debug', action='store_true', default=default_config.save_debug)
    ap.add_argument('--save_points', dest='save_points', action='store_true', default=default_config.save_points)    
    ap.add_argument('--save_error', dest='save_error', action='store_true', default=default_config.save_error)    
    
    # outlier removal ---------------------------------------------------------
    ap.add_argument('--remove_outliers', dest='remove_outliers', action='store_true', default=default_config.remove_outliers)
    ap.add_argument('--outlier_threshold', dest='outlier_threshold', type=float, default=default_config.outlier_threshold)
    
    ap.add_argument('--verbose', dest='verbose', action='store_true', default=default_config.verbose)    
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

