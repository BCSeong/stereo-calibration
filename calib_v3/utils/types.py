from __future__ import annotations

from dataclasses import dataclass, fields, replace, is_dataclass
from pathlib import Path
from typing import List, Optional, Tuple, TypeVar
import numpy as np

from .config import BlobDetectorConfig, AffineCandidateConfig, ScoringConfig, GridConfig, DebugConfig

T = TypeVar('T')


@dataclass
class FrameData:
    """get_points.detector → calibrate_points.calibrator 입력.
    get_points.detector에서 검출된 점들과 객체 점들을 저장하는 클래스

    사용처:
    - calibrate_shared: image_points/object_points로 캘리브레이션
    - main.run: 전체 프레임 리스트를 구성
    """
    image_path: Path
    image_points: Optional[np.ndarray]  # (N,2) or None if spooled
    object_points: Optional[np.ndarray]  # (N,3) or None if spooled
    points_file: Optional[Path] = None  # npz file when spooled


@dataclass
class CalibResult:
    """calibrate_points.calibrator: OpenCV 결과 번들.

    사용처:
    - main.run: std_intr/std_extr/per_view_errs 포함 모든 산출 이용
    """
    K: np.ndarray
    dist: np.ndarray
    rvecs: List[np.ndarray]
    tvecs: List[np.ndarray]
    std_intr: Optional[np.ndarray]
    std_extr: Optional[np.ndarray]
    per_view_errs: Optional[np.ndarray]


@dataclass
class AppConfig:
    """전체 파이프라인 실행 설정(단일 진입점).

    사용처:
    - main.run: CLI 인자 수집 및 하위 모듈로 전달(blob/grid/lut 등)
    - 모든 CLI default 값과 하드코딩된 설정을 중앙 관리
    """
    # Required paths
    src: Path 
    dst: Path
    debug_dir: Path
    skip: int = 10 # 프레임 중 일부만 사용하여 고속화 할 때 사용. 1이면 모든 프레임을 처리. 10이면 10번째 프레임마다 1회 처리.

    
    #########################################
    # BlobDetectorConfig
    #########################################    
    # 아래 파라미터는 캘리브레이션 타겟에 따라 변경되어야 합니다.        
    blob_dia_in_px = 37.5
    min_area: float = 0.75 * 3.14* (blob_dia_in_px/2)**2 if blob_dia_in_px is not None else 300.0 #  - 지름 20px와 동등: π*(9^2) ≈ 300        
    max_area: float = 4.00 * 3.14* (blob_dia_in_px/2)**2 if blob_dia_in_px is not None else 8000.0 #  - 지름 100px와 동등: π*(50^2) ≈ 8000

    # 아래 파라미터는 조정이 필요할 수 있는 blob detector 파라미터입니다.
    binarize_thd = None # if 0 or None, Otsu is used
    retrieval: str = 'list' # 'SBD' | 'external' | 'list', list is default

    
    #########################################
    # GridConfig
    #########################################           
    # 아래 파라미터는 캘리브레이션 타겟에 따라 반드시 변경되어야 합니다.        
    dot_pitch_um: float = 700 # blob 사이 거리 in um
    max_grid_size: int = 100 # grid 최대 크기가 N X N cells 인지. N 보다 max_grid_size 가 커야 안정적.


    #########################################
    # TransportConfig
    #########################################      
    # LUT settings
    # LUT 는 rectified 이미지를 생성하는 맵입니다.
    # 이미지 외곽 등에서 품질이 저하된다면 crop_margin 증가로 stereo 연산에 사용되는 이미지 크기를 줄이는 것도 방법입니다.
    lut_policy: str = 'crop'  # 'expand' | 'crop', crop is default
    lut_crop_margin: float = 0.0 

    #########################################
    # 결과 저장 관련 파라미터
    #########################################      
    # Runtime/control flags
    save_debug: bool = True
    save_points: bool = False
    save_error: bool = True
    verbose: bool = True
    
    # obj point 와 img point 쌍으로 camera calibration 시 outlier 제거를 위한 임계값 설정.
    # 현제 시스템에서는 사용하지 않는 것이 더 좋은 성능을 보임.
    remove_outliers: bool = False
    outlier_threshold: float = 1.0
    
    # Legacy settings (for compatibility)
    mode: str = 'fast'  # 'fast' | 'normal' | 'best'    
    debug_sample_rate: int = 0  # 0=off, N>0이면 N프레임마다 1회 저장
    debug_shard_size: int = 0   # 0=off, N>0이면 디버그를 shard_XXXX 하위로 분산
    errors: bool = True
    spool_points: bool = True
    use_guess: bool = False
    K_guess: Optional[np.ndarray] = None



@dataclass
class RuntimeState:
    """실행 중 파생되는 상태 변수 집합(로그/보고용).

    사용처:
    - main.run: 해상도, LUT 크기/중심, 크롭 bbox, RMS 등 기록
    """
    image_size: Optional[Tuple[int, int]] = None
    num_frames: int = 0
    K: Optional[np.ndarray] = None
    dist: Optional[np.ndarray] = None
    rvecs: Optional[List[np.ndarray]] = None
    tvecs: Optional[List[np.ndarray]] = None
    rms_reproj: float = 0.0
    transport: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    resolution_um_per_px: float = 0.0
    min_xy: Optional[Tuple[float, float]] = None
    max_xy: Optional[Tuple[float, float]] = None
    map_shape: Optional[Tuple[int, int]] = None  # (H,W)
    did_flip: bool = False
    crop_bbox: Optional[Tuple[int, int, int, int]] = None  # (y0,y1,x0,x1)
    cx_rect: Optional[float] = None
    cy_rect: Optional[float] = None


def merge_config_from_app(app: AppConfig, default_cfg: T) -> T:
    """범용 Config 병합 함수: AppConfig에서 Config를 업데이트.
    
    Args:
        app: AppConfig 인스턴스
        default_cfg: 기본 Config 인스턴스 (frozen dataclass도 지원)
    
    Returns:
        병합된 새로운 Config 인스턴스
    """
    if not is_dataclass(default_cfg):
        raise TypeError('default_cfg must be a dataclass instance')
    
    # 기본값으로 시작
    updates = {}
    
    # AppConfig에서 일치하는 필드 찾아서 업데이트
    for f in fields(default_cfg):
        if hasattr(app, f.name):
            app_value = getattr(app, f.name)
            # None이 아니면 업데이트
            if app_value is not None:
                updates[f.name] = app_value
    
    # replace를 사용하여 새로운 인스턴스 생성 (frozen dataclass도 지원)
    if updates:
        return replace(default_cfg, **updates)
    else:
        return default_cfg

