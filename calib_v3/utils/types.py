from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from .config import DetectorConfig, CandidateConfig, ScoringConfig, GridConfig, PatternConfig, DebugConfig


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
    
    # Blob detection parameters (CLI defaults)
    min_area: float = 5.0
    min_fill: float = 0.8
    max_eccentricity: float = 0.85
    max_area: Optional[float] = None
    retrieval: str = 'list'  # 'SBD' | 'external' | 'list'
    bin_threshold: Optional[float] = None
    
    # Pattern/camera scale
    dot_pitch_mm: float = 0.5
    
    # LUT settings
    lut_policy: str = 'crop'  # 'expand' | 'crop'
    lut_crop_margin: float = 0.0
    
    # Runtime/control flags
    save_debug: bool = True
    save_points: bool = True
    save_error: bool = False
    verbose: bool = True
    
    # Outlier removal
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
class DetectorParams:
    """실제 검출에 사용되는 확정 파라미터(런타임 병합 결과).

    - Optional, None이 없도록 값을 보유합니다.
    - 기본은 `DetectorConfig`에서 가져오고, `AppConfig`가 제공하는 값으로 덮어씁니다.
    """
    # 공통
    min_fill: float
    min_area: float
    max_area: float
    max_eccentricity: float
    retrieval: str
    bin_threshold: Optional[float]

    # FNN
    fnn_filter: int
    fnn_radius_beta: float
    fnn_local_k: int
    fnn_rollback_min_kept: int
    fnn_rollback_fraction: float

    # SBD 전용
    sbd_min_dist_between_blobs: float
    sbd_fixed_threshold: Optional[float] | str
    sbd_blob_color: Optional[int]
    sbd_min_repeatability: int
    sbd_band_halfwidth: float
    sbd_threshold_step: float


def build_detector_params(app: AppConfig, det_cfg: DetectorConfig) -> DetectorParams:
    """DetectorConfig 기본값과 AppConfig(사용자 입력)를 병합해 확정 파라미터 생성."""
    # AppConfig에 없으면 DetectorConfig 기본값 사용
    def pick(name: str):
        return getattr(app, name) if hasattr(app, name) and getattr(app, name) is not None else getattr(det_cfg, name)

    return DetectorParams(
        min_fill=float(pick('min_fill')),
        min_area=float(pick('min_area')),
        max_area=float(pick('max_area')),
        max_eccentricity=float(pick('max_eccentricity')),
        retrieval=str(pick('retrieval')),
        bin_threshold=getattr(app, 'bin_threshold', None),  # None이면 Otsu
        fnn_filter=int(getattr(app, 'fnn_filter', det_cfg.fnn_filter)),
        fnn_radius_beta=float(getattr(app, 'fnn_radius_beta', det_cfg.fnn_radius_beta)),
        fnn_local_k=int(getattr(app, 'fnn_local_k', det_cfg.fnn_local_k)),
        fnn_rollback_min_kept=int(getattr(app, 'fnn_rollback_min_kept', det_cfg.fnn_rollback_min_kept)),
        fnn_rollback_fraction=float(getattr(app, 'fnn_rollback_fraction', det_cfg.fnn_rollback_fraction)),
        sbd_min_dist_between_blobs=float(getattr(app, 'sbd_min_dist_between_blobs', det_cfg.sbd_min_dist_between_blobs)),
        sbd_fixed_threshold=getattr(app, 'sbd_fixed_threshold', det_cfg.sbd_fixed_threshold),
        sbd_blob_color=getattr(app, 'sbd_blob_color', det_cfg.sbd_blob_color),
        sbd_min_repeatability=int(getattr(app, 'sbd_min_repeatability', det_cfg.sbd_min_repeatability)),
        sbd_band_halfwidth=float(getattr(app, 'sbd_band_halfwidth', det_cfg.sbd_band_halfwidth)),
        sbd_threshold_step=float(getattr(app, 'sbd_threshold_step', det_cfg.sbd_threshold_step)),
    )


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
    resolution_mm_per_px: float = 0.0
    min_xy: Optional[Tuple[float, float]] = None
    max_xy: Optional[Tuple[float, float]] = None
    map_shape: Optional[Tuple[int, int]] = None  # (H,W)
    did_flip: bool = False
    crop_bbox: Optional[Tuple[int, int, int, int]] = None  # (y0,y1,x0,x1)
    cx_rect: Optional[float] = None
    cy_rect: Optional[float] = None


@dataclass
class UnifiedConfig:
    """ 통합 설정 클래스 """
    # Core modules
    detector: DetectorConfig
    candidate: CandidateConfig
    scoring: ScoringConfig
    grid: GridConfig
    pattern: PatternConfig
    debug: DebugConfig
    
    # Runtime settings
    app: AppConfig
    state: RuntimeState
    
    @classmethod
    def create_default(cls) -> 'UnifiedConfig':
        """기본 설정으로 UnifiedConfig 생성"""
        return cls(
            detector=DetectorConfig(),
            candidate=CandidateConfig(),
            scoring=ScoringConfig(),
            grid=GridConfig(),
            pattern=PatternConfig(),
            debug=DebugConfig(),
            app=AppConfig(src=Path('.'), dst=Path('.'), debug_dir=Path('.')),
            state=RuntimeState()
        )
