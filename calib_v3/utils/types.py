from __future__ import annotations

from dataclasses import dataclass, fields, replace, is_dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, TypeVar, Dict
import numpy as np

from .config import BlobDetectorConfig, AffineCandidateConfig, ScoringConfig, GridConfig, DebugConfig

T = TypeVar('T')

@dataclass(frozen=True)
class VersionEntry:
    date: str
    author: str
    note: str

VERSION_HISTORY: List[VersionEntry] = [
    VersionEntry("2025-12-07", "b.seong@kohyoung.com", "first version"),
    VersionEntry("2025-12-10", "b.seong@kohyoung.com", "Refactor calibration pipeline"),
    VersionEntry("2025-12-11", "b.seong@kohyoung.com", "Edit RuntimeState to include additional information and identical with AIT-ICI convention"),
    VersionEntry("2025-12-12", "b.seong@kohyoung.com", "Update backward LUT generation to include additional information"),
    VersionEntry("2025-12-14", "b.seong@kohyoung.com", "Add additaional _compute_per_view_errors to CalibResult to utilze opencv's calibrateCamera not extended version"),
    VersionEntry("2025-12-14", "b.seong@kohyoung.com", "Add base disparity and working distance info on LotaCalibrationResult.json")
]

CURRENT_VERSION: VersionEntry = VERSION_HISTORY[-1]
VERSION: str = CURRENT_VERSION.date + '-' + CURRENT_VERSION.author

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
    skip: int = 1 # 프레임 중 일부만 사용하여 고속화 할 때 사용. 1이면 모든 프레임을 처리. 10이면 10번째 프레임마다 1회 처리.

    #########################################
    # GridConfig
    #########################################           
    # 아래 파라미터는 캘리브레이션 타겟에 따라 반드시 변경되어야 합니다.        
    dot_pitch_um: float = 700 # blob 사이 거리 in um
    max_grid_size: int = 100 # grid 최대 크기가 N X N cells 인지. N 보다 max_grid_size 가 커야 안정적.

    #########################################
    # BlobDetectorConfig
    #########################################    
    # 아래 파라미터는 캘리브레이션 타겟에 따라 변경되어야 합니다.        
    blob_dia_in_px: float = 35 # 일반적으로 dot_pitch_um/pixel_resolution_um_per_px/2.0 로 계산될 수 있습니다 (실제 타겟 dimension)
    min_area: Optional[float] = None
    max_area: Optional[float] = None

    # 아래 파라미터는 조정이 필요할 수 있는 blob detector 파라미터입니다.
    binarize_thd: Optional[float] = None # if 0 or None, Otsu is used
    retrieval: str = 'list' # 'SBD' | 'external' | 'list', list is default

    #########################################
    # TransportConfig
    #########################################      
    # LUT settings
    # LUT 는 rectified 이미지를 생성하는 맵입니다.
    # 이미지 외곽 등에서 품질이 저하된다면 crop_margin 증가로 stereo 연산에 사용되는 이미지 크기를 줄이는 것도 방법입니다.
    lut_policy: str = 'crop'  # 'expand' | 'crop', crop is default
    lut_crop_margin: Optional[float] = 0.0 

    #########################################
    # 결과 저장 관련 파라미터
    #########################################      
    # Runtime/control flags
    save_debug: bool = True # 주요 디버깅
    save_reproj_png: bool = True # 디버깅에 큰 도움 안됨
    save_points: bool = False # 디버깅에 큰 도움 안됨
    save_error: bool = True # 각 이미지 간 물리적 거리, 필수
    verbose: bool = True
    
    # obj point 와 img point 쌍으로 camera calibration 시 outlier 제거를 위한 임계값 설정.
    # 현제 시스템에서는 사용하지 않는 것이 더 좋은 성능을 보임.
    remove_outliers: bool = False
    outlier_threshold: float = 2.0
    
    # opencv calibrateCamera 사용 시 초기 추정치 사용 여부
    use_guess: bool = False
    K_guess: Optional[np.ndarray] = None

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
    camera_matrix: np.ndarray # camera matrix from opencv
    distortion: np.ndarray # distortion coefficients from opencv
    rvecs: List[np.ndarray] # rotation vectors from opencv (number of views, 3(XYZ))
    tvecs: List[np.ndarray] # translation vectors from opencv (number of views, 3(XYZ))
    std_intrinsic: Optional[np.ndarray] # standard deviation of intrinsic parameters
    std_extrinsic: Optional[np.ndarray] # standard deviation of extrinsic parameters
    reprojected: float # RMS reprojection error
    per_view_errors: Optional[np.ndarray] # per view reprojection errors
    kept_indices: Optional[List[int]] = None # indices of kept frames if removed outliers


@dataclass
class RuntimeState:
    """실행 중 파생되는 상태 변수 집합(로그/보고용).

    사용처:
    - main.run: 해상도, LUT 크기/중심, 크롭 bbox, RMS 등 기록
    """
    # LotaCalibrationResult.json 산출물: camera calibration result    
    camera_matrix: Optional[np.ndarray] = None # camera matrix from opencv
    distortion: Optional[np.ndarray] = None # distortion coefficients from opencv
    resolution: Optional[float] = None # estimated resolution at center fiducial Z-position 
    reprojected: Optional[float] = None # RMS reprojection error, refer accuracy of calibration
    size: Optional[Tuple[int, int]] = None # native image size    
    cam_width: Optional[int] = None # native image width
    cam_height: Optional[int] = None # native image height    
    cam_focal: Optional[float] = None # focal length in pixels
    transport: Optional[Tuple[float, float, float]] = None # transport vector (x, y, z)    
    mean_Z_um: Optional[float] = None # mean Z-position of the center fiducial in um
    mean_disparity: Optional[dict] = None # mean disparity in predefined baseline (mm)

    # calibration_lut_forward.json 산출물: remap lut related
    cam_center_x: Optional[float] = None # LUT center x coordinate
    cam_center_y: Optional[float] = None # LUT center y coordinate
    map_width: Optional[int] = None # LUT width
    map_height: Optional[int] = None # LUT height
    map_x: Optional[np.ndarray] = None # LUT map for X
    map_y: Optional[np.ndarray] = None # LUT map for Y
    lut_info: Optional[dict] = None # LUT info, including map shape, cam center, crop bbox, did flip, policy, margin

    # Backward 정보 저장 할 시 LotaCalibrationResult.json 산출물
    transport_backward: Optional[Tuple[float, float, float]] = None # backward transport vector (x, y, z)    
    cam_center_x_backward: Optional[float] = None # backward LUT center x coordinate
    cam_center_y_backward: Optional[float] = None # backward LUT center y coordinate    

    # 저장 용도의 리스트와 딕셔너리
    CALIB_RESULT: Optional[CalibResult] = None # camera calibration result dataclass
    FRAME_DATA_LIST: List[FrameData] = field(default_factory=list) # 각 native image 프레임 별 image_points 와 object_points 저장
    image_size: Optional[Tuple[int, int]] = None # naitve 이미지 크기 저장
    num_frames: int = 0     # outlier 제거 후 남은 프레임 개수 저장
    frame_names_list: List[str] = field(default_factory=list) # outlier 제거 후 남은 프레임 이름 리스트 저장
    object_points_list: List[np.ndarray] = field(default_factory=list) # outlier 제거 후 남은 객체 포인트 리스트 저장
    image_points_list: List[np.ndarray] = field(default_factory=list) # outlier 제거 후 남은 이미지 포인트 리스트 저장
    folder_index_list: Dict[str, List[int]] = field(default_factory=dict) # outlier 제거 후 남은 폴더별 프레임 인덱스 매핑 저장

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

def update_runtime_state_by_kept_indices(RuntimeState: RuntimeState, kept_indices: List[int]) -> RuntimeState:
    if kept_indices is not None:
        kept = kept_indices
    else:
        kept = list(range(len(RuntimeState.FRAME_DATA_LIST)))

    frame_names_all = [f"{f.image_path.parent.name}/{f.image_path.name}" for f in RuntimeState.FRAME_DATA_LIST]
    RuntimeState.frame_names_list = [frame_names_all[i] for i in kept]
    RuntimeState.object_points_list = [RuntimeState.FRAME_DATA_LIST[i].object_points for i in kept]
    RuntimeState.image_points_list = [RuntimeState.FRAME_DATA_LIST[i].image_points for i in kept]
    
    # by_folder 필터링
    RuntimeState.folder_index_list = {}
    for new_idx, old_idx in enumerate(kept):
        folder_name = RuntimeState.FRAME_DATA_LIST[old_idx].image_path.parent.name
        RuntimeState.folder_index_list.setdefault(folder_name, []).append(new_idx)

    return RuntimeState