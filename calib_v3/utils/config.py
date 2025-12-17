from __future__ import annotations

import cv2
from dataclasses import dataclass


@dataclass(frozen=True)
class BlobDetectorConfig:
    """get_points.three_dots: 블롭/컨투어 검출 및 전처리 파라미터.

    사용처:
    - ThreeDotDetector._create_blob_detector / detect_blobs
    - get_points.detector (폴더 처리)
    """
    
    # -------------------------------------------------------------
    # 공통 blob detector 파라미터    
    blob_dia_in_px: float = 37.5 # 타겟에 따라 blob 크기 다를 수 있음
    min_area: float = None # 타겟에 따라 blob 크기 다를 수 있음
    max_area: float = None # 타겟에 따라 blob 크기 다를 수 있음
    min_fill: float = 0.66
    max_eccentricity: float = 0.66

    bin_threshold: float = None # if 0 or None, Otsu is used
    retrieval: str = "list"

    # -------------------------------------------------------------
    # FNN 옵션, 중앙 fiducail 검색 도구
    fnn_filter: int = 25 # fiducail point 인접 dot 개수 constraint
    fnn_grid_explore_size: int = 10 # 격자 공간에서 중앙 점을 중심으로 이 크기만큼의 격자를 탐색하여 최근접 이웃을 찾음
    # FNN 반경 = beta * local_d (local_d는 k-NN 기반 지역 스케일)
    fnn_radius_beta: float = 10.0 # 반경을 정의,  R(px) = fnn_radius_beta × local_d(px), 여기서 local_d는 k-NN(기본 k≈5)의 2번째 이웃 거리의 중앙값
    # 지역 스케일 산출용 k (자기 자신 포함). 작을수록 빠르고, 너무 작으면 불안정
    fnn_local_k: int = 5
    # FNN 필터가 과도하게 적용되어 남는 점이 너무 적을 때 롤백 기준
    fnn_rollback_min_kept: int = 9
    fnn_rollback_fraction: float = 0.25

    # -------------------------------------------------------------
    # retreival 옵션 SBD 에서 사용    
    sbd_min_dist_between_blobs: float = 1.50 * blob_dia_in_px
    sbd_fixed_threshold: int = 'otsu' # 'otsu' or int
    sbd_blob_color: int = 0 # None → opencv blobColor 미적용, 0: dark, 255: light
    sbd_min_repeatability: int = 1   # SBD는 단일 임계 사용 시 최소 1 권장
    sbd_band_halfwidth: float = 10.0 # Otsu/fixed 기준으로 좌우 스윕 폭
    sbd_threshold_step: float = 5.0  # 스윕 간격 (px)

    # -------------------------------------------------------------
    # retreival 옵션 list, external 에서 사용
    # 서브픽셀 정밀화 사용 여부 (컨투어 경로에서 moments+cornerSubPix 적용)
    enable_subpixel: bool = False
    # cornerSubPix 파라미터
    subpix_window: tuple = (5, 5)
    subpix_zero_zone: tuple = (-1, -1)
    subpix_criteria: tuple = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    
@dataclass(frozen=True)
class GridConfig:
    """get_points.three_dots: 격자 확장 반경 및 추정 방법.

    사용처:
    - grid_assign / dilate_with_local_refine (max_radius, use_lmeds)
    """    
    estimatation_affine_method_opencv: int = cv2.LMEDS # cv2.LMEDS or cv2.RANSAC both are acceptable
    dot_pitch_um: float = None # 타겟에 따라 blob 간격 다를 수 있음 
    max_grid_size: int = 100 # 격자 공간에서 (0,0)에서 최대 거리, 이보다 먼 좌표의 격자는 할당하지 않음, 타겟과 시스템의 FOV에 따라 총 격자 크기 다를 수 있음


@dataclass(frozen=True)
class AffineCandidateConfig:
    """get_points.three_dots: 중앙 3점 후보(집합 B) 분리 파라미터.

    사용처:
    - ThreeDotDetector.find_central_candidates (Step 3)
    """
    topk: int = 5 # 집합 B 후보를 찾기 위해 상위 topk 개의 점을 선택
    uplift_all_med: float = 1.10 # 모든 점의 크기 중앙값을 찾은 뒤 그보다 얼마나 큰 점을 thresholding하여 후보로 선택할지 결정
    ensure_top3_factor: float = 0.99 # 3번째로 큰 점을 찾은 뒤 이 비율보다 작은 점은 무시
    big_size_median_mult: float = 1.35 # 모든 점의 크기 중앙값대비 이 비율보다 작은 점은 무시
    neighbors_central: int = 24 # 각 중앙 후보 점(kidx) 주변에서 k24개의 최근접 이웃을 찾습니다. k24개 이웃 중 집합 B에 속하는 점이 정확히 2개인 경우만 유효한 triplet 후보로 선택합니다


@dataclass(frozen=True)
class ScoringConfig:
    """get_points.three_dots: 아핀/시드/그리드/팽창 스코어와 임계값.

    사용처:
    - _score_affine (Step 4), grid_assign (Step 5), dilate_with_local_refine (Step 6)
    """
    # choose_Tc의 _score_affine의 히트 판정 임계값.
    # - 아핀으로 맵핑된 점군 P에서 8-이웃 정수 격자까지의 거리 d가 이 값 이하일 때 “hit”로 간주.
    # - 단위: 격자 공간(1.0 = 도트 간격 1칸). 작을수록 더 엄격.
    # 더 큰 값으로 조정하면 더 많은 점이 affine 점수 계산에 포함됨 (누락 방지)
    affine_score_thr: float = 0.2

    # choose_Tc의 아핀 평가에서 선택된 최적 변환(best)의 평균 격자거리(mean_error) 상한. 
    # 의미: 아핀으로 맵핑된 점군 P에서 8-이웃 정수 격자까지의 히트들 거리 평균.
    # - 단위는 격자 공간(1.0 = 도트 간격 1칸)이며, 값이 작을수록 기준 그리드에 잘 정렬됨을 의미.
    # - 예: 0.05면 그리드 간격의 5% 이내 평균 오차만 허용. 더 크면 FAIL 처리.
    # 더 큰 값으로 조정하면 더 많은 프레임이 성공으로 처리됨 (누락 방지)
    max_affine_mean_error: float = 0.05

    # grid_assign의 시드 선택 시, (0,0)과 가장 가까운 점의 정수격자 스냅 거리 임계값. 
    # - P[cidx]가 round(P[cidx])와 떨어진 거리가 이 값(또는 grid_assign_thr와의 max)보다 크면 시드 부적합.
    # - 단위: 격자 공간.
    # 더 큰 값으로 조정하면 더 많은 프레임에서 시드 선택이 성공함
    seed_thr: float = 0.4 # 0.4 -> 0.8 (격자 좌표계에서 더 관대하게)

    # grid_assign에서 인접 정수격자 키에 점을 배정할 때 허용되는 최대 거리.
    # - 거리 ≤ grid_assign_thr일 때만 배정. 단위: 격자 공간.
    # - 더 큰 값으로 조정하면 더 많은 점이 grid에 할당됨 (누락 방지)
    grid_assign_thr: float = 0.4 # 0.4 -> 0.8 (격자 좌표계에서 더 관대하게)

    # dilate_with_local_refine에서 지역 아핀으로 맵핑된 점의 배정 허용 거리.
    # - 지역 정합으로 업데이트된 Pn과 정수 키 사이 거리 ≤ dilation_thr일 때 확장 배정. 단위: 격자 공간.
    # - 더 큰 값으로 조정하면 더 많은 점이 확장 단계에서 포함됨
    # - 이 값이 커지면 jump 발생 확률 증가
    dilation_thr: float = 0.4 # 0.5 -> 0.6 (격자 좌표계에서 더 관대하게) 

    # 지역 아핀 재적합 후 잔차(residual)에 대한 3σ(표준편차) 제거에서 σ 계수.
    # - keep 조건: resid ≤ mean + sigma_prune * std. 값이 클수록 느슨한 제거.
    sigma_prune: float = 6.0

    # periodic refit을 수행하기 위한 최소 시드 페어 수.
    # - uv/idx 쌍이 이 값 이상일 때만 LMEDS로 아핀 재적합 및 outlier pruning 수행.
    min_seed_pairs: int = 6


@dataclass(frozen=True)
class DebugConfig:
    """debug.visuals 및 get_points.three_dots 오버레이 스타일.

    사용처:
    - save_detection_overlay, save_grid_report, plot_series 등 시각화
    """
    point_radius: int = 10
    point_thickness: int = 2
    label_font_scale: float = 3.0
    label_thickness: int = 4
    line_thickness: int = 2
    grid_circle_radius: int = 4
    arrow_scale: float = 10.0
    plot_figsize: tuple = (10, 4)
    plot_dpi: int = 150
    
    # 색상 설정 (BGR 형식)
    point_color: tuple = (0, 255, 0)         # 일반 점들 색상 (초록)
    triplet_i_color: tuple = (0, 0, 255)     # triplet center (i) 색상 (빨강)
    triplet_iplus_color: tuple = (0, 255, 255)  # triplet (i+) 색상 (노랑)
    triplet_iminus_color: tuple = (255, 0, 0)   # triplet (i-) 색상 (파랑)
    grid_color: tuple = (0, 255, 0)          # grid 점들 색상 (초록)

    reproj_actual_color: tuple = (0, 255, 0) # 재투영 실제 점 색상 (초록)
    reproj_projected_color: tuple = (0, 0, 255)  # 재투영된 점 색상 (빨강)
    reproj_error_color: tuple = (0, 255, 255)  # 재투영 에러 벡터 색상 (노랑)

    # Grid 에러 화살표 강조 설정
    grid_error_thr: float = 3.0  # 격자 공간 단위(1.0 = 도트 간격 1칸)
    grid_error_color: tuple = (0, 0, 255)  # 큰 에러 벡터 색상 (빨강)
    grid_error_thickness: int = 2  # 큰 에러 벡터 두께


@dataclass(frozen=True)
class TransportConfig:
    """main.run: 이동 벡터 출력 좌표계 부호 설정.
    기본 좌표계는 OpenCV 관례 가정:
    - X: +는 이미지 오른쪽(right) 방향
    - Y: +는 이미지 아래(down) 방향
    - Z: +는 카메라 전방(forward) 방향

    사용처:
    - main.run 내 transport 계산 후 axis_sign 적용
    예) (1,-1,1)이면 Y축 부호를 반전해 새로운 규약(Y up)을 제공 할 수 있음.
    """
    baseline_mm: float = 0.8*8 # predefined baseline in mm for calculating mean disparity

    axis_sign: tuple = (1.0, 1.0, 1.0)
    # LUT 적용 이후 프레임 증가(시간 진행)에 따라 배경 disparity는 X+ 방향(오른쪽)으로 발생해야 함.
    # transport vector 의 max component 가 Y-axis 인 경우 Y-axis 와 X-axis 를 서로 바꿈 (swap).
    # trasnport[max_component] 가 axis_sign[max_component] 와 부호가 다른 경우 LUT가 플립을 적용해 disparity 가 X+ 으로 발생하게 보정함.
    # !! 최종적으로 rectified (LUT가 적용된) 이미지는 frame 증가에 따라 이미지 feature 가 왼쪽에서 오른쪽으로 움직여야함.    
    # 이 정책 토글(하드코딩 기준) 관리:
    hflip_on_negative_mean_trel_x: bool = True

    # LUT 정책 설정
    # LUT 는 rectified 이미지를 생성하는 맵입니다.
    # 이미지 외곽 등에서 품질이 저하된다면 crop_margin 증가로 stereo 연산에 사용되는 이미지 크기를 줄이는 것도 방법입니다.
    lut_policy: str = 'crop' # 'expand' | 'crop', crop is default
    lut_crop_margin: float = 0.0
    


