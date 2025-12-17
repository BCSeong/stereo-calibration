## 요구사항

- **Python**: 3.7 이상 (권장: 3.9)

### 1. 필수 패키지

```bash
pip install -r requirements.txt
```

### 2. 실행 예제

기본 실행 명령어:

```bash
python -m calib_v3.main sample_imgs_KYICI_4M output_4M_KY
```

#### 명령어 구조

```bash
python -m calib_v3.main <입력_이미지_폴더> <출력_폴더>
```

- **입력 이미지 폴더** (`sample_imgs_KYICI_4M`): 캘리브레이션용 이미지가 포함된 폴더
  - 하위 폴더 구조 지원 (예: `_1/`, `_2/`, `_3/`, `_4/`)
  - 지원 형식: `.bmp`, `.png`
  - !!!중요 : 이미지는 debayer mono8 필수 (raw 입력하면 이진화 시 에러 발생)
- **출력 폴더** (`output_4M`): 캘리브레이션 결과가 저장될 폴더
- **Validation 코드** 현재 광학 일정 문제로 코드 예시는 제공 예정 없음. 요구사항 전달 예정.
- **속도 최적화** 현재 cpp 로 구현 뒤 속도가 너무 느리다면 target 을 신규 제작 할 예정임. dot 크기와 spacing 을 키울 예정

#### 주요 옵션

```bash
# 디버그 이미지 저장 활성화 (기본값: True)
python -m calib_v3.main sample_imgs_KYICI_4M output_4M --save_debug

# Transport 에러 플롯 저장 (json 에도 포함됨, 기본값: save_error)
python -m calib_v3.main sample_imgs_KYICI_4M output_4M --save_error

# 재투영 리포트 저장 (기본값: True)
python -m calib_v3.main sample_imgs_KYICI_4M output_4M --save_reproj_png

# Remap LUT 미리보기 이미지 저장 (기본값: True)
python -m calib_v3.main sample_imgs_KYICI_4M output_4M --save_remap_preview

# point npz 저장
python -m calib_v3.main sample_imgs_KYICI_4M output_4M --save_points

# 상세 로그 출력
python -m calib_v3.main sample_imgs_KYICI_4M output_4M --verbose

# skip 기능은 이미지 스킵하여 고속화 <-> 정밀도 타협
# 도트 간격 및 blob 크기 설정 : 현재는 두개만 설정하면 2종류의 타겟 모두 성공 확인
python -m calib_v3.main sample_imgs_KYICI_4M output_4M_KY --dot_pitch_um 500 --blob_dia_in_px 25 --skip 10
python -m calib_v3.main sample_imgs_16M_lota_unit1 output_16M_lota_unit1_KY --dot_pitch_um 700 --blob_dia_in_px 35 --skip 10
python -m calib_v3.main sample_imgs_16M_lota_unit2 output_16M_lota_unit2_KY --dot_pitch_um 700 --blob_dia_in_px 35 --skip 10



### 출력 파일

- `calibration.json`: 카메라 내부/외부 파라미터 및 메타데이터
- `calibration_lut_forward.json`: Forward LUT 정보 (정사투영 파라미터 포함)
- `calibration_lut_backward.json`: Backward LUT 정보 (선택적)
- `calibration_err_dots.json`: 프레임별 재투영 오차 정보
- `calibration.log`: 실행 로그
- `calibration_lut/`
  - `map_x.tiff`, `map_y.tiff`: Undistort/Rectify LUT (TIFF 형식)
  - `map_x.bin`, `map_y.bin`: Undistort/Rectify LUT (바이너리 형식)
- `calibration_lut_backward/` (선택적)
  - `map_x.tiff`, `map_y.tiff`: Backward LUT (TIFF 형식)
  - `map_x.bin`, `map_y.bin`: Backward LUT (바이너리 형식)

### 디버그 출력 (--save_debug 옵션 사용 시, 기본값: True)

- `report/bin/`: 이진화된 이미지
- `report/Blob_Success/`: 성공적으로 검출된 blob 오버레이
- `report/Blob_Fails/`: 검출 실패한 이미지
- `report/Grid_Report/`: 그리드 할당 리포트 (점 ID 및 mesh 시각화)
- `report/Grid_Path_Report/`: 그리드 경로 리포트 (나선 경로 및 u/v 축 화살표)
- `report/Blob_Grid_Assignment_Summary.html`: 프레임별 통계 요약 (HTML 테이블)
- `report/reprojection_reports/`: 재투영 리포트 (이미지 및 플롯, --save_reproj_png 옵션)
- `report/*_remap_preview_*.png`: Remap LUT 적용 결과 미리보기 (--save_remap_preview 옵션, 기본값: True)

## 입력 이미지 폴더 구조 예시

```
sample_imgs_KYICI_4M/
├── _1/
│   ├── img00000.png
│   ├── img00001.png
│   └── ...
├── _2/
│   ├── img00000.png
│   └── ...
├── _3/
│   └── ...
└── _4/
    └── ...
```

## 주요 기능

### 1. Blob 검출 및 그리드 할당
- 자동 blob 검출 (SimpleBlobDetector 기반)
- 3-dot 패턴 기반 그리드 할당
- 나선 경로를 통한 그리드 순회 및 시각화

### 2. Camera Calibration
- OpenCV 기반 카메라 캘리브레이션
- Outlier 제거 옵션 (`--remove_outliers`)
- 프레임별 재투영 오차 분석

### 3. LUT (Look-Up Table) 생성
- Undistort/Rectify LUT 생성
- Forward/Backward LUT 지원
- Crop/Expand 정책 지원
- Transport vector 기반 자동 방향 조정

### 4. 디버그 시각화
- 그리드 mesh 시각화
- 나선 경로 및 u/v 축 방향 표시
- 재투영 에러 시각화
- Remap 결과 미리보기

## 참고

- 모든 옵션의 기본값은 `calib_v3/utils/types.py`의 `AppConfig` 클래스에서 관리
- 상세한 로그는 `output_4M/report/calibration.log` 파일에서 확인
- LUT는 TIFF 및 바이너리 형식으로 저장되며, `cv2.remap()` 함수로 사용 가능
- 정사투영 파라미터는 `calibration_lut_forward.json`에 포함됨 (target_Z_um, resolution_at_target_Z_um 등)

