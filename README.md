# Stereo Calibration

English · 한국어 (Korean version below)

## Requirements

- **Python**: 3.7 or later (recommended: 3.9)

## Setup and run (Windows / other PCs)

So that this folder alone is enough to run the tool on another PC, we recommend creating a virtual environment (`.venv`) inside the folder.

**1. Create a virtual environment (first time only)** — Requires Python 3.7 or later (3.9 recommended) (tested with: Python 3.9.12)

```bat
cd stereo-calibration
py -3.9 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**2. Run**

- Double-click `run_stereo_calibration.bat`, or run it from a command prompt with arguments: `run_stereo_calibration.bat sample_imgs_KYICI_4M output_4M_KY --dot_pitch_um 500 --blob_dia_in_px 25`
- If you run it without arguments, it asks for the required values interactively.
- When it finishes, it shows the result code (Exit code) and the window stays open and waits.

**Python used by `run_stereo_calibration.bat` (search order)**

1. `.venv` in this folder (the environment created in step 1 above)
2. `.venv39` in the parent folder (shared lota_tools environment, the old method)
3. `py -3` or `python` on PATH

If none of the three is found, it prints how to create a virtual environment and exits.

On **Linux / macOS**, run it directly instead of using the bat file.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m calib_v3.main <input_image_folder> <output_folder>
```

## Example data

The example inputs and outputs are large, so they are not included in git. Download them from the shared folder, put them in this folder, and run.

- Shared folder: [kyg_rnd_opt_group SharePoint](https://kohyoung.sharepoint.com/:f:/s/kyg_rnd_opt_group/IgDWvBFS3ChETbErtMCnhAz2Afvzfe8ZOv6SDDnS94_GDss?e=N0KTJo)
- `apply_stereo_camera_remap/map/` (example remap LUT tiff files) is also excluded from git.

### 1. Required packages

```bash
pip install -r requirements.txt
```

### 2. Run example

Basic command:

```bash
python -m calib_v3.main sample_imgs_KYICI_4M output_4M_KY
```

#### Command structure

```bash
python -m calib_v3.main <input_image_folder> <output_folder>
```

- **Input image folder** (`sample_imgs_KYICI_4M`): folder that contains the images for calibration
  - Subfolder structure is supported (e.g. `_1/`, `_2/`, `_3/`, `_4/`)
  - Supported formats: `.bmp`, `.png`
  - !!!Important: images must be debayered mono8 (raw input causes an error during binarization)
- **Output folder** (`output_4M`): folder where the calibration results are saved
- **Validation code** Because of the current optics schedule, no code example is planned. Requirements will be provided instead.
- **Speed optimization** After the current C++ implementation, if it is still too slow, a new target will be made. The dot size and spacing will be increased.

#### Main options

```bash
# When the object sampling resolution mm/px from Camera Calibration is given
python -m calib_v3.main sample_imgs_KYICI_4M output_4M -r 10

# Enable saving of debug images (default: True)
python -m calib_v3.main sample_imgs_KYICI_4M output_4M --save_debug

# Save Transport error plot (also included in json, default: save_error)
python -m calib_v3.main sample_imgs_KYICI_4M output_4M --save_error

# Save reprojection report (default: True)
python -m calib_v3.main sample_imgs_KYICI_4M output_4M --save_reproj_png

# Save Remap LUT preview images (default: True)
python -m calib_v3.main sample_imgs_KYICI_4M output_4M --save_remap_preview

# Save point npz
python -m calib_v3.main sample_imgs_KYICI_4M output_4M --save_points

# Print detailed log
python -m calib_v3.main sample_imgs_KYICI_4M output_4M --verbose

# skip: skips images to run faster <-> trade-off with accuracy
# Set dot pitch and blob size: currently, setting only these two was confirmed to work for both target types
python -m calib_v3.main sample_imgs_KYICI_4M output_4M_KY --dot_pitch_um 500 --blob_dia_in_px 25 --skip 10
python -m calib_v3.main sample_imgs_16M_lota_unit1 output_16M_lota_unit1_KY --dot_pitch_um 700 --blob_dia_in_px 35 --skip 10
python -m calib_v3.main sample_imgs_16M_lota_unit2 output_16M_lota_unit2_KY --dot_pitch_um 700 --blob_dia_in_px 35 --skip 10
```

### Output files

- `calibration.json`: camera intrinsic/extrinsic parameters and metadata
- `calibration_lut_forward.json`: Forward LUT information (includes orthographic projection parameters)
- `calibration_lut_backward.json`: Backward LUT information (optional)
- `calibration_err_dots.json`: per-frame reprojection error information
- `calibration.log`: run log
- `calibration_lut/`
  - `map_x.tiff`, `map_y.tiff`: Undistort/Rectify LUT (TIFF format)
  - `map_x.bin`, `map_y.bin`: Undistort/Rectify LUT (binary format)
- `calibration_lut_backward/` (optional)
  - `map_x.tiff`, `map_y.tiff`: Backward LUT (TIFF format)
  - `map_x.bin`, `map_y.bin`: Backward LUT (binary format)

### Debug output (with the --save_debug option, default: True)

- `report/bin/`: binarized images
- `report/Blob_Success/`: overlay of successfully detected blobs
- `report/Blob_Fails/`: images where detection failed
- `report/Grid_Report/`: grid assignment report (point IDs and mesh visualization)
- `report/Grid_Path_Report/`: grid path report (spiral path and u/v axis arrows)
- `report/Blob_Grid_Assignment_Summary.html`: per-frame statistics summary (HTML table)
- `report/reprojection_reports/`: reprojection reports (images and plots, --save_reproj_png option)
- `report/*_remap_preview_*.png`: preview of the result of applying the Remap LUT (--save_remap_preview option, default: True)

## Example input image folder structure

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

## Main features

### 1. Blob detection and grid assignment
- Automatic blob detection (based on SimpleBlobDetector)
- Grid assignment based on a 3-dot pattern
- Grid traversal along a spiral path, and visualization

### 2. Camera Calibration
- OpenCV-based camera calibration
- Outlier removal option (`--remove_outliers`)
- Per-frame reprojection error analysis

### 3. LUT (Look-Up Table) generation
- Undistort/Rectify LUT generation
- Forward/Backward LUT support
- Crop/Expand policy support
- Automatic direction adjustment based on transport vectors

### 4. Debug visualization
- Grid mesh visualization
- Spiral path and u/v axis direction display
- Reprojection error visualization
- Remap result preview

## Notes

- Default values of all options are managed in the `AppConfig` class in `calib_v3/utils/types.py`
- See the detailed log in the `output_4M/report/calibration.log` file
- LUTs are saved in TIFF and binary formats and can be used with the `cv2.remap()` function
- Orthographic projection parameters are included in `calibration_lut_forward.json` (target_Z_um, resolution_at_target_Z_um, etc.)

---

# 한국어

## 요구사항

- **Python**: 3.7 이상 (권장: 3.9)

## 환경 설정과 실행 (Windows / 다른 PC)

다른 PC에서도 이 폴더만 받으면 실행할 수 있도록, 폴더 안에 가상환경(`.venv`)을 만드는 것을 권장합니다.

**1. 가상환경 만들기 (최초 1회)** — Python 3.7 (권장 3.9) 이상 필요 (검증 환경: Python 3.9.12)

```bat
cd stereo-calibration
py -3.9 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**2. 실행**

- `run_stereo_calibration.bat` 를 더블클릭하거나, 명령창에서 인자를 붙여 실행합니다: `run_stereo_calibration.bat sample_imgs_KYICI_4M output_4M_KY --dot_pitch_um 500 --blob_dia_in_px 25`
- 인자 없이 실행하면 필요한 값을 대화형으로 묻습니다.
- 끝나면 결과 코드(Exit code)를 표시하고 창이 닫히지 않고 대기합니다.

**`run_stereo_calibration.bat` 가 사용하는 Python (찾는 순서)**

1. 이 폴더의 `.venv` (위 1번에서 만든 환경)
2. 상위 폴더의 `.venv39` (lota_tools 공용 환경, 기존 방식)
3. PATH의 `py -3` 또는 `python`

셋 다 없으면 가상환경 만드는 방법을 출력하고 종료합니다.

**Linux / macOS** 는 bat 대신 직접 실행합니다.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m calib_v3.main <입력_이미지_폴더> <출력_폴더>
```

## 예제 데이터

예제 입력·출력은 용량이 커서 git에 포함하지 않습니다. 공유 폴더에서 받아 이 폴더에 두고 실행합니다.

- 공유 폴더: [kyg_rnd_opt_group SharePoint](https://kohyoung.sharepoint.com/:f:/s/kyg_rnd_opt_group/IgDWvBFS3ChETbErtMCnhAz2Afvzfe8ZOv6SDDnS94_GDss?e=N0KTJo)
- `apply_stereo_camera_remap/map/` (remap LUT 예제 tiff)도 git에서 제외되어 있습니다.

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
# Camera Calibration 에서 object sampleing resolution mm/px 가 주어진 경우
python -m calib_v3.main sample_imgs_KYICI_4M output_4M -r 10

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
```

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

