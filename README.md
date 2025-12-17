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
- **출력 폴더** (`output_4M`): 캘리브레이션 결과가 저장될 폴더

#### 주요 옵션

```bash
# 디버그 이미지 저장 활성화
python -m calib_v3.main sample_imgs_KYICI_4M output_4M --save_debug

# 에러 플롯 저장 (json 에도 포함됨)
python -m calib_v3.main sample_imgs_KYICI_4M output_4M --save_error

# 상세 로그 출력
python -m calib_v3.main sample_imgs_KYICI_4M output_4M --verbose

# 도트 간격 설정 (um 단위)
python -m calib_v3.main sample_imgs_KYICI_4M output_4M_KY -dot_pitch_um 500 -blob_dia_in_px 25 --skip 10
python -m calib_v3.main sample_imgs_16M_lota_unit1 output_16M_lota_unit1_KY -dot_pitch_um 700 -blob_dia_in_px 35 --skip 10
python -m calib_v3.main sample_imgs_16M_lota_unit2 output_16M_lota_unit2_KY -dot_pitch_um 700 -blob_dia_in_px 35 --skip 10



#### Blob 검출 파라미터 조정

```bash
# 최소 영역 크기
python -m calib_v3.main sample_imgs_KYICI_4M output_4M -min_area 10.0

# 최소 채움 비율
python -m calib_v3.main sample_imgs_KYICI_4M output_4M -min_fill 0.5

# 최대 이심률
python -m calib_v3.main sample_imgs_KYICI_4M output_4M -max_ecc 0.8

# 이진화 임계값
python -m calib_v3.main sample_imgs_KYICI_4M output_4M -binarize_thd 127.0
```


### 출력 파일

- `calibration.json`: 카메라 내부/외부 파라미터 및 메타데이터
- `calibration_err_dots.json`: 프레임별 재투영 오차 정보
- `calibration.log`: 실행 로그
- `calibration_lut/`
  - `map_x.tiff`, `map_y.tiff`: Undistort LUT (TIFF 형식)
  - `map_x.bin`, `map_y.bin`: Undistort LUT (바이너리 형식)

### 디버그 출력 (--save_debug 옵션 사용 시)

- `debug/bin/`: 이진화된 이미지
- `debug/Blob_Success/`: 성공적으로 검출된 blob 오버레이
- `debug/Blob_Fails/`: 검출 실패한 이미지
- `debug/reprojection_reports/`: 재투영 리포트 (이미지 및 플롯)

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

## 참고

- 모든 옵션의 기본값은 `calib_v3/utils/types.py`의 `AppConfig` 클래스에서 관리
- 상세한 로그는 `output_4M/calibration.log` 파일에서 확인

