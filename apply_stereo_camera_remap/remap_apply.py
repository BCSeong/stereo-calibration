from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import tifffile as tiff


def _walk_subfolders(root: Path) -> List[Path]:
    """`root` 하위에서 이름이 `_`로 시작하는 서브폴더만 나열."""
    folders: List[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.startswith('_'):
            folders.append(p)
    return folders


def _load_map_pair(map_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """map_dir에서 map_x.tiff, map_y.tiff를 float32로 로드."""
    mx_path = map_dir / 'map_x.tiff'
    my_path = map_dir / 'map_y.tiff'
    if not mx_path.exists() or not my_path.exists():
        raise FileNotFoundError(f"Missing map files in {map_dir}: map_x.tiff or map_y.tiff")
    map_x = tiff.imread(str(mx_path)).astype(np.float32, copy=False)
    map_y = tiff.imread(str(my_path)).astype(np.float32, copy=False)
    if map_x.shape != map_y.shape:
        raise ValueError(f"map_x and map_y shape mismatch: {map_x.shape} vs {map_y.shape}")
    return map_x, map_y


def _remap_image(src_img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, interpolation: int, border_mode: int, border_value: float) -> np.ndarray:
    """cv2.remap을 사용해 src_img를 재배치. 채널 수/비트심도 보존."""
    dst = cv2.remap(src_img, map_x, map_y, interpolation=interpolation, borderMode=border_mode, borderValue=border_value)
    return dst


def _gather_images(folder: Path) -> List[Path]:
    """폴더에서 지원 포맷 이미지를 정렬된 리스트로 수집."""
    imgs: List[Path] = []
    imgs += sorted(folder.glob('*.bmp'))
    imgs += sorted(folder.glob('*.png'))
    imgs += sorted(folder.glob('*.tif'))
    imgs += sorted(folder.glob('*.tiff'))
    return imgs


def _read_image(path: Path) -> np.ndarray:
    """고정밀 유지: TIF는 tifffile로, 그 외는 OpenCV로 로드."""
    suffix = path.suffix.lower()
    if suffix in ('.tif', '.tiff'):
        arr = tiff.imread(str(path))
        return arr
    # OpenCV는 PNG/BMP 8/16 지원. 그대로 사용
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Failed to read image: {path}")
    return img


def _write_tiff(path: Path, arr: np.ndarray) -> None:
    """TIFF로 저장. dtype/부호/채널 보존."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(path), arr)


def _interp_code(name: str) -> int:
    if name == 'nearest':
        return cv2.INTER_NEAREST
    if name == 'cubic':
        return cv2.INTER_CUBIC
    return cv2.INTER_LINEAR


def _border_code(name: str) -> int:
    if name == 'replicate':
        return cv2.BORDER_REPLICATE
    if name == 'reflect':
        return cv2.BORDER_REFLECT
    if name == 'reflect101':
        return cv2.BORDER_REFLECT_101
    if name == 'wrap':
        return cv2.BORDER_WRAP
    return cv2.BORDER_CONSTANT


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser(description='Apply remap LUTs (reference vs ours) to dataset images')
    ap.add_argument('src', type=str, help='dataset root containing subfolders like _1, _2, ...')
    ap.add_argument('out', type=str, help='output root directory')
    ap.add_argument('--map_dir', type=str, required=True, help='directory containing map_x.tiff and map_y.tiff (e.g., reference)')
    ap.add_argument('--label', type=str, default='remap', help='label name for output subdir')
    ap.add_argument('--interp', type=str, default='linear', choices=['linear','nearest','cubic'], help='interpolation method')
    ap.add_argument('--border', type=str, default='constant', choices=['constant','replicate','reflect','reflect101','wrap'], help='border handling')
    ap.add_argument('--border_value', type=float, default=0.0, help='constant border value when border=constant')
    args = ap.parse_args(argv)

    src_root = Path(args.src)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    map_dir = Path(args.map_dir)
    map_x, map_y = _load_map_pair(map_dir)
    print(f"[map] size={map_x.shape[::-1]} (W,H)")

    interp = _interp_code(args.interp)
    border = _border_code(args.border)
    bval = float(args.border_value)

    folders = _walk_subfolders(src_root)
    if not folders:
        print('[warn] no subfolders found starting with "_"; applying to root')
        folders = [src_root]

    for sub in folders:
        imgs = _gather_images(sub)
        print(f"[info] folder {sub.name}: {len(imgs)} images")
        for img_path in imgs:
            src = _read_image(img_path)
            h_src, w_src = src.shape[:2]

            # mapA 적용 ------------------------------------------------------
            if (map_x.shape[0], map_x.shape[1]) != (h_src, w_src):
                # remap의 타깃 크기는 map 크기. src가 더 작거나 크더라도 remap은 src에서 샘플링함.
                pass
            remap = _remap_image(src, map_x, map_y, interp, border, bval)
            out = out_root / args.label / sub.name / f"{img_path.stem}.tiff"
            _write_tiff(out, remap)

    print('[done] remap finished')


if __name__ == '__main__':
    main()


