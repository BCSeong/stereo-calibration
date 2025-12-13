from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import cv2
from .logger import get_logger


def compute_transport_vector(
    tvecs: List[np.ndarray], 
    by_folder: Dict[str, List[int]],
    transport_axis_sign: Tuple[float, float, float]
) -> List[float]:
    """Transport vector 계산 (calib_v2와 동일한 로직)
    
    Args:
        tvecs: 각 프레임의 translation vector 리스트
        by_folder: 폴더별 프레임 인덱스 매핑
        
    Returns:
        transport: 정규화된 transport vector [x, y, z]
    """
    trs = []
    for idxs in by_folder.values():
        for a, b in zip(idxs[:-1], idxs[1:]):
            va = np.array(tvecs[b]).ravel() - np.array(tvecs[a]).ravel()
            n = np.linalg.norm(va)
            if n > 0:
                trs.append(va / n)
    
    if trs:
        transport_vec = np.mean(np.vstack(trs), axis=0)
        n = np.linalg.norm(transport_vec)
        transport_vec = (transport_vec / n) if n > 0 else transport_vec
        sign = np.array(transport_axis_sign, dtype=np.float64)
        transport = (transport_vec * sign).tolist()
    else:
        transport = [0.0, 0.0, 0.0]
    
    return transport


def compute_resolution_mm_per_px(
    K: np.ndarray, 
    tvecs: List[np.ndarray]
) -> float:
    """Resolution 계산 (mm/px) (calib_v2와 동일한 로직)
    
    Args:
        K: 카메라 내부 파라미터 행렬
        tvecs: 각 프레임의 translation vector 리스트
        
    Returns:
        resolution_mm_per_px: 픽셀당 mm 단위
    """
    fx = float(K[0, 0])
    z_vals = [float(tvecs[i].ravel()[2]) for i in range(len(tvecs))]
    mean_Z = float(np.mean(z_vals)) if z_vals else 0.0
    resolution_mm_per_px = (mean_Z / fx) if fx > 0 else 0.0
    return float(resolution_mm_per_px)

# deprecated, instead use compute_relative_transforms_without_rotation
def compute_relative_transforms(
    rvecs: List[np.ndarray], 
    tvecs: List[np.ndarray], 
    by_folder: Dict[str, List[int]]
) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """상대 변환 계산 (calib_v2와 동일한 로직)
    
    b.seong comment: 작은 회전행렬의 오차가 translation error 를 크게 증폭시킬 위험이 있음.
    수학적으로는 motion stage의 실제 이동량을 대변함.
    Args:
        rvecs: 각 프레임의 rotation vector 리스트
        tvecs: 각 프레임의 translation vector 리스트
        by_folder: 폴더별 프레임 인덱스 매핑
        
    Returns:
        by_folder_series: 폴더별 상대 변환 시퀀스
    """
    by_folder_series = {}
    for key, idxs in by_folder.items():
        seq = []
        for a, b in zip(idxs[:-1], idxs[1:]):
            Ri, _ = cv2.Rodrigues(rvecs[a])
            Rj, _ = cv2.Rodrigues(rvecs[b])
            ti = tvecs[a].reshape(3, 1)
            tj = tvecs[b].reshape(3, 1)
            Rrel = Rj @ Ri.T
            trel = (tj - Rj @ Ri.T @ ti).ravel()
            seq.append((Rrel, trel))
        by_folder_series[key] = seq
    return by_folder_series

# deprecated, instead use compute_relative_transforms_without_rotation
def compute_relative_transforms_world(
    rvecs: List[np.ndarray],
    tvecs: List[np.ndarray],
    by_folder: Dict[str, List[int]]
) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """상대 변환 계산 (월드/타겟 좌표계 기준)

    - 카메라 중심 C = -R^T t (월드 좌표계) 사용
    - ΔC = C_j - C_i 를 translation으로 사용
    - 회전은 Rrel_world = R_i^T R_j

    b.seong comment: 칼리브레이션 타겟의 XYZ 좌표와 motion stage의 XYZ좌표간의 변환행렬을 알 수 없으므로
    이 결과로 나오는 XYZ 의 상대이동량으로 실제 motion stage의 직선성을 판단할 순 없음.
    더욱이 회전행렬의 작은 오차가 traslation error 를 크게 증폭시킬 위험이 있음.
    """
    by_folder_series = {}
    for key, idxs in by_folder.items():
        seq = []
        for a, b in zip(idxs[:-1], idxs[1:]):
            Ri, _ = cv2.Rodrigues(rvecs[a])
            Rj, _ = cv2.Rodrigues(rvecs[b])
            ti = tvecs[a].reshape(3, 1)
            tj = tvecs[b].reshape(3, 1)
            Ci = -Ri.T @ ti
            Cj = -Rj.T @ tj
            dC = (Cj - Ci).ravel()
            Rrel_world = Ri.T @ Rj
            seq.append((Rrel_world, dC))
        by_folder_series[key] = seq
    return by_folder_series


def compute_relative_transforms_without_rotation(
    rvecs: List[np.ndarray],
    tvecs: List[np.ndarray],
    by_folder: Dict[str, List[int]]
) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """회전을 무시하고 순수 tvec 차분만 사용하는 상대 변환.

    - trel_naive = tj - ti (카메라 좌표계가 다름에도 불구하고 그대로 차분)
    - Rrel은 항등행렬을 반환

    b.seong comment: 회전을 무시하고 순수 tvec 차분만 사용하는 상대 변환.
    - trel_naive = tj - ti (카메라 좌표계가 다름에도 불구하고 그대로 차분)
    - Rrel은 항등행렬을 반환
    각 frame 간 카메라의 회전이 없음을 가정함.
    """
    by_folder_series = {}
    I = np.eye(3, dtype=np.float64)
    for key, idxs in by_folder.items():
        seq = []
        for a, b in zip(idxs[:-1], idxs[1:]):
            ti = np.asarray(tvecs[a], dtype=np.float64).reshape(3)
            tj = np.asarray(tvecs[b], dtype=np.float64).reshape(3)
            trel = tj - ti
            seq.append((I, trel))
        by_folder_series[key] = seq
    return by_folder_series

def plot_series(
    rel_list: Dict[str, List[Tuple[np.ndarray, np.ndarray]]], 
    out_dir: Path
) -> None:
    """상대 변환 시리즈 플롯 생성 (calib_v2와 동일한 로직)"""
    try:
        logger = get_logger()
        logger.info('[ENTER] plot_series -> %s', str(out_dir))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # translations
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        comps = [(0, 'X'), (1, 'Y'), (2, 'Z')]
        
        for ax_idx, (c, name) in enumerate(comps):
            ax = axes[ax_idx]
            for folder_name, seq in rel_list.items():
                if not seq:
                    continue
                steps = [float(np.linalg.norm(t)) for (_, t) in seq]
                xs = np.cumsum(steps)
                # Matplotlib은 '_'로 시작하는 레이블을 legend에서 무시하므로 안전 라벨 적용
                safe_label = folder_name if not str(folder_name).startswith('_') else f" {folder_name}"
                ax.plot(xs, [t[c] for (_, t) in seq], 'o-', label=safe_label, markersize=3)
            ax.set_ylabel(f'Translation {name} (mm)')
            ax.grid(True, alpha=0.3)
            if ax.get_legend_handles_labels()[0]:  # legend가 있을 때만 추가
                ax.legend()
        
        axes[-1].set_xlabel('Cumulative Distance (mm)')
        plt.tight_layout()
        plt.savefig(out_dir / 'translation_series.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # rotations
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        for ax_idx, (c, name) in enumerate(comps):
            ax = axes[ax_idx]
            for folder_name, seq in rel_list.items():
                if not seq:
                    continue
                steps = [float(np.linalg.norm(t)) for (_, t) in seq]
                xs = np.cumsum(steps)
                # Rodrigues vector에서 각도(도) 및 부호 추출 (부호는 rvec의 해당 성분 기준)
                angles_signed = []
                for (R, _) in seq:
                    rvec, _ = cv2.Rodrigues(R)
                    r = rvec.reshape(-1)
                    angle_deg = float(np.linalg.norm(r)) * 180.0 / np.pi
                    sgn = 1.0 if float(r[c]) >= 0.0 else -1.0
                    angles_signed.append(angle_deg * sgn)
                safe_label = folder_name if not str(folder_name).startswith('_') else f" {folder_name}"
                ax.plot(xs, angles_signed, 'o-', label=safe_label, markersize=3)
            ax.set_ylabel(f'Rotation {name} (deg)')
            ax.grid(True, alpha=0.3)
            if ax.get_legend_handles_labels()[0]:  # legend가 있을 때만 추가
                ax.legend()
        
        axes[-1].set_xlabel('Cumulative Distance (mm)')
        plt.tight_layout()
        plt.savefig(out_dir / 'rotation_series.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger = get_logger()
        logger.exception('Error creating series plots: %s', e)


def plot_series_with_prefix(
    rel_list: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    out_dir: Path,
    prefix: str = 'camera',
    apply_abs: bool = False
) -> None:
    """상대 변환 시리즈 플롯 생성 (파일 접두어 지정)"""
    try:
        logger = get_logger()
        logger.info('[ENTER] plot_series_with_prefix(%s) -> %s', prefix, str(out_dir))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # translations
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        comps = [(0, 'X'), (1, 'Y'), (2, 'Z')]
        for ax_idx, (c, name) in enumerate(comps):
            ax = axes[ax_idx]
            for folder_name, seq in rel_list.items():
                if not seq:
                    continue
                steps = [float(np.linalg.norm(t)) for (_, t) in seq]
                xs = np.cumsum(steps)
                vals = []
                for (_, t) in seq:
                    v = float(t[c])
                    if apply_abs:
                        v = abs(v)
                    vals.append(v)
                safe_label = folder_name if not str(folder_name).startswith('_') else f" {folder_name}"
                ax.plot(xs, vals, 'o-', label=safe_label, markersize=3)
            ax.set_ylabel(f'Translation {name} (mm)')
            ax.grid(True, alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend()
        axes[-1].set_xlabel('Cumulative Distance (mm)')
        plt.tight_layout()
        plt.savefig(out_dir / f'{prefix}_translation_series.png', dpi=150, bbox_inches='tight')
        plt.close()

        # rotations
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        for ax_idx, (c, name) in enumerate(comps):
            ax = axes[ax_idx]
            for folder_name, seq in rel_list.items():
                if not seq:
                    continue
                steps = [float(np.linalg.norm(t)) for (_, t) in seq]
                xs = np.cumsum(steps)
                angles_signed = []
                for (R, _) in seq:
                    rvec, _ = cv2.Rodrigues(R)
                    r = rvec.reshape(-1)
                    angle_deg = float(np.linalg.norm(r)) * 180.0 / np.pi
                    sgn = 1.0 if float(r[c]) >= 0.0 else -1.0
                    val = angle_deg if (apply_abs) else (angle_deg * sgn)
                    angles_signed.append(val)
                safe_label = folder_name if not str(folder_name).startswith('_') else f" {folder_name}"
                ax.plot(xs, angles_signed, 'o-', label=safe_label, markersize=3)
            ax.set_ylabel(f'Rotation {name} (deg)')
            ax.grid(True, alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend()
        axes[-1].set_xlabel('Cumulative Distance (mm)')
        plt.tight_layout()
        plt.savefig(out_dir / f'{prefix}_rotation_series.png', dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger = get_logger()
        logger.exception('Error creating series plots with prefix: %s', e)


def compute_trel_stats(
    rel_list: Dict[str, List[Tuple[np.ndarray, np.ndarray]]]
) -> Tuple[np.ndarray, np.ndarray, int]:
    """상대 변환 trel의 축별 평균/표준편차와 샘플 수를 계산.

    Returns:
        mean (3,), std (3,), count: trel 샘플들의 축별 평균/표준편차와 개수
    """
    try:
        all_t = []
        for _, seq in rel_list.items():
            if not seq:
                continue
            for _, t in seq:
                all_t.append(np.asarray(t, dtype=np.float64).reshape(3))
        if not all_t:
            return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 0
        T = np.vstack(all_t)
        mean = np.mean(T, axis=0)
        std = np.std(T, axis=0, ddof=1) if T.shape[0] > 1 else np.zeros(3, dtype=np.float64)
        return mean, std, T.shape[0]
    except Exception as e:
        logger = get_logger()
        logger.exception('compute_trel_stats failed: %s', e)
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 0


