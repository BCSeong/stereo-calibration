from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
from itertools import permutations
from scipy.spatial import cKDTree as KDTree
from ..utils.config import CAND, GRID, SCORE, DETECTOR
from ..utils.types import DetectorParams
from ..utils.logger import get_logger


@dataclass
class BlobParams:  # Deprecated: kept for minimal compatibility if imported elsewhere
    pass


@dataclass
class GridPreset:
    """ Pattern scale parameters 
    - dot_pitch_mm: dot pitch in mm
    """
    dot_pitch_mm: float = 0.5


@dataclass
class DetectionResult:
    """ single image detection result """
    Tc: Optional[np.ndarray]
    center_index: Optional[int]
    grid_assign: Dict[Tuple[int, int], int]
    points_xy: np.ndarray
    diameters: np.ndarray
    chosen_triplet: Optional[Tuple[int, int, int]]
    all_keypoints: List[cv2.KeyPoint]
    nn24_indices: Optional[np.ndarray] = None
    binarized_image: Optional[np.ndarray] = None  # 디버그용 이진화 이미지

@dataclass
class CentralCandidate:
    """central candidate and 24-NN indices from step 3"""
    center_idx: int
    plus_idx: int
    minus_idx: int
    nn24: np.ndarray  # indices of 24 nearest neighbors (excluding self)


class ThreeDotDetector:
    def __init__(
        self,
        blob_params: DetectorParams,
        grid_preset: GridPreset,
        debug_dir: Path,
        debug_sample_rate: int = 0,
        debug_shard_size: int = 0,
    ):
        self.params = blob_params
        self.grid = grid_preset
        self.debug_dir = debug_dir
        self.debug_sample_rate = int(debug_sample_rate or 0)
        self.debug_shard_size = int(debug_shard_size or 0)
        self._debug_counter = 0
        self._last_bw: Optional[np.ndarray] = None
        # DetectorParams는 이미 병합된 확정 파라미터이므로 추가 보정 없음

    def _apply_default_params(self) -> None:
        return  # deprecated path; no-op


    # ---------- Step 1: Blob detection ----------
    def detect_blobs(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[cv2.KeyPoint]]:
        H, W = int(gray.shape[0]), int(gray.shape[1])
        # Contour retrieval path
        if (self.params.retrieval in ("external", "list")):
            # binarize: 사용자가 지정하지 않으면 Otsu(THRESH_BINARY), 지정 시 고정 임계
            if self.params.bin_threshold and 0.0 < float(self.params.bin_threshold) < 1.0:
                thr = int(round(255.0 * float(self.params.bin_threshold)))
                _, bw = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
            else:
                _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # contour mode: external=바깥 컨투어만, list=모든 컨투어(holes 포함)
            mode = cv2.RETR_EXTERNAL if self.params.retrieval == "external" else cv2.RETR_LIST
            self._last_bw = bw
            contours, _ = cv2.findContours(bw, mode, cv2.CHAIN_APPROX_SIMPLE)
            n_total = int(len(contours))
            removed_area = 0
            removed_radius = 0
            removed_fill = 0
            removed_ecc = 0            
            kps: List[cv2.KeyPoint] = []
            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                # 필터 1) 최소 면적: 너무 작은 컨투어 제외 (min_area={self.params.min_area})
                if area < float(self.params.min_area):
                    removed_area += 1
                    continue
                (x, y), r = cv2.minEnclosingCircle(cnt)
                # 필터 2) 반지름 0/음수: 퇴화 컨투어 제거
                if r <= 0:
                    removed_radius += 1
                    continue
                # 필터 2b) 면적 상한: 먼지/큰 아티팩트 제거 (SBD와 정합)
                if self.params.max_area is not None and area > float(self.params.max_area):
                    removed_area += 1
                    continue
                # 필터 3) 채움률: 도넛(속이 빈) 형태 배제
                #  - 원 기준 채움률: fill_circle = area / (π r^2)
                circle_area = float(np.pi * (r ** 2))
                if circle_area <= 0:
                    removed_radius += 1
                    continue
                fill_circle = float(area / circle_area)
                if fill_circle < float(self.params.min_fill):
                    removed_fill += 1
                    continue
                # 필터 4) 이심률 기반(원형도는 여기서 판단): fitEllipse로 관성비(b/a) 계산하여 max_ecc 적용
                if len(cnt) >= 5:
                    try:
                        (cx, cy), (MA, ma), ang = cv2.fitEllipse(cnt)
                        a = float(max(MA, ma)) * 0.5
                        b = float(min(MA, ma)) * 0.5
                        if a > 0:
                            inertia_ratio = b / a
                            min_inertia = float(np.sqrt(max(0.0, 1.0 - float(self.params.max_eccentricity) ** 2)))
                            if inertia_ratio < min_inertia:
                                removed_ecc += 1
                                continue
                    except Exception:
                        get_logger().warning('[WARN] cv2.fitEllipse failed')                        
                        pass
                
                # 서브픽셀 정밀화: 설정 시 moments 중심을 초기값으로 cornerSubPix로 보정
                if DETECTOR.enable_subpixel:
                    try:
                        M = cv2.moments(cnt)
                        if M['m00'] != 0:
                            cx0 = float(M['m10'] / M['m00'])
                            cy0 = float(M['m01'] / M['m00'])
                            # 윈도우 추출 범위 (이미지 경계 클램프)
                            win_w = int(max(3, DETECTOR.subpix_window[0]))
                            win_h = int(max(3, DETECTOR.subpix_window[1]))
                            x0 = max(0, min(W - 1, int(round(cx0))))
                            y0 = max(0, min(H - 1, int(round(cy0))))
                            x1 = max(0, min(W - 1, x0 - win_w))
                            y1 = max(0, min(H - 1, y0 - win_h))
                            x2 = max(0, min(W - 1, x0 + win_w))
                            y2 = max(0, min(H - 1, y0 + win_h))
                            roi = gray[y1:y2+1, x1:x2+1]
                            if roi.size > 0:
                                # cornerSubPix는 float32 Nx1x2 포맷 포인트 필요
                                pts_in = np.array([[[cx0 - x1, cy0 - y1]]], dtype=np.float32)
                                criteria = (int(DETECTOR.subpix_criteria[0]), int(DETECTOR.subpix_criteria[1]), float(DETECTOR.subpix_criteria[2]))
                                cv2.cornerSubPix(roi, pts_in, (win_w, win_h), (int(DETECTOR.subpix_zero_zone[0]), int(DETECTOR.subpix_zero_zone[1])), criteria)
                                cx_ref = float(pts_in[0,0,0] + x1)
                                cy_ref = float(pts_in[0,0,1] + y1)
                                x, y = cx_ref, cy_ref
                    except Exception:
                        get_logger().warning('[WARN] cv2.cornerSubPix failed')
                        pass
                        # 실패 시 원래 중심 유지
                        
                kps.append(cv2.KeyPoint(float(x), float(y), float(2 * r)))
            # FNN(이웃 밀도) 필터: 주변에 충분한 이웃이 없는 점 제거
            pts = np.array([k.pt for k in kps], dtype=np.float32) if kps else np.zeros((0, 2), np.float32)
            diam = np.array([k.size for k in kps], dtype=np.float32) if kps else np.zeros((0,), np.float32)
            removed_fnn = 0
            if len(pts) > 0 and self.params.fnn_filter and self.params.fnn_filter > 1:
                pts0, diam0, kps0 = pts.copy(), diam.copy(), kps[:]
                k = min(max(2, int(self.params.fnn_filter)), len(pts))
                try:
                    tree = KDTree(pts)
                    d4, _ = tree.query(pts, k=min(int(max(1, self.params.fnn_local_k)), len(pts)))
                    local_d = float(np.median(d4[:, 1])) if len(pts) >= 2 else 1.0
                    R = float(max(0.0, self.params.fnn_radius_beta) * local_d)
                    dk, _ = tree.query(pts, k=k)
                    required_neighbors = max(1, min(int(self.params.fnn_filter) - 1, k - 1))
                    valid = (dk[:, 1:] <= R).sum(axis=1) >= required_neighbors
                    removed_fnn = int((~valid).sum())
                    pts = pts[valid]
                    diam = diam[valid]
                    kps = [kp for kp, ok in zip(kps, valid) if ok]
                    # 과도 필터링 시 롤백
                    min_kept_abs = int(max(0, self.params.fnn_rollback_min_kept))
                    min_kept_rel = int(max(1, round(len(pts0) * float(self.params.fnn_rollback_fraction))))
                    if len(pts) < max(min_kept_abs, min_kept_rel):
                        pts, diam, kps = pts0, diam0, kps0
                        removed_fnn = 0
                except Exception:
                    get_logger().warning('[WARN] KDTree.query failed')
                    pass            
            get_logger().debug('\t\t[DBG_BLOB] contours: %d kept: %d removed_area=%d removed_r=%d removed_fill=%d removed_ecc=%d removed_fnn=%d', n_total, len(kps), removed_area, removed_radius, removed_fill, removed_ecc, removed_fnn)            
        
        elif self.params.retrieval == "SBD": # SimpleBlobDetector            
            def _create_blob_detector() -> cv2.SimpleBlobDetector:
                p = cv2.SimpleBlobDetector_Params()
                p.filterByArea = True
                p.minArea = float(self.params.min_area)
                p.maxArea = float(getattr(self.params, 'max_area', DETECTOR.max_area))
                p.filterByCircularity = True
                p.minCircularity = float(max(0.0, min(1.0, self.params.min_fill)))
                p.filterByConvexity = True
                p.minConvexity = float(max(0.0, min(1.0, self.params.min_fill)))
                e = float(max(0.0, min(0.9999, self.params.max_eccentricity)))
                min_inertia = float(np.sqrt(max(1e-6, 1.0 - e * e)))
                p.filterByInertia = True
                p.minInertiaRatio = min_inertia

                p.minDistBetweenBlobs = float(self.params.sbd_min_dist_between_blobs)                
                if self.params.sbd_blob_color is not None:
                    p.filterByColor = True               
                    p.blobColor = int(self.params.sbd_blob_color)                
                # repeatability 요구 사항을 충족하도록 threshold 밴드를 구성
                if self.params.sbd_fixed_threshold is not None:
                    if self.params.sbd_fixed_threshold == 'otsu':
                        ret, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                        thr0 = float(ret)
                    else:
                        thr0 = float(self.params.sbd_fixed_threshold)
                    thr0 = max(1.0, min(254.0, thr0))
                    # 밴드 폭과 스텝은 설정에서 가져옴
                    half = max(0.5, float(self.params.sbd_band_halfwidth))
                    step = max(0.5, float(self.params.sbd_threshold_step))
                    p.minThreshold = float(max(1.0, thr0 - half))
                    p.maxThreshold = float(min(255.0, thr0 + half))
                    p.thresholdStep = float(step)
                # 반복성 설정: 단일 임계에 가깝다면 1로 낮춰 경고 방지
                p.minRepeatability = int(max(1, int(self.params.sbd_min_repeatability)))
                return cv2.SimpleBlobDetector_create(p)
            
            det = _create_blob_detector()
            kps = det.detect(gray)
            pts = np.array([k.pt for k in kps], dtype=np.float32) if kps else np.zeros((0, 2), np.float32)
            diam = np.array([k.size for k in kps], dtype=np.float32) if kps else np.zeros((0,), np.float32)
            self._last_bw = None
        
            # Optional neighborhood density filter
            if len(pts) > 0 and self.params.fnn_filter and self.params.fnn_filter > 1:
                pts0, diam0, kps0 = pts, diam, kps
                k = min(max(2, self.params.fnn_filter), len(pts))
                tree = KDTree(pts)
                d4, _ = tree.query(pts, k=min(5, len(pts)))
                local_d = float(np.median(d4[:, 1])) if len(pts) >= 2 else 1.0
                R = float(10.0 * local_d)
                dk, _ = tree.query(pts, k=k)
                required_neighbors = max(1, min(self.params.fnn_filter - 1, k - 1))
                valid = (dk[:, 1:] <= R).sum(axis=1) >= required_neighbors
                pts = pts[valid]
                diam = diam[valid]
                kps = [kp for kp, ok in zip(kps, valid) if ok]
                if len(pts) < max(9, len(pts0) // 4):
                    pts, diam, kps = pts0, diam0, kps0
        else: # invalid retrieval option
            get_logger().error('[FAIL] Invalid retrieval option: %s', self.params.retrieval)
            raise ValueError(f"Invalid retrieval option: {self.params.retrieval}")
        return pts, diam, kps

    # ---------- Step 2: Ratio sort (4-NN) ----------
    @staticmethod
    def ratio_sort(pts: np.ndarray, diam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """4-NN 평균 대비 지름 비율을 계산해 내림차순 정렬."""
        n = len(pts)
        if n == 0:
            return np.zeros((0,), np.int32), np.zeros((0,), np.float32)
        k = min(5, n)
        if k > 1:
            tree = KDTree(pts)
            _, idx = tree.query(pts, k=k)  # includes self in [:,0]
            m4 = diam[idx[:, 1:]].mean(axis=1)
        else:
            med = float(np.median(diam)) if diam.size else 1.0
            m4 = np.full((n,), med, dtype=np.float32)
        ratio = diam / np.maximum(m4, 1e-6)
        order = np.argsort(-ratio)
        return order.astype(np.int32), ratio.astype(np.float32)

    # ---------- Step 3: Central candidates ----------
    def find_central_candidates(self, pts: np.ndarray, order: np.ndarray, ratio: np.ndarray, diam: np.ndarray) -> List[CentralCandidate]:
        """Step3: 중앙 3점 후보 집합 B 정의 후, 각 후보의 24-NN에서 정확히 2개의 B를 갖는 경우만 수집."""
        # Robust separation using medians:
        # - r_med_all: median of all ratios (robust to dust outliers)
        # - r_med_topk: median of top-K ratios (K=5 or available count)
        # Threshold = midpoint between these medians, with a gentle uplift (>= 1.1 * r_med_all)
        n = len(order)
        if n == 0:
            return []
        r_sorted = ratio[order]
        r_med_all = float(np.median(ratio)) if len(ratio) else 0.0
        k = min(CAND.topk, len(r_sorted))
        r_med_topk = float(np.median(r_sorted[:k])) if k > 0 else r_med_all
        thr = 0.5 * (r_med_all + r_med_topk)
        thr = max(thr, r_med_all * CAND.uplift_all_med)  # ensure separation from the bulk
        # Guarantee at least 3 elements when clear separation exists
        if len(r_sorted) >= 3 and (ratio >= thr).sum() < 3:
            thr = r_sorted[2] * CAND.ensure_top3_factor
        # Additional absolute size constraint: diameter >= 1.35 * global median diameter
        med_d = float(np.median(diam)) if len(diam) else 0.0
        mask_size = diam >= (med_d * CAND.big_size_median_mult)
        mask_B = (ratio >= float(thr)) & mask_size
        try:
            get_logger().debug('\t\t[DBG_CANDS] r_med_all=%.3f r_med_topk=%.3f thr=%.3f B_count=%d n_pts=%d', r_med_all, r_med_topk, thr, int(mask_B.sum()), len(pts))
        except Exception:
            get_logger().warning('[WARN] KDTree.query failed')
            pass

        # Step3 uses fixed 24-NN per paper; independent of blob-filter FNN
        k24 = max(2, CAND.neighbors_central)
        k24 = min(k24, max(1, len(pts) - 1))
        tree = KDTree(pts)
        cands: List[CentralCandidate] = []
        for kidx in order:  # biggest ratio first
            if not mask_B[kidx]:
                continue
            _, nn = tree.query(pts[kidx], k=k24 + 1)
            nn = np.array(nn, dtype=np.int32)[1:]
            inB = [j for j in nn if mask_B[j]]
            if len(inB) == 2:
                cands.append(CentralCandidate(center_idx=kidx, plus_idx=inB[0], minus_idx=inB[1], nn24=nn))
        get_logger().debug('\t\t[DBG_CANDS] cands=%d', len(cands))
        return cands

    # ---------- Step 4: Evaluate 6 affine transforms ----------
    @staticmethod
    def _affines_for_triplet(Pc: np.ndarray, Pp: np.ndarray, Pm: np.ndarray) -> List[np.ndarray]:
        """(i,i+,i-)를 이상 격자 {(0,0),(0,-2),(2,1)}로 보내는 6개의 아핀 변환 생성."""
        src = np.float32([Pc, Pp, Pm])
        # Target grid basis around central element i (paper-correct geometry):
        #   i  -> (0,  0)
        #   i+ -> (0, -2)    (두 칸 위)
        #   i- -> (2,  1)    (오른쪽 두 칸, 아래 한 칸)
        # 주의: 실제 (Pc,Pp,Pm)의 순서는 알 수 없으므로 3! 모든 대응을 평가해 최적을 선택한다.
        target = np.float32([[0.0, 0.0], [0.0, -2.0], [2.0, 1.0]])
        affs: List[np.ndarray] = []
        for perm in permutations(range(3)):
            A = cv2.getAffineTransform(src[list(perm)], target)
            affs.append(A)
        return affs

    @staticmethod
    def _score_affine(A: np.ndarray, pts_centered: np.ndarray, thr: float = 0.35) -> Tuple[Tuple[float, int], np.ndarray]:
        """아핀으로 맵핑된 점군이 8-이웃 정수 격자에 얼마나 가깝게 히트하는지 점수화."""
        # Map to grid coords
        P = np.hstack([pts_centered, np.ones((len(pts_centered), 1), np.float32)]) @ A.T
        neigh = np.array(
            [
                [1, 0],
                [-1, 0],
                [0, 1],
                [0, -1],
                [1, 1],
                [1, -1],
                [-1, 1],
                [-1, -1],
            ],
            dtype=np.int32,
        )
        used: set[int] = set()
        dsum = 0.0
        hit = 0
        for t in neigh:
            d = np.linalg.norm(P - t, axis=1)
            j = int(np.argmin(d))
            if j in used:
                continue
            if d[j] <= thr:
                used.add(j)
                dsum += float(d[j])
                hit += 1
        return ((1e9, 0) if hit < 8 else (dsum / hit, hit), P) # 8개의 이웃 격자 탐색에 실패하면 FAIL 반환.

    def choose_Tc(self, pts: np.ndarray, cands: List[CentralCandidate]) -> Tuple[Optional[np.ndarray], Optional[int], Optional[Tuple[int, int, int]], Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[float, int]], int, float]:
        """모든 후보에 대해 6개 아핀 평가 후 최적 Tc/center/chosen_triplet/P/nn24 선택."""
        if not cands:
            get_logger().error('[FAIL] cands is None')
            return None, None, None, None, None, None, 0, 0.0
        best = None
        best_P = None
        examined = 0
        all_scores = []  # 모든 후보의 점수 저장
        
        for cand in cands:
            c, p, m = cand.center_idx, cand.plus_idx, cand.minus_idx
            Pc, Pp, Pm = pts[c], pts[p], pts[m]
            for A in self._affines_for_triplet(Pc, Pp, Pm):
                score, P = self._score_affine(A, pts, thr=SCORE.affine_score_thr)
                all_scores.append((score, c, (c, p, m), A, cand.nn24, P))
                
                if np.isfinite(score[0]):
                    if best is None or score < best[0]:
                        best = (score, c, (c, p, m), A, cand.nn24)
                        best_P = P
                examined += 1
        
        '''
        # DEBUG: 모든 후보의 점수 출력
        print(f"[temp] Tc candidates scores (total={len(all_scores)}):") # 디버그 출력
        for i, (score, c, triplet, A, nn24, P) in enumerate(all_scores):
            mean_error, hit = score if score is not None else (float('inf'), 0)
            print(f"  [{i+1}] mean_error={mean_error:.6f}, hit={hit}, triplet={triplet}")
            if A is not None:
                print(f"      Tc={A.flatten()}")
        '''
        # center_offset 검사 후 적절한 Tc 선택
        if best is None:
            get_logger().error('[FAIL] Tc: best is None')
            return None, None, None, None, None, None, examined
        
        # 유효한 점수만 필터링
        valid_scores = [(score, c, triplet, A, nn24, P) for score, c, triplet, A, nn24, P in all_scores 
                       if score is not None and np.isfinite(score[0])]
        
        if not valid_scores:
            get_logger().error('[FAIL] No valid scores found')
            return None, None, None, None, None, None, examined
        
        # 점수 기준으로 정렬 (낮은 점수가 좋음)
        valid_scores.sort(key=lambda x: x[0])
        
        # center_offset 검사하여 적절한 Tc 선택
        selected_tc = None
        for i, (score, c, triplet, A, nn24, P) in enumerate(valid_scores):
            if P is not None and c < len(P):
                center_offset = P[c].copy()
                # print(f"[temp] Tc[{i+1}] center_offset={center_offset}") # 디버그 출력
                
                # center_offset이 (0,0)에 가까운지 검사 (threshold: 0.1)
                if np.linalg.norm(center_offset) < 0.1:
                    selected_tc = (score, c, triplet, A, nn24, P)
                    # print(f"[temp] Selected Tc[{i+1}] - center_offset is close to (0,0)") # 디버그 출력
                    break
                else:
                    # print(f"[temp] Tc[{i+1}] rejected - center_offset too far from (0,0)") # 디버그 출력
                    pass
        
        if selected_tc is None:
            get_logger().error('[FAIL] No Tc found with center_offset close to (0,0)')
            return None, None, None, None, None, None, examined
        
        score_tuple, cidx, triplet, Tc, nn24, best_P = selected_tc
        
        # center_offset 검사로 이미 (0,0)에 가까운 Tc를 선택했으므로 추가 보정 불필요
        
        get_logger().debug('\t\t[DBG_CHOOSE_TC] score_hit=%d, mean_error=%.3f, examined=%d', score_tuple[1], score_tuple[0], examined)
        return Tc, cidx, triplet, best_P, nn24, score_tuple, examined

    # ---------- Steps 5–6: Gridding and dilation ----------
    @staticmethod
    def grid_assign(P: np.ndarray, thr: float = 0.35, max_radius: int = GRID.max_radius) -> Dict[Tuple[int, int], int]:
        """Begin gridding (Step 5) with fixed transform.

        단위 주의: thr는 격자 공간 단위(1.0 = 도트 간격 1칸). 픽셀(px) 아님.
        - Seed at integer key nearest to (0,0) if close enough (≤ max(thr, SCORE.seed_thr)).
        - Assign 8-neighborhood integer keys iteratively with unique nearest points (distance ≤ thr).
        - This matches Step 5 behavior (no local transform updates).
        """
        n = len(P)
        if n == 0:
            return {}
        # find center seed
        d0 = np.linalg.norm(P - np.array([0.0, 0.0], dtype=np.float32), axis=1)
        cidx = int(np.argmin(d0))
        center_key = tuple(np.round(P[cidx]).astype(np.int32))
        center_dist = float(np.linalg.norm(P[cidx] - np.array(center_key, np.float32)))
        get_logger().debug('[DEBUG] Grid assign: center_idx=%d, center_key=%s, center_dist=%.3f, threshold=%.3f', cidx, str(center_key), center_dist, max(thr, SCORE.seed_thr))
        
        if center_dist > max(thr, SCORE.seed_thr):
            get_logger().debug('[DEBUG] Grid assign FAILED: center too far from (0,0)')
            return {}
        assigned: Dict[Tuple[int, int], Tuple[int, float]] = {}
        used_idx: set[int] = set()
        assigned[center_key] = (cidx, center_dist)
        used_idx.add(cidx)
        frontier: List[Tuple[int, int]] = [center_key]

        def nearest_idx_to_key(key: Tuple[int, int]) -> Tuple[int, float]:
            # choose nearest UNUSED point to integer key
            tgt = np.array(key, dtype=np.float32)
            d = np.linalg.norm(P - tgt, axis=1)
            # mask out used indices by inflating distance
            if used_idx:
                for ui in used_idx:
                    d[ui] = 1e9
            j = int(np.argmin(d))
            return j, float(d[j])

        while frontier:
            u0, v0 = frontier.pop()
            for du in (-1, 0, 1):
                for dv in (-1, 0, 1):
                    if du == 0 and dv == 0:
                        continue
                    key = (u0 + du, v0 + dv)
                    if key in assigned:
                        continue
                    if abs(key[0]) > GRID.max_radius or abs(key[1]) > GRID.max_radius:
                        continue
                    j, dist = nearest_idx_to_key(key)
                    if dist <= thr:
                        assigned[key] = (j, dist)
                        used_idx.add(j)
                        frontier.append(key)
        return {k: v[0] for k, v in assigned.items()}

    @staticmethod
    def dilate_with_local_refine(pts: np.ndarray, base_transform: np.ndarray, initial_grid: Dict[Tuple[int, int], int] = None, thr: float = 0.25, max_radius: int = GRID.max_radius) -> Dict[Tuple[int, int], int]:
        """Dilate gridded area (Step 6) using local 2x2 squares to refine transform.

        단위 주의: thr는 격자 공간 단위(1.0 = 도트 간격 1칸). 픽셀(px) 아님.
        Algorithm:
        - Start with mapping P via base transform and initial assignments from grid_assign.
        - Repeatedly find 2x2 integer squares whose four corners are assigned but not explored.
        - For each such square, compute a local affine T_next that maps the 4 image points to their integer keys
          (estimateAffine2D with 4 pairs).
        - Using P_next (pts -> grid via T_next), assign 8-neighborhood around those four corners (unique, ≤ thr).
        - Continue until no growth.
        """
        # Use provided initial grid or compute from scratch
        if initial_grid is not None:
            assigned = initial_grid.copy()
        else:
            # Step 5 seed (fallback)
            P = np.hstack([pts, np.ones((len(pts), 1), np.float32)]) @ base_transform.T
            assigned = ThreeDotDetector.grid_assign(P, thr=thr, max_radius=max_radius)
        
        if not assigned:
            return {}

        explored: set[Tuple[int, int]] = set()

        def list_squares(keys: set[Tuple[int, int]]):
            keys_list = list(keys)
            keyset = set(keys_list)
            for (u, v) in keys_list:
                sq = ((u, v), (u + 1, v), (u, v + 1), (u + 1, v + 1))
                if all(k in keyset for k in sq):
                    yield sq

        def assign_from_transform(T: np.ndarray, seeds: List[Tuple[int, int]], assigned_map: Dict[Tuple[int, int], int]) -> int:
            Pn = np.hstack([pts, np.ones((len(pts), 1), np.float32)]) @ T.T
            uv_assigned = set(assigned_map.keys())
            used_idx = set(assigned_map.values())
            grew = 0
            for (u0, v0) in seeds:
                for du in (-1, 0, 1):
                    for dv in (-1, 0, 1):
                        if du == 0 and dv == 0:
                            continue
                        key = (u0 + du, v0 + dv)
                        if key in uv_assigned:
                            continue
                        if abs(key[0]) > GRID.max_radius or abs(key[1]) > GRID.max_radius:
                            continue
                        tgt = np.array(key, np.float32)
                        d = np.linalg.norm(Pn - tgt, axis=1)
                        # exclude used indices
                        if used_idx:
                            dd = d.copy()
                            for ui in used_idx:
                                dd[ui] = 1e9
                            d = dd
                        j = int(np.argmin(d))
                        if float(d[j]) <= thr:
                            assigned_map[key] = j
                            uv_assigned.add(key)
                            used_idx.add(j)
                            grew += 1
            return grew

        grew_total = True
        while grew_total:
            grew_total = False
            keys = set(assigned.keys())
            for sq in list_squares(keys):
                if sq in explored:
                    continue
                # build correspondences (image pts -> integer coords)
                src = np.float32([pts[assigned[sq[0]]], pts[assigned[sq[1]]], pts[assigned[sq[2]]], pts[assigned[sq[3]]]])
                dst = np.float32([[sq[0][0], sq[0][1]], [sq[1][0], sq[1][1]], [sq[2][0], sq[2][1]], [sq[3][0], sq[3][1]]])
                method = cv2.LMEDS if GRID.use_lmeds else cv2.RANSAC
                T_next, _ = cv2.estimateAffine2D(src, dst, method=method)
                if T_next is None:
                    explored.add(sq)
                    continue
                grew = assign_from_transform(T_next, list(sq), assigned)
                if grew > 0:
                    grew_total = True
                explored.add(sq)

        # periodic refit and 3σ pruning
        uv = np.array(list(assigned.keys()), np.float32)
        idx = np.array(list(assigned.values()), np.int32)
        if len(uv) >= SCORE.min_seed_pairs:
            src = pts[idx]
            dst = uv
            T_refit, _ = cv2.estimateAffine2D(src, dst, method=cv2.LMEDS)
            if T_refit is not None:
                Pn = np.hstack([pts, np.ones((len(pts), 1), np.float32)]) @ T_refit.T
                resid = np.linalg.norm(Pn[idx] - dst, axis=1)
                mu, sig = float(np.mean(resid)), float(np.std(resid))
                keep = resid <= (mu + SCORE.sigma_prune * sig)
                uv_kept = [tuple(map(int, uv[i])) for i in range(len(uv)) if keep[i]]
                idx_kept = [int(idx[i]) for i in range(len(idx)) if keep[i]]
                assigned = {uv_kept[i]: idx_kept[i] for i in range(len(uv_kept))}
        return assigned


    def run_on_image(self, image_path: Path, gray: np.ndarray) -> DetectionResult:
        """단일 이미지에 대해 전체 파이프라인 실행 후 DetectionResult 반환."""
        
        # Debug 정보 추출
        sub = image_path.parent.name
        stem = image_path.stem              
        self._debug_counter += 1
        
        # step 1: blob detection
        pts, diam, kps = self.detect_blobs(gray)

        # binarized image는 return으로 전달 (외부에서 저장)
        bw = self._last_bw

        # step 2: ratio sort
        order, ratio = self.ratio_sort(pts, diam)
        
        # step 3: central candidates
        cands = self.find_central_candidates(pts, order, ratio, diam)
        if not cands:
            # 시각화는 main에서 처리 
            # if should_save: # debug for step 3: blobs only
            #     # save blobs only
            #     out_vis_dir = self.debug_dir / 'Blob_Fails'
            #     out_vis_dir.mkdir(parents=True, exist_ok=True)
            #     out_png = out_vis_dir / f"{sub}_{stem}_blobs.png"
            #     save_detection_overlay(gray, pts, None, out_png, None, None)

            get_logger().error('[FAIL] Central element search failed: %s', str(image_path))             
            return DetectionResult(
                Tc=None,
                center_index=None,
                grid_assign={},
                points_xy=pts,
                diameters=diam,
                chosen_triplet=None,
                all_keypoints=kps,
                nn24_indices=None,
                binarized_image=bw
            )

        
        # step 4: choose Tc
        Tc, cidx, triplet, best_P, nn24, score_tuple, examined = self.choose_Tc(pts, cands)
        # 선택된 아핀 변환의 평균 격자거리(mean_error) 검증.
        flag_TC_err = False
        if score_tuple is None or not np.isfinite(score_tuple[0]) or float(score_tuple[0]) > float(SCORE.max_affine_mean_error):
            get_logger().warning('[WARN] mean_error too large: %.3f > %.3f', (np.nan if score_tuple is None else float(score_tuple[0])), float(SCORE.max_affine_mean_error))
            flag_TC_err = True
        get_logger().debug('\t\t[DBG_CHOOSE_TC] triplet=%s (len=%d)', str(triplet), (0 if triplet is None else len(triplet)))
        get_logger().debug('\t\t[DBG_CHOOSE_TC] Tc=%s (len=%d)', str(Tc), (0 if Tc is None else len(Tc)))
        
        # step 5: grid assign
        grid = self.grid_assign(best_P, thr=SCORE.grid_assign_thr) if best_P is not None else {}
        
        # DEBUG: triplet (i) 점의 할당 결과 출력
        if triplet is not None and grid:
            c, p, m = triplet  # c = (i) 점
            # triplet (i) 점의 실제 좌표 확인
            if best_P is not None and c < len(best_P):
                triplet_i_coord = best_P[c]
                get_logger().debug('\t\t[DBG_CHOOSE_TC] Triplet (i) point %d coordinate: (%.3f, %.3f)', c, triplet_i_coord[0], triplet_i_coord[1])
                
                # 추가: (0,0)에서의 거리 확인
                dist_from_origin = np.linalg.norm(triplet_i_coord)
                get_logger().debug('\t\t[DBG_CHOOSE_TC] Distance from (0,0): %.6f', dist_from_origin)
            
            # triplet (i) 점이 어떤 격자 좌표에 할당되었는지 확인
            for (u, v), idx in grid.items():
                if idx == c:  # triplet의 (i) 점
                    get_logger().debug('\t\t[DBG_GRID_ASSIGN] Triplet (i) point %d assigned to grid (%d,%d)', c, u, v)
                    break
            else:
                get_logger().debug('\t\t[DBG_GRID_ASSIGN] Triplet (i) point %d NOT assigned to any grid!', c)  
        
        # step 6: dilate with local refine
        if grid and Tc is not None:
            # DEBUG: step6 전에 triplet (i) 할당 결과 확인
            if triplet is not None:
                c, p, m = triplet  # c = (i) 점
                get_logger().debug('\t\t[DBG_GRID_ASSIGN] Before step6 - triplet (i) point %d assignment check:', c)
                
                # Tc와 pts를 사용하여 (i) 점이 어디에 할당되었는지 확인
                P_test = np.hstack([pts, np.ones((len(pts), 1), np.float32)]) @ Tc.T
                
                # center_offset 검사로 이미 (0,0)에 가까운 Tc를 선택했으므로 추가 보정 불필요
                d0_test = np.linalg.norm(P_test - np.array([0.0, 0.0], dtype=np.float32), axis=1)
                cidx_test = int(np.argmin(d0_test))
                center_key_test = tuple(np.round(P_test[cidx_test]).astype(np.int32))
                center_dist_test = float(np.linalg.norm(P_test[cidx_test] - np.array(center_key_test, np.float32)))
                
                get_logger().debug('\t\t[DBG_GRID_ASSIGN] Tc+pts: center_idx=%d, center_key=%s, center_dist=%.3f', cidx_test, str(center_key_test), center_dist_test)
                
                # triplet (i) 점의 실제 할당 확인
                for (u, v), idx in grid.items():
                    if idx == c:  # triplet의 (i) 점
                        get_logger().debug('\t\t[DBG_GRID_ASSIGN] Tc+pts: Triplet (i) point %d assigned to grid (%d,%d)', c, u, v)
                        break
                else:
                    get_logger().debug('\t\t[DBG_GRID_ASSIGN] Tc+pts: Triplet (i) point %d NOT assigned to any grid!', c)
            
            # 임시: dilate_with_local_refine 사용 (이미 할당된 grid 재사용)
            grid = self.dilate_with_local_refine(pts, Tc, initial_grid=grid, thr=SCORE.dilation_thr)
        
        # 시각화는 main에서 처리 
        # if should_save:            
        #     out_vis_dir = self.debug_dir / 'Blob_Success' if not flag_TC_err else self.debug_dir / 'Blob_Fails'
        #     out_vis_dir.mkdir(parents=True, exist_ok=True)
        #     out_png = out_vis_dir / f"{sub}_{stem}_grid.png"
        #     save_detection_overlay(gray, pts, triplet, out_png, best_P, nn24)
        #     save_grid_report(image_path, gray, pts, grid, Tc, out_vis_dir, image_path.stem)
            

        get_logger().debug('[SUCCESS] Central element search succeeded: %s, num grid=%d, score_hit=%s', str(image_path), len(grid), (None if score_tuple is None else str(score_tuple[1])))

        return DetectionResult(Tc=Tc, center_index=cidx,
                                grid_assign=grid, points_xy=pts,
                                diameters=diam, chosen_triplet=triplet,
                                all_keypoints=kps, nn24_indices=nn24,
                                binarized_image=bw
                                )



def example(debug_dir, image_path):
    """단일 이미지 디버그 실행 예제."""
    blob_params = BlobParams()
    preset = GridPreset()
    det = ThreeDotDetector(blob_params, preset, Path(debug_dir))
    
    # load image
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    res = det.run_on_image(Path(image_path), gray)
    get_logger().info('%s', str(res))

if __name__ == "__main__":
    example(debug_dir='./debug_test', image_path='./debug_test/image.bmp')