"""
공통 특징 추출 모듈 (학습/실시간 인식이 반드시 같은 함수를 쓰도록 분리)

특징 = 흑백 HOG(모양) + HSV 색 히스토그램(색) + 작은 컬러 썸네일(배치)
"""

import cv2
import numpy as np


def make_hog(size):
    return cv2.HOGDescriptor((size, size), (16, 16), (8, 8), (8, 8), 9)


def extract_feature(img_bgr, size=64, hog=None):
    """BGR 이미지 1장 -> 1차원 float32 특징 벡터."""
    if hog is None:
        hog = make_hog(size)
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)

    # 1) 모양: HOG
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f_hog = hog.compute(gray).ravel().astype(np.float32)

    # 2) 색: HSV 채널별 히스토그램 (조명 변화에 비교적 강함)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    f_col = []
    for ch, rng in zip(range(3), ((0, 180), (0, 256), (0, 256))):
        hist = cv2.calcHist([hsv], [ch], None, [16], rng).ravel()
        f_col.append(hist / (hist.sum() + 1e-6))
    f_col = np.concatenate(f_col).astype(np.float32)

    # 3) 배치: 12x12 컬러 썸네일
    thumb = cv2.resize(img, (12, 12), interpolation=cv2.INTER_AREA)
    f_thumb = (thumb.astype(np.float32) / 255.0).ravel()

    return np.concatenate([f_hog, f_col, f_thumb])