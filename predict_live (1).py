
import argparse
import os
import time
from collections import deque

import cv2
import joblib
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from hand_features import (
    extract_hand_features,
    get_feature_columns
)


# =====================================
# 기본 설정
# =====================================

WINDOW_NAME = "Air Guitar - Real Time Prediction"

EMA_ALPHA = 0.25

CONFIDENCE_THRESHOLD = 0.55
MARGIN_THRESHOLD = 0.10

STABLE_DURATION = 0.5
STABLE_MIN_FRAMES = 5

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# =====================================
# 실행 옵션
# =====================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Air Guitar Real-time Prediction"
    )

    parser.add_argument(
        "--model",
        default="outputs/landmark_model.joblib"
    )

    parser.add_argument(
        "--hand-model",
        default="hand_landmarker.task"
    )

    parser.add_argument(
        "--cam",
        type=int,
        default=0
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD
    )

    parser.add_argument(
        "--max-hands",
        type=int,
        default=1
    )

    return parser.parse_args()


# =====================================
# 모델 불러오기
# =====================================

def load_model(model_path):

    if not os.path.exists(model_path):
        raise SystemExit(
            f"학습 모델을 찾을 수 없습니다: {model_path}"
        )

    bundle = joblib.load(model_path)

    required_keys = [
        "model",
        "classes",
        "feature_columns",
        "input_type",
        "feature_count"
    ]

    for key in required_keys:

        if key not in bundle:
            raise SystemExit(
                f"모델 파일에 {key} 정보가 없습니다."
            )

    if bundle["input_type"] != "mediapipe_hand_landmarks":
        raise SystemExit(
            "이 모델은 MediaPipe 랜드마크 모델이 아닙니다."
        )

    expected_columns = get_feature_columns()

    if bundle["feature_count"] != 77:
        raise SystemExit(
            "모델 특징 개수가 77개가 아닙니다."
        )

    if bundle["feature_columns"] != expected_columns:
        raise SystemExit(
            "학습 모델과 현재 hand_features.py의 "
            "특징 구성이 다릅니다."
        )

    model = bundle["model"]

    if not hasattr(model, "predict_proba"):
        raise SystemExit(
            "모델에 predict_proba() 기능이 없습니다."
        )

    classes = list(bundle["classes"])

    if len(classes) < 2:
        raise SystemExit(
            "학습된 클래스가 2개 미만입니다."
        )

    print("\n========== 모델 정보 ==========")

    print("모델:", bundle.get("model_name", "Unknown"))
    print("클래스:", classes)
    print("특징 개수:", bundle["feature_count"])

    return model, classes


# =====================================
# MediaPipe 초기화
# =====================================

def create_landmarker(model_path, max_hands):

    if not os.path.exists(model_path):
        raise SystemExit(
            f"MediaPipe 모델 파일을 찾을 수 없습니다: {model_path}"
        )

    base_options = python.BaseOptions(
        model_asset_path=model_path
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=max_hands,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return vision.HandLandmarker.create_from_options(
        options
    )


# =====================================
# 손 랜드마크 그리기
# =====================================

def draw_hand(frame, landmarks):

    height, width = frame.shape[:2]

    points = []

    for landmark in landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    # 관절 연결선
    for start, end in CONNECTIONS:

        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2
        )

    # 관절
    for i, point in enumerate(points):

        color = (
            (0, 0, 255)
            if i in [4, 8, 12, 16, 20]
            else (0, 165, 255)
        )

        cv2.circle(
            frame,
            point,
            5,
            color,
            -1
        )

    return frame


# =====================================
# 특징 추출
# =====================================

def make_feature(landmarks):

    features = extract_hand_features(
        landmarks
    )

    if features is None:
        return None

    features = np.asarray(
        features,
        dtype=np.float32
    ).reshape(1, -1)

    if features.shape != (1, 77):
        raise ValueError(
            f"특징 크기 오류: {features.shape}"
        )

    if not np.isfinite(features).all():
        return None

    return features


# =====================================
# 확률 예측
# =====================================

def predict(model, classes, features):

    raw_probability = model.predict_proba(
        features
    )[0]

    model_classes = list(model.classes_)

    probability = np.zeros(
        len(classes),
        dtype=np.float32
    )

    for i, label in enumerate(model_classes):

        if label in classes:

            index = classes.index(label)

            probability[index] = raw_probability[i]

    total = probability.sum()

    if total <= 0:
        return None

    probability /= total

    return probability


# =====================================
# EMA 확률 안정화
# =====================================

class ProbabilitySmoother:

    def __init__(self, alpha=EMA_ALPHA):

        self.alpha = alpha
        self.value = None

    def update(self, probability):

        if self.value is None:

            self.value = probability.copy()

        else:

            self.value = (
                self.alpha * probability
                + (1 - self.alpha) * self.value
            )

        return self.value.copy()

    def reset(self):

        self.value = None


# =====================================
# 연속 인식 안정화
# =====================================

class StablePredictor:

    def __init__(
        self,
        classes,
        threshold,
        margin
    ):

        self.classes = classes
        self.threshold = threshold
        self.margin = margin

        self.candidate = None
        self.candidate_start = None
        self.candidate_frames = 0

        self.stable_label = None

    def update(self, probability, current_time):

        order = np.argsort(probability)[::-1]

        best_index = int(order[0])

        best_label = self.classes[best_index]

        best_score = float(
            probability[best_index]
        )

        second_score = float(
            probability[order[1]]
        )

        confidence_ok = (
            best_score >= self.threshold
        )

        margin_ok = (
            best_score - second_score >= self.margin
        )

        if not confidence_ok or not margin_ok:

            self.candidate = None
            self.candidate_start = None
            self.candidate_frames = 0

            return self.stable_label, False

        if self.candidate != best_label:

            self.candidate = best_label
            self.candidate_start = current_time
            self.candidate_frames = 1

        else:

            self.candidate_frames += 1

        elapsed = (
            current_time - self.candidate_start
        )

        if (
            elapsed >= STABLE_DURATION
            and self.candidate_frames >= STABLE_MIN_FRAMES
        ):

            self.stable_label = best_label

            return self.stable_label, True

        return self.stable_label, False

    def reset(self):

        self.candidate = None
        self.candidate_start = None
        self.candidate_frames = 0
        self.stable_label = None


# =====================================
# 화면 출력
# =====================================

def draw_text(
    frame,
    text,
    position,
    color=(255, 255, 255),
    scale=0.7
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA
    )


def draw_probability(
    frame,
    classes,
    probability
):

    if probability is None:
        return

    height, width = frame.shape[:2]

    x = width - 280
    y = 50

    order = np.argsort(probability)[::-1]

    draw_text(
        frame,
        "TOP PREDICTIONS",
        (x, y),
        (255, 255, 255),
        0.6
    )

    for rank, index in enumerate(order[:5]):

        label = classes[index]
        score = float(probability[index])

        bar_y = y + 35 + rank * 45

        draw_text(
            frame,
            f"{label}: {score * 100:.1f}%",
            (x, bar_y),
            (255, 255, 255),
            0.55
        )

        bar_width = 180
        filled = int(bar_width * score)

        cv2.rectangle(
            frame,
            (x, bar_y + 8),
            (x + bar_width, bar_y + 17),
            (70, 70, 70),
            -1
        )

        cv2.rectangle(
            frame,
            (x, bar_y + 8),
            (x + filled, bar_y + 17),
            (0, 220, 0),
            -1
        )


# =====================================
# 메인
# =====================================

def main():

    args = parse_args()

    if not 0 <= args.threshold <= 1:
        raise SystemExit(
            "--threshold는 0~1 사이여야 합니다."
        )

    # 모델 로드
    model, classes = load_model(
        args.model
    )

    # MediaPipe 생성
    landmarker = create_landmarker(
        args.hand_model,
        args.max_hands
    )

    # 카메라
    cap = cv2.VideoCapture(
        args.cam,
        cv2.CAP_DSHOW
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        args.width
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        args.height
    )

    if not cap.isOpened():

        landmarker.close()

        raise SystemExit(
            "웹캠을 열 수 없습니다."
        )

    smoother = ProbabilitySmoother()

    stable_predictor = StablePredictor(
        classes,
        args.threshold,
        MARGIN_THRESHOLD
    )

    frame_count = 0
    fps = 0.0
    fps_start = time.monotonic()

    last_timestamp_ms = -1

    probability = None
    stable_label = None

    print("\n========== 실시간 예측 시작 ==========")

    print("Q : 종료")
    print("R : 인식 상태 초기화")
    print("+ : 인식 임계값 증가")
    print("- : 인식 임계값 감소")
    print("M : 좌우 반전 표시 전환")

    mirror = True

    try:

        while True:

            success, frame = cap.read()

            if not success:
                print("카메라 프레임을 읽을 수 없습니다.")
                break

            # 거울 모드
            if mirror:
                frame = cv2.flip(frame, 1)

            current_time = time.monotonic()

            # FPS 계산
            frame_count += 1

            elapsed_fps = current_time - fps_start

            if elapsed_fps >= 1.0:

                fps = frame_count / elapsed_fps

                frame_count = 0
                fps_start = current_time

            # BGR -> RGB
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            rgb = np.ascontiguousarray(rgb)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int(
                current_time * 1000
            )

            timestamp_ms = max(
                timestamp_ms,
                last_timestamp_ms + 1
            )

            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            probability = None

            # =================================
            # 손 검출
            # =================================

            if result.hand_landmarks:

                landmarks = result.hand_landmarks[0]

                handedness = "Unknown"

                if result.handedness:

                    handedness = (
                        result.handedness[0][0].category_name
                    )

                draw_hand(
                    frame,
                    landmarks
                )

                # 특징 추출
                features = make_feature(
                    landmarks
                )

                if features is not None:

                    # 예측
                    raw_probability = predict(
                        model,
                        classes,
                        features
                    )

                    if raw_probability is not None:

                        # 확률 안정화
                        probability = smoother.update(
                            raw_probability
                        )

                        stable_label, confirmed = (
                            stable_predictor.update(
                                probability,
                                current_time
                            )
                        )

                draw_text(
                    frame,
                    f"Hand: {handedness}",
                    (20, 40),
                    (0, 255, 255)
                )

                # 최종 인식 코드
                if stable_label is not None:

                    draw_text(
                        frame,
                        f"CHORD: {stable_label}",
                        (20, 100),
                        (0, 255, 0),
                        1.2
                    )

                else:

                    draw_text(
                        frame,
                        "CHORD: Recognizing...",
                        (20, 100),
                        (0, 200, 255),
                        0.8
                    )

                draw_probability(
                    frame,
                    classes,
                    probability
                )

            else:

                # 손이 사라지면 상태 초기화
                smoother.reset()
                stable_predictor.reset()

                stable_label = None

                draw_text(
                    frame,
                    "No Hand Detected",
                    (20, 80),
                    (0, 0, 255),
                    0.9
                )

            # =================================
            # 정보 표시
            # =================================

            draw_text(
                frame,
                f"FPS: {fps:.1f}",
                (20, frame.shape[0] - 70),
                (255, 255, 255)
            )

            draw_text(
                frame,
                f"Threshold: {stable_predictor.threshold:.2f}",
                (20, frame.shape[0] - 40),
                (255, 255, 255)
            )

            draw_text(
                frame,
                "Q: Quit | R: Reset | +/-: Threshold | M: Mirror",
                (20, frame.shape[0] - 10),
                (255, 255, 255),
                0.45
            )

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            # =================================
            # 키 입력
            # =================================

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            elif key == ord("r"):

                smoother.reset()
                stable_predictor.reset()

                stable_label = None
                probability = None

            elif key in [ord("+"), ord("=")]:

                stable_predictor.threshold = min(
                    0.95,
                    stable_predictor.threshold + 0.05
                )

            elif key in [ord("-"), ord("_")]:

                stable_predictor.threshold = max(
                    0.1,
                    stable_predictor.threshold - 0.05
                )

            elif key == ord("m"):

                mirror = not mirror

    finally:

        cap.release()

        landmarker.close()

        cv2.destroyAllWindows()

        print("\n실시간 예측 종료")


if __name__ == "__main__":
    main()
