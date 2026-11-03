
import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from hand_features import (
    extract_hand_features,
    get_feature_columns
)


# =====================================
# 설정
# =====================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="MediaPipe 77개 손 랜드마크 특징 데이터 수집기"
    )

    parser.add_argument(
        "--labels",
        default="A,C,D,E,G,Am,Dm,Em",
        help="수집할 코드 라벨"
    )

    parser.add_argument(
        "--out",
        default="dataset"
    )

    parser.add_argument(
        "--file",
        default="landmarks_v2.csv"
    )

    parser.add_argument(
        "--model",
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
        "--interval",
        type=float,
        default=0.3,
        help="연속 수집 간격(초)"
    )

    parser.add_argument(
        "--max-hands",
        type=int,
        default=1
    )

    parser.add_argument(
        "--min-detection",
        type=float,
        default=0.7
    )

    return parser.parse_args()


# =====================================
# 랜드마크 연결선
# =====================================

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (5, 9), (9, 10), (10, 11), (11, 12),

    (9, 13), (13, 14), (14, 15), (15, 16),

    (13, 17), (17, 18), (18, 19), (19, 20),

    (0, 17)
]


# =====================================
# 화면에 손 랜드마크 그리기
# =====================================

def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]

    points = []

    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)

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

    # 관절 점
    for i, point in enumerate(points):
        color = (
            (0, 0, 255)
            if i in [4, 8, 12, 16, 20]
            else (255, 100, 0)
        )

        cv2.circle(
            frame,
            point,
            5,
            color,
            -1
        )


# =====================================
# CSV 헤더
# =====================================

def make_header():
    return [
        "label",
        "timestamp",
        "handedness",
        "session_id"
    ] + get_feature_columns()


# =====================================
# 기존 데이터 개수 확인
# =====================================

def load_counts(csv_path, labels):
    counts = {label: 0 for label in labels}

    if not os.path.exists(csv_path):
        return counts

    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            label = row.get("label")

            if label in counts:
                counts[label] += 1

    return counts


# =====================================
# CSV 파일 준비
# =====================================

def prepare_csv(csv_path):
    header = make_header()

    if not os.path.exists(csv_path):
        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.writer(f)
            writer.writerow(header)

        print("[생성] CSV 파일:", csv_path)

        return

    # 기존 CSV 헤더 검사
    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.reader(f)
        existing_header = next(reader, [])

    if existing_header != header:
        raise SystemExit(
            "\n[오류] 기존 CSV의 구조가 현재 특징 구조와 다릅니다.\n"
            "기존 CSV를 다른 이름으로 백업하고 새 파일을 사용하세요.\n"
            f"현재 필요한 특징 개수: {len(header) - 4}개\n"
            f"기존 컬럼 개수: {len(existing_header) - 4}개"
        )


# =====================================
# 데이터 저장
# =====================================

def save_data(
    writer,
    csv_file,
    label,
    landmarks,
    handedness,
    session_id,
    counts
):
    if landmarks is None:
        print("[저장 실패] 손이 검출되지 않았습니다.")
        return False

    # 77개 특징 계산
    features = extract_hand_features(landmarks)

    if features is None:
        print("[저장 실패] 특징 계산 오류")
        return False

    if len(features) != 77:
        print(
            f"[저장 실패] 특징 개수 오류: {len(features)}"
        )
        return False

    if not all(map(
        lambda x: __import__("math").isfinite(float(x)),
        features
    )):
        print("[저장 실패] 좌표에 잘못된 값이 있습니다.")
        return False

    timestamp = datetime.now().isoformat(
        timespec="milliseconds"
    )

    row = [
        label,
        timestamp,
        handedness,
        session_id
    ] + features.tolist()

    writer.writerow(row)

    # 즉시 디스크에 반영
    csv_file.flush()

    counts[label] += 1

    print(
        f"[저장 완료] {label} | "
        f"누적 {counts[label]}개 | "
        f"특징 {len(features)}개"
    )

    return True


# =====================================
# 화면 정보 표시
# =====================================

def draw_info(
    frame,
    label,
    labels,
    counts,
    handedness,
    detected,
    burst,
    interval
):
    h, w = frame.shape[:2]

    # 상단 정보 영역
    cv2.rectangle(
        frame,
        (0, 0),
        (w, 125),
        (25, 25, 25),
        -1
    )

    status = (
        "HAND DETECTED"
        if detected
        else "NO HAND"
    )

    status_color = (
        (0, 255, 0)
        if detected
        else (0, 0, 255)
    )

    cv2.putText(
        frame,
        f"CURRENT LABEL: {label}",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"STATUS: {status}",
        (15, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2
    )

    cv2.putText(
        frame,
        f"HAND: {handedness}",
        (15, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    # 우측 수집 상태
    burst_text = "ON" if burst else "OFF"

    cv2.putText(
        frame,
        f"BURST: {burst_text}",
        (w - 250, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0) if burst else (200, 200, 200),
        2
    )

    cv2.putText(
        frame,
        f"INTERVAL: {interval:.1f}s",
        (w - 250, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    # 하단 라벨별 개수
    y = h - 100

    cv2.rectangle(
        frame,
        (0, y - 10),
        (w, h),
        (25, 25, 25),
        -1
    )

    x = 15

    for i, name in enumerate(labels):
        text = f"{i + 1}:{name}({counts[name]})"

        color = (
            (0, 255, 255)
            if name == label
            else (220, 220, 220)
        )

        cv2.putText(
            frame,
            text,
            (x, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

        x += 150

    cv2.putText(
        frame,
        "SPACE: Save | B: Burst | [ ]: Change Label | Q: Quit",
        (15, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


# =====================================
# 메인
# =====================================

def main():
    args = parse_args()

    labels = [
        item.strip()
        for item in args.labels.split(",")
        if item.strip()
    ]

    if len(labels) < 2:
        raise SystemExit("최소 2개 이상의 라벨이 필요합니다.")

    if len(set(labels)) != len(labels):
        raise SystemExit("중복된 라벨이 있습니다.")

    if args.interval <= 0:
        raise SystemExit("수집 간격은 0보다 커야 합니다.")

    os.makedirs(args.out, exist_ok=True)

    csv_path = os.path.join(
        args.out,
        args.file
    )

    # CSV 준비
    prepare_csv(csv_path)

    # 기존 수집 개수
    counts = load_counts(csv_path, labels)

    # 촬영 세션 ID
    session_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    print("\n== 데이터 수집 ==")

    for label in labels:
        print(f"{label}: {counts[label]}개")

    print("\n세션 ID:", session_id)
    print("CSV:", csv_path)

    # CSV 열기
    csv_file = open(
        csv_path,
        "a",
        newline="",
        encoding="utf-8-sig"
    )

    writer = csv.writer(csv_file)

    # MediaPipe 모델 설정
    if not os.path.exists(args.model):
        csv_file.close()

        raise SystemExit(
            f"모델 파일을 찾을 수 없습니다: {args.model}"
        )

    base_options = python.BaseOptions(
        model_asset_path=args.model
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.max_hands,
        min_hand_detection_confidence=args.min_detection,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    landmarker = vision.HandLandmarker.create_from_options(
        options
    )

    # 웹캠 설정
    cap = cv2.VideoCapture(args.cam)

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
        csv_file.close()

        raise SystemExit(
            f"카메라 {args.cam}번을 열 수 없습니다."
        )

    # 초기 상태
    current_index = 0
    current_label = labels[current_index]

    current_landmarks = None
    current_handedness = "Unknown"

    burst = False
    interval = args.interval

    last_save = 0.0

    start_time = time.monotonic()
    last_timestamp_ms = -1

    print("\n수집기를 시작합니다.")
    print("스페이스바를 눌러 데이터를 저장하세요.")

    try:
        while True:

            ok, frame = cap.read()

            if not ok:
                print("[오류] 카메라 프레임을 읽지 못했습니다.")
                break

            # BGR -> RGB
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            # VIDEO 모드 timestamp
            timestamp_ms = max(
                int((time.monotonic() - start_time) * 1000),
                last_timestamp_ms + 1
            )

            last_timestamp_ms = timestamp_ms

            # 손 검출
            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            current_landmarks = None
            current_handedness = "Unknown"

            if result.hand_landmarks:

                # 첫 번째 검출 손 사용
                current_landmarks = result.hand_landmarks[0]

                # handedness
                if result.handedness:
                    current_handedness = (
                        result.handedness[0][0].category_name
                    )

                draw_hand(
                    frame,
                    current_landmarks
                )

            detected = current_landmarks is not None

            # 연속 수집
            now = time.monotonic()

            if burst and detected:
                if now - last_save >= interval:

                    save_data(
                        writer,
                        csv_file,
                        current_label,
                        current_landmarks,
                        current_handedness,
                        session_id,
                        counts
                    )

                    last_save = now

            # 화면 정보
            draw_info(
                frame,
                current_label,
                labels,
                counts,
                current_handedness,
                detected,
                burst,
                interval
            )

            cv2.imshow(
                "Air Guitar Landmark Collector",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            # 종료
            if key in (ord("q"), 27):
                break

            # 한 장 저장
            elif key == ord(" "):

                if detected:
                    save_data(
                        writer,
                        csv_file,
                        current_label,
                        current_landmarks,
                        current_handedness,
                        session_id,
                        counts
                    )

                    last_save = time.monotonic()

                else:
                    print("[실패] 손이 검출되지 않았습니다.")

            # 연속 수집
            elif key in (ord("b"), ord("B")):
                burst = not burst

                print(
                    "[연속 수집]",
                    "시작" if burst else "중지"
                )

                last_save = 0.0

            # 다음 라벨
            elif key == ord("]"):

                current_index = (
                    current_index + 1
                ) % len(labels)

                current_label = labels[current_index]

                print("[라벨 변경]", current_label)

            # 이전 라벨
            elif key == ord("["):

                current_index = (
                    current_index - 1
                ) % len(labels)

                current_label = labels[current_index]

                print("[라벨 변경]", current_label)

            # 숫자키로 라벨 선택
            elif ord("1") <= key <= ord("9"):

                index = key - ord("1")

                if index < len(labels):
                    current_index = index
                    current_label = labels[current_index]

                    print("[라벨 선택]", current_label)

            # 수집 간격 증가
            elif key in (ord("+"), ord("=")):

                interval = min(
                    interval + 0.1,
                    5.0
                )

                print(
                    "[수집 간격]",
                    f"{interval:.1f}초"
                )

            # 수집 간격 감소
            elif key in (ord("-"), ord("_")):

                interval = max(
                    interval - 0.1,
                    0.1
                )

                print(
                    "[수집 간격]",
                    f"{interval:.1f}초"
                )

    finally:
        cap.release()
        landmarker.close()
        csv_file.close()
        cv2.destroyAllWindows()

    print("\n== 수집 종료 ==")

    for label in labels:
        print(f"{label}: {counts[label]}개")

    print("저장 위치:", csv_path)


if __name__ == "__main__":
    main()
