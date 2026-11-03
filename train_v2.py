
import argparse
import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from hand_features import get_feature_columns


# =====================================
# 실행 옵션
# =====================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Air Guitar 77-feature training"
    )

    parser.add_argument(
        "--data",
        default="dataset/landmarks_v2.csv"
    )

    parser.add_argument(
        "--out",
        default="outputs"
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--gap",
        type=int,
        default=5,
        help="시간순 분할 시 경계에서 제외할 샘플 수"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    return parser.parse_args()


# =====================================
# 데이터 불러오기
# =====================================

def load_data(path):

    if not os.path.exists(path):
        raise SystemExit(
            f"CSV 파일을 찾을 수 없습니다: {path}"
        )

    df = pd.read_csv(path)

    feature_columns = get_feature_columns()

    required_columns = [
        "label",
        "timestamp",
        "session_id"
    ] + feature_columns

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise SystemExit(
            "CSV 컬럼이 올바르지 않습니다.\n"
            f"누락된 컬럼: {missing}"
        )

    # 필요한 데이터가 없는 행 제거
    before = len(df)

    df = df.dropna(
        subset=required_columns
    ).copy()

    if len(df) < before:
        print(
            f"[경고] 결측 데이터 {before - len(df)}개 제거"
        )

    df["label"] = df["label"].astype(str)
    df["session_id"] = df["session_id"].astype(str)

    # 숫자형 변환
    for col in feature_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=feature_columns
    ).copy()

    X = df[feature_columns].to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(X).all():
        raise SystemExit(
            "좌표 또는 각도에 NaN/무한대 값이 있습니다."
        )

    if X.shape[1] != 77:
        raise SystemExit(
            f"특징 개수 오류: {X.shape[1]}개 (77개 필요)"
        )

    if len(df) == 0:
        raise SystemExit("학습 데이터가 없습니다.")

    return df, feature_columns


# =====================================
# 데이터 현황
# =====================================

def show_data_info(df):

    classes = sorted(df["label"].unique())

    print("\n========== 데이터 현황 ==========")

    print("전체 데이터:", len(df))
    print("클래스:", classes)
    print("특징 개수: 77")

    print("\n[라벨별 데이터]")

    counts = df["label"].value_counts()

    for label in classes:
        count = int(counts[label])

        print(f"{label:10s}: {count:5d}개")

        if count < 30:
            print(
                f"  [경고] {label} 데이터가 적습니다. "
                "추가 수집을 권장합니다."
            )

    print("\n[세션별 데이터]")

    print(
        df.groupby("session_id")["label"]
        .count()
        .to_string()
    )

    min_count = counts.min()
    max_count = counts.max()

    if min_count > 0 and max_count > min_count * 3:
        print(
            "\n[경고] 라벨별 데이터 불균형이 큽니다."
        )

    if df["session_id"].nunique() < 2:
        print(
            "\n[주의] 촬영 세션이 하나뿐입니다."
            "\n세션 단위 검증이 불가능하므로 시간순 분할을 사용합니다."
        )

    return classes


# =====================================
# 세션 단위 데이터 분할
# =====================================

def split_by_session(df, feature_columns, val_ratio, seed):

    classes = sorted(df["label"].unique())

    X = df[feature_columns].to_numpy(
        dtype=np.float32
    )

    y = df["label"].to_numpy()

    groups = df["session_id"].to_numpy()

    unique_sessions = np.unique(groups)

    # 세션이 2개 이상이면 그룹 분할 시도
    if len(unique_sessions) >= 2:

        splitter = GroupShuffleSplit(
            n_splits=200,
            test_size=val_ratio,
            random_state=seed
        )

        for train_idx, val_idx in splitter.split(
            X, y, groups
        ):

            train_classes = set(y[train_idx])
            val_classes = set(y[val_idx])

            # 모든 클래스가 학습과 검증에 있어야 함
            if (
                train_classes == set(classes)
                and val_classes == set(classes)
            ):

                print(
                    "\n[분할 방식] 촬영 세션 단위 분할"
                )

                print(
                    "학습 세션:",
                    sorted(set(groups[train_idx]))
                )

                print(
                    "검증 세션:",
                    sorted(set(groups[val_idx]))
                )

                return (
                    X[train_idx],
                    X[val_idx],
                    y[train_idx],
                    y[val_idx]
                )

        print(
            "\n[주의] 모든 클래스가 포함되는 세션 분할을 "
            "찾지 못했습니다."
        )

    # =================================
    # 시간순 분할로 대체
    # =================================

    print(
        "\n[분할 방식] 라벨별 시간순 분할"
    )

    train_indices = []
    val_indices = []

    gap = args_gap = 5

    # main에서 설정된 gap은 별도 전달
    # 아래 split_data에서 직접 전달받도록 사용

    raise RuntimeError(
        "시간순 분할은 split_chronologically()를 사용해야 합니다."
    )


# =====================================
# 시간순 분할
# =====================================

def split_chronologically(
    df,
    feature_columns,
    val_ratio,
    gap
):

    train_indices = []
    val_indices = []

    classes = sorted(df["label"].unique())

    print("\n[분할 방식] 라벨별 시간순 분할")

    # 시간순 정렬
    df = df.copy()

    df["_time"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if df["_time"].isna().any():
        raise SystemExit(
            "timestamp를 날짜 형식으로 읽지 못했습니다."
        )

    for label in classes:

        indices = (
            df[df["label"] == label]
            .sort_values("_time")
            .index.to_numpy()
        )

        n = len(indices)

        n_val = max(
            1,
            int(n * val_ratio)
        )

        n_train = n - n_val - gap

        if n_train < 1:
            raise SystemExit(
                f"{label} 데이터가 부족합니다. "
                f"최소 {gap + 2}개 이상 필요합니다."
            )

        train_part = indices[:n_train]

        val_part = indices[n_train + gap:]

        train_indices.extend(train_part)
        val_indices.extend(val_part)

        print(
            f"{label}: 전체 {n} / "
            f"학습 {len(train_part)} / "
            f"검증 {len(val_part)} / "
            f"제외 {gap}"
        )

    train_df = df.loc[train_indices]
    val_df = df.loc[val_indices]

    X_train = train_df[feature_columns].to_numpy(
        dtype=np.float32
    )

    X_val = val_df[feature_columns].to_numpy(
        dtype=np.float32
    )

    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()

    return X_train, X_val, y_train, y_val


# =====================================
# 데이터 분할 통합
# =====================================

def split_data(
    df,
    feature_columns,
    val_ratio,
    gap,
    seed
):

    classes = sorted(df["label"].unique())

    X = df[feature_columns].to_numpy(
        dtype=np.float32
    )

    y = df["label"].to_numpy()

    groups = df["session_id"].to_numpy()

    unique_sessions = np.unique(groups)

    if len(unique_sessions) >= 2:

        splitter = GroupShuffleSplit(
            n_splits=200,
            test_size=val_ratio,
            random_state=seed
        )

        for train_idx, val_idx in splitter.split(
            X, y, groups
        ):

            if (
                set(y[train_idx]) == set(classes)
                and set(y[val_idx]) == set(classes)
            ):

                print(
                    "\n[분할 방식] 촬영 세션 단위 분할"
                )

                print(
                    "학습 세션:",
                    sorted(set(groups[train_idx]))
                )

                print(
                    "검증 세션:",
                    sorted(set(groups[val_idx]))
                )

                return (
                    X[train_idx],
                    X[val_idx],
                    y[train_idx],
                    y[val_idx]
                )

        print(
            "\n[주의] 세션 분할에서 모든 라벨을 "
            "포함하지 못했습니다."
        )

    return split_chronologically(
        df,
        feature_columns,
        val_ratio,
        gap
    )


# =====================================
# 모델 생성
# =====================================

def get_models(seed):

    models = {

        "Logistic": make_pipeline(
            StandardScaler(),

            LogisticRegression(
                C=1.0,
                max_iter=3000,
                class_weight="balanced"
            )
        ),

        "SVM(rbf)": make_pipeline(
            StandardScaler(),

            SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=seed
            )
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed
        ),

        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed
        ),

        "MLP": make_pipeline(
            StandardScaler(),

            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                max_iter=700,
                early_stopping=False,
                random_state=seed
            )
        )
    }

    return models


# =====================================
# 모델 학습 및 비교
# =====================================

def train_models(
    X_train,
    X_val,
    y_train,
    y_val,
    classes,
    seed
):

    models = get_models(seed)

    best = None

    print("\n========== 모델 비교 ==========")

    for name, model in models.items():

        print(f"\n[학습 시작] {name}")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            model.fit(
                X_train,
                y_train
            )

        for warning in caught:
            print(
                "[학습 경고]",
                str(warning.message)
            )

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_acc = accuracy_score(
            y_train,
            train_pred
        )

        val_acc = accuracy_score(
            y_val,
            val_pred
        )

        # predict_proba 클래스 순서 보정
        raw_proba = model.predict_proba(X_val)

        model_classes = model.classes_

        proba = np.zeros(
            (len(X_val), len(classes)),
            dtype=np.float64
        )

        for i, cls in enumerate(model_classes):
            class_index = classes.index(cls)

            proba[:, class_index] = raw_proba[:, i]

        ll = log_loss(
            y_val,
            proba,
            labels=classes
        )

        predicted_classes = set(
            val_pred.tolist()
        )

        flag = ""

        if len(predicted_classes) == 1:
            flag += " [경고: 한 클래스만 예측]"

        spread = float(
            proba.std(axis=0).mean()
        )

        if spread < 0.05:
            flag += " [경고: 확률 변동이 작음]"

        print(
            f"{name:14s} "
            f"Train: {train_acc:.3f} | "
            f"Val: {val_acc:.3f} | "
            f"LogLoss: {ll:.3f} | "
            f"확률변동: {spread:.3f}"
            f"{flag}"
        )

        # 검증 정확도 우선, logloss가 낮은 모델 우선
        key = (
            val_acc,
            -ll
        )

        if best is None or key > best["key"]:

            best = {
                "key": key,
                "name": name,
                "model": model,
                "pred": val_pred,
                "proba": proba,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "log_loss": ll
            }

    return best


# =====================================
# 모델 상세 평가
# =====================================

def evaluate_model(best, y_val, classes):

    print("\n========== 최고 모델 ==========")

    print("모델:", best["name"])

    print(
        f"학습 정확도: {best['train_acc']:.4f}"
    )

    print(
        f"검증 정확도: {best['val_acc']:.4f}"
    )

    print(
        f"LogLoss: {best['log_loss']:.4f}"
    )

    print("\n========== 클래스별 평가 ==========")

    print(
        classification_report(
            y_val,
            best["pred"],
            labels=classes,
            target_names=classes,
            zero_division=0
        )
    )

    print("\n========== 혼동 행렬 ==========")

    cm = confusion_matrix(
        y_val,
        best["pred"],
        labels=classes
    )

    print("라벨 순서:", classes)
    print("행 = 실제 정답 / 열 = 모델 예측")
    print(cm)

    print("\n========== 오분류 분석 ==========")

    for i, actual in enumerate(classes):

        for j, predicted in enumerate(classes):

            if i != j and cm[i, j] > 0:

                print(
                    f"{actual} -> {predicted}: "
                    f"{cm[i, j]}개"
                )


# =====================================
# 모델 저장
# =====================================

def save_model(
    best,
    classes,
    feature_columns,
    out
):

    os.makedirs(
        out,
        exist_ok=True
    )

    save_path = os.path.join(
        out,
        "landmark_model.joblib"
    )

    bundle = {
        "model": best["model"],
        "classes": classes,
        "feature_columns": feature_columns,
        "model_name": best["name"],
        "version": 1,
        "input_type": "mediapipe_hand_landmarks",
        "feature_count": 77
    }

    joblib.dump(
        bundle,
        save_path
    )

    print("\n모델 저장 완료:")
    print(save_path)


# =====================================
# 메인
# =====================================

def main():

    args = parse_args()

    if not 0 < args.val_ratio < 1:
        raise SystemExit(
            "--val-ratio는 0과 1 사이여야 합니다."
        )

    if args.gap < 0:
        raise SystemExit(
            "--gap은 0 이상이어야 합니다."
        )

    # 데이터 불러오기
    df, feature_columns = load_data(
        args.data
    )

    # 데이터 현황
    classes = show_data_info(df)

    if len(classes) < 2:
        raise SystemExit(
            "최소 2개 이상의 클래스가 필요합니다."
        )

    # 분할
    X_train, X_val, y_train, y_val = split_data(
        df,
        feature_columns,
        args.val_ratio,
        args.gap,
        args.seed
    )

    print("\n========== 분할 결과 ==========")

    print("학습 데이터:", len(X_train))
    print("검증 데이터:", len(X_val))
    print("특징 차원:", X_train.shape[1])

    # 학습 데이터 클래스 확인
    for label in classes:

        count = int(
            np.sum(y_train == label)
        )

        if count < 2:
            raise SystemExit(
                f"{label} 학습 데이터가 부족합니다."
            )

    # 모델 학습
    best = train_models(
        X_train,
        X_val,
        y_train,
        y_val,
        classes,
        args.seed
    )

    # 평가
    evaluate_model(
        best,
        y_val,
        classes
    )

    # 저장
    save_model(
        best,
        classes,
        feature_columns,
        args.out
    )


if __name__ == "__main__":
    main()
