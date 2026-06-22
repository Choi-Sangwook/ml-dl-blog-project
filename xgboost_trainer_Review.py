"""XGBoost 고객 이탈 학습기 코드 리뷰 복사본.

원본: src/models/churn/xgboost_trainer.py
판정: 현재 model.json, preprocessor.joblib, 평가 산출물과 학습 로직이 일치하며
      실행을 막는 오류나 현재 모델을 무효화하는 누수는 발견되지 않음.

유지보수 권고(현재 결과를 무효화하는 오류는 아님):
1. trainer가 model.json과 preprocessor.joblib은 생성하지만 model_config.json과
   feature_schema.json은 직접 갱신하지 않는다. 하이퍼파라미터 override 실행 후에는
   정적 설정 파일과 실제 모델이 달라지지 않는지 확인해야 한다.
2. XGBoost 트리에는 scaling이 필수는 아니지만 현재 저장 모델은 RobustScaler를 전제로
   학습됐으므로 운영에서도 반드시 동일한 preprocessor.joblib을 적용해야 한다.
3. threshold=0.43은 F1 기준이라 Test 사용자의 약 97%를 Positive로 분류한다.
   쿠폰 비용이 중요하면 별도의 비즈니스 임계값 정책이 필요하다.

XGBoost의 모델 구조는 라이브러리 XGBClassifier가 제공하므로 별도 model.py는 없다.
저장된 models/churn/xgboost/model.json은 주석을 넣을 수 없는 네이티브 모델 산출물이다.
"""
import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from src.common.data import FEATURE_ORDER_V2, load_tabular_v2, make_scaler
from src.common.evaluation import evaluate_and_save
from src.common.manifest import write_manifest
from src.common.registry import artifact_rel_path, log_run, resolve_dirs

MODEL_KEY = "xgboost"
MODEL_NAME = "XGBoost_Churn_v2"
MODEL_TYPE = "tree"
SEED = 42


def _oof_threshold(X_raw, y, common, scaler_name):
    """Train 5-Fold OOF 확률에서 F1 최대 임계값을 선택한다."""
    model = xgb.XGBClassifier(**common)
    scaler = make_scaler(scaler_name)

    # [리뷰] Pipeline 안에서 Fold마다 scaler를 fit하므로 검증 Fold 통계가 학습에 섞이지 않는다.
    estimator = (
        Pipeline([("scaler", scaler), ("model", model)])
        if scaler is not None
        else model
    )

    # [리뷰] StratifiedKFold는 약 83%인 이탈 비율을 각 Fold에 비슷하게 유지한다.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(
        estimator, X_raw, y, cv=cv, method="predict_proba", n_jobs=4
    )[:, 1]

    # [리뷰] OOT Test가 아니라 Train OOF로 운영 임계값을 결정하므로 Test 누수를 막는다.
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    f1 = [
        precision_recall_fscore_support(
            y, (oof >= threshold).astype(int), average="binary", zero_division=0
        )[2]
        for threshold in grid
    ]
    return float(grid[int(np.argmax(f1))])


# [리뷰] 올바른 원시 Train/Test 전처리 위에서 5-Fold CV PR-AUC로 선택된 정식 설정이다.
# 얕은 트리, 낮은 학습률, 높은 min_child_weight와 L2 규제가 과적합을 함께 제어한다.
DEFAULTS = {
    "scaler": "robust",
    "n_estimators": 250,          # 낮은 학습률을 보완하는 Boosting Tree 수
    "max_depth": 3,               # 복잡한 개별 트리보다 여러 얕은 트리로 일반화
    "learning_rate": 0.0359,      # 각 새 트리의 기여도를 작게 제한
    "subsample": 0.8,             # 트리마다 사용자 80%를 사용해 확률적 규제
    "colsample_bytree": 1.0,      # 22개 피처가 적어 전체 후보 피처 허용
    "min_child_weight": 20,       # 소수 표본만 포함하는 리프 생성 억제
    "gamma": 0,                   # 추가 분할 손실 제한은 사용하지 않음
    "reg_alpha": 0,               # L1 규제 비활성화
    "reg_lambda": 10,             # 강한 L2 규제로 리프 가중치 안정화
    "scale_pos_weight": 1,        # churn=1이 이미 다수라 추가 Positive 가중치 불필요
    "early_stopping_rounds": 0,   # 최종 모델은 전체 Train으로 refit
}


def train(run_tag=None, **overrides):
    """XGBoost 학습, OOT 평가, 모델·전처리·분석 산출물을 저장한다."""
    hp = dict(DEFAULTS)
    hp.update(overrides)

    # [리뷰] 공통 로더가 train/test 모두 cohort_recency7==1로 필터하고,
    # user_id를 피처에서 제외한 동일 22개 FEATURE_ORDER_V2를 반환한다.
    (X_full, y_full, _), (X_test, y_test, test_user_ids) = load_tabular_v2(MODEL_KEY)
    X_full_raw = X_full

    # [리뷰] scaler는 원시 Train에만 fit하고 Test에는 transform만 적용한다.
    # 현재 preprocessor.joblib과 model.json도 이 순서를 전제로 하므로 서빙에서 동일해야 한다.
    scaler = make_scaler(hp["scaler"])
    if scaler is not None:
        X_full = scaler.fit_transform(X_full)
        X_test = scaler.transform(X_test)

    early_stopping_rounds = hp["early_stopping_rounds"]
    common = dict(
        n_estimators=hp["n_estimators"],
        max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"],
        subsample=hp["subsample"],
        colsample_bytree=hp["colsample_bytree"],
        min_child_weight=hp["min_child_weight"],
        gamma=hp["gamma"],
        reg_alpha=hp["reg_alpha"],
        reg_lambda=hp["reg_lambda"],
        scale_pos_weight=hp["scale_pos_weight"],
        objective="binary:logistic",  # churn=1 확률을 출력하는 이진 분류
        eval_metric="logloss",
        tree_method="hist",           # 빠르고 메모리 효율적인 Histogram 학습
        random_state=SEED,
    )

    if early_stopping_rounds and early_stopping_rounds > 0:
        # [리뷰] 튜닝 실행에서만 validation 15%를 분리하고 조기 종료한다.
        X_train, X_val, y_train, y_val = train_test_split(
            X_full,
            y_full,
            test_size=0.15,
            stratify=y_full,
            random_state=SEED,
        )
        classifier = xgb.XGBClassifier(
            early_stopping_rounds=early_stopping_rounds, **common
        )
        classifier.fit(
            X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False
        )
        eval_result = classifier.evals_result()
        train_loss = [float(value) for value in eval_result["validation_0"]["logloss"]]
        val_loss = [float(value) for value in eval_result["validation_1"]["logloss"]]
        n_train = len(y_train)
    else:
        # [리뷰] 정식 모델은 선택된 설정으로 전체 109,378명을 다시 학습한다.
        # 별도 validation이 없으므로 val_loss가 빈 배열인 것은 의도된 계약이다.
        classifier = xgb.XGBClassifier(**common)
        classifier.fit(X_full, y_full, eval_set=[(X_full, y_full)], verbose=False)
        eval_result = classifier.evals_result()
        train_loss = [float(value) for value in eval_result["validation_0"]["logloss"]]
        val_loss = []
        n_train = len(y_full)

    # [리뷰] predict_proba의 두 번째 열은 positive class인 churn=1 확률이다.
    y_score = classifier.predict_proba(X_test)[:, 1]

    artifact_dir, eval_dir = resolve_dirs(MODEL_KEY, run_tag)

    # [리뷰] 네이티브 JSON은 Python pickle보다 XGBoost 버전 간 호환성이 좋다.
    classifier.save_model(str(artifact_dir / "model.json"))
    if scaler is not None:
        joblib.dump(scaler, artifact_dir / "preprocessor.joblib")

    training_history = {
        "epoch": list(range(1, len(train_loss) + 1)),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    # [리뷰] pred_contribs=True는 gain proxy가 아니라 실제 TreeSHAP 기여도를 계산한다.
    dtest = xgb.DMatrix(X_test, feature_names=list(FEATURE_ORDER_V2))
    contributions = classifier.get_booster().predict(dtest, pred_contribs=True)
    mean_abs = np.abs(contributions[:, :-1]).mean(axis=0)
    order = np.argsort(-mean_abs)
    shap_summary = {
        "feature": [FEATURE_ORDER_V2[index] for index in order],
        "mean_abs_shap": [float(mean_abs[index]) for index in order],
        "rank": list(range(1, len(FEATURE_ORDER_V2) + 1)),
        "note": "XGBoost native TreeSHAP (mean|SHAP|) on test set",
    }

    # [리뷰] 현재 정식 산출물에서는 이 과정으로 threshold=0.43이 선택됐다.
    oof_threshold = _oof_threshold(X_full_raw, y_full, common, hp["scaler"])

    # [리뷰] ROC/PR 곡선, 임계값 지표, 혼동행렬, calibration, lift, SHAP,
    # business value와 사용자별 평가 예측을 한 번에 저장한다.
    metrics = evaluate_and_save(
        eval_dir,
        model_name=MODEL_NAME,
        model_key=MODEL_KEY,
        model_type=MODEL_TYPE,
        user_id=test_user_ids,
        y_true=y_test,
        y_score=y_score,
        n_train=n_train,
        training_history=training_history,
        shap_summary=shap_summary,
        fixed_threshold=oof_threshold,
        threshold_grid_step=0.01,
    )

    # [리뷰] 백엔드 등록용 manifest에 실제 데이터·모델·평가 경로와 전처리 계약을 기록한다.
    write_manifest(
        eval_dir,
        model_name=MODEL_NAME,
        model_key=MODEL_KEY,
        model_type=MODEL_TYPE,
        input_train="data/processed/churn/train_tabular_v2.parquet",
        input_test="data/processed/churn/test_tabular_v2.parquet",
        artifact_path=artifact_rel_path(MODEL_KEY, run_tag, "model.json"),
        metrics=metrics,
        preprocessing_config={
            "input_format": "parquet",
            "cohort_filter": "cohort_recency7==1",
            "scale": hp["scaler"],
            "feature_order": FEATURE_ORDER_V2,
            "id_column": "user_id",
            "target_column": "churn",
            "objective": "binary:logistic",
            **{
                key: hp[key]
                for key in (
                    "n_estimators",
                    "max_depth",
                    "learning_rate",
                    "subsample",
                    "colsample_bytree",
                    "min_child_weight",
                    "gamma",
                    "reg_alpha",
                    "reg_lambda",
                    "scale_pos_weight",
                )
            },
        },
    )

    # [리뷰] run_tag별 지표와 파라미터를 leaderboard CSV에 누적한다.
    log_run(MODEL_KEY, run_tag, hp, metrics)
    print(
        f"[{MODEL_KEY}] run={run_tag or 'baseline'} n_train={n_train} "
        f"ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
        f"F1={metrics['best_f1']:.4f}"
    )
    return metrics


if __name__ == "__main__":
    train()
