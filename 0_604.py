# ======================== IMPORTS & CONFIG ========================
import os
import sys

import numpy as np
import pandas as pd
import polars as pl

from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Путь к данным соревнования
DATA_DIR = "/kaggle/input/nfl-big-data-bowl-2026-prediction"

# Делаем модуль с evaluation API доступным для импорта (если он есть)
sys.path.append(DATA_DIR)
try:
    from kaggle_evaluation.nfl_inference_server import NFLInferenceServer
    HAS_EVAL_SERVER = True
except ModuleNotFoundError:
    NFLInferenceServer = None
    HAS_EVAL_SERVER = False
    print(
        "WARNING: kaggle_evaluation not found. "
        "Если ты запускаешь скрипт локально (не на Kaggle), это ожидаемо: "
        "сервер инференса и автогенерация submission.csv работать не будут."
    )

# Фиксируем seed для воспроизводимости
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 70)
print("NFL BIG DATA BOWL 2026 - LGBM+MLP v6 (ballistic + neighbor feats, Eval API)")
print("=" * 70)

# ======================== FEATURE LISTS ===========================
# Явный список числовых фичей, которые должны быть в train / test матрице
FEATURES = [
    # геометрия текущего состояния
    "x_last", "y_last",
    "s", "a", "o", "dir",

    # компоненты скорости и ускорения в осях поля
    "vx", "vy",
    "ax_comp", "ay_comp",
    "dir_sin", "dir_cos",
    "o_sin", "o_cos",

    # последний шаг до паса (динамика)
    "last_step_dx", "last_step_dy",
    "last_step_dist", "last_step_speed",
    "ds_last", "da_last",
    "d_dir_last", "d_o_last",

    # временные признаки
    "frame_offset", "time_offset",
    "num_frames_output",
    "frac_of_flight",
    "frames_left",
    "time_to_land",
    "remaining_flight_frac",

    # отношение к точке приземления мяча
    "dist_to_ball_land",
    "angle_to_ball_land",
    "dist_to_ball_land_per_frame",
    "cos_dir_to_ball",
    "cos_orient_to_ball",
    "req_speed_to_ball",
    "speed_minus_req",
    "last_step_speed_minus_req",

    # баллистические фичи: проекции движения к моменту приземления мяча
    "proj_x_vel", "proj_y_vel",
    "proj_x_acc", "proj_y_acc",
    "proj_vel_dx_to_ball", "proj_vel_dy_to_ball",
    "proj_acc_dx_to_ball", "proj_acc_dy_to_ball",
    "dist_proj_vel_to_ball", "dist_proj_acc_to_ball",

    # стандартизированные по направлению x
    "x_std",
    "ball_land_x_std",
    "dx_to_land_std",
    "dy_to_land",

    # позиция по ширине и длине поля
    "dist_to_sideline",
    "dist_to_center",
    "yardline_100",
    "yardline_norm",
    "dist_to_endzone",

    # таргет ресивер
    "dist_to_target_last",
    "dx_to_target_last",
    "dy_to_target_last",
    "angle_to_target",
    "cos_dir_to_target",
    "cos_orient_to_target",
    "is_target",

    # контекст розыгрыша / игрока
    "absolute_yardline_number",
    "player_height", "player_weight",
    "bmi",

    # новые фичи окружения игроков
    "min_dist_any",
    "min_dist_teammate",
    "min_dist_opponent",
]

# Категориальные признаки (для LGBM и последующего OHE в MLP)
CAT_FEATS = ["player_role", "player_side", "play_direction", "frame_bin"]

# Какие колонки тянем из "последнего наблюдения до паса"
BASE_COLS = [
    "game_id", "play_id", "nfl_id",
    "x_last", "y_last",
    "x_prev", "y_prev",
    "s", "a", "o", "dir",
    "s_prev", "a_prev", "o_prev", "dir_prev",
    "player_role", "player_side",
    "num_frames_output",
    "ball_land_x", "ball_land_y",
    "target_last_x", "target_last_y", "target_nfl_id",
    "play_direction",
    "absolute_yardline_number",
    "player_height", "player_weight",
    "player_to_predict",          # фильтрация train

    # фичи окружения
    "min_dist_any",
    "min_dist_teammate",
    "min_dist_opponent",
]

# Глобальные модели и объекты, чтобы их использовал predict()
MODEL_DX_LGBM = None
MODEL_DY_LGBM = None
MODEL_DX_MLP = None
MODEL_DY_MLP = None
MLP_SCALER = None
MLP_COLUMNS = None


# ======================== STEP 1: LOAD RAW TRAIN DATA =============
def load_train(data_dir: str):
    """
    Загружаем train input/output по всем неделям.
    Возвращаем concat по всем неделям:
      - df_in: трекинг до паса (train/input_2023_wXX.csv)
      - df_out: истинные координаты во время полёта мяча (train/output_2023_wXX.csv)
    """
    train_dir = os.path.join(data_dir, "train")
    df_in_list = []
    df_out_list = []

    print("\n[STEP 1] Loading training inputs/outputs by week...")
    for w in range(1, 19):
        ip = os.path.join(train_dir, f"input_2023_w{w:02d}.csv")
        op = os.path.join(train_dir, f"output_2023_w{w:02d}.csv")
        if os.path.exists(ip) and os.path.exists(op):
            df_i = pd.read_csv(ip)
            df_o = pd.read_csv(op)
            df_in_list.append(df_i)
            df_out_list.append(df_o)
            print(f"  Week {w:02d}: input {df_i.shape}, output {df_o.shape}")
        else:
            print(f"  Week {w:02d}: files not found, skipping")

    if not df_in_list or not df_out_list:
        raise FileNotFoundError(
            f"No train CSV files found in {train_dir}. "
            "Проверь, что датасет nfl-big-data-bowl-2026-prediction подключен в разделе Data."
        )

    df_in = pd.concat(df_in_list, ignore_index=True)
    df_out = pd.concat(df_out_list, ignore_index=True)
    print(f"[STEP 1] Train inputs: {df_in.shape}, train outputs: {df_out.shape}")
    return df_in, df_out


# ======================== STEP 2: LAST OBSERVATION + NEIGHBORS ==== 
def height_to_inches(ht):
    """Переводим рост из формата '6-2' в дюймы (6*12 + 2)."""
    if isinstance(ht, str) and "-" in ht:
        try:
            feet, inches = ht.split("-")
            return int(feet) * 12 + int(inches)
        except Exception:
            return np.nan
    return np.nan


def add_neighbor_features(df_last: pd.DataFrame) -> pd.DataFrame:
    """
    Фичи окружения на последнем pre-throw кадре:
      - min_dist_any: минимальное расстояние до любого другого игрока
      - min_dist_teammate: минимальное расстояние до игрока той же стороны (Offense/Defense)
      - min_dist_opponent: минимальное расстояние до игрока противоположной стороны
    """
    df_last = df_last.copy()

    def per_play(group: pd.DataFrame) -> pd.DataFrame:
        coords = group[["x_last", "y_last"]].to_numpy(dtype=float)
        n = coords.shape[0]
        if n <= 1:
            group["min_dist_any"] = 0.0
            group["min_dist_teammate"] = 0.0
            group["min_dist_opponent"] = 0.0
            return group

        # попарные расстояния между игроками в розыгрыше
        dists = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1))
        np.fill_diagonal(dists, np.inf)

        min_any = dists.min(axis=1)

        side = group.get("player_side")
        if side is None:
            group["min_dist_any"] = np.where(np.isfinite(min_any), min_any, 0.0)
            group["min_dist_teammate"] = group["min_dist_any"]
            group["min_dist_opponent"] = group["min_dist_any"]
            return group

        side_vals = side.to_numpy()
        same_mask = side_vals[:, None] == side_vals[None, :]
        opp_mask = ~same_mask

        d_same = np.where(same_mask, dists, np.inf)
        d_opp = np.where(opp_mask, dists, np.inf)

        min_same = d_same.min(axis=1)
        min_opp = d_opp.min(axis=1)

        group["min_dist_any"] = np.where(np.isfinite(min_any), min_any, 0.0)
        group["min_dist_teammate"] = np.where(np.isfinite(min_same), min_same, 0.0)
        group["min_dist_opponent"] = np.where(np.isfinite(min_opp), min_opp, 0.0)
        return group

    df_last = df_last.groupby(["game_id", "play_id"], group_keys=False).apply(per_play)
    return df_last


def prepare_last_obs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Перевод трекинга в "последнее наблюдение до паса":
      - сортируем по времени,
      - считаем признаки предыдущего кадра (x_prev, s_prev, ...),
      - берём последнюю строку по (game_id, play_id, nfl_id),
      - конвертируем рост и добавляем фичи окружения.
    """
    # Сортируем по времени внутри (game_id, play_id, nfl_id)
    df_sorted = (
        df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
          .reset_index(drop=True)
    )
    group_keys = ["game_id", "play_id", "nfl_id"]

    # Признаки предыдущего шага
    for col in ["x", "y", "s", "a", "o", "dir"]:
        df_sorted[f"{col}_prev"] = df_sorted.groupby(group_keys)[col].shift(1)

    # Берём последнее наблюдение (до паса) по игроку в розыгрыше
    df_last = (
        df_sorted
        .groupby(group_keys, as_index=False)
        .last()
    )
    df_last = df_last.rename(columns={"x": "x_last", "y": "y_last"})

    # Рост в дюймах
    if "player_height" in df_last.columns:
        df_last["player_height"] = df_last["player_height"].apply(height_to_inches)
    else:
        df_last["player_height"] = np.nan

    # Фичи окружения
    df_last = add_neighbor_features(df_last)

    print(f"[STEP 2] Last observations table: {df_last.shape}")
    return df_last


def add_target_info(df_last: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляем координаты таргет ресивера для каждого игрока в розыгрыше.
    """
    df_last = df_last.copy()
    mask_target = df_last.get("player_role", "") == "Targeted Receiver"
    targets = df_last.loc[
        mask_target,
        ["game_id", "play_id", "nfl_id", "x_last", "y_last"],
    ].copy()

    targets = targets.rename(
        columns={
            "nfl_id": "target_nfl_id",
            "x_last": "target_last_x",
            "y_last": "target_last_y",
        }
    )

    df_last = df_last.merge(
        targets[["game_id", "play_id", "target_last_x", "target_last_y", "target_nfl_id"]],
        on=["game_id", "play_id"],
        how="left",
    )

    print(
        "[STEP 2] Added target receiver info: "
        f"{df_last['target_nfl_id'].notna().sum()} plays with target"
    )
    return df_last


# ======================== STEP 3: CORE FEATURE ENGINEERING ========
def _angle_diff_deg(a_deg, b_deg):
    """Разность углов в градусах в диапазоне [-180, 180]."""
    diff = (a_deg - b_deg + 180.0) % 360.0 - 180.0
    return diff


def create_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Генерация признаков:
      - динамика последнего шага (dx/dy, дельты скоростей и углов),
      - временные признаки полёта мяча,
      - геометрия/углы относительно точки приземления,
      - требуемая скорость и её зазор с текущей,
      - баллистические проекции траектории игрока до точки приземления,
      - стандартизированные координаты по направлению розыгрыша,
      - позиция по полю,
      - признаки относительно таргет ресивера,
      - BMI и фичи окружения,
      - при is_train=True формируются таргеты dx, dy.
    """
    df = df.copy()

    # -------- Базовые величины и углы --------
    s = df["s"].fillna(0.0)
    a = df["a"].fillna(0.0)
    dir_deg = df["dir"].fillna(0.0)
    o_deg = df["o"].fillna(0.0)
    dir_rad = np.deg2rad(dir_deg)
    o_rad = np.deg2rad(o_deg)

    # -------- Компоненты скорости и ускорения --------
    df["vx"] = s * np.cos(dir_rad)
    df["vy"] = s * np.sin(dir_rad)
    df["ax_comp"] = a * np.cos(dir_rad)
    df["ay_comp"] = a * np.sin(dir_rad)

    df["dir_sin"] = np.sin(dir_rad)
    df["dir_cos"] = np.cos(dir_rad)
    df["o_sin"] = np.sin(o_rad)
    df["o_cos"] = np.cos(o_rad)

    # -------- Динамика последнего шага до паса --------
    if "x_prev" in df.columns:
        dx_last = (df["x_last"] - df["x_prev"]).fillna(0.0)
        dy_last = (df["y_last"] - df["y_prev"]).fillna(0.0)
        df["last_step_dx"] = dx_last
        df["last_step_dy"] = dy_last
        df["last_step_dist"] = np.sqrt(dx_last ** 2 + dy_last ** 2)
        # 10 кадров в секунду
        df["last_step_speed"] = df["last_step_dist"] * 10.0
    else:
        df["last_step_dx"] = 0.0
        df["last_step_dy"] = 0.0
        df["last_step_dist"] = 0.0
        df["last_step_speed"] = 0.0

    if "s_prev" in df.columns:
        df["ds_last"] = (df["s"] - df["s_prev"]).fillna(0.0)
    else:
        df["ds_last"] = 0.0

    if "a_prev" in df.columns:
        df["da_last"] = (df["a"] - df["a_prev"]).fillna(0.0)
    else:
        df["da_last"] = 0.0

    if "dir_prev" in df.columns:
        df["d_dir_last"] = _angle_diff_deg(df["dir"], df["dir_prev"]).fillna(0.0)
    else:
        df["d_dir_last"] = 0.0

    if "o_prev" in df.columns:
        df["d_o_last"] = _angle_diff_deg(df["o"], df["o_prev"]).fillna(0.0)
    else:
        df["d_o_last"] = 0.0

    # -------- Время полёта / идентификатор кадра --------
    df["frame_offset"] = df["frame_id"]
    df["time_offset"] = df["frame_offset"] / 10.0  # 10 кадров в секунду

    if "num_frames_output" in df.columns:
        nfo = df["num_frames_output"].replace(0, np.nan)
        df["frac_of_flight"] = (df["frame_offset"] / nfo).clip(lower=0, upper=1)
        df["frac_of_flight"] = df["frac_of_flight"].fillna(0.0)
        df["frames_left"] = (nfo - df["frame_offset"]).clip(lower=0).fillna(0.0)
    else:
        df["frac_of_flight"] = 0.0
        df["frames_left"] = 0.0

    df["time_to_land"] = df["frames_left"] / 10.0
    df["remaining_flight_frac"] = (1.0 - df["frac_of_flight"]).clip(lower=0.0, upper=1.0)

    # -------- Дискретизированная фаза полёта (категория) --------
    df["frame_bin"] = 0
    df.loc[df["frac_of_flight"] > 0.33, "frame_bin"] = 1
    df.loc[df["frac_of_flight"] > 0.66, "frame_bin"] = 2

    # -------- Геометрия относительно точки приземления мяча --------
    df["dist_to_ball_land"] = np.sqrt(
        (df["ball_land_x"] - df["x_last"]) ** 2 +
        (df["ball_land_y"] - df["y_last"]) ** 2
    )
    df["angle_to_ball_land"] = np.arctan2(
        df["ball_land_y"] - df["y_last"],
        df["ball_land_x"] - df["x_last"],
    )

    frames_left_safe = df["frames_left"].replace(0, np.nan)
    df["dist_to_ball_land_per_frame"] = df["dist_to_ball_land"] / frames_left_safe
    df["dist_to_ball_land_per_frame"] = (
        df["dist_to_ball_land_per_frame"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    df["cos_dir_to_ball"] = np.cos(df["angle_to_ball_land"] - dir_rad)
    df["cos_orient_to_ball"] = np.cos(df["angle_to_ball_land"] - o_rad)

    # Требуемая средняя скорость до мяча и зазор с текущей
    time_to_land_safe = df["time_to_land"].replace(0, np.nan)
    df["req_speed_to_ball"] = df["dist_to_ball_land"] / time_to_land_safe
    df["req_speed_to_ball"] = (
        df["req_speed_to_ball"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    df["speed_minus_req"] = s - df["req_speed_to_ball"]
    df["last_step_speed_minus_req"] = df["last_step_speed"] - df["req_speed_to_ball"]

    # -------- Баллистические проекции --------
    dt = df["time_to_land"]
    # Траектория при постоянной скорости
    df["proj_x_vel"] = df["x_last"] + df["vx"] * dt
    df["proj_y_vel"] = df["y_last"] + df["vy"] * dt
    # Траектория при учёте ускорения
    df["proj_x_acc"] = df["x_last"] + df["vx"] * dt + 0.5 * df["ax_comp"] * (dt ** 2)
    df["proj_y_acc"] = df["y_last"] + df["vy"] * dt + 0.5 * df["ay_comp"] * (dt ** 2)

    df["proj_vel_dx_to_ball"] = df["ball_land_x"] - df["proj_x_vel"]
    df["proj_vel_dy_to_ball"] = df["ball_land_y"] - df["proj_y_vel"]
    df["proj_acc_dx_to_ball"] = df["ball_land_x"] - df["proj_x_acc"]
    df["proj_acc_dy_to_ball"] = df["ball_land_y"] - df["proj_y_acc"]

    df["dist_proj_vel_to_ball"] = np.sqrt(
        df["proj_vel_dx_to_ball"] ** 2 + df["proj_vel_dy_to_ball"] ** 2
    )
    df["dist_proj_acc_to_ball"] = np.sqrt(
        df["proj_acc_dx_to_ball"] ** 2 + df["proj_acc_dy_to_ball"] ** 2
    )

    # -------- Стандартизируем координаты по направлению розыгрыша --------
    play_dir = df.get("play_direction", "right").fillna("right")
    is_left = (play_dir == "left").astype(int)

    # x_std: всегда направление атаки "вправо"
    df["x_std"] = np.where(is_left == 1, 120.0 - df["x_last"], df["x_last"])
    df["ball_land_x_std"] = np.where(
        is_left == 1, 120.0 - df["ball_land_x"], df["ball_land_x"]
    )

    df["dx_to_land_std"] = df["ball_land_x_std"] - df["x_std"]
    df["dy_to_land"] = df["ball_land_y"] - df["y_last"]

    # -------- Позиция по ширине/длине поля --------
    df["dist_to_sideline"] = np.minimum(df["y_last"], 53.3 - df["y_last"])
    df["dist_to_center"] = np.abs(df["y_last"] - 53.3 / 2.0)

    yard = df["absolute_yardline_number"].fillna(50.0)
    yard_100 = yard.clip(lower=0.0, upper=100.0)
    df["yardline_100"] = yard_100
    df["yardline_norm"] = yard_100 / 100.0
    df["dist_to_endzone"] = 100.0 - yard_100

    # -------- Таргет ресивер --------
    df["dist_to_target_last"] = np.sqrt(
        (df["target_last_x"] - df["x_last"]) ** 2 +
        (df["target_last_y"] - df["y_last"]) ** 2
    )

    df["dx_to_target_last"] = df["target_last_x"] - df["x_last"]
    df["dy_to_target_last"] = df["target_last_y"] - df["y_last"]
    df["angle_to_target"] = np.arctan2(
        df["target_last_y"] - df["y_last"],
        df["target_last_x"] - df["x_last"],
    )

    df["cos_dir_to_target"] = np.cos(df["angle_to_target"] - dir_rad)
    df["cos_orient_to_target"] = np.cos(df["angle_to_target"] - o_rad)

    df["is_target"] = (df["nfl_id"] == df["target_nfl_id"]).astype(int)

    # -------- Физика игрока --------
    h = df["player_height"].replace(0, np.nan)
    w = df["player_weight"].replace(0, np.nan)
    df["bmi"] = 703.0 * (w / (h ** 2))
    df["bmi"] = df["bmi"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # -------- Таргеты только в train --------
    if is_train:
        df["dx"] = df["x"] - df["x_last"]
        df["dy"] = df["y"] - df["y_last"]

    return df


# ======================== STEP 4: BUILD TRAIN TABLE ===============
def prepare_train(df_in: pd.DataFrame, df_out: pd.DataFrame) -> pd.DataFrame:
    """
    Формируем обучающую выборку:
      - считаем last_obs и фичи окружения,
      - добавляем таргет ресивера,
      - мержим с output (истинные координаты во время полёта),
      - строим фичи,
      - фильтруем по player_to_predict == True (если есть).
    """
    print("\n[STEP 4] Preparing training features...")
    last_obs = prepare_last_obs(df_in)
    last_obs = add_target_info(last_obs)

    cols_to_keep_existing = [c for c in BASE_COLS if c in last_obs.columns]

    train = df_out.merge(
        last_obs[cols_to_keep_existing],
        on=["game_id", "play_id", "nfl_id"],
        how="left",
    )

    train = create_features(train, is_train=True)

    # Фильтрация только на игроков, которые реально участвуют в метрике
    if "player_to_predict" in train.columns:
        before = len(train)
        train = train[train["player_to_predict"].astype(bool)].copy()
        after = len(train)
        print(f"[STEP 4] Filtered to player_to_predict==True: {before} -> {after} rows")

    print(f"[STEP 4] Final training DataFrame shape: {train.shape}")
    return train


def sanity_check_train(train: pd.DataFrame):
    """
    Базовые sanity-checks по train:
      - размеры и уникальные сущности,
      - краткая статистика,
      - NaN по основным фичам.
    """
    print("\n[STEP 4] Sanity checks on training data...")
    print(f"  train shape: {train.shape}")

    for col in ["game_id", "play_id", "nfl_id"]:
        if col in train.columns:
            print(f"  # unique {col}: {train[col].nunique()}")

    key_numeric = [c for c in ["s", "a", "dist_to_ball_land", "time_to_land", "vx", "vy"]
                   if c in train.columns]
    if key_numeric:
        print("\n  Basic stats for key numeric features:")
        print(train[key_numeric].describe(percentiles=[0.25, 0.5, 0.75]).T)

    # NaN по числовым фичам
    numeric_cols = [c for c in FEATURES if c in train.columns]
    if numeric_cols:
        nan_counts = train[numeric_cols].isna().sum()
        nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
        if not nan_counts.empty:
            print("\n  Top features by NaN count:")
            print(nan_counts.head(10))
        else:
            print("\n  No NaNs detected in numeric FEATURES subset.")


# ======================== STEP 5: MODEL TRAINING ==================
def train_models(train: pd.DataFrame):
    """
    Обучаем две модели LGBM (dx, dy) + MLP для обучения на остатках.
    Итоговый предикт = LGBM + 0.5 * MLP_residual.
    """
    global MODEL_DX_LGBM, MODEL_DY_LGBM
    global MODEL_DX_MLP, MODEL_DY_MLP
    global MLP_SCALER, MLP_COLUMNS

    print("\n[STEP 5] Training models (LGBM + MLP residual)...")

    # Гарантируем наличие всех фич
    for col in FEATURES:
        if col not in train.columns:
            train[col] = 0.0
    for col in CAT_FEATS:
        if col not in train.columns:
            train[col] = "unknown"

    X = train[FEATURES + CAT_FEATS].copy()
    for c in CAT_FEATS:
        X[c] = X[c].astype("category")

    y_dx = train["dx"].values
    y_dy = train["dy"].values

    # -------- LGBM базовые модели --------
    lgbm_params = dict(
        objective="regression",
        boosting_type="gbdt",
        n_estimators=1600,
        learning_rate=0.04,
        num_leaves=127,
        min_data_in_leaf=80,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        reg_alpha=0.1,
        reg_lambda=0.2,
        n_jobs=-1,
        verbosity=-1,
        random_state=RANDOM_STATE,
    )

    print("  Training LGBM for dx...")
    model_dx = LGBMRegressor(**lgbm_params)
    model_dx.fit(X, y_dx, categorical_feature=CAT_FEATS)

    print("  Training LGBM for dy...")
    model_dy = LGBMRegressor(**lgbm_params)
    model_dy.fit(X, y_dy, categorical_feature=CAT_FEATS)

    MODEL_DX_LGBM = model_dx
    MODEL_DY_LGBM = model_dy

    # -------- MLP по остаткам --------
    print("  Preparing data for MLP residual models...")
    X_dense = pd.get_dummies(X, columns=CAT_FEATS, dummy_na=True)
    MLP_COLUMNS = list(X_dense.columns)

    MLP_SCALER = StandardScaler()
    X_scaled = MLP_SCALER.fit_transform(X_dense)

    lgbm_dx_pred = model_dx.predict(X)
    lgbm_dy_pred = model_dy.predict(X)

    res_dx = y_dx - lgbm_dx_pred
    res_dy = y_dy - lgbm_dy_pred

    mlp_params = dict(
        hidden_layer_sizes=(128, 128),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=40,
        batch_size=1024,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=RANDOM_STATE,
        verbose=False,
    )

    print("  Training MLP for residual dx...")
    mlp_dx = MLPRegressor(**mlp_params)
    mlp_dx.fit(X_scaled, res_dx)

    print("  Training MLP for residual dy...")
    mlp_dy = MLPRegressor(**mlp_params)
    mlp_dy.fit(X_scaled, res_dy)

    MODEL_DX_MLP = mlp_dx
    MODEL_DY_MLP = mlp_dy

    # -------- Offline sanity metric (на train) --------
    try:
        res_dx_mlp = mlp_dx.predict(X_scaled)
        res_dy_mlp = mlp_dy.predict(X_scaled)

        train_dx_pred = lgbm_dx_pred + 0.5 * res_dx_mlp
        train_dy_pred = lgbm_dy_pred + 0.5 * res_dy_mlp

        rmse_dx = np.sqrt(np.mean((y_dx - train_dx_pred) ** 2))
        rmse_dy = np.sqrt(np.mean((y_dy - train_dy_pred) ** 2))
        rmse_total = np.sqrt(np.mean(
            (y_dx - train_dx_pred) ** 2 + (y_dy - train_dy_pred) ** 2
        ))

        print("\n[STEP 5] Offline train metrics (RMSE):")
        print(f"  dx RMSE: {rmse_dx:.4f}")
        print(f"  dy RMSE: {rmse_dy:.4f}")
        print(f"  combined RMSE (dx,dy): {rmse_total:.4f}")
        print("  На Kaggle это решение даёт публичный LB ~ 0.604 (по описанию автора).")
    except Exception as e:
        print(f"[STEP 5] Failed to compute offline metrics: {e}")

    print("\n[STEP 5] ✓ Models (LGBM + MLP residual) trained on full dataset")
    return model_dx, model_dy, mlp_dx, mlp_dy


# ======================== STEP 6: INFERENCE HELPERS ===============
def prepare_inference_batch(test_pd: pd.DataFrame, test_input_pd: pd.DataFrame) -> pd.DataFrame:
    """
    Готовим строки для инференса:
      - строим last_obs по test_input (одно наблюдение до паса на игрока),
      - добавляем таргет ресивера и фичи окружения,
      - мержим с текущим батчем test (id, game_id, play_id, nfl_id, frame_id),
      - строим признаки как в train (create_features с is_train=False).
    """
    last_obs = prepare_last_obs(test_input_pd)
    last_obs = add_target_info(last_obs)

    cols_to_keep_existing = [c for c in BASE_COLS if c in last_obs.columns]

    test_rows = test_pd.merge(
        last_obs[cols_to_keep_existing],
        on=["game_id", "play_id", "nfl_id"],
        how="left",
    )

    test_rows = create_features(test_rows, is_train=False)
    print(f"[STEP 6] Inference batch prepared: {test_rows.shape}")
    return test_rows


# ======================== STEP 7: PREDICT API (EVAL) ==============
def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pl.DataFrame:
    """
    Основная функция, которую вызывает Kaggle evaluation API.
    На входе:
      - test: Polars DataFrame с колонками [id, game_id, play_id, nfl_id, frame_id]
      - test_input: Polars DataFrame с трекингом до паса для соответствующих розыгрышей.
    На выходе:
      - Polars DataFrame с колонками ["x", "y"] той же длины, что и test.
    """
    assert MODEL_DX_LGBM is not None and MODEL_DY_LGBM is not None, "Models are not trained"

    # Переводим в pandas для переиспользования наших функций
    test_pd = test.to_pandas()
    test_input_pd = test_input.to_pandas()

    # Формируем фичи для текущего батча
    test_rows = prepare_inference_batch(test_pd, test_input_pd)

    # Гарантируем наличие всех фич
    for col in FEATURES:
        if col not in test_rows.columns:
            test_rows[col] = 0.0
    for col in CAT_FEATS:
        if col not in test_rows.columns:
            test_rows[col] = "unknown"

    X_test = test_rows[FEATURES + CAT_FEATS].copy()
    for c in CAT_FEATS:
        X_test[c] = X_test[c].astype("category")

    # Предсказания LGBM
    pred_dx_lgbm = MODEL_DX_LGBM.predict(X_test)
    pred_dy_lgbm = MODEL_DY_LGBM.predict(X_test)

    # Коррекция MLP по остаткам (если обучен)
    if (MODEL_DX_MLP is not None) and (MLP_SCALER is not None) and (MLP_COLUMNS is not None):
        X_test_dense = pd.get_dummies(X_test, columns=CAT_FEATS, dummy_na=True)
        # Выравниваем признаки под трейн
        X_test_dense = X_test_dense.reindex(columns=MLP_COLUMNS, fill_value=0.0)
        X_test_scaled = MLP_SCALER.transform(X_test_dense)

        res_dx_mlp = MODEL_DX_MLP.predict(X_test_scaled)
        res_dy_mlp = MODEL_DY_MLP.predict(X_test_scaled)

        # Шаг "градиентного бустинга" поверх LGBM
        pred_dx = pred_dx_lgbm + 0.5 * res_dx_mlp
        pred_dy = pred_dy_lgbm + 0.5 * res_dy_mlp
    else:
        pred_dx = pred_dx_lgbm
        pred_dy = pred_dy_lgbm

    # Восстанавливаем координаты
    x_pred = test_rows["x_last"].to_numpy() + pred_dx
    y_pred = test_rows["y_last"].to_numpy() + pred_dy

    # Ограничиваем полем
    x_pred = np.clip(x_pred, 0, 120)
    y_pred = np.clip(y_pred, 0, 53.3)

    # Возвращаем Polars DataFrame только с x, y
    return pl.DataFrame({"x": x_pred, "y": y_pred})


# ======================== STEP 8: MAIN (TRAIN + SERVER) ===========
if __name__ == "__main__":
    # 1) Обучаемся на публичном train
    df_in, df_out = load_train(DATA_DIR)
    train_df = prepare_train(df_in, df_out)
    sanity_check_train(train_df)
    train_models(train_df)

    # 2) Поднимаем inference server, только если модуль есть
    if HAS_EVAL_SERVER and NFLInferenceServer is not None:
        inference_server = NFLInferenceServer(predict)

        # В режиме скоринга Kaggle выставляет переменную окружения
        if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
            # Боевой режим: Kaggle будет стучаться к серверу и сам соберёт submission
            print("\n[STEP 8] Starting inference server for Kaggle rerun...")
            inference_server.serve()
        else:
            # Локальный режим на Kaggle: можно прогнать mock test и получить submission.csv
            print("\n[STEP 8] Running local gateway to generate submission.csv on public mock test...")
            inference_server.run_local_gateway((DATA_DIR,))
            print("[STEP 8] ✓ submission.parquet / submission.csv should now be created")
    else:
        print(
            "\n[STEP 8] NFLInferenceServer недоступен (kaggle_evaluation не найден). "
            "Локально можно тренировать модель и дебажить фичи, "
            "но для генерации submission нужно запускать код в Kaggle "
            "с подключённым датасетом 'nfl-big-data-bowl-2026-prediction' во вкладке Data."
        )
