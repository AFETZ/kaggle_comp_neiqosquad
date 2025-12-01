# NFL Big Data Bowl 2026 – LGBM+MLP v6 Data Pipeline

Этот документ описывает полный путь данных от сырых файлов соревнования **NFL Big Data Bowl 2026** до обучения и инференса модели **LGBM + MLP по остаткам**. Фокус — на пайплайне подготовки данных и структуре кода, а не на деталях реализации.

---

## 1. Краткое резюме пайплайна

Скрипт загружает все недельные `train`-файлы (`input` и `output`), строит для каждого игрока в каждом розыгрыше последнее наблюдение до паса с динамикой предыдущего шага и фичами окружения (минимальные дистанции до тиммейтов и оппонентов). Затем это последнее наблюдение мержится с истинными конечными координатами игроков из `output`, и поверх объединённой таблицы считается большой набор геометрических, баллистических, временных и контекстных признаков, включая отношение игрока к траектории мяча и таргет-ресиверу. Пропуски обрабатываются локально в блоках фичей (заполнение нулями, клиппинг, очистка бесконечностей), так чтобы LGBM мог потреблять сырые значения, а MLP — стандартизированные, плотные признаки без NaN. Итоговая обучающая матрица строится без явного разбиения на train/valid: две модели LGBM и две MLP по остаткам обучаются на всём train и используются в инференсе через Kaggle evaluation API. На тесте повторяется тот же путь: строится таблица последних наблюдений до паса, считаются те же фичи, предсказываются смещения `dx`, `dy`, и координаты `x, y` восстанавливаются и клиппятся полем.

---

## 2. Структура пайплайна (по шагам)

Ниже — поэтапная схема того, как данные проходят путь от сырых файлов до обученной модели и функции `predict` для Kaggle evaluation API.

### Шаг 1. Загрузка и первичная агрегация train-данных

**Назначение:** собрать единый `df_in` (трекинг до паса) и `df_out` (таргеты) по всем неделям.

1.1. Источники данных  
- `train/input_2023_wXX.csv`  
  Содержит трекинг игроков **до паса**: координаты, скорость, ускорение, ориентация, идентификаторы кадров (`frame_id`), игроков (`nfl_id`), розыгрышей (`game_id`, `play_id`) и т. д.  
- `train/output_2023_wXX.csv`  
  Содержит истинные координаты игроков на кадрах **полёта мяча** (после выпуска мяча).

1.2. Объединение по неделям  
- Для недель `w = 1..18`:
  - Если оба файла (`input_2023_wXX.csv` и `output_2023_wXX.csv`) существуют — читаются в `df_i`, `df_o`.
  - Формируется список `df_in_list` и `df_out_list`.  
- В конце:
  - `df_in = concat(df_in_list)` — полный трекинг до паса.
  - `df_out = concat(df_out_list)` — полный набор таргетов.

1.3. Ключи и структура  
Основные ключи, используемые дальше:
- `game_id`, `play_id` — идентификаторы розыгрыша;
- `nfl_id` — игрок;
- `frame_id` — кадр (номер кадра полёта в `df_out`, номер кадра до паса в `df_in`).

---

### Шаг 2. Построение таблицы «последнее наблюдение до паса» и фичей окружения

**Назначение:** свести трекинг до единого снимка на момент перед пасом для каждого `(game_id, play_id, nfl_id)` и добавить геометрию окружения.

#### 2.1. Нормализация роста

- Вспомогательная функция переводит `player_height` из строкового формата `"6-2"` в дюймы (`6 * 12 + 2`).
- При невозможности распарсить строку — возвращается `NaN`.

#### 2.2. Последнее наблюдение до паса (`prepare_last_obs`)

1. Сортировка  
   - Данные `df_in` сортируются по:
     - `["game_id", "play_id", "nfl_id", "frame_id"]`
   - Это гарантирует корректный порядок кадров во времени по игроку.

2. Признаки предыдущего кадра (lag-фичи)  
   - Для каждой группы `(game_id, play_id, nfl_id)` через `groupby().shift(1)` считаются:
     - `x_prev`, `y_prev`,
     - `s_prev`, `a_prev`,
     - `o_prev`, `dir_prev`.  
   - Эти признаки нужны для динамики последнего шага.

3. Выбор последнего кадра  
   - Берётся по одной последней строке на комбинацию `(game_id, play_id, nfl_id)`:
     - Это «последнее наблюдение до паса» для данного игрока.
   - Переименование:
     - `x -> x_last`, `y -> y_last`.

4. Конвертация роста  
   - `player_height` приводится к дюймам через функцию из п. 2.1.
   - Если нет колонки роста — создаётся как `NaN`.

#### 2.3. Фичи окружения (`add_neighbor_features`)

1. Группировка по розыгрышу  
   - Для каждого `(game_id, play_id)` рассматриваются все игроки на последнем кадре до паса.

2. Попарные расстояния  
   - Используются `x_last`, `y_last` для расчёта матрицы расстояний между игроками.
   - Диагональ ставится в `inf`, чтобы игнорировать расстояние до самого себя.

3. Минимальные расстояния  
   - `min_dist_any` — минимум по строке матрицы (любой другой игрок).
   - По `player_side` (Offense / Defense):
     - `min_dist_teammate` — минимум до игрока той же стороны.
     - `min_dist_opponent` — минимум до игрока противоположной стороны.
   - При отсутствии `player_side` все три расстояния сводятся к одному (`min_dist_any`).

4. Для групп размера 0–1  
   - Все три расстояния устанавливаются в 0.

#### 2.4. Итоговая таблица last_obs

- Одна строка на `(game_id, play_id, nfl_id)`:
  - Содержит:
    - `x_last`, `y_last`,
    - lag-фичи (`x_prev`, `y_prev`, `s_prev`, `a_prev`, `o_prev`, `dir_prev`),
    - `s`, `a`, `o`, `dir` на последнем кадре,
    - `player_side`, `player_role`,
    - `num_frames_output`, `ball_land_x`, `ball_land_y`,
    - `absolute_yardline_number`, `play_direction`,
    - `player_height`, `player_weight`,
    - `player_to_predict` (маркер для фильтрации train),
    - фичи окружения `min_dist_any`, `min_dist_teammate`, `min_dist_opponent` (добавлены на этом шаге).

---

### Шаг 3. Встраивание информации о таргет-ресивере и сбор обучающей таблицы

**Назначение:** дать каждому игроку знание о положении таргет-ресивера на последнем pre-throw кадре и связать это с таргетами `df_out`.

#### 3.1. Информация о таргет-ресивере (`add_target_info`)

1. Выбор таргет-ресиверов  
   - В `df_last` выбираются строки, где `player_role == "Targeted Receiver"`.

2. Формирование таргет-таблицы  
   - Из выбранных строк берутся:
     - `game_id`, `play_id`, `nfl_id`, `x_last`, `y_last`.
   - Переименование:
     - `nfl_id -> target_nfl_id`,
     - `x_last -> target_last_x`,
     - `y_last -> target_last_y`.

3. Merge обратно к `df_last`  
   - Объединение по `(game_id, play_id)` с режимом `left`.
   - Каждый игрок получает:
     - координаты таргета (`target_last_x`, `target_last_y`),
     - идентификатор (`target_nfl_id`).
   - Если в розыгрыше таргета нет (аномалия) — значения будут `NaN`.

#### 3.2. Формирование обучающей таблицы (`prepare_train`)

1. Построение last_obs  
   - `last_obs = prepare_last_obs(df_in)`  
   - `last_obs = add_target_info(last_obs)`

2. Выбор нужных колонок  
   - Список `BASE_COLS` описывает набор колонок, которые переносятся из `last_obs` в будущий train.
   - Фактически выбираются только те, которые реально присутствуют в `last_obs`.

3. Merge с таргетами  
   - `train = df_out.merge(last_obs[BASE_COLS_actual], on=["game_id", "play_id", "nfl_id"], how="left")`
   - На каждую строку `df_out` (кадр полёта мяча для игрока) навешивается состояние игрока на последнем кадре до паса.

4. Генерация фичей  
   - Вызов `create_features(train, is_train=True)`:
     - Строит полный набор фичей (см. Шаг 4).
     - Дополнительно создаёт таргеты `dx`, `dy`:
       - `dx = x - x_last`,
       - `dy = y - y_last`.

5. Фильтрация по `player_to_predict`  
   - Если колонка есть:
     - оставляются только строки, где `player_to_predict == True`.
   - Это приведение к тому же пространству игроков, по которым Kaggle считает метрику.

6. Итоговая обучающая таблица  
   - `train` с колонками:
     - ключи (`game_id`, `play_id`, `nfl_id`, `frame_id`),
     - все engineered-фичи из `FEATURES + CAT_FEATS`,
     - таргеты `dx` и `dy`.

---

### Шаг 4. Feature engineering: геометрия, баллистика, время, таргеты

**Назначение:** превратить сырые координаты, скорости и контекст в богатый набор признаков, отражающих геометрию розыгрыша и положение игрока относительно мяча и таргета.

#### 4.1. Базовая кинематика и углы

- Из исходных колонок:
  - `s` (скорость), `a` (ускорение), `dir` (направление бега), `o` (ориентация тела),
  - NaN → 0.
- Углы переводятся в радианы: `dir_rad`, `o_rad`.
- Компоненты:
  - `vx = s * cos(dir_rad)`, `vy = s * sin(dir_rad)`,
  - `ax_comp = a * cos(dir_rad)`, `ay_comp = a * sin(dir_rad)`.
- Тригонометрические фичи:
  - `dir_sin`, `dir_cos`, `o_sin`, `o_cos`.

#### 4.2. Динамика последнего шага до паса

- Если есть `x_prev`, `y_prev`:
  - `last_step_dx = x_last - x_prev`,
  - `last_step_dy = y_last - y_prev`,
  - `last_step_dist = sqrt(last_step_dx^2 + last_step_dy^2)`,
  - `last_step_speed = last_step_dist * 10.0` (10 кадров в секунду).
- Если `_prev` нет — эти фичи обнуляются.
- Дельты по скорости и ускорению:
  - `ds_last = s - s_prev` (иначе 0),
  - `da_last = a - a_prev` (иначе 0).
- Угловые дельты:
  - `d_dir_last = angle_diff(dir, dir_prev)` с нормировкой в `[-180, 180]`,
  - `d_o_last = angle_diff(o, o_prev)`.

#### 4.3. Временные признаки траектории мяча

- `frame_offset = frame_id` — номер текущего кадра полёта.
- `time_offset = frame_offset / 10` — время с начала полёта мяча (сек).
- Если есть `num_frames_output`:
  - `nfo = num_frames_output` с 0 → NaN.
  - `frac_of_flight = frame_offset / nfo`, клиппинг в `[0, 1]`, NaN → 0.
  - `frames_left = (nfo - frame_offset)`, клиппинг снизу 0, NaN → 0.
- Далее:
  - `time_to_land = frames_left / 10`,
  - `remaining_flight_frac = 1 - frac_of_flight` (с клиппингом в `[0, 1]`).

#### 4.4. Дискретизация фазы полёта (категориальная фича)

- `frame_bin`:
  - 0 — ранняя фаза (`frac_of_flight <= 0.33`),
  - 1 — средняя (`0.33 < frac_of_flight <= 0.66`),
  - 2 — поздняя (`> 0.66`).
- Используется как категориальный признак для LGBM (и будет one-hot’нут для MLP).

#### 4.5. Геометрия относительно точки приземления мяча

- Расстояние до точки приземления:
  - `dist_to_ball_land = sqrt((ball_land_x - x_last)^2 + (ball_land_y - y_last)^2)`.
- Угол направления:
  - `angle_to_ball_land = atan2(ball_land_y - y_last, ball_land_x - x_last)`.
- Нормированная на кадры дистанция:
  - `frames_left_safe = frames_left` c заменой 0 → NaN.
  - `dist_to_ball_land_per_frame = dist_to_ball_land / frames_left_safe`,
  - Inf/NaN → 0.
- Выравнивание направления и ориентации по мячу:
  - `cos_dir_to_ball = cos(angle_to_ball_land - dir_rad)`,
  - `cos_orient_to_ball = cos(angle_to_ball_land - o_rad)`.
- Требуемая скорость:
  - `time_to_land_safe = time_to_land` с 0 → NaN.
  - `req_speed_to_ball = dist_to_ball_land / time_to_land_safe`, Inf/NaN → 0.
- Дефицит скорости:
  - `speed_minus_req = s - req_speed_to_ball`,
  - `last_step_speed_minus_req = last_step_speed - req_speed_to_ball`.

#### 4.6. Баллистические проекции траектории игрока

- Временной горизонт:
  - `dt = time_to_land`.
- Линейная модель (постоянная скорость):
  - `proj_x_vel = x_last + vx * dt`,
  - `proj_y_vel = y_last + vy * dt`.
- Модель с ускорением:
  - `proj_x_acc = x_last + vx * dt + 0.5 * ax_comp * dt^2`,
  - `proj_y_acc = y_last + vy * dt + 0.5 * ay_comp * dt^2`.
- Делта до мяча:
  - `proj_vel_dx_to_ball = ball_land_x - proj_x_vel`,
  - `proj_vel_dy_to_ball = ball_land_y - proj_y_vel`,
  - `proj_acc_dx_to_ball = ball_land_x - proj_x_acc`,
  - `proj_acc_dy_to_ball = ball_land_y - proj_y_acc`.
- Дистанции:
  - `dist_proj_vel_to_ball = sqrt(proj_vel_dx_to_ball^2 + proj_vel_dy_to_ball^2)`,
  - `dist_proj_acc_to_ball = sqrt(proj_acc_dx_to_ball^2 + proj_acc_dy_to_ball^2)`.

#### 4.7. Стандартизация координат по направлению розыгрыша

- По `play_direction`:
  - `is_left = 1`, если игра идёт «left», иначе 0.
- Стандартизированная x:
  - `x_std = 120 - x_last`, если `is_left == 1`,
  - иначе `x_std = x_last`.
- Стандартизированная точка приземления мяча:
  - `ball_land_x_std = 120 - ball_land_x` при left,
  - иначе `ball_land_x_std = ball_land_x`.
- Разности:
  - `dx_to_land_std = ball_land_x_std - x_std`,
  - `dy_to_land = ball_land_y - y_last`.

#### 4.8. Положение на поле

- Вертикальное положение:
  - `dist_to_sideline = min(y_last, 53.3 - y_last)`,
  - `dist_to_center = abs(y_last - 53.3 / 2)`.
- Продольное положение:
  - `yard = absolute_yardline_number`, NaN → 50,
  - `yardline_100 = yard.clip(0, 100)`,
  - `yardline_norm = yardline_100 / 100`,
  - `dist_to_endzone = 100 - yardline_100`.

#### 4.9. Признаки относительно таргет-ресивера

- Расстояние до таргета:
  - `dist_to_target_last = sqrt((target_last_x - x_last)^2 + (target_last_y - y_last)^2)`.
- Проекции:
  - `dx_to_target_last = target_last_x - x_last`,
  - `dy_to_target_last = target_last_y - y_last`.
- Угол:
  - `angle_to_target = atan2(target_last_y - y_last, target_last_x - x_last)`.
- Выравнивание по таргету:
  - `cos_dir_to_target = cos(angle_to_target - dir_rad)`,
  - `cos_orient_to_target = cos(angle_to_target - o_rad)`.
- Индикатор таргета:
  - `is_target = 1`, если `nfl_id == target_nfl_id`, иначе 0.

#### 4.10. Физические характеристики

- Имеются:
  - `player_height` (в дюймах), `player_weight`.
- Индекс массы тела:
  - `bmi = 703 * weight / height^2` (если height > 0).
  - Inf/NaN → 0.

#### 4.11. Формирование таргетов (только на train)

- Если `is_train=True`:
  - `dx = x - x_last`,
  - `dy = y - y_last`.
- На инференсе (`is_train=False`) таргеты не создаются.

---

### Шаг 5. Обработка пропусков, категориальных признаков и нормализация

**Назначение:** подготовить данные к LGBM и MLP, минимизировав влияние пропусков и обеспечив корректную работу категорий.

#### 5.1. Пропуски

- Внутри feature engineering:
  - Скорости/ускорения, углы: NaN → 0.
  - Доли полёта, времена, «на кадр/на секунду»: деления на ноль дают Inf/NaN, которые затем заменяются на 0.
  - BMI и прочие derived-фичи: Inf/NaN → 0.
- В итоговом `train`:
  - LGBM допускает NaN в числовых фичах, но после обработки в фич-инжиниринге их практически нет.
  - Перед MLP применяется `get_dummies` + `StandardScaler`, что требует плотной числовой матрицы (NaN минимизированы или отсутствуют).

#### 5.2. Категориальные признаки

- `CAT_FEATS = ["player_role", "player_side", "play_direction", "frame_bin"]`.
- Для LGBM:
  - Эти колонки приводятся к типу `category`.
  - Передаются в параметр `categorical_feature`.
- Для MLP:
  - Категории one-hot кодируются `pd.get_dummies(..., dummy_na=True)`.
  - Список признаков после OHE сохраняется и используется для выравнивания матрицы на тесте.

#### 5.3. Нормализация

- Только для MLP:
  - Применяется `StandardScaler`, обученный на обучающей `X_dense`.
  - Преобразованные данные (`X_scaled`) подаются в MLP.
- Для LGBM отдельная нормализация не нужна.

---

### Шаг 6. Итоговая обучающая выборка и sanity-checks

**Назначение:** проверить корректность построенного train-набора и наличие ключевых статистик.

1. Размеры и уникальности  
   - Печатается `train.shape`.
   - Количество уникальных `game_id`, `play_id`, `nfl_id`.

2. Базовые статистики  
   - Для ключевых числовых фич (`s`, `a`, `dist_to_ball_land`, `time_to_land`, `vx`, `vy` и т. п.) выводится `describe()`:
     - min, max, mean, median, квартильные значения.

3. NaN-анализ  
   - Рассматриваются только числовые фичи из `FEATURES`.
   - Считается количество NaN по каждой фиче.
   - Выводятся топ-фичи по числу пропусков (если есть).

---

### Шаг 7. Обучение моделей (LGBM + MLP по остаткам)

**Назначение:** обучить базовую модель LGBM на `dx` и `dy`, затем обучить MLP, корректирующий остатки LGBM, и оценить offline-качество.

#### 7.1. Подготовка признаков

- Гарантируется наличие всех фич из `FEATURES` и категориальных `CAT_FEATS`:
  - Отсутствующие числовые фичи создаются с 0.
  - Отсутствующие категориальные — со строкой `"unknown"`.
- Формируется:
  - `X = train[FEATURES + CAT_FEATS]`,
  - `y_dx = train["dx"]`,
  - `y_dy = train["dy"]`.
- Категории:
  - `X[CAT_FEATS]` приводятся к типу `category`.

#### 7.2. LGBM–модели

- Общие параметры (не изменяются при рефакторинге):
  - `objective="regression"`,
  - `n_estimators=1600`,
  - `learning_rate=0.04`,
  - `num_leaves=127`,
  - `min_data_in_leaf=80`,
  - `feature_fraction=0.9`,
  - `bagging_fraction=0.9`, `bagging_freq=1`,
  - `reg_alpha=0.1`, `reg_lambda=0.2`,
  - `n_jobs=-1`, `random_state=42`.
- Обучаются два регрессора:
  - `MODEL_DX_LGBM` — на `y_dx`,
  - `MODEL_DY_LGBM` — на `y_dy`.
- Обучение на всём train (без явной валидации или KFold).
- После обучения:
  - `lgbm_dx_pred = model_dx.predict(X)`,
  - `lgbm_dy_pred = model_dy.predict(X)`.

#### 7.3. MLP по остаткам

- Построение плотной матрицы:
  - `X_dense = get_dummies(X, columns=CAT_FEATS, dummy_na=True)`.
  - Список колонок сохраняется как `MLP_COLUMNS`.
- Масштабирование:
  - `X_scaled = StandardScaler().fit_transform(X_dense)`; scaler сохраняется.
- Остатки:
  - `res_dx = y_dx - lgbm_dx_pred`,
  - `res_dy = y_dy - lgbm_dy_pred`.
- Параметры MLP:
  - `hidden_layer_sizes=(128, 128)`,
  - `activation="relu"`,
  - `solver="adam"`,
  - `learning_rate_init=1e-3`,
  - `max_iter=40`,
  - `batch_size=1024`,
  - `early_stopping=True`, `n_iter_no_change=5`,
  - `random_state=42`.
- Обучаются:
  - `MODEL_DX_MLP` — на `(X_scaled, res_dx)`,
  - `MODEL_DY_MLP` — на `(X_scaled, res_dy)`.

#### 7.4. Offline-метрики (sanity-check)

- После обучения MLP:
  - `res_dx_mlp = MODEL_DX_MLP.predict(X_scaled)`,
  - `res_dy_mlp = MODEL_DY_MLP.predict(X_scaled)`.
- Итоговые train-предсказания:
  - `train_dx_pred = lgbm_dx_pred + 0.5 * res_dx_mlp`,
  - `train_dy_pred = lgbm_dy_pred + 0.5 * res_dy_mlp`.
- Вычисляются RMSE:
  - `RMSE_dx`, `RMSE_dy`,
  - комбинированный RMSE по `(dx, dy)`.
- Эти метрики печатаются как offline-оценка. По описанию автора, решение выдаёт публичный LB ≈ `0.604`.

---

### Шаг 8. Подготовка данных и предсказание на тесте (инференс)

**Назначение:** повторить всю логику подготовки фичей на тесте и получить предсказанные координаты `x`, `y` для Kaggle evaluation API.

#### 8.1. Подготовка батча для инференса (`prepare_inference_batch`)

1. Входные данные:
   - `test_pd` — pandas-версия `test` (из polars DataFrame), содержит:
     - `id`, `game_id`, `play_id`, `nfl_id`, `frame_id`.
   - `test_input_pd` — pandas-версия `test_input`, содержит трекинг до паса.

2. Логика:
   - `last_obs = prepare_last_obs(test_input_pd)` — как в train.
   - `last_obs = add_target_info(last_obs)` — добавление таргет-инфо.
   - Выбор колонок из `BASE_COLS`, которые реально присутствуют.
   - `test_rows = test_pd.merge(last_obs[BASE_COLS_actual], on=["game_id", "play_id", "nfl_id"], how="left")`.
   - `test_rows = create_features(test_rows, is_train=False)`:
     - строятся **все те же фичи**, что и для train, кроме таргетов `dx`, `dy`.

3. Результат:
   - `test_rows` — все строки теста с нужными фичами.

#### 8.2. Основная функция `predict` (API для Kaggle)

1. Вход:
   - `test` — Polars DataFrame: `[id, game_id, play_id, nfl_id, frame_id]`.
   - `test_input` — Polars DataFrame: трекинг до паса по соответствующим розыгрышам.

2. Преобразование:
   - Конвертация в pandas:
     - `test_pd = test.to_pandas()`,
     - `test_input_pd = test_input.to_pandas()`.
   - `test_rows = prepare_inference_batch(test_pd, test_input_pd)`.

3. Подготовка матрицы признаков:
   - Гарантируется наличие всех `FEATURES` и `CAT_FEATS` (отсутствующие создаются).
   - `X_test = test_rows[FEATURES + CAT_FEATS]`.
   - Категории:
     - `X_test[CAT_FEATS]` → тип `category`.

4. Предсказания LGBM:
   - `pred_dx_lgbm = MODEL_DX_LGBM.predict(X_test)`,
   - `pred_dy_lgbm = MODEL_DY_LGBM.predict(X_test)`.

5. MLP-коррекция (если обучен MLP):
   - `X_test_dense = get_dummies(X_test, columns=CAT_FEATS, dummy_na=True)`.
   - Выравнивание колонок:
     - `X_test_dense = X_test_dense.reindex(columns=MLP_COLUMNS, fill_value=0)`.
   - Масштабирование:
     - `X_test_scaled = MLP_SCALER.transform(X_test_dense)`.
   - Остатки:
     - `res_dx_mlp = MODEL_DX_MLP.predict(X_test_scaled)`,
     - `res_dy_mlp = MODEL_DY_MLP.predict(X_test_scaled)`.
   - Итоговые смещения:
     - `pred_dx = pred_dx_lgbm + 0.5 * res_dx_mlp`,
     - `pred_dy = pred_dy_lgbm + 0.5 * res_dy_mlp`.

6. Восстановление координат:
   - `x_pred = x_last + pred_dx`,
   - `y_pred = y_last + pred_dy`,
   - затем клиппинг:
     - `x_pred ∈ [0, 120]`,
     - `y_pred ∈ [0, 53.3]`.

7. Выход:
   - Polars DataFrame с колонками `["x", "y"]`, длина совпадает с входным `test`.

---

### Шаг 9. Интеграция с Kaggle Evaluation API

**Назначение:** обеспечить работу пайплайна как в локальном режиме (для отладки и генерации submission), так и в режиме официального скоринга Kaggle.

1. Главный блок запуска:
   - При старте как `__main__`:
     - `df_in, df_out = load_train(DATA_DIR)`,
     - `train_df = prepare_train(df_in, df_out)`,
     - выполнение sanity-checks по train,
     - `train_models(train_df)`.

2. Использование `NFLInferenceServer`:
   - Если модуль `kaggle_evaluation` доступен:
     - создаётся `NFLInferenceServer(predict)`.
   - Если выставлена переменная `KAGGLE_IS_COMPETITION_RERUN`:
     - `server.serve()` — боевой режим, Kaggle дергает `predict` на скрытом тесте.
   - Иначе:
     - `run_local_gateway((DATA_DIR,))` — прогон mock test и создание `submission.csv` (или `.parquet`) локально на Kaggle.

3. Локальный запуск вне Kaggle:
   - При отсутствии `kaggle_evaluation`:
     - выполняется только обучение (для отладки features и моделей).
     - Генерация submission возможна только на Kaggle с подключённым датасетом.

---

## 3. Итог

- Данные движутся по чётко определённому маршруту:
  1. Загрузка сырых недельных файлов (`input`, `output`).
  2. Строительство таблицы последнего наблюдения до паса, lag-фичей и окружения.
  3. Добавление информации о таргет-ресивере и merge с таргетами.
  4. Массовый feature engineering: геометрия, баллистика, время, позиции, отношение к мячу и таргету, окружение, BMI.
  5. Обработка пропусков, категоризация и нормализация для MLP.
  6. Обучение LGBM по `dx` и `dy`, затем MLP по остаткам.
  7. Полное воспроизведение того же пайплайна на тесте в функции `predict`.
- Все параметры модели, сетап LGBM/MLP и формат работы с Kaggle evaluation API сохраняются в соответствии с исходным решением, обеспечивая сопоставимое качество (публичный LB ~ 0.604).
