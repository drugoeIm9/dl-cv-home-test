# **0. Саммари**
- Для стерео сейчас рационально ориентироваться на две ветки: компактные real-time сети (например, BGNet, AnyNet и тд) для встраивания, и современные итеративные и foundation-модели (RAFT-style, IGEV, FoundationStereo) как источники высококачественной разметки или teacher-моделей для distillation и pseudo-labels.
- Для задач, подобных второй, предлагаю следущий воркфлоу: использовать BGNet или другую лёгкую архитектуру как базу; предобучить/зафайнтюнить на смеси синтетики + InStereo2K + паре driving датасетов; применить knowledge distillation, фильтрацию псевдо-меток и пост-оптимизации (prune/quantize) перед деплоем.
-----
# **1. Какие модели хороши**
**Модели подходящие как on-device:**

BGNet — CVPR2021: компактный real-time stereo, хорошая точность/скорость для embedded. Подойдёт как стартовая лёгкая модель или как teacher для distillation в задачах с ограничениями по FPS/RAM.

AnyNet, FADNet, DeepPruner-Fast — ориентированы на экономию вычислений (простое cost-volume + мелкие сети).

**Учителя для генерации разметки или pseudo-labels:**\
RAFT-Stereo, CREStereo, IGEV — итеративные рефайнеры: отличная точность и temporal-friendly поведение; хорошие кандидаты на роль учителя.\
FoundationStereo — крупная foundation-модель для stereo, ориентированная на zero-shot generalization (тренирована на ~1M синтетических пар), отлична как источник качественных pseudo-label'ей и как эталон.

Monocular:\
MiDaS, DPT, AdaBins — лёгкие варианты monocular, хороши как fallback и для side-priors в side-tuning и teacher pipelines.

Кратко: для задания 2 разумно брать BGNet или AnyNet как backbone лёгкой модели и использовать RAFT-style, CREStereo и FoundationStereo как источник меток/учителя.

-----
# **2. Какие датасеты добавить в обучающую выборку для задания 2**
(цель: покрыть indoor и outdoor, синтетику для редких кейсов, и наборы с плотным и длинным range)

Лучшие:

1. InStereo2K (Indoor, real, 2k pairs) — высококачественные disparity maps для indoor, важен для fine detail и indoor-навигции.
1. SceneFlow / FlyingThings3D (synthetic) — предобучение для cost-volume моделей, даёт разнообразие геометрии и движения.
1. DrivingStereo (large driving stereo) — масштабный набор улиц и городских сцен, помогает generalization для outdoor.
1. KITTI (Stereo/Depth) — стандартный outdoor бенчмарк.
1. DDAD (Dense Depth for Autonomous Driving) — long-range dense depth (до 250 метров), обязательно если робот должен видеть далеко.
1. TartanAir — погодные и текстур-edge cases, полезны для corruptions и augmentation.
1. Middlebury, ETH3D — для проверки fine-detail, важны при оценке краевых случаев.

Почему именно этот микс:

- InStereo2K покрывает indoor сцены, SceneFlow даёт масштабную синтетику для устойчивого предобучения, DrivingStereo/KITTI/DDAD обеспечивают outdoor и long-range покрытие. Такие наборы минимизируют domain-gap и дают хороший компромисс точности и покрытия.
-----
# **3. Какие бенчмарки и тесты использовать, чтобы оценить робастность модели**
Метрики (стандарт):

- Stereo: D1-all (% bad), EPE, (pixel error > 3px / >1% и т.п.).
- Monocular: AbsRel, RMSE, δ<1.25 и тд.

Обязательный набор тестов, который надо применять на валидации:

1. Cross-dataset — train на нашем датасете, test на новом датасете (напр., train без Middlebury → тест на Middlebury / ETH3D). Это показывает generalization.
1. Corruption sweep — brightness, contrast, gaussian noise, motion blur, JPEG compression, fog/rain (синтетически через TartanAir).
1. Range-split evaluation — для driving: разбить по дистанции (0–10m, 10–50m, >50m) и считать метрики отдельно (DDAD особенно полезен).
1. Occlusion / textureless regions — сравнить performance в occluded/reflective/low-texture областях (Middlebury).
1. Temporal stability (для видео) — frame-to-frame variance / flicker metrics, важно для onboard-robotics.
-----
# **4. Какие датасеты более релевантны для robotics, а какие — менее**
Более релевантны:

- DDAD, KITTI, DrivingStereo, Waymo, Argoverse — для мобильных роботов / автономного вождения (дальний диапазон, уличные условия).
- InStereo2K, Replica, HM3D, Matterport3D, NYUv2 — indoor-navigation / manipulation / SLAM.
- TartanAir / SceneFlow — для редких случае, погодных условий, data augmentation, обобщения.

Менее релевантны:

- Нiche/very-small академические наборы, снимки с сильно необычной оптикой (телескопы, medical stereo microscopy) — полезны для конкретных задач, но не для robotics в целом.
- Вест-плейс датасеты с малой геометрической вариативностью (малый набор сцен без реальных движений) — дают ограниченную пользу для generalization.
