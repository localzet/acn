# Adaptive Core Network (ACN)

Учебно-исследовательский прототип continual learning / adaptation на базе PyTorch.

## Идея

`Adaptive Core Network (ACN)` демонстрирует адаптацию классификатора к сдвигу распределения входных данных без полного переобучения всей модели с одинаковым learning rate.

Сравниваются два подхода:
- `Baseline`: обычный fine-tuning CNN по этапам потока данных.
- `ACN`: та же CNN, но с динамической пластичностью слоёв и ручным layer-wise обновлением весов.

Формула пластичности:

```text
plasticity = sigmoid(alpha * uncertainty + beta * grad_norm + gamma * novelty)
```

Где:
- `uncertainty`: энтропия softmax-выхода;
- `grad_norm`: норма градиента слоя;
- `novelty`: расстояние embedding batch до centroid в `MemoryBank`.

Обновление параметров в ACN:

```text
W = W - lr * plasticity_layer * grad
```

## Какую проблему решает

При последовательном обучении на новых доменах (rotation/noise/invert/blur) базовая модель часто теряет качество на прошлых данных (`catastrophic forgetting`).

ACN снижает деградацию за счёт адаптивной пластичности:
- при высокой неопределённости/новизне слой становится более пластичным;
- при стабильном режиме пластичность ниже, что помогает сохранять старые знания.

## Структура проекта

```text
.
├── README.md
├── requirements.txt
├── main.py
├── acn
│   ├── __init__.py
│   ├── model.py
│   ├── trainer.py
│   ├── plasticity.py
│   ├── memory.py
│   ├── transforms.py
│   ├── metrics.py
│   └── visualization.py
└── outputs
```

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --epochs-clean 3 --epochs-adapt 2 --batch-size 128 --lr 0.001 --device cpu
```

Аргументы CLI:
- `--epochs-clean`
- `--epochs-adapt`
- `--batch-size`
- `--lr`
- `--device` (`cpu` или `cuda`)
- `--seed`

Если `--device cuda` недоступен, код автоматически переключается на CPU.

## Датасет-поток

Используется `FashionMNIST` и последовательные этапы сдвига:
1. `clean`
2. `rotated`
3. `noisy`
4. `inverted`
5. `blurred`

Каждый этап имеет отдельные train/test dataloader.

## Метрики

В `main.py` считаются и печатаются:
- `accuracy per stage` (через историю оценок по этапам);
- `old task accuracy`;
- `new task accuracy`;
- `forgetting score`;
- `adaptation speed` (номер эпохи, когда достигнуто 90% финального stage-качества).

## Какие графики смотреть

После запуска сохраняются в `outputs/`:
- `baseline_vs_acn_accuracy.png` — средняя точность по уже увиденным этапам;
- `forgetting_score.png` — динамика forgetting;
- `plasticity_over_time.png` — layer-wise пластичность ACN;
- `stage_examples.png` — визуальные примеры сдвигов распределения.

## Почему это демонстрирует continual learning / test-time adaptation

- Обучение идёт **последовательно** по изменяющимся этапам потока данных.
- После каждого этапа проверяется качество на предыдущих этапах.
- ACN не делает полный retrain, а адаптирует веса через динамический layer-wise effective LR, что имитирует механизм контролируемой пластичности.

## Экспериментальная новизна прототипа

- Явное объединение трёх сигналов (`uncertainty`, `grad_norm`, `novelty`) в единый `plasticity score`.
- Ручное обновление весов по слоям вместо стандартного `optimizer.step()`.
- Простой, расширяемый `MemoryBank` (глобальный centroid), который можно расширить до class-centroids без изменения общей архитектуры.
