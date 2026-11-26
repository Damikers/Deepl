from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def write_forecast(directory: Path, ranges: dict) -> dict:
    text = (
        f"Медіана відношень діапазонів каналів r={ranges['median_r']:.6f}; "
        f"відповідна втрата роздільності становить {ranges['median_bits_lost']:.6f} біта. "
        f"Найбільше r={ranges['top_three'][0]['r']:.6f}. "
        "Очікую, що потензорна квантизація знизить згоду з FP32 порівняно з поканальною, "
        "оскільки крок для слабших каналів стане грубішим. За такого розкиду діапазонів "
        "очікую невеликої зміни macro-F1 та зміни accuracy в межах шуму за правилом "
        "|b−c| ≤ 2√(b+c). Тобто не прогнозую помітного погіршення якості за критерієм роботи. "
        "Сам критерій перевіряє accuracy; окремої стандартної похибки macro-F1 він не задає."
    )
    readme = directory / "README.md"
    readme.write_text("# ЛР4. FP32 проти INT8\n\n## Прогноз до потензорного досліду\n\n" + text + "\n", encoding="utf-8")
    forecast = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "after FP32 channel analysis, before either INT8 quantization or test evaluation",
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "readme_before_experiment_sha256": hashlib.sha256(readme.read_bytes()).hexdigest(),
    }
    (directory / "lab4_prediction.json").write_text(
        json.dumps(forecast, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return forecast


def paired_verdict(row: dict) -> str:
    if not row["accuracy_change_exceeds_noise"]:
        return "зміна accuracy не відрізняється від шуму"
    direction = "зростання" if row["delta_accuracy"] > 0 else "падіння"
    return f"{direction} accuracy відрізняється від шуму"


def write_report(path: Path, result: dict) -> None:
    env = result["environment"]
    data = result["data"]
    export = result["export"]
    equivalence = result["export_equivalence"]
    rows = result["quality"]
    fp32, int8, tensor = (rows[key] for key in ("fp32", "int8", "int8_per_tensor"))
    latency = result["latency"]
    ranges = result["channel_ranges"]
    names = {"fp32": "FP32", "int8": "INT8 per-channel", "int8_per_tensor": "INT8 per-tensor"}
    quality_rows = "\n".join(
        f"| {names[key]} | {row['macro_f1']:.6f} | {row['accuracy']:.6f} | "
        f"{row['agreement_with_fp32']:.6f} | {row['b']} | {row['c']} | "
        f"{row['delta_accuracy']:+.6f} ± {row['paired_standard_error']:.6f} |"
        for key, row in rows.items()
    )
    channel_rows = "\n".join(
        f"| `{row['name']}` | {row['alpha_max']:.6f} | {row['alpha_median']:.6f} | "
        f"{row['r']:.6f} | {row['median_channel_bits_lost']:.6f} |"
        for row in ranges["top_three"]
    )
    latency_rows = "\n".join(
        f"| {names[key]} | {result['artifacts'][key]['megabytes']:.6f} | "
        f"{latency[key]['p50_ms']:.3f} | {latency[key]['p95_ms']:.3f} | "
        + "; ".join(f"{row['p50_ms']:.3f}" for row in latency[key]["rounds"]) + " |"
        for key in ("fp32", "int8")
    )
    control_rows = "\n".join(
        f"| {part} | {control['count']} | {control['filename_sum']:,} |".replace(",", " ")
        for part, control in data["split_control"].items()
    )
    calibration_rows = "\n".join(f"| {label} | {count} |" for label, count in data["calibration"]["class_counts"].items())
    fp_interval = latency["fp32"]["round_p50_interval_ms"]
    int_interval = latency["int8"]["round_p50_interval_ms"]
    latency_verdict = "не перетинаються: зміна затримки виміряна" if latency["intervals_disjoint"] else "перетинаються: зміна затримки не підтверджена правилом роботи"
    agreement_forecast = "підтвердився" if tensor["agreement_with_fp32"] < int8["agreement_with_fp32"] else "не підтвердився"
    quality_forecast = "не підтвердився" if tensor["accuracy_change_exceeds_noise"] else "підтвердився"
    cpu_features = ", ".join(f"{key}={value}" for key, value in env["cpu_features"].items()) or "не визначалися цією ОС"
    before = latency["context_before"]["power"] or "статус живлення недоступний"
    after = latency["context_after"]["power"] or "статус живлення недоступний"
    ac_power = "AC Power" in before and "AC Power" in after
    power_statement = "До і після вимірювання `pmset` підтвердив живлення від мережі." if ac_power else "Статус живлення до й після вимірювання збережено в JSON; живлення від мережі автоматично не підтверджено."
    lab3_statement = (
        "Отримана macro-F1 збігається зі збереженим test-результатом тієї самої моделі."
        if equivalence.get("matches_lab3") or equivalence.get("matches_recorded_run") else ""
    )
    arm_supported = all(env["cpu_features"].get(name) == "1" for name in ("FEAT_DotProd", "FEAT_I8MM"))
    hardware_statement = (
        f"На {env['cpu']} підтримку ARM DotProd/I8MM підтверджено через sysctl."
        if arm_supported else
        "Підтримку конкретних INT8-інструкцій у цьому запуску автоматично не підтверджено."
    )
    se_ratio = int8["fp32_single_se_over_paired_se"]
    se_comparison = (
        f"відношення цих похибок — {se_ratio:.3f}."
        if se_ratio is not None else
        "відношення похибок не визначено, оскільки парна похибка дорівнює нулю."
    )
    paired_se_statement = (
        "У цьому досліді спільність відповідей робить парну похибку меншою."
        if int8["paired_standard_error"] < fp32["single_model_standard_error"] else
        "Розмір парної похибки залежить від кількості розбіжностей у правильності."
    )
    suitability = (
        "INT8 придатна для цієї задачі на цьому ноутбуці: "
        "отримано менший файл і підтверджене прискорення без зміни accuracy понад шум."
        if latency["intervals_disjoint"] and latency["p50_speedup"] > 1 and not int8["accuracy_change_exceeds_noise"]
        else "Доцільність INT8 залежить від допустимої зміни якості та виміряного виграшу ресурсів у наведених таблицях."
    )
    text = rf"""# ЛР4. FP32 проти INT8

## Мета роботи

Експортувати вибрану ResNet-18 із ЛР3 у ONNX, виконати статичну
квантизацію після навчання та порівняти FP32 з INT8 на CPU:

1. перевірити відповідність PyTorch та ONNX FP32;
2. оцінити якість INT8 на тих самих test-зображеннях із парним порівнянням;
3. дослідити поканальні й потензорні ваги;
4. виміряти розмір файлу, p50 і p95 затримки за фіксованим протоколом.

Модель не донавчалася. Використано модель `{result['model']['regime']}`, seed={result['model']['seed']},
вибрану в ЛР3 за validation.

## Відтворення

Потрібні Python 3.12 та uv. Із теки `lab4/`:

```bash
uv run --frozen python main.py
```

Модель `lab3_model.pt` і поділ `split.csv` включені до цієї теки.
Під час першого запуску програма завантажує Intel Image Classification
в `data/`, перевіряє SHA-256 архіву й розпаковує тільки потрібні
200 калібрувальних та 1502 тестових зображення. Для завантаження потрібен інтернет;
наступні запуски використовують локальні дані. Залежності встановлюються за `uv.lock`.

Результати, README та ONNX-моделі записуються в нову теку `reproduced/`.
Повторний запуск автоматично створює `reproduced_run2/`, потім
`reproduced_run3/` тощо. Наявні результати не перезаписуються.
ONNX-файли й набір даних виключені з Git.

Власні шляхи можна задати параметрами:

```bash
uv run --frozen python main.py \
  --model lab3_model.pt --split split.csv --data data --out experiment
```

Наведені нижче числа походять із `lab4_results.json` поряд із цим README.
На іншому CPU затримка відрізнятиметься; якість INT8 також може залежати
від архітектури та версії ONNX Runtime. Контрольні числа даних і діапазони
ваг мають відтворюватися з тією самою моделлю.

## Структура файлів

| Файл | Призначення |
|---|---|
| `lab3_model.pt` | Копія вибраної моделі ЛР3 з незміненими вагами |
| `split.csv` | Поділ даних із ЛР3 |
| `download_data.py` | Завантаження й перевірка архіву, підготовка потрібних зображень |
| `data.py` | Модель ЛР3, контроль поділу, калібрування, RGB і перетворення |
| `quantization.py` | Експорт ONNX, діапазони каналів, підготовка й статична квантизація |
| `metrics.py` | CPU-сесії, якість, парні лічильники та вимірювання затримки |
| `main.py` | Повний експеримент, середовище, хеші й запис JSON |
| `reporting.py` | Незмінений прогноз до досліду та README з результатів JSON |
| `test_metrics.py` | Незалежні перевірки парного порівняння та протоколу вимірювання |
| `lab4_prediction.json` | Прогноз, час його запису й хеш до запуску INT8 |
| `lab4_results.json` | Усі результати, 600 вимірів затримки, контрольні числа й середовище |

## 1. Дані та контроль поділу

Використано [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
та `split.csv` із ЛР3. Порядок класів: buildings, forest, glacier,
mountain, sea, street. Test зберігає порядок відповідних рядків CSV.

| Частина | Зображень | Сума числових імен |
|---|---:|---:|
{control_rows}

Контрольні числа збіглися з умовою. Імена унікальні, перетинів частин немає.
Для кожного зображення застосовано Pillow `.convert("RGB")`,
`Resize((160,160))` із bilinear та antialias, `ToTensor()`,
`Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))`.
Входи мають тип float32. Аугментацій немає.

SHA-256 моделі:

```text
{result['model']['sha256']}
```

SHA-256 поділу:

```text
{data['split_sha256']}
```

## 2. Експорт і відповідність

Модель завантажено через `torch.load(..., map_location="cpu", weights_only=True)`.
Побудовано `resnet18(weights=None)`, голову замінено на `Linear(512,6)`,
завантажено `state_dict`, увімкнено `eval()`.

Експорт: `torch.onnx.export`, `dynamo=False`, opset=17, вхід `input`,
вихід `logits`, змінна вісь 0 для обох. Решта параметрів за замовчуванням.
Порівняння виконано на всіх 1502 test-зображеннях пакетами по 64,
PyTorch працював під `torch.no_grad()`.

| Перевірка | Результат |
|---|---:|
| Максимальна абсолютна різниця логітів | {equivalence['maximum_absolute_logit_difference']:.9g} |
| Збіг передбачених класів | {equivalence['prediction_agreement']:.6f} |
| Test macro-F1 PyTorch | {equivalence['pytorch_macro_f1']:.6f} |
| Test macro-F1 ONNX FP32 | {equivalence['onnx_fp32_macro_f1']:.6f} |
| Conv в експортованому графі | {export['conv_nodes']} |
| BatchNormalization в експортованому графі | {export['batch_normalization_nodes']} |
| Час експорту, с | {export['seconds']:.6f} |

Різниця логітів менша за 0.001; усі класи та macro-F1 збігаються.
{lab3_statement}
BatchNormalization у режимі eval використовує сталі статистики,
тому експортер злив її зі згортками; повторного навчання для цього немає.

## 3. Калібрування й квантизація

1200 train-рядків відсортовано за іменем файлу як рядком. Індекси
`np.random.default_rng(0).permutation(1200)[:200]` визначають 200
калібрувальних зображень. Сума числових імен — {format(data['calibration']['filename_sum'], ',').replace(',', ' ')}.

| Клас | Зображень |
|---|---:|
{calibration_rows}

Усі контрольні числа калібрування збіглися. Обидва INT8-варіанти
отримали ті самі зображення в тому самому порядку, по одному:
`1×3×160×160`. Калібрування використовує тільки train, бо оцінені
діапазони активацій стають частиною моделі.

Перед квантизацією виконано `quant_pre_process(fp32, prep, skip_symbolic_shape=True)`.
Далі `quantize_static`: QDQ, активації QUInt8, ваги QInt8,
`CalibrationMethod.MinMax`. Основний варіант має `per_channel=True`,
дослід — `False`; інші аргументи за замовчуванням.

| Етап | Час, с |
|---|---:|
| Підготовка графа | {result['preprocessing']['seconds']:.6f} |
| Поканальна квантизація | {result['quantization']['int8']['seconds']:.6f} |
| Потензорна квантизація | {result['quantization']['int8_per_tensor']['seconds']:.6f} |

Ці часи включають відповідні виклики бібліотеки; квантизація включає
читання й перетворення калібрувальних зображень. Вони не входять у затримку виконання моделі.

## 4. Якість і парне порівняння

Усі варіанти оцінено на тих самих 1502 зображеннях, пакетами по 64.
`b` — INT8 правильна, FP32 помиляється; `c` — навпаки.
У колонці Δ наведено зміну accuracy та одну парну стандартну похибку.

| Варіант | macro-F1 | Accuracy | Згода з FP32 | b | c | Δaccuracy ± SE |
|---|---:|---:|---:|---:|---:|---:|
{quality_rows}

```math
\Delta_{{acc}}=\frac{{b-c}}{{1502}},\qquad
SE(\Delta_{{acc}})\approx\frac{{\sqrt{{b+c}}}}{{1502}}.
```

Для поканальної INT8: `|b−c|={abs(int8['b']-int8['c'])}`,
`2√(b+c)={2 * (int8['b']+int8['c'])**0.5:.6f}`:
**{paired_verdict(int8)}**. Зміна macro-F1 — {int8['delta_macro_f1']:+.6f}.

Стандартна похибка accuracy FP32 як однієї моделі:
`sqrt(p(1−p)/1502)={fp32['single_model_standard_error']:.6f}`.
Парна похибка INT8 — {int8['paired_standard_error']:.6f};
{se_comparison}
У парному порівнянні враховуються лише `b+c={int8['b']+int8['c']}`
зображень із різною правильністю. Спільні правильні та спільні помилкові
відповіді віднімаються. {paired_se_statement}

Правило `|b−c| > 2√(b+c)` стосується accuracy. Для macro-F1 наведено
саму зміну; це правило не є окремим статистичним тестом macro-F1.

## 5. Дослід із потензорними вагами

Для кожної з 20 Conv взято ваги з `graph.initializer` за другим входом
вузла, після злиття BN. Для вихідного каналу:

```math
\alpha_c=\max_{{i,h,w}}|W_{{c,i,h,w}}|,\qquad
r=\frac{{\max_c\alpha_c}}{{\operatorname{{median}}_c\alpha_c}}.
```

Три згортки з найбільшим r:

| Згортка | max α | median α | r | log₂r, біта |
|---|---:|---:|---:|---:|
{channel_rows}

Медіана r за всіма 20 згортками — **{ranges['median_r']:.6f}**,
`log₂(median r)={ranges['median_bits_lost']:.6f}` біта. Це наближена
втрата роздільності медіанного каналу від спільного кроку ваг.
Повні діапазони всіх згорток збережено в JSON.

### Прогноз до запуску

{result['forecast']['text']}

Прогноз записано до обох INT8-квантизацій та оцінювання якості.
Первісний текст і його хеш збережені в `lab4_prediction.json`.

### Результат і зіставлення

Для per-tensor згода з FP32 — {tensor['agreement_with_fp32']:.6f}
проти {int8['agreement_with_fp32']:.6f} для per-channel:
прогноз щодо падіння згоди **{agreement_forecast}**.
macro-F1 змінилася на {tensor['delta_macro_f1']:+.6f} відносно FP32.
`b={tensor['b']}`, `c={tensor['c']}`, `|b−c|={abs(tensor['b']-tensor['c'])}`,
`2√(b+c)={2 * (tensor['b']+tensor['c'])**0.5:.6f}`:
**{paired_verdict(tensor)}**. Прогноз щодо зміни accuracy в межах шуму
**{quality_forecast}**.

Поканальна квантизація змінила передбачення на {int8['changed_predictions']} зображеннях,
потензорна — на {tensor['changed_predictions']}. Згода реагує на кожну зміну
класу, включно з переходами між двома хибними відповідями:
таких переходів {int8['changed_both_wrong']} та {tensor['changed_both_wrong']} відповідно.
Виправлення й нові помилки можуть компенсуватися в `b−c`;
macro-F1 додатково залежить від того, між якими класами вони розподілені.
Тому нижча згода не обов'язково означає пропорційне падіння macro-F1.
Потензорний варіант не використовувався для вибору кращої конфігурації за test.

## 6. Затримка й середовище

| Параметр | Значення |
|---|---|
| Процесор | {env['cpu']} |
| ОС | {env['operating_system']} |
| Архітектура | {env['architecture']} |
| Python | {env['python']} |
| PyTorch / torchvision | {env['torch']} / {env['torchvision']} |
| ONNX / ONNX Runtime | {env['onnx']} / {env['onnxruntime']} |
| NumPy / scikit-learn / Pillow | {env['numpy']} / {env['scikit_learn']} / {env['pillow']} |
| Execution provider | CPUExecutionProvider |
| Потоки inference | intra-op=1, inter-op=1; PyTorch=1/1 |
| Оптимізація графа ORT | {env['graph_optimization_level']} за замовчуванням |
| Пакет якості / затримки | 64 / 1 |
| ARM INT8-функції, sysctl | {cpu_features} |

Налаштування 1/1 задані для сесій оцінювання та вимірювання.
Службові сесії підготовки й калібрування ORT використовують внутрішні
налаштування за замовчуванням. Відхилень від фіксованих параметрів умови немає.

Обидві сесії створено заздалегідь. Вхід — перше test-зображення
`{latency['input_file']}`, підготовлений float32-масив `1×3×160×160`.
Три раунди, у кожному спершу FP32, потім INT8. На кожен варіант
20 прогрівальних викликів, потім 100 вимірів. Між двома
`time.perf_counter()` виконується тільки один `session.run`.
Читання JPEG, перетворення, створення сесій і квантизація в час не входять.
Перцентилі обчислено за всіма 300 вимірами кожного варіанта.

{power_statement}
Стан живлення й навантаження процесів до та після вимірювання збережено в JSON.

| Варіант | Файл, МБ | p50, мс | p95, мс | p50 за раундами, мс |
|---|---:|---:|---:|---|
{latency_rows}

Розмір у МБ — байти / 10⁶. Інтервали p50 за раундами:
FP32 [{fp_interval[0]:.3f}; {fp_interval[1]:.3f}] мс,
INT8 [{int_interval[0]:.3f}; {int_interval[1]:.3f}] мс.
Вони **{latency_verdict}**.

## 7. Висновок

Файл зменшився у **{result['size_reduction_fp32_over_int8']:.3f} раза**:
{result['artifacts']['fp32']['megabytes']:.6f} → {result['artifacts']['int8']['megabytes']:.6f} МБ.
Відношення FP32/INT8 для p50 — **{latency['p50_speedup']:.3f}**,
для p95 — **{latency['p95_speedup']:.3f}**.
Діапазони p50 {latency_verdict}.

Поканальна INT8 має macro-F1 {int8['macro_f1']:.6f} проти {fp32['macro_f1']:.6f},
згоду з FP32 {int8['agreement_with_fp32']:.6f} та
Δaccuracy={int8['delta_accuracy']:+.6f} ± {int8['paired_standard_error']:.6f}:
{paired_verdict(int8)}.

Чотириразове зменшення байтів окремої ваги не означає обов'язкового
чотириразового прискорення всієї мережі. За моделлю даху
`t ≥ max(C/P, Q/S)`, де C — кількість операцій, P — обчислювальна
продуктивність, Q — обсяг обміну даними, S — пропускна здатність пам'яті.
INT8 зменшує трафік ваг і дає змогу використовувати цілочисельні ядра CPU.
{hardware_statement}
Водночас активації, накопичення, перетворення й requantization,
залишкові операції, кеш і виклики runtime мають власну вартість.
Виміряні {latency['p50_speedup']:.3f}× для p50 —
{abs(latency['p50_speedup'] / 4 - 1) * 100:.2f}% {'вище' if latency['p50_speedup'] > 4 else 'нижче'} за 4×.
Зменшення трафіку та зміна обчислювальної продуктивності діють разом,
тому відношення затримок не обмежується самим відношенням байтів ваг.
Без апаратного профілювання не можна
приписати різницю одному конкретному обмеженню.

{suitability}
Цей висновок стосується {env['cpu']} та ONNX Runtime {env['onnxruntime']}.
Дослід per-tensor дав macro-F1 {tensor['macro_f1']:.6f} і згоду
{tensor['agreement_with_fp32']:.6f}; він демонструє вплив спільного кроку ваг,
а його затримка за умовою не вимірювалася.

## Використані джерела та інструменти

- Умова ЛР4 та лекції 8–9 курсу; модель, поділ і дані ЛР3.
- [PyTorch 2.8: експорт ONNX](https://docs.pytorch.org/docs/2.8/onnx.html).
- [ONNX Runtime: статична квантизація й підготовка графа](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).
- [ONNX Runtime: налаштування потоків](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).
- OpenAI Codex.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Відновити README зі збережених результатів ЛР4")
    parser.add_argument("--results", type=Path, default=Path("lab4_results.json"))
    parser.add_argument("--out", type=Path, default=Path("README.md"))
    arguments = parser.parse_args()
    result = json.loads(arguments.results.read_text(encoding="utf-8"))
    forecast_text = result["forecast"]["text"]
    if hashlib.sha256(forecast_text.encode()).hexdigest() != result["forecast"]["text_sha256"]:
        raise ValueError("Saved forecast text does not match its original hash")
    write_report(arguments.out, result)


if __name__ == "__main__":
    main()
