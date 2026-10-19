# Данные

## Пакет для полного обучения

**`solution_external_verification.zip`** размещается в разделе **Releases** этого репозитория.

- Размер: 582010693 байт.
- SHA256: `b8b0110d3d47703e5e9b3e07b6dd670add8f233173eae279d06afa173745378b`.

Пакет содержит competition train/test/sample, `content_features`, `source_features` и три внешних URL-датасета. Код извлекает входы в отдельную рабочую папку и сверяет хеши. Параметры модели зафиксированы в `configs/external.json`.

## Источники

HTML организатора: [Яндекс.Диск](https://disk.yandex.ru/d/WhQO9dUapBEeOA). Исходный архив — 1 773 004 430 байт, SHA256 `5e001d2b843f79a5eabfd1b0a30be45fcfa4658c77815c693eb2be5f395549d1`.

| Внешний набор | Лицензия | Преобразование |
| --- | --- | --- |
| [Malicious URLs, Manu Siddhartha](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset), v1 | CC0 | benign → 0, phishing → 1; прочие классы исключены |
| [URL Dataset, TeseRact](https://www.kaggle.com/datasets/teseract/urldataset), v1 | CC0 | good → 0, bad → 1 |
| [PhiUSIIL](https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset) | CC BY 4.0 | legitimate (label=1) → 0, phishing (label=0) → 1 |

PhiUSIIL: Prasad, A. & Chandra, S. (2024). *PhiUSIIL Phishing URL (Website) [Dataset]*. UCI Machine Learning Repository. [DOI](https://doi.org/10.1016/j.cose.2023.103545), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Остальные 54 признака PhiUSIIL не использовались. Ссылки, версии и хеши архивов — [external_sources.json](../reference/external_sources.json).

Перед построением внешних host rules удалены совпадения нормализованных URL со всеми competition train URL. При прогнозе test применяются согласованные exact URL matches и однозначные host labels в прежней области уверенности. Приватные метки соревнования не использовались.

## Запуск из исходных данных

Оригинальные train/test/sample поместите в `data/competition/`:

```powershell
.\.venv\Scripts\python.exe reproduce.py train --competition-dir "data\competition"
```

HTML организатора и фиксированные внешние наборы будут скачаны автоматически. Если архивы уже локально:

```powershell
.\.venv\Scripts\python.exe reproduce.py train --competition-dir "data\competition" --organizer-archive "D:\data\organizer_pages.zip" --external-dir "D:\data\external"
```

В `external-dir` нужны `malicious_urls_v1.zip`, `url_dataset_v1.zip`, `phiusiil_uci_967.zip`. Используются исходные файлы с фиксированными SHA256. Сайты по URL не посещаются, JavaScript не выполняется.

## Быстрый режим

`artifacts/external_test_scores.npz` содержит test scores refined/source и решения known rules, полученные из train и внешних источников. Эталонный CSV лежит в `reference/`. Быстрый режим не обучает классификаторы; F1 на private локально без приватных меток не пересчитывается.
