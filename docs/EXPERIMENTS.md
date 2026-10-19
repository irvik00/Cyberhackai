# Эксперименты

## Основные этапы

1. URL char TF-IDF и гибридная URL-модель на LinearSVC.
2. Добавление заголовков, текста и структуры сохранённого HTML.
3. Числовой бустинг и однозначные HTML-template matches.
4. Source/attributes LinearSVC и ограниченные правила по внешним URL — выбранный **external**.
5. Коррекции коротких страниц, brand/domain features и совместный short+brand вариант.
6. Short experts, HTML representation и анализ ближайших соседей ошибок.
7. Эксперименты с длинными FN: внешние примеры, контекст form/password, language и verified brand features, аудит правил и URL/content consistency.

## Результаты отправок

| Файл | Public F1 | Private F1 |
| --- | ---: | ---: |
| `submission_short_brand_joint.csv` | 0.994750 | 0.992641 |
| `submission_brand_domain.csv` | 0.994752 | 0.992558 |
| `submission_short_pages.csv` | 0.995078 | 0.992389 |
| `submission_competition_short_s0.csv` | 0.994750 | 0.992387 |
| `submission_external.csv` | 0.995079 | 0.992305 |
| `submission_url_content_consistency.csv` | 0.995408 | 0.992223 |
| `submission_html_family_threshold_only.csv` | 0.995078 | 0.992135 |
| `submission_url_html.csv` | 0.994421 | 0.990549 |
| `submission_html_refined.csv` | 0.994420 | 0.990459 |
| `submission_hybrid_svm.csv` | 0.974984 | 0.974253 |
| `submission_char_svm.csv` | 0.970760 | 0.968649 |

**External** был выбран для зачёта: private 0.992305. После раскрытия private максимальный результат среди всех 11 отправок оказался у **short+brand joint — 0.992641**. Максимальный public был у URL/content — 0.995408, private этого варианта — 0.992223.

## Дополнительные проверки

| Эксперимент | Локальный результат |
| --- | --- |
| Short+brand joint | 477 ошибок при обычном OOF-подборе; 494 при выборе параметров без query-фолда |
| Competition short expert <100 | 500 ошибок после leave-one-fold-out selection |
| HTML short expert | Улучшение отдельной модели не дало устойчивого выигрыша ансамбля; отправленный HTML-family вариант использовал только порог |
| Targeted PhreshPhish pilot | 508 ошибок — без выигрыша относительно external |
| HTML-контекст form/password | 516 ошибок |
| Language features | 519 ошибок |
| Verified brand features | 511 ошибок; совместная language+brand модель не обучалась |
| Ограничение принудительных правил | 506 ошибок; улучшение только на двух фолдах |
| URL/content consistency | 504 ошибки; исправлены 14 FN и 1 FP, появились 4 FN и 7 FP |

Базовый external — 508 ошибок. Для neighbours и новых обучаемых признаков использовались fitting части соответствующих фолдов. Пилоты внешних данных и дополнительные коррекции не входят в воспроизводимый external.

Ранжирование близких вариантов на public и private разошлось. Исторический OOF неоднократно использовался для исследований и подбора параметров, поэтому его значение не является независимой оценкой. Параметры отправленных файлов после раскрытия private не изменялись.
