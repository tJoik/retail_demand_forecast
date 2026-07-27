# Прогнозирование продаж хозяйственного магазина

Пайплайн прогнозирования недельных, месячных и квартальных продаж для розничного магазина. От выгрузки из 1С до FastAPI-сервиса с UI, откуда можно получить прогноз в разрезе категорий и выгрузить заявку на закупку в Excel.

## Стек

- **ML**: XGBoost (loss = MAE), scikit-learn Pipeline, TargetEncoder, TimeSeriesSplit, GridSearchCV.
- **Метрики**: WAPE (метрика выбора гиперпараметров на CV), MAE / MSE / R².
- **Сервис**: FastAPI + Jinja2, экспорт в Excel через openpyxl.
- **Данные**: 1С (COM-коннектор), Open-Meteo (погода), внутренний справочник поставщиков, данные о праздниках.

## Результаты моделей

| Горизонт | MAE | WAPE | n_test |
|---|---|---|---|
| Неделя | 0.287 | 0.83 | 94 105 |
| Месяц | 1.020 | 0.82 | 36 785 |
| Квартал | 3.487 | 1.38 | 26 116 |

## Структура

```
retail_demand_forecast/
├── features_import_one_week.ipynb   # Выгрузка из 1С в папку features/features_one_week.csv
├── eda_weeks.ipynb                  # Разведочный анализ
├── make_one_week_model.ipynb        # Обучение недельной модели
├── make_one_month_model.ipynb       # Обучение месячной модели
├── make_3_month_model.ipynb         # Обучение квартальной модели
│
├── features/
│   └── features_one_week.csv        # Основной датасет (для недель)
├── nomenclature_import/
│   └── nomenclature_with_supplier.csv  # Справочник поставщиков
├── models/
│   ├── weeks.pkl                    # создаются "make" ноутбуками
│   ├── month.pkl
│   └── 3month.pkl
│
└── service/                         # FastAPI-сервис
    ├── app.py
    ├── predict.py
    ├── precompute.py
    ├── run.py
    ├── templates/
    ├── static/
    └── precomputed/
```

## Как запустить

### 1. Окружение

Разработка велась на **Python 3.13.5** (Anaconda). Рекомендуется 3.13.x.

```bash
pip install -r requirements.txt
```

Если планируешь заново выгружать данные из 1С — раскомментируй `pywin32` в `requirements.txt` (только Windows, а также нужна установленная 1С:Предприятие 8.3 и файловая база).

### 2. Данные

Для полного пайплайна нужен `features/features_one_week.csv`. Его собирает [features_import_one_week.ipynb](features_import_one_week.ipynb) из живой базы 1С. Ни база 1С (1c_files/) ни файл `features/features_one_week.csv` в репозиторий не коммитятся (Данные принадлежат компании и не публикуются. Доступно по запросу.).

### 3. Обучение

Требуется `features/features_one_week.csv` из шага 2! Без него ноутбуки упадут на первой ячейке чтения данных!

Открой любой из трёх ноутбуков и прогони все ячейки:
- [make_one_week_model.ipynb](make_one_week_model.ipynb)
- [make_one_month_model.ipynb](make_one_month_model.ipynb)
- [make_3_month_model.ipynb](make_3_month_model.ipynb)

Каждый сохраняет обученную модель в `models/*.pkl`.

### 4. Сервис

```bash
python service/precompute.py     # пересчёт матриц признаков для ближайшего прогнозного периода
python service/run.py            # запуск uvicorn
```

Открой `http://127.0.0.1:8000`, выбери горизонт, выбери необходимые товары, получи прогноз и выгрузи Excel.

## Пайплайн подготовки данных

1. **1С COM-коннектор** — три параметризованных запроса: продажи по неделям, движения приходов, история цен.
2. **Погода** — Open-Meteo daily.
3. **Merge**: для цен и даты последнего поступления.
4. **Признаки в ноутбуках**:
   - TF-IDF по наименованию товара (`tfidf_mean`);
   - подмешивание бренда из справочника поставщиков;
   - лаговые статистики по окнам [2, 4, 8, 12] недель (и сдвинутые на 1 неделю);
   - для месячной / квартальной моделей — агрегация недельных строк + свои лаги.

## Выбор параметров модели XGboost

GridSearchCV по сетке `{n_estimators: [300, 600], max_depth: [4, 6]}` с TimeSeriesSplit-3, scoring — WAPE. Победители по CV-WAPE:

| Горизонт | n_estimators | max_depth | CV-WAPE |
|---|---|---|---|
| Неделя | 600 | 6 | 0.732 |
| Месяц | 300 | 4 | 0.701 |
| Квартал | 300 | 4 | 0.669 |