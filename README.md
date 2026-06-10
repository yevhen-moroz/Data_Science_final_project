# Telecom Churn Prediction

Проєкт прогнозує ймовірність відтоку клієнтів телекомунікаційної компанії на основі історичних даних про підписку, рахунки, контракт, якість сервісу та використання інтернету.

## Структура проєкту

```text
telecom_churn_app/
│
├── data/
│   ├── raw/
│   │   └── internet_service_churn.csv
│   └── processed/
│       └── cleaned_churn_data.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── train_model.py
│   ├── predict.py
│   └── app.py
│
├── models/
│   ├── churn_model.pkl
│   └── metrics.json
│
├── assets/
│   ├── churn_distribution.png
│   ├── correlation_heatmap.png
│   ├── subscription_age_by_churn.png
│   ├── bill_avg_by_churn.png
│   └── eda_summary.txt
│
├── requirements.txt
├── Dockerfile
└── README.md
```

## Логіка роботи

У проєкті використовується такий pipeline:

```text
notebooks/EDA.ipynb
        ↓
data/processed/cleaned_churn_data.csv
        ↓
src/train_model.py
        ↓
models/churn_model.pkl
        ↓
src/app.py
```

## Що робить EDA.ipynb

Після запуску ноутбука створюється очищений датасет `data/processed/cleaned_churn_data.csv`, і вже на ньому навчається модель.

Ноутбук виконує:

- завантаження сирого датасету з `data/raw/internet_service_churn.csv`;
- перегляд структури даних;
- перевірку пропущених значень;
- аналіз розподілу цільової змінної `churn`;
- побудову графіків;
- аналіз кореляцій;
- очищення датасету;
- збереження результату в `data/processed/cleaned_churn_data.csv`.

Правила очищення:

- колонка `id` видаляється;
- `reamining_contract` заповнюється значенням `0`;
- `download_avg` заповнюється медіаною;
- `upload_avg` заповнюється медіаною.

## Встановлення залежностей

```bash
pip install -r requirements.txt
```

## Порядок запуску

### 1. Запустити EDA-ноутбук

Відкрийте файл:

```text
notebooks/EDA.ipynb
```

і виконайте всі клітинки. Після цього має зʼявитися файл:

```text
data/processed/cleaned_churn_data.csv
```

### 2. Навчити модель

```bash
python src/train_model.py
```

Після навчання будуть створені або оновлені файли:

```text
models/churn_model.pkl
models/metrics.json
```

### 3. Запустити Streamlit-додаток

```bash
streamlit run src/app.py
```

Після запуску відкриється вебінтерфейс, де можна ввести дані нового клієнта й отримати прогноз:

```text
Високий / середній / низький ризик відтоку
```

## Docker

Збірка контейнера:

```bash
docker build -t telecom-churn-app .
```

Запуск контейнера:

```bash
docker run -p 8501:8501 telecom-churn-app
```

Після запуску застосунок буде доступний у браузері:

```text
http://localhost:8501
```

## Моделі

У `src/train_model.py` порівнюються кілька моделей:

- Logistic Regression;
- Decision Tree;
- Random Forest.

Найкраща модель вибирається за значенням F1-score і зберігається у `models/churn_model.pkl`.

## Метрики

Для оцінки використовуються:

- Accuracy;
- Precision;
- Recall;
- F1-score;
- Classification report;
- Confusion matrix.

Для задачі прогнозування відтоку клієнтів особливо важливі `Recall` та `F1-score`, оскільки компанії важливо не пропустити клієнтів, які справді можуть припинити користування послугами.
