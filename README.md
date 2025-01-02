# Прогнозирование продаж в магазинах (Time Series Forecasting)

Этот проект является решением для соревнования Kaggle [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting). Соревнование направлено на прогнозирование ежедневных продаж для нескольких магазинов с использованием данных временных рядов.

## Структура проекта

```
TimeSeriesCompetition/
├── data/
│   ├── holidays_events.csv      # Данные о праздниках и событиях.
│   ├── oil.csv                  # Данные о ценах на нефть.
│   ├── stores.csv               # Метаданные магазинов.
│   ├── test.csv                 # Тестовые данные.
│   ├── train.csv                # Обучающие данные.
│   ├── transactions.csv         # Данные о транзакциях в магазинах.
├── modeling.ipynb               # Ноутбук для построения и оценки моделей.
├── preprocessing_data.ipynb     # Ноутбук для предобработки данных и генерации признаков.
├── nn_toolkits/                 # Пользовательские инструменты и утилиты для моделирования.
├── requirements.txt             # Список зависимостей.
```

## Как использовать

### 1. Клонируйте репозиторий

Склонируйте этот репозиторий на ваш локальный компьютер и перейдите в директорию проекта.

```bash
git clone <repository-url>
cd TimeSeriesCompetition
```

### 2. Установите зависимости

Создайте виртуальное окружение и установите необходимые Python-пакеты:

```bash
python -m venv env
source env/bin/activate   # В Windows: .\env\Scripts\activate
pip install -r requirements.txt
```

### 3. Изучите данные

Папка `data/` содержит наборы данных, предоставленные для соревнования. Изучите эти файлы, чтобы понять доступные признаки.

### 4. Предобработка данных

Запустите ноутбук `preprocessing_data.ipynb` для подготовки данных к моделированию. Это включает обработку пропущенных значений, создание новых признаков и масштабирование.

### 5. Построение и оценка моделей

Откройте ноутбук `modeling.ipynb`, чтобы экспериментировать с различными моделями для прогнозирования временных рядов. Этот ноутбук включает обучение, проверку и оценку моделей.

### 6. Кастомизация с помощью `nn_toolkits`

В папке `nn_toolkits/` содержатся модули для продвинутого моделирования, например, нейронных сетей. Вы можете модернизировать или создать на их основе собственные модули для других задач.

## Описание набора данных

- **holidays_events.csv**: Данные о праздниках и событиях.
- **oil.csv**: Ежедневные цены на нефть, которые могут влиять на продажи.
- **stores.csv**: Метаданные магазинов.
- **train.csv**: Обучающие данные с историей продаж.
- **test.csv**: Тестовые данные для прогнозирования.

## Цель соревнования

Цель — спрогнозировать ежедневные продажи для нескольких магазинов, используя предоставленные наборы данных. Итоговая отправка должна включать прогнозы для тестового набора данных.

## Результаты:
Оценка **RMSLE: 0.43953** _(≈top 15%)_
