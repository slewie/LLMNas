# LLM NAS for Time Series Forecasting

* Ручной подбор архитектур и параметров нейросетей - это долго, дорого и требует экспертных знаний
* Классические NAS требуют много ресурсов
* Популярные Optuna/Hyperopt:
  * Работают в жестко заданном search space
  * Относятся к модели как к "Черному ящику", не понимая ее внутренней логики
  * Не интерпретируемы
* LLM (по API) не требуют дополнительных ресурсов
* LLM могут без заданного search space
* LLM могут оптимизировать вычисления


## Стек технологий

* Язык: Python 3.10+
* ML Framework: PyTorch
* Оптимизация: Optuna (TPE Sampler)
* LLM Integration: LangChain / OpenRouter API
* LLM Модели: Gemini-2.5-Flash-Lite / Gemini-2.5-Pro / Grok-4.1-Fast
* Архитектура модели: Informer / InformerStack / LSTM


## Результаты экспериментов

Я проводил эксперименты на датасетах [ETTh1 и ETTh2](https://github.com/zhouhaoyi/ETDataset).
Целевая метрика: **MSE**.

Перебираемые параметры:
- model_type: ['informer', 'informerstack', 'lstm']
- d_model: [128, 256, 512, 768]
- n_heads: [4, 8, 16]
- e_layers: [1, 2, 3, 4, 5]
- d_layers: [1, 2, 3]
- d_ff: [512, 1024, 2048]
- factor: [3, 5]
- learning_rate: [1e-5, 1e-4, 1e-3]


### ETTh1
| Метод | Среднее MSE (по 3 запускам) | Best MSE | Конфигурация |
|-------|----------|---------------------|---------------------|
| **Optuna** (TPE) |  0.58 | 0.55 |"model_type": "informer", "d_model": 512, "n_heads": 16,"e_layers": 1,"d_layers": 2,"d_ff": 1024,"factor": 3,"learning_rate": 0.0001
| **LLM** (Gemini 2.5 Pro) |  0.55 | 0.52 | "model_type": "lstm", "d_model": "256", "n_heads": "4", "e_layers": "1", "d_layers": "3", "d_ff": "1024", "factor": "3", "learning_rate": "0.001" |
| **LLM** (Gemini 2.5 Flash) |  0.63 | 0.60 | "model_type": "informer", "d_model": "256", "n_heads": "8", "e_layers": "3", "d_layers": "2", "d_ff": "2048", "factor": "3", "learning_rate": "0.0001" |
| **LLM** (Grok 4.1 Fast) | 0.6 | 0.58 | "model_type": "lstm", "d_model": "256",  "n_heads": "8", "e_layers": "2", "d_layers": "1",  "d_ff": "512", "factor": "5", "learning_rate": "1e-3" |

### ETTh2
| Метод | Best MSE | Конфигурация |
|-------|----------|---------------------|
| **Optuna** (TPE) |   0.3 | 0.28 | "d_model": 512,  "n_heads": 4,  "e_layers": 1, "d_layers": 3,  "d_ff": 1024,  "factor": 5,  "learning_rate": 0.0001 |
| **LLM** (Gemini 2.5 Flash) |   0.247 | 0.24 | "model_type": "informerstack", "d_model": "256", "n_heads": "8", "e_layers": "5", "d_layers": "2", "d_ff": "2048", "factor": "5", "learning_rate": "0.0001" |

## Пример цепочки рассуждений LLM
The first trial with the 'informer' model established a baseline MSE of 0.631. To better understand the problem landscape, it is valuable to test a fundamentally different and simpler architecture. I will choose the 'lstm' model to see if a classic rnn can perform competitively... -> LSTM 0.57

The history shows that the 'lstm' model (MSE: 0.5737) significantly outperformed the 'informer' model (MSE: 0.6310). This suggests that a simpler recurrent architecture is more effective for this specific dataset than the transformer-based one. To build upon this success, I will continue with the 'lstm' model but increase its capacity to see if it can capture more complex patterns without overfitting... -> LSTM 0.68

The history shows that a moderately sized 'lstm' model (MSE: 0.5737) significantly outperformed both a larger 'informer' model (MSE: 0.6310) and a deeper 'lstm' model (MSE: 0.6848) ... I will now try the 'informer' architecture again, but this time with hyperparameters inspired by the successful LSTM run... -> Informer 0.68

The history shows a clear pattern: a simple 'lstm' model is superior ... model benefits from a more powerful decoder to process the information captured by the simple encoder. To exploit this successful finding, I will continue this trend. I will keep the architecture and all other hyperparameters from the best run identical and make a single, incremental change: increasing 'd_layers' from 2 to its maximum value of 3. This is a direct exploitation of the most promising trend observed so far.
-> LSTM 0.52


## Запуск проекта

1.  **Установка зависимостей:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Настройка окружения:**
    Создайте файл `.env` по подобию `.env.example`:
    ```bash
    OPENROUTER_API_KEY=sk-or-v1-...
    ```

3.  **Запуск:**
   Запуск NAS:
    ```bash
    python run_nas.py --model_name "google/gemini-2.5-flash-lite" --nas_iter 5
    ```
    Запуск Optuna:
    ```bash
    python run_optuna.py --n_trials 20
    ```

## Дальнейшие шаги

1. Попробовать вместе LLM и Optuna. 
   - Либо использовать LLM для выбора пространства поиска, а Optuna для перебора этого пространства.
   - Либо через Optuna найти лучшую конфигурацию, а затем использовать LLM с этой стартовой конфигурации.
2. Применить Agentic Tree Search.
3. Больше экспериментов на большем search space и большем количестве итераций.

