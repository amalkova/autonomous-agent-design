# HW1 — Demand & Discovery ReAct Agent

Навчальний ReAct-агент на базі LangGraph, LangChain і Gemini API для супроводу demand-ініціатив та попередньої оцінки discovery.

Агент уміє:

- знаходити поточний статус ініціативи;
- перевіряти повноту intake;
- розраховувати Discovery Points за шкалою Fibonacci;
- рекомендувати Light, Standard або Deep discovery;
- розраховувати priority score за оцінками, заданими людиною;
- запитувати відсутні дані замість їх вигадування.

## Архітектура

```mermaid
flowchart TD
    A[User request] --> B[Gemini decision]
    B -->|Tool required| C[Tool execution]
    C --> D[Observation]
    D --> B
    B -->|Enough information| E[Structured response]
    E --> F[Trajectory log]
```

ReAct-цикл реалізований вручну через `StateGraph`:

1. LLM аналізує запит.
2. Обирає потрібний інструмент.
3. LangGraph виконує tool call.
4. Результат повертається моделі як observation.
5. Модель продовжує цикл або формує відповідь.
6. Фінальна відповідь перевіряється Pydantic-схемою.

## Інструменти

| Tool | Призначення |
|---|---|
| `get_initiative_status` | Повертає статус синтетичної demand-ініціативи |
| `check_intake_completeness` | Перевіряє обов’язкові поля Gate 0 |
| `classify_discovery_scope` | Розраховує Discovery Points та рекомендує scope |
| `calculate_priority_score` | Розраховує priority score за людськими оцінками |

Усі інструменти використовують Pydantic-схеми та повертають стандартний JSON:

```json
{
  "status": "success",
  "data": {},
  "error": null
}
```

## Discovery Points

Для оцінювання використовуються лише явно надані параметри:

- кількість систем;
- зрозумілість ownership;
- технічна невизначеність;
- зовнішні залежності;
- регуляторний вплив;
- готовність даних.

Raw score переводиться у Fibonacci:

| Raw score | Discovery Points |
|---:|---:|
| 0–1 | 1 |
| 2–3 | 2 |
| 4–5 | 3 |
| 6–7 | 5 |
| 8–10 | 8 |
| 11–14 | 13 |

Відповідність scope:

| Discovery Points | Scope |
|---:|---|
| 1–2 | Light |
| 3–5 | Standard |
| 8–13 | Deep |

Ініціатива, яка зачіпає три або більше систем, отримує щонайменше Standard scope.

Результат є рекомендацією агента і потребує підтвердження Lead Business Analyst.

## Priority Score

Критерії оцінюються людиною від 1 до 5:

| Criterion | Weight |
|---|---:|
| Strategic alignment | 30% |
| Customer impact | 25% |
| Financial impact | 20% |
| Regulatory urgency | 15% |
| Implementation feasibility | 10% |

Агент не визначає значення критеріїв самостійно — він лише виконує розрахунок.

## Safety controls

Реалізовано три рівні захисту:

- `max_steps=8` — максимальна кількість LLM-кроків;
- `timeout_seconds=60` — обмеження тривалості запуску;
- `repeated_call_limit=2` — зупинка після повторення однакового tool call.

Для кожного запуску створюється окремий `SafetyController`.

## Встановлення

Потрібен Python 3.13 або сумісна версія.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Створити файл `.env`:

```env
GOOGLE_API_KEY=your_api_key
```

Файл `.env` виключений з Git і не повинен потрапляти до репозиторію.

## Запуск

Інтерактивний режим:

```bash
python agent.py
```

Приклад запиту:

```text
Какой статус у инициативы DEM-001?
```

Для завершення:

```text
exit
```

## Тестування

Unit-тести:

```bash
pytest -q
```

Поточний результат:

```text
11 passed
```

End-to-end сценарії з реальною моделлю:

```bash
python test_runner.py
```

Поточний результат:

```text
5/5 passed
```

Між сценаріями використовується пауза через обмеження Gemini Free Tier.

## Артефакти

- `trajectory.json` — повна ReAct-траєкторія: human → AI tool call → tool observation → AI response.
- `test_results.json` — результати п’яти end-to-end сценаріїв.
- `test_tools.py` — unit-тести інструментів.
- `test_safety.py` — unit-тести safety controls.

## Структура проєкту

```text
hw1_react_agent/
├── agent.py
├── tools.py
├── safety.py
├── trajectory_logger.py
├── test_runner.py
├── test_tools.py
├── test_safety.py
├── trajectory.json
├── test_results.json
├── requirements.txt
├── README.md
├── .env
└── .gitignore
```

## Модель

Використовується `gemini-3.1-flash-lite`, оскільки рекомендована в початковому завданні `gemini-2.5-flash` більше не доступна новим користувачам Gemini API.

Модель підтримує function calling та structured output.