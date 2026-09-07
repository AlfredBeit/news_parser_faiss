# Vedomosti Risk Analyzer

Скрипт загружает RSS-ленту «Ведомостей», отбирает новости с признаками риска,
индексирует их в FAISS и формирует краткий анализ с помощью OpenAI.

## Установка и запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENAI_API_KEY="ваш-новый-ключ"
python analyze_news.py
```

