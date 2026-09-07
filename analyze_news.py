#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Анализ рисков компаний по RSS-ленте «Ведомостей»."""

import logging
import os
from pathlib import Path


import feedparser
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

RSS_URL = "https://www.vedomosti.ru/rss/news.xml"
INDEX_PATH = Path("vedomosti_faiss_index")
MAX_RESULTS = 5

KEYWORDS = (
    "банкрот", "риск", "санкц", "уголовн", "арест", "обыск",
    "дефолт", "ликвидац", "мошеннич", "суд",
)

QUERY = """
Найди новости о компаниях, связанные с банкротством, санкциями,
уголовными делами, судебными проблемами и репутационными рисками.
""".strip()


def load_news(url: str) -> list[str]:
    """Загружает из RSS непустые заголовки и описания."""
    feed = feedparser.parse(url)

    if feed.bozo:
        logging.warning("Проблема при чтении RSS: %s", feed.bozo_exception)
    if not feed.entries:
        raise RuntimeError("RSS-лента не содержит новостей.")

    news_items = []
    for entry in feed.entries:
        title = entry.get("title", "").strip()
        summary = (entry.get("summary") or entry.get("description") or "").strip()
        news = ". ".join(part for part in (title, summary) if part)
        if news:
            news_items.append(news)

    return news_items


def filter_news(news_items: list[str]) -> list[str]:
    """Предварительно фильтрует новости по маркерам риска."""
    filtered = [
        news
        for news in news_items
        if any(keyword in news.casefold() for keyword in KEYWORDS)
    ]
    if not filtered:
        raise RuntimeError("В RSS-ленте нет новостей по заданным ключевым словам.")
    return filtered


def build_context(news_items: list[str]) -> str:
    """Создаёт FAISS-индекс и возвращает релевантные документы."""
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        max_retries=2,
    )
    vector_db = FAISS.from_texts(texts=news_items, embedding=embeddings)
    vector_db.save_local(str(INDEX_PATH))

    documents = vector_db.similarity_search(
        query=QUERY,
        k=min(MAX_RESULTS, len(news_items)),
    )
    return "\n\n".join(
        f"{number}. {document.page_content}"
        for number, document in enumerate(documents, start=1)
    )


def analyze_news(context: str) -> str:
    """Анализирует найденные документы с помощью языковой модели."""
    prompt = f"""
Проанализируй новости, найденные с помощью FAISS.

Для каждой релевантной новости укажи:
1. компанию или организацию;
2. категорию риска;
3. факты, указывающие на риск;
4. уровень риска: низкий, средний или высокий.

Возможные категории:
- банкротство;
- санкции;
- уголовное дело;
- судебный или регуляторный риск;
- репутационный риск.

Не придумывай факты, которых нет в тексте. Если данных недостаточно, укажи это.

Новости:
{context}
""".strip()

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5.6-terra"),
        max_retries=2,
    )
    return str(llm.invoke(prompt).content)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Задайте новый OPENAI_API_KEY в переменной окружения.")

    all_news = load_news(RSS_URL)
    filtered_news = filter_news(all_news)
    logging.info("Отобрано новостей: %d из %d", len(filtered_news), len(all_news))
    print(analyze_news(build_context(filtered_news)))


if __name__ == "__main__":
    main()
