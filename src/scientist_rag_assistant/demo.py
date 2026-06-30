from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = ROOT / "data" / "samples"
FIGURES_DIR = ROOT / "docs" / "figures"

FIGURES: list[tuple[str, str]] = [
    ("рисунок-1-rag.png", "Общая схема RAG"),
    ("рисунок-1-конвейер.png", "Конвейер обработки источников"),
    ("рисунок-2-интерфейс.png", "Интерфейс модуля обработки документов через Paddle OCR"),
    ("рисунок-2-подготовка.png", "Подготовка данных"),
    ("рисунок-3-знания.png", "База знаний"),
    ("рисунок-3-результат.png", "Результат обработки документа через Paddle OCR"),
    ("рисунок-4-rag-демо.png", "Демонстрация RAG-сценария"),
    ("рисунок-4-конвейер-чб.png", "Чёрно-белая схема конвейера"),
    ("рисунок-5-структуризация.png", "Структуризация материалов"),
    ("рисунок-6-агенты.png", "Агентная организация"),
]


def tokenize(text: str) -> set[str]:
    return {part.lower() for part in re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text)}


def load_fragments() -> list[tuple[str, str]]:
    fragments: list[tuple[str, str]] = []
    for path in sorted(SAMPLES_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for paragraph in text.split("\n\n"):
            cleaned = paragraph.strip()
            if cleaned:
                fragments.append((path.name, cleaned))
    return fragments


def answer(question: str, top_k: int = 3) -> str:
    query_tokens = tokenize(question)
    scored: list[tuple[int, str, str]] = []
    for source, fragment in load_fragments():
        score = len(query_tokens & tokenize(fragment))
        if score:
            scored.append((score, source, fragment))

    scored.sort(reverse=True, key=lambda item: item[0])
    hits = scored[:top_k]
    if not hits:
        return "В загруженных источниках недостаточно данных для точного ответа."

    lines = ["Ответ сформирован по найденным фрагментам:\n"]
    for index, (_, source, fragment) in enumerate(hits, start=1):
        lines.append(f"{index}. {fragment}\nИсточник: {source}\n")
    return "\n".join(lines)


def list_figures() -> str:
    missing = [filename for filename, _ in FIGURES if not (FIGURES_DIR / filename).is_file()]
    if missing:
        files = ", ".join(missing)
        raise SystemExit(f"Не найдены рисунки демонстрации: {files}")

    lines = ["Рисунки, подключённые к демонстрации:\n"]
    for index, (filename, title) in enumerate(FIGURES, start=1):
        lines.append(f"{index}. {title}\nФайл: docs/figures/{filename}\n")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Минимальная демонстрация RAG-поиска.")
    parser.add_argument("question", nargs="?", help="Вопрос к базе знаний")
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Показать рисунки, подключённые к демонстрации",
    )
    args = parser.parse_args()
    if args.figures:
        print(list_figures())
        return
    if not args.question:
        parser.error("укажите вопрос или используйте --figures")
    print(answer(args.question))


if __name__ == "__main__":
    main()
