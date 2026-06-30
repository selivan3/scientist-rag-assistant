from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = ROOT / "data" / "samples"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Минимальная демонстрация RAG-поиска.")
    parser.add_argument("question", help="Вопрос к базе знаний")
    args = parser.parse_args()
    print(answer(args.question))


if __name__ == "__main__":
    main()

