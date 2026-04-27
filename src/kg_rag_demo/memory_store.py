from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryTurn:
    turn_id: str
    question: str
    answer: str
    final_prompt: str
    intent: str
    used_retrieval: bool
    created_at: str


@dataclass
class LongTermMemoryItem:
    item_id: str
    content: str
    source_turn_id: str
    created_at: str


class MemoryStore:
    """Maintain short-term and long-term conversation memories."""

    def __init__(
        self,
        short_term_path: Path,
        long_term_path: Path,
        short_term_max_turns: int = 8,
        long_term_max_turns: int = 1000,
    ) -> None:
        self.short_term_path = short_term_path
        self.long_term_path = long_term_path
        self.short_term_max_turns = max(1, short_term_max_turns)
        self.long_term_max_turns = max(10, long_term_max_turns)
        self.short_turns: list[MemoryTurn] = self._load_jsonl(self.short_term_path)
        self.long_items: list[LongTermMemoryItem] = self._load_long_items(self.long_term_path)
        if len(self.long_items) > self.long_term_max_turns:
            self.long_items = self.long_items[-self.long_term_max_turns :]
            self._save_long_items(self.long_term_path, self.long_items)

    def recent_context(self, limit: int = 4) -> list[dict[str, Any]]:
        rows = self.short_turns[-max(1, limit) :]
        return [
            {
                "turn_id": row.turn_id,
                "question": row.question,
                "answer": row.answer,
                "intent": row.intent,
                "used_retrieval": row.used_retrieval,
                "created_at": row.created_at,
            }
            for row in rows
        ]

    def recent_retrieved_turns(self, limit: int = 8) -> list[MemoryTurn]:
        rows = [row for row in self.short_turns if row.used_retrieval and row.final_prompt.strip()]
        return rows[-max(1, limit) :]

    def add_turn(
        self,
        question: str,
        answer: str,
        final_prompt: str,
        intent: str,
        used_retrieval: bool,
    ) -> MemoryTurn:
        turn = MemoryTurn(
            turn_id=f"turn_{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            question=question.strip(),
            answer=answer.strip(),
            final_prompt=final_prompt.strip(),
            intent=intent.strip(),
            used_retrieval=bool(used_retrieval),
            created_at=_utc_now_iso(),
        )
        self.short_turns.append(turn)

        overflow = len(self.short_turns) - self.short_term_max_turns
        if overflow > 0:
            self.short_turns = self.short_turns[overflow:]

        self._save_jsonl(self.short_term_path, self.short_turns)
        return turn

    def add_long_term_memory(self, content: str, source_turn_id: str) -> LongTermMemoryItem | None:
        text = content.strip()
        if not text:
            return None
        item = LongTermMemoryItem(
            item_id=f"ltm_{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            content=text,
            source_turn_id=source_turn_id.strip(),
            created_at=_utc_now_iso(),
        )
        self.long_items.append(item)
        if len(self.long_items) > self.long_term_max_turns:
            self.long_items = self.long_items[-self.long_term_max_turns :]
        self._save_long_items(self.long_term_path, self.long_items)
        return item

    def get_long_term_memories(self) -> list[LongTermMemoryItem]:
        return list(self.long_items)

    def reset_short_term_memory(self) -> None:
        self.short_turns = []
        self._save_jsonl(self.short_term_path, self.short_turns)

    def reset_long_term_memory(self) -> None:
        self.long_items = []
        self._save_long_items(self.long_term_path, self.long_items)

    def total_long_term_chars(self) -> int:
        return sum(len(item.content) for item in self.long_items)

    def compress_oldest_half_long_memories(self) -> None:
        if not self.long_items:
            return
        half = max(1, len(self.long_items) // 2)
        for idx in range(half):
            item = self.long_items[idx]
            text = item.content.strip()
            if len(text) <= 40:
                continue
            item.content = self._shrink_text(text)
        self._save_long_items(self.long_term_path, self.long_items)

    @staticmethod
    def _shrink_text(text: str) -> str:
        compact = " ".join(text.split())
        if len(compact) <= 80:
            return compact
        return f"{compact[:35]} ... {compact[-35:]}"

    def recent_short_turns_for_prompt(self, limit: int = 3) -> list[dict[str, str]]:
        rows = self.short_turns[-max(1, limit) :]
        return [
            {
                "question": row.question,
                "answer": row.answer,
            }
            for row in rows
        ]

    @staticmethod
    def _load_jsonl(path: Path) -> list[MemoryTurn]:
        if not path.exists():
            return []
        rows: list[MemoryTurn] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    rows.append(
                        MemoryTurn(
                            turn_id=str(payload.get("turn_id", "")),
                            question=str(payload.get("question", "")),
                            answer=str(payload.get("answer", "")),
                            final_prompt=str(payload.get("final_prompt", "")),
                            intent=str(payload.get("intent", "")),
                            used_retrieval=bool(payload.get("used_retrieval", False)),
                            created_at=str(payload.get("created_at", "")),
                        )
                    )
        except Exception:
            return []
        return rows

    @staticmethod
    def _load_long_items(path: Path) -> list[LongTermMemoryItem]:
        if not path.exists():
            return []
        rows: list[LongTermMemoryItem] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    # Backward compatibility: tolerate old MemoryTurn-shaped long memory file.
                    if "content" not in payload:
                        question = str(payload.get("question", "")).strip()
                        answer = str(payload.get("answer", "")).strip()
                        combined = f"{question} => {answer}".strip(" =>")
                        if not combined:
                            continue
                        rows.append(
                            LongTermMemoryItem(
                                item_id=str(payload.get("turn_id", f"ltm_{len(rows)+1}")),
                                content=combined,
                                source_turn_id=str(payload.get("turn_id", "")),
                                created_at=str(payload.get("created_at", "")),
                            )
                        )
                        continue

                    rows.append(
                        LongTermMemoryItem(
                            item_id=str(payload.get("item_id", f"ltm_{len(rows)+1}")),
                            content=str(payload.get("content", "")),
                            source_turn_id=str(payload.get("source_turn_id", "")),
                            created_at=str(payload.get("created_at", "")),
                        )
                    )
        except Exception:
            return []
        return rows

    @staticmethod
    def _save_jsonl(path: Path, turns: list[MemoryTurn]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for turn in turns:
                f.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")

    @staticmethod
    def _save_long_items(path: Path, items: list[LongTermMemoryItem]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
