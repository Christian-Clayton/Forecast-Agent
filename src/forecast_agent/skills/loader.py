import re
from dataclasses import dataclass
from pathlib import Path

_SKILL_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


@dataclass(frozen=True)
class SkillDocument:
    skill_id: str
    path: Path
    content: str
    token_estimate: int


class SkillLoader:
    def __init__(self, directory: str | Path, max_bytes: int = 131_072) -> None:
        self._directory = Path(directory).resolve()
        self._max_bytes = max_bytes
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")

    def load(self, skill_id: str) -> SkillDocument:
        if not _SKILL_ID.fullmatch(skill_id):
            raise ValueError(f"invalid skill id: {skill_id!r}")

        candidates = (
            self._directory / skill_id / "SKILL.md",
            self._directory / f"{skill_id}.md",
        )
        for candidate in candidates:
            if not candidate.is_file():
                continue
            resolved_path = candidate.resolve()
            if self._directory not in resolved_path.parents:
                raise ValueError("skill path escapes the configured skills directory")
            size = resolved_path.stat().st_size
            if size > self._max_bytes:
                raise ValueError(
                    f"skill {skill_id!r} exceeds the configured size limit"
                )
            content = resolved_path.read_text(encoding="utf-8")
            return SkillDocument(
                skill_id=skill_id,
                path=resolved_path,
                content=content,
                token_estimate=(len(content) + 3) // 4,
            )
        raise FileNotFoundError(f"skill not found: {skill_id}")
