"""Research projects you can save, close and pick up later.

Each project is a folder under projects/ (next to app.py, or RESEARCHPILOT_PROJECTS_DIR):
  projects/<name>/project.json   everything needed to resume: papers, reviews, plan, progress, lab log
  projects/<name>/files/         files attached to lab-log entries
Nothing leaves your computer.
"""
import json
import os
import re
from datetime import datetime
from pathlib import Path

VERSION = 1
TEXT_TYPES = {".csv", ".tsv", ".txt", ".md", ".json", ".log", ".dat", ".xy", ".cif"}


def projects_dir():
    root = os.getenv("RESEARCHPILOT_PROJECTS_DIR") or Path(__file__).resolve().parent / "projects"
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(name):
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(name or "")).strip("-").lower()[:60]
    return slug or "research-project"


def new_project_dir(name):
    base = projects_dir() / slugify(name)
    path, n = base, 2
    while path.exists():
        path = Path(f"{base}-{n}")
        n += 1
    (path / "files").mkdir(parents=True)
    return path


def list_projects():
    """Saved projects, most recently saved first: [{"name", "dir", "saved_at", "question"}]."""
    found = []
    for file in projects_dir().glob("*/project.json"):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        found.append({"name": data.get("name") or file.parent.name, "dir": str(file.parent),
                      "saved_at": data.get("saved_at", ""), "question": data.get("question", "")})
    return sorted(found, key=lambda p: p["saved_at"], reverse=True)


def save_project(project_dir, data):
    """Write atomically, so a crash mid-save never leaves a half-written project."""
    project_dir = Path(project_dir)
    (project_dir / "files").mkdir(parents=True, exist_ok=True)
    data = {**data, "version": VERSION, "saved_at": datetime.now().isoformat(timespec="seconds")}
    text = json.dumps(data, indent=1, ensure_ascii=False, default=str)
    tmp = project_dir / "project.json.tmp"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(project_dir / "project.json")
    return data["saved_at"]


def load_project(project_dir):
    data = json.loads((Path(project_dir) / "project.json").read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "papers" not in data:
        raise ValueError("This isn't a ResearchPilot project file.")
    return data


def save_attachment(project_dir, entry_id, name, content):
    """Store a lab-log attachment; returns metadata, including a text excerpt the AI can read for data files."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:80] or "file"
    path = Path(project_dir) / "files" / f"{entry_id}_{safe}"
    path.write_bytes(content)
    meta = {"name": name, "path": str(path.relative_to(project_dir)), "size": len(content)}
    if path.suffix.lower() in TEXT_TYPES and len(content) <= 2_000_000:
        meta["excerpt"] = content[:4000].decode("utf-8", errors="replace")
    return meta


def merge_papers(existing, incoming, note=None):
    """Add newly found papers, skipping duplicates, with IDs that continue after the existing ones."""
    def key(p):
        return (p.get("doi") or "").lower() or re.sub(r"\W+", " ", p.get("title", "").lower()).strip()

    seen = {key(p) for p in existing}
    next_n = max([int(p["id"][1:]) for p in existing if re.fullmatch(r"P\d+", str(p.get("id")))] or [0]) + 1
    added = []
    for p in incoming:
        k = key(p)
        if not k or k in seen:
            continue
        seen.add(k)
        q = {**p, "id": f"P{next_n:03d}"}
        if note:
            q["added_for"] = note
        added.append(q)
        next_n += 1
    return existing + added, added
