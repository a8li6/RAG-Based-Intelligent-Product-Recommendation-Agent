"""The module-boundary rules from the design, enforced as tests rather than as prose."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "recommender"
WEB = {"fastapi", "streamlit", "uvicorn", "starlette"}
ENTRY_POINTS = {"api.py", "cli.py", "client.py"}


def imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module if node.level == 0 else f"recommender.{node.module}")
    return found


def top(modules: set[str]) -> set[str]:
    return {m.split(".")[0] for m in modules}


def test_core_modules_never_import_web_frameworks():
    for path in SRC.glob("*.py"):
        if path.name in ENTRY_POINTS:
            continue
        assert not (top(imports(path)) & WEB), f"{path.name} imports a web framework"


def test_only_the_api_module_imports_fastapi():
    for path in SRC.glob("*.py"):
        uses = "fastapi" in top(imports(path)) or "uvicorn" in top(imports(path))
        assert uses == (path.name == "api.py"), path.name


def test_the_ui_reaches_the_core_only_through_the_http_client_and_contracts():
    allowed = {"recommender.client", "recommender.schemas"}
    used = {m for m in imports(ROOT / "ui" / "app.py") if m.startswith("recommender")}
    assert used <= allowed | {"recommender"}, used - allowed


def test_the_ui_never_imports_retrieval_or_storage_modules():
    forbidden = {
        "service",
        "index",
        "retrieval",
        "catalog",
        "embeddings",
        "local_embedder",
        "llm",
        "grounding",
        "ranking",
        "query",
    }
    used = {
        m.split(".")[1] for m in imports(ROOT / "ui" / "app.py") if m.startswith("recommender.")
    }
    assert not (used & forbidden), used & forbidden


def test_api_and_cli_call_the_same_service_function():
    for name in ("api.py", "cli.py"):
        tree = ast.parse((SRC / name).read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "recommender.service"
            for alias in node.names
        }
        assert "recommend" in imported, name


def test_importing_a_module_never_downloads_or_loads_a_model():
    # Heavy optional dependencies are imported lazily, inside functions only.
    for path in SRC.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # module level only
            if isinstance(node, ast.Import | ast.ImportFrom):
                names = {a.name for a in node.names} | {getattr(node, "module", "") or ""}
                assert "sentence_transformers" not in " ".join(names), path.name


def test_no_module_reads_the_environment_except_settings():
    for path in SRC.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if path.name in {"config.py"}:
            continue
        assert "os.environ" not in text and "os.getenv" not in text, path.name


def test_no_unsafe_deserialization_or_dynamic_execution_in_the_package():
    for path in SRC.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import pickle" not in text and "pickle.load" not in text, path.name
        assert "allow_pickle=True" not in text, path.name
        tree = ast.parse(text)
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert not (called & {"eval", "exec", "compile", "__import__"}), path.name
