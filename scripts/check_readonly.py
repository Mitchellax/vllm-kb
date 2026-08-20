"""验证知识库只读姿态（结构层面，不依赖提示词）。

检查项：
1. SQLite 只读连接拒绝写入（URI mode=ro，INSERT 必然失败）；
2. 向量库只读包装拒绝写入（ReadOnlyVectorStore）；
3. API 源码审计：无写操作调用、不导入可写模块、SQLite 连接均使用 mode=ro。

用法：python scripts/check_readonly.py
"""
import ast
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_kb.config import AppConfig  # noqa: E402
from vllm_kb.vectorstore import ReadOnlyVectorStore  # noqa: E402

# API 源码中禁止出现的可写操作名（AST 调用审计）
_WRITE_CALLS = {
    "open", "write", "write_text", "write_bytes", "unlink", "rename", "mkdir",
    "add_items", "delete_doc", "update_doc_meta", "clear", "pull",
    "subprocess", "Popen", "system", "check_output", "run",
}
# API 禁止导入的（可写）模块
_WRITE_MODULES = {"ingest", "github_pull", "pipeline", "sources"}


def check_sqlite_readonly(sqlite_path: Path) -> list[str]:
    problems = []
    try:
        conn = sqlite3.connect(f"file:{sqlite_path.as_posix()}?mode=ro", uri=True)
    except sqlite3.OperationalError as e:
        problems.append(f"SQLite 只读连接失败（库可能不存在）: {e}")
        return problems
    try:
        conn.execute("INSERT INTO docs (source_id, source_type) VALUES ('x','y')")
        problems.append("SQLite mode=ro 竟然允许写入！")
    except sqlite3.OperationalError:
        pass  # 预期：只读连接拒绝写入
    finally:
        conn.close()
    return problems


def check_vector_store_readonly() -> list[str]:
    problems = []
    store = ReadOnlyVectorStore.__new__(ReadOnlyVectorStore)
    store._store = None
    try:
        store.add_items([])
        problems.append("ReadOnlyVectorStore.add_items 未抛错！")
    except Exception as e:
        if not isinstance(e, Exception) or "只读" not in str(e):
            problems.append(f"ReadOnlyVectorStore 抛错类型异常: {type(e).__name__}")
    return problems


def check_api_source_audit(api_path: Path) -> list[str]:
    problems = []
    source = api_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in _WRITE_CALLS:
                problems.append(f"api.py 出现可写调用: {name}（第 {node.lineno} 行）")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                mod = (alias.name or "").split(".")[0]
                if mod in _WRITE_MODULES:
                    problems.append(f"api.py 导入了可写模块: {mod}（第 {node.lineno} 行）")
    # 所有 sqlite3.connect 必须带 mode=ro
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == "connect")
        ):
            src = ast.get_source_segment(source, node) or ""
            if "mode=ro" not in src:
                problems.append(f"api.py 存在未使用 mode=ro 的 sqlite 连接（第 {node.lineno} 行）")
    return problems


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    cfg = AppConfig.load(root / "config.json")
    sqlite_path = cfg.resolve(cfg.storage.sqlite_path)
    api_path = root / "vllm_kb" / "api.py"

    all_problems: list[str] = []
    all_problems += check_sqlite_readonly(sqlite_path)
    all_problems += check_vector_store_readonly()
    all_problems += check_api_source_audit(api_path)

    print(f"[readonly] SQLite 路径: {sqlite_path}")
    print(f"[readonly] 向量库后端: {cfg.storage.vector_backend}")
    if all_problems:
        print(f"[readonly] [!] 发现 {len(all_problems)} 个问题：")
        for p in all_problems:
            print(f"    - {p}")
        sys.exit(1)
    print("[readonly] OK：SQLite 拒绝写入 / 向量库写操作抛错 / API 源码无写代码，只读姿态成立")


if __name__ == "__main__":
    main()
