"""向量存储：统一接口 + 两个后端。

- lancedb：正式后端（需 pip install lancedb），本地嵌入式列式向量库；
- python：纯标准库兜底（离线/未安装 lancedb 时），JSON 持久化，暴力余弦检索。
  —— 语义检索能力完整，仅性能与规模有限（万级以内可用），用于开发与演示。

任选后端的检索结果都返回 SearchHit(id, score, meta, text)。
"""
from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .config import AppConfig


@dataclass
class VectorItem:
    id: str
    vector: list[float]
    meta: dict[str, Any] = field(default_factory=dict)
    text: str = ""


@dataclass
class SearchHit:
    id: str
    score: float  # 相似度 0..1（越大越相关）
    meta: dict[str, Any] = field(default_factory=dict)
    text: str = ""


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


class BaseVectorStore(ABC):
    @abstractmethod
    def add_items(self, items: list[VectorItem]) -> None: ...

    @abstractmethod
    def delete_doc(self, doc_id: str) -> None: ...

    def delete_docs(self, doc_ids: list[str]) -> None:
        """批量删除（默认逐条，子类可覆盖为一条 IN 语句，LanceDB 单条删除开销极大）。"""
        for did in doc_ids:
            self.delete_doc(did)

    @abstractmethod
    def update_doc_meta(self, doc_id: str, meta: dict[str, Any]) -> None:
        """不重新嵌入，只刷新某文档所有 chunk 的元数据（增量入库的 meta_refresh 用）。"""

    @abstractmethod
    def search(self, vector: list[float], top_k: int = 10) -> list[SearchHit]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def clear(self) -> None: ...


class PythonVectorStore(BaseVectorStore):
    """纯标准库兜底：内存 + JSON 持久化。"""

    def __init__(self, path: Path):
        self.path = path
        self._items: dict[str, VectorItem] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for rec in data:
                self._items[rec["id"]] = VectorItem(
                    id=rec["id"], vector=rec["vector"], meta=rec.get("meta", {}), text=rec.get("text", "")
                )

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {"id": it.id, "vector": it.vector, "meta": it.meta, "text": it.text}
            for it in self._items.values()
        ]
        self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def add_items(self, items: list[VectorItem]) -> None:
        for it in items:
            self._items[it.id] = it
        self._save()

    def delete_doc(self, doc_id: str) -> None:
        before = len(self._items)
        self._items = {k: v for k, v in self._items.items() if v.meta.get("doc_id") != doc_id}
        if len(self._items) != before:
            self._save()

    def update_doc_meta(self, doc_id: str, meta: dict[str, Any]) -> None:
        changed = False
        for it in self._items.values():
            if it.meta.get("doc_id") == doc_id:
                it.meta = dict(meta)
                changed = True
        if changed:
            self._save()

    def search(self, vector: list[float], top_k: int = 10) -> list[SearchHit]:
        scored = [(it.id, _cosine(vector, it.vector), it) for it in self._items.values()]
        scored.sort(key=lambda t: t[1], reverse=True)
        return [
            SearchHit(id=it.id, score=max(0.0, s), meta=it.meta, text=it.text)
            for _, s, it in scored[:top_k]
        ]

    def count(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items = {}
        if self.path.exists():
            self.path.unlink()


class LanceDBVectorStore(BaseVectorStore):
    """正式后端（lancedb）。表结构：id, doc_id, text, meta_json, vector。"""

    def __init__(self, path: Path, table_name: str = "chunks"):
        import lancedb  # 延迟导入：未安装时由 build_vector_store 回退

        self._lancedb = lancedb
        self.path = path
        self.table_name = table_name
        self.db = lancedb.connect(str(path))
        self._table = None
        if table_name in self.db.table_names():
            self._table = self.db.open_table(table_name)

    def _ensure_table(self, sample: VectorItem) -> None:
        if self._table is None:
            self._table = self.db.create_table(
                self.table_name, data=[self._row(sample)], mode="overwrite"
            )

    @staticmethod
    def _row(it: VectorItem) -> dict[str, Any]:
        return {
            "id": it.id,
            "doc_id": it.meta.get("doc_id", ""),
            "text": it.text,
            "meta_json": json.dumps(it.meta, ensure_ascii=False),
            "vector": it.vector,
        }

    def add_items(self, items: list[VectorItem]) -> None:
        if not items:
            return
        if self._table is None:
            self._ensure_table(items[0])
            rest = items[1:]
        else:
            rest = items
        if rest:
            self._table.add([self._row(it) for it in rest])

    def delete_doc(self, doc_id: str) -> None:
        if self._table is None:
            return
        try:
            self._table.delete(f"doc_id = '{doc_id.replace(chr(39), chr(39)*2)}'")
        except Exception:
            pass  # 无匹配行时部分版本抛错，忽略

    def delete_docs(self, doc_ids: list[str]) -> None:
        """批量删除：单条 delete 每次都要 commit 版本文件（大表下单条约 0.04s+ 且随表增长），
        合并为一条 IN 语句可提速 30~40 倍。"""
        if self._table is None or not doc_ids:
            return
        esc = [d.replace(chr(39), chr(39) * 2) for d in doc_ids]
        # 分块避免 IN 列表过长；每块仍是一条语句
        chunk = 200
        for i in range(0, len(esc), chunk):
            part = esc[i : i + chunk]
            quoted = ",".join(f"'{d}'" for d in part)
            try:
                self._table.delete(f"doc_id IN ({quoted})")
            except Exception:
                pass  # 无匹配行时部分版本抛错，忽略

    def update_doc_meta(self, doc_id: str, meta: dict[str, Any]) -> None:
        if self._table is None:
            return
        esc = doc_id.replace(chr(39), chr(39) * 2)
        self._table.update(
            where=f"doc_id = '{esc}'",
            values={"meta_json": json.dumps(meta, ensure_ascii=False)},
        )

    def search(self, vector: list[float], top_k: int = 10) -> list[SearchHit]:
        if self._table is None:
            return []
        q = (
            self._table.search(vector)
            .metric("cosine")
            .limit(top_k)
            .select(["id", "doc_id", "text", "meta_json", "_distance"])
        )
        try:
            # 显式声明 _distance 由我们自己选择，避免新版自动附带/未来移除的行为差异
            q = q.disable_scoring_autoprojection()
        except AttributeError:
            pass  # 旧版无此方法，保持默认（_distance 已显式 select，两种行为下都可用）
        rows = q.to_list()
        hits = []
        for row in rows:
            dist = float(row.get("_distance", 1.0))
            sim = max(0.0, 1.0 - dist)
            try:
                meta = json.loads(row.get("meta_json", "{}"))
            except (TypeError, ValueError):
                meta = {}
            hits.append(SearchHit(id=row["id"], score=sim, meta=meta, text=row.get("text", "")))
        return hits

    def count(self) -> int:
        return self._table.count_rows() if self._table is not None else 0

    def clear(self) -> None:
        if self._table is not None:
            self.db.drop_table(self.table_name)
            self._table = None


class ReadOnlyError(RuntimeError):
    """只读模式下尝试写操作。"""


class ReadOnlyVectorStore(BaseVectorStore):
    """只读包装：搜索/计数委托给底层 store，一切写操作抛 ReadOnlyError。

    用于检索 API —— 结构上杜绝 agent 通过 API 修改向量库
    （即使提示注入指示"修改"，写入路径也会硬失败）。
    """

    def __init__(self, store: BaseVectorStore):
        self._store = store

    def add_items(self, items: list[VectorItem]) -> None:
        raise ReadOnlyError("知识库为只读：禁止写入向量库")

    def delete_doc(self, doc_id: str) -> None:
        raise ReadOnlyError("知识库为只读：禁止删除向量")

    def update_doc_meta(self, doc_id: str, meta: dict[str, Any]) -> None:
        raise ReadOnlyError("知识库为只读：禁止修改向量元数据")

    def clear(self) -> None:
        raise ReadOnlyError("知识库为只读：禁止清空")

    def search(self, vector: list[float], top_k: int = 10) -> list[SearchHit]:
        return self._store.search(vector, top_k)

    def count(self) -> int:
        return self._store.count()


def build_vector_store(cfg: AppConfig) -> BaseVectorStore:
    """按配置构建向量存储；lancedb 未安装时自动回退纯 Python 后端并告警。"""
    sc = cfg.storage
    if sc.vector_backend == "python":
        return PythonVectorStore(cfg.resolve(sc.lancedb_path + "_py.json"))
    try:
        return LanceDBVectorStore(cfg.resolve(sc.lancedb_path))
    except ImportError:
        print("[warn] 未安装 lancedb，回退纯 Python 向量后端（pip install lancedb 可启用正式后端）")
        return PythonVectorStore(cfg.resolve(sc.lancedb_path + "_py.json"))
