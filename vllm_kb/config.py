"""统一配置入口：唯一 config.json 加载与校验。

数据源（sources）是多来源可配置的：github（REST）、markdown、excel 等。
- 每个来源有独立 id、type 与来源特有字段（extra 透传）；
- 旧版单 github 配置（顶层 "github" 段）自动折叠为等效的 sources[0]，保持向后兼容；
- 原始数据按来源分目录存储；canonical 保持统一单文件（storage.canonical_file）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def ensure_url_scheme(url: str, what: str = "url") -> str:
    """URL 缺 scheme（裸 ip:port）时补全 http://，避免 requests/urllib 报
    'no connection adapters were found' / 'unknown url type' 且请求从未发出。

    内网 vLLM/OCR 服务常见裸地址配置；https 应显式写全（本函数只补 http://）。
    """
    if url and "://" not in url:
        fixed = "http://" + url
        print(f"[warn] {what} 缺少 http(s):// 前缀，已自动补全为 {fixed}（https 请显式写全）")
        return fixed
    return url


class ProjectCfg(BaseModel):
    name: str = "vllm-kb"
    data_root: str = "data"


class GithubCfg(BaseModel):
    """旧版单 github 源的配置（仅用于向后兼容折叠，新配置请用 sources）。"""

    repo: str = "vllm-project/vllm"
    token: str = ""
    token_env: str = "GITHUB_TOKEN"
    api_base: str = "https://api.github.com"
    per_page: int = 100
    max_issues: int = 0  # 0 = 全量历史（issue + PR + 评论）
    issue_state: str = "all"  # open | closed | all
    include_prs: bool = True
    sort: str = "created"
    direction: str = "desc"
    fetch_comments: bool = True
    request_timeout_seconds: int = 30
    max_retries: int = 3
    retry_backoff_seconds: int = 5
    checkpoint_file: str = "data/checkpoints/github_issues.json"
    raw_dir: str = "data/raw/github"

    @property
    def effective_token(self) -> str:
        return self.token or os.environ.get(self.token_env, "")


class SourceCfg(BaseModel):
    """数据源配置。

    - id/type/enabled 为通用字段；其余字段按 type 由对应 Source 实现读取（extra 透传）。
    - 新增来源类型时无需改这里：配置里写 type + 来源特有字段即可。
    """

    model_config = ConfigDict(extra="allow")

    id: str
    type: str  # github | markdown | excel | ...
    enabled: bool = True

    def get(self, key: str, default=None):
        return getattr(self, key, default)


class EmbeddingCfg(BaseModel):
    provider: str = "openai_compatible"  # openai_compatible | echo
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = "EMBEDDING_API_KEY"
    model: str = "BAAI/bge-m3"
    dimensions: int = 1024
    batch_size: int = 32
    max_input_chars: int = 12000
    timeout_seconds: int = 60
    max_retries: int = 3

    @property
    def effective_api_key(self) -> str:
        return self.api_key or os.environ.get(self.api_key_env, "")


class ChunkingCfg(BaseModel):
    max_chunk_chars: int = 4000
    overlap_chars: int = 200


class StorageCfg(BaseModel):
    vector_backend: str = "lancedb"  # lancedb | python
    lancedb_path: str = "data/lancedb"
    sqlite_path: str = "data/kb.sqlite3"
    canonical_file: str = "data/raw/canonical.jsonl"  # 统一单文件，不按来源拆分
    companion_file: str = "data/compatibility/vllm-ascend.json"  # 组件配套矩阵（vllm-ascend 主键）
    release_calendar: str = ""  # Phase 1：tag->日期 JSON；空则版本置信度退化为默认值
    code_root: str = "data/code"  # 版本化代码仓：zips/{version}.zip + snapshots/{version}/ + index.sqlite3
    graph_path: str = "data/graph"  # Phase 2：Kùzu 图存储目录（scripts/build_graph.py）
    review_path: str = "data/review.sqlite3"  # Phase 2：审核工作台队列（独立于只读 kb.sqlite3）


class CodeCfg(BaseModel):
    """版本化代码仓配置（预存主要镜像版本源码快照）。"""

    repo: str = "vllm-project/vllm-ascend"  # 快照来源仓库
    # 预存版本列表：留空 = 构建脚本默认取全部 tag；填列表则只预存这些
    versions: list[str] = Field(default_factory=list)
    # zip 下载临时目录（构建时用，与存储无关）
    download_workers: int = 2


class ConfidenceCfg(BaseModel):
    alpha: float = 0.6
    beta: float = 0.4
    gamma: float = 0.6
    half_life_days: float = 365.0
    time_floor: float = 0.15
    version_sigma: float = 1.5  # 版本距离衰减，单位：小版本数
    unknown_version_weight: float = 0.5
    reliability: dict[str, float] = Field(default_factory=lambda: {
        "merged_fix": 0.9, "closed": 0.6, "open": 0.4,
        "official_doc": 0.85, "wiki": 0.7, "discussion": 0.5,
    })
    # 验证状态因子（维度 B）：expert 官方认证 / tested 已测试有效 / unverified 未验证。
    # 融合公式（V1）：w_rel = max(规则可靠度, verification_factor)——验证状态作为
    # 可靠度下限提升（如官方手册 status=open→0.4，但 verification=expert→0.95）。
    # 具体融合公式留待 Phase 5 用真实故障案例标定（可改乘法/加权）。
    verification_weights: dict[str, float] = Field(default_factory=lambda: {
        "unverified": 0.5, "tested": 0.85, "expert": 0.95,
    })


class RetrievalCfg(BaseModel):
    vector_top_k: int = 50
    fts_top_k: int = 50
    final_top_k: int = 10
    min_similarity: float = 0.0
    dedupe_by_doc: bool = True  # 同一 issue/doc 的多个 chunk 只保留最高分的一条
    default_target_version: str = ""  # 查询未传版本时的兜底；建议查询时显式传版本
    prefer_unresolved_without_resolved: bool = True  # 无强匹配已解决问题时，优先列未解决（含规避方案）
    resolved_min_similarity: float = 0.5  # 已解决问题相似度低于此值时触发未解决优先


class ApiCfg(BaseModel):
    """只读检索 API 配置。read_only 恒为 True（结构只读，不提供关闭入口）。"""

    host: str = "127.0.0.1"
    port: int = 8000


class VerifyCfg(BaseModel):
    queries: list[str] = Field(default_factory=list)


class LoggingCfg(BaseModel):
    """总日志：打屏（默认）+ 可选落盘分卷（RotatingFileHandler）。"""

    console: bool = True
    file: bool = False  # 默认不落盘；配置开启后写 file_path（按 max_bytes 分卷）
    file_path: str = "logs/vllm-kb.log"
    max_bytes: int = 10485760  # 10MB/卷
    backup_count: int = 5  # 保留 5 个历史卷


class AppConfig(BaseModel):
    project: ProjectCfg = Field(default_factory=ProjectCfg)
    github: Optional[GithubCfg] = None  # 旧版单源（向后兼容）；优先使用 sources
    sources: list[SourceCfg] = Field(default_factory=list)
    embedding: EmbeddingCfg = Field(default_factory=EmbeddingCfg)
    chunking: ChunkingCfg = Field(default_factory=ChunkingCfg)
    storage: StorageCfg = Field(default_factory=StorageCfg)
    confidence: ConfidenceCfg = Field(default_factory=ConfidenceCfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)
    api: ApiCfg = Field(default_factory=ApiCfg)
    verify: VerifyCfg = Field(default_factory=VerifyCfg)
    code: CodeCfg = Field(default_factory=CodeCfg)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)

    @classmethod
    def load(cls, path: Optional[str | os.PathLike] = None, require_keys: bool = True) -> "AppConfig":
        p = Path(path) if path else PROJECT_ROOT / "config.json"
        if not p.exists():
            raise FileNotFoundError(
                f"未找到配置文件: {p}\n请复制 config.example.json 为 config.json 并填写参数。"
            )
        data = json.loads(p.read_text(encoding="utf-8"))
        cfg = cls.model_validate(data)
        # 先注入本地 secrets（data/secrets.local.json）到环境变量，再校验——
        # 否则写入路径（require_keys=True）校验时看不到工作台配置的 key
        cfg._load_local_secrets()
        cfg.validate_runtime(require_keys=require_keys)
        cfg._warn_synced_data_paths()
        return cfg

    def _load_local_secrets(self) -> None:
        """自动加载本地密钥文件（data/secrets.local.json，审核工作台写入）。

        覆盖**所有入口**（build_kb 采集/嵌入、serve_api、review_ui、build_* 等）：
        config 一经加载即把文件中未设置的环境变量注入 os.environ，
        之后 effective_api_key / OCR 读取 / GITHUB_TOKEN 全部自动生效。
        """
        try:
            from .secrets import load_secrets

            load_secrets(self)
        except Exception:
            pass  # 密钥文件缺失/损坏不影响启动（缺 key 时各自降级）

    # 数据根目录覆盖：存算分离时，API 服务端可通过环境变量指向远程数据目录，
    # 所有 data/* 相对路径都重定向到该目录（config.json 本身仍从项目根读取）。
    @property
    def effective_data_root(self) -> Optional[Path]:
        env = os.environ.get("VLLM_KB_DATA_ROOT", "").strip()
        return Path(env) if env else None

    def resolve(self, p: str | os.PathLike) -> Path:
        """相对路径一律基于数据根解析，保证任何 cwd 下运行一致。

        存算分离：设了 VLLM_KB_DATA_ROOT 时，data/ 前缀剥掉后重定向到该目录
        （VLLM_KB_DATA_ROOT 指向 data/ 目录本身：data/lancedb -> {root}/lancedb）；
        config.json 等项目文件仍按项目根解析（绝对路径不受影响）。
        """
        path = Path(p)
        if path.is_absolute():
            return path
        root = self.effective_data_root
        if root is not None:
            parts = list(path.parts)
            if parts and parts[0] == "data":
                parts = parts[1:]
            return root.joinpath(*parts) if parts else root
        return PROJECT_ROOT / path

    def effective_sources(self) -> list[SourceCfg]:
        """生效的数据源列表：优先 sources；无则把旧版 github 单源折叠为等效 source。"""
        if self.sources:
            return self.sources
        if self.github:
            return [SourceCfg(id="vllm", type="github", **self.github.model_dump())]
        return []

    def _warn_synced_data_paths(self) -> None:
        """Windows 常见坑：数据目录位于 OneDrive/Documents 同步目录下时，
        文件可能被同步/杀软短暂锁定，导致 LanceDB 版本提示文件写入失败
        （LanceError IO: 拒绝访问, os error 5）。数据不受影响，但建议移出同步目录。"""
        candidates = [
            self.project.data_root,
            self.storage.lancedb_path,
            self.storage.sqlite_path,
            self.storage.canonical_file,
        ]
        for src in self.effective_sources():
            candidates.append(src.get("raw_dir", f"data/raw/{src.id}"))
            candidates.append(src.get("checkpoint_file", f"data/checkpoints/{src.id}.json"))
        synced = False
        for c in candidates:
            p = str(self.resolve(c)).lower()
            if "onedrive" in p:
                synced = True
            elif "documents" in p and os.environ.get("OneDrive"):
                synced = True
        if synced:
            print(
                "[warn] 数据目录疑似位于 OneDrive/Documents 同步目录下：可能引发文件锁，"
                "导致 LanceDB 报 '拒绝访问 (os error 5)' 写入失败（数据不受影响，仅提示）。"
                "建议把 storage 路径移到非同步磁盘（如 D:\\vllm-kb-data），或对数据目录添加杀软排除。"
            )

    def validate_runtime(self, require_keys: bool = True) -> None:
        """配置校验。

        require_keys=True（默认）：写操作（采集/入库）前调用，强制要求密钥；
        require_keys=False：只读检索（API/skill）启动用，密钥缺失仅警告——
        检索不依赖密钥，embedding 不可用时自动降级为全文检索。
        """
        if self.embedding.provider not in ("openai_compatible", "echo"):
            raise ValueError(f"embedding.provider 不合法: {self.embedding.provider}（支持 openai_compatible | echo）")
        if self.embedding.provider == "openai_compatible":
            if not self.embedding.base_url:
                raise ValueError("embedding.base_url 为空：OpenAI 兼容端点必填（如 https://api.siliconflow.cn/v1）")
            self.embedding.base_url = ensure_url_scheme(
                self.embedding.base_url, "embedding.base_url"
            )
            if not self.embedding.effective_api_key:
                msg = (
                    "embedding.api_key 为空且环境变量 "
                    f"{self.embedding.api_key_env} 未设置"
                )
                if require_keys:
                    raise ValueError(f"{msg}：请在 config.json 填写或设置环境变量")
                print(f"[warn] {msg}——检索将降级为全文检索（向量召回不可用）；更新数据前请先设置")
        for src in self.effective_sources():
            if src.type == "github":
                st = src.get("issue_state", "all")
                if st not in ("open", "closed", "all"):
                    raise ValueError(f"来源 {src.id}: github.issue_state 不合法: {st}")
            if src.type == "image":
                base = src.get("ocr_api_base", "")
                if base:
                    src.ocr_api_base = ensure_url_scheme(base, f"来源 {src.id} 的 ocr_api_base")
        if self.storage.vector_backend not in ("lancedb", "python"):
            raise ValueError(f"storage.vector_backend 不合法: {self.storage.vector_backend}")
        if not (0 < self.confidence.alpha + self.confidence.beta <= 1.0 + 1e-9):
            raise ValueError("confidence.alpha + beta 应约等于 1")
