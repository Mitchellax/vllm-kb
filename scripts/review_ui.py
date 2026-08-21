"""审核工作台（轻量级 Web UI，FastAPI 单页，无构建步骤）。

功能：
1. **审核队列**：6 类人工确认项（verification_pending / case_title_flag / ocr_mismatch /
   low_confidence_ocr / equivalence_candidate / table_join_candidate）统一入口；
   详情页可预览原图（assets 静态服务）+ 提交标注（确认/拒绝/修改 + 备注）；
2. **API 配置中心**：集中展示 embedding / OCR / GitHub 等 API 配置状态（key 脱敏），
   embedding 支持连通性测试。

用法：
    pip install fastapi uvicorn
    python scripts/review_ui.py [--port 8010] [--no-seed]   # 启动（默认自动 seed 补单，幂等）
    python scripts/review_ui.py --seed-only                 # 只补单不启动

与只读检索 API（serve_api.py）分离端口；审核库 data/review.sqlite3 独立，检索 API 不碰。
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_kb.config import AppConfig
from vllm_kb.review import ReviewStore, api_configs, default_review_path, seed_all

_HTML = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>vllm-kb 审核工作台</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#f5f6f8;color:#222}
header{background:#1f2937;color:#fff;padding:10px 20px;display:flex;gap:24px;align-items:center}
header a{color:#9ca3af;text-decoration:none;cursor:pointer;font-size:14px}
header a.active{color:#fff;font-weight:600}
main{padding:20px;max-width:1100px;margin:0 auto}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:14px;margin-bottom:14px}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;margin-right:6px}
.badge.pending{background:#fef3c7;color:#92400e}.badge.approved{background:#d1fae5;color:#065f46}
.badge.suspected{background:#ede9fe;color:#5b21b6}.badge.deleted{background:#fee2e2;color:#991b1b}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #f0f0f0}
button{background:#2563eb;color:#fff;border:0;border-radius:6px;padding:6px 14px;cursor:pointer;font-size:13px}
button.sec{background:#6b7280}button.ok{background:#059669}button.danger{background:#dc2626}
input,select,textarea{padding:6px;border:1px solid #d1d5db;border-radius:6px;font-size:13px}
pre{background:#111827;color:#e5e7eb;padding:10px;border-radius:6px;overflow:auto;font-size:12px}
.mono{font-family:ui-monospace,monospace;font-size:12px}
.item{display:flex;justify-content:space-between;gap:10px;cursor:pointer;padding:8px;border-radius:6px}
.item:hover{background:#f3f4f6}
.muted{color:#6b7280;font-size:12px}
</style></head><body>
<header><strong>vllm-kb 审核工作台</strong>
<a id="nav-stats" class="active" onclick="tab('stats')">概览</a>
<a id="nav-queue" onclick="tab('queue')">审核队列</a>
<a id="nav-docs" onclick="tab('docs')">文档管理</a>
<a id="nav-configs" onclick="tab('configs')">API 配置</a></header>
<div id="toast" style="position:fixed;top:12px;right:20px;background:#111827;color:#fff;padding:8px 14px;border-radius:6px;display:none;z-index:99;font-size:13px"></div>
<main>
<div id="view-stats"></div><div id="view-queue" style="display:none"></div><div id="view-docs" style="display:none"></div><div id="view-configs" style="display:none"></div>
</main>
<script>
const $=s=>document.querySelector(s);
async function j(url,opt){const r=await fetch(url,opt);const d=await r.json();if(!r.ok)throw new Error(d.detail||r.status);return d}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function badge(s){return `<span class="badge ${s}">${s}</span>`}
function tab(name){['stats','queue','docs','configs'].forEach(t=>{document.getElementById('view-'+t).style.display=t===name?'':'none';document.getElementById('nav-'+t).className=t===name?'active':''});if(name==='stats')loadStats();if(name==='queue')loadQueue();if(name==='docs')loadDocs();if(name==='configs')loadConfigs()}
async function loadStats(){const s=await j('/api/stats');const cats=Object.entries(s).sort((a,b)=>b[1].pending-a[1].pending);
$('#view-stats').innerHTML=`<h3>待办概览（${cats.reduce((a,[k,v])=>a+v.pending,0)} 待办 / ${cats.reduce((a,[k,v])=>a+v.suspected,0)} 存疑）</h3>`+cats.map(([k,v])=>`<div class="card"><b>${esc(k)}</b> <span class="badge pending">待办 ${v.pending}</span>${v.suspected?`<span class="badge suspected">存疑 ${v.suspected}</span>`:''} <span class="muted">共 ${v.total}</span></div>`).join('')||'<div class="card">暂无审核项（导入文档后自动补单）</div>'}
async function loadQueue(cat=''){const q=await j('/api/queue'+(cat?'?category='+cat:''));
const del=await j('/api/queue?status=deleted&limit=100');
const opts=(await (async()=>{const r=await fetch('/api/categories');return r.ok?await r.json():[]})());
$('#view-queue').innerHTML=`<h3>审核队列</h3><div class="card"><label>类别: </label><select id="qcat" onchange="loadQueue(this.value)"><option value="">全部</option>${opts.map(c=>`<option ${c===cat?'selected':''}>${esc(c)}</option>`).join('')}</select> <span class="muted">共 ${q.length} 条（未审核在前，存疑在后）</span></div>`+q.map(i=>`<div class="item" onclick="showItem(${i.id})"><div><span class="badge ${i.status}">${i.status}</span> <b>${esc(i.category)}</b> ${esc((i.payload||{}).title||i.item_ref)}</div><div class="muted">#${i.id} · ${(i.created_at||'').slice(0,16)}</div></div>`).join('')||'<div class="card muted">队列为空</div>'
+`<h3 style="margin-top:26px">待实际删除（${del.length}）— 数据库记录已删，原始文件请手动本地删除</h3>`+del.map(i=>`<div class="card" style="border-left:3px solid #dc2626"><div><b>${esc((i.payload||{}).title||i.item_ref)}</b> <span class="mono muted">${esc((i.payload||{}).source_id||i.item_ref)}</span></div>
<div class="muted mono">${esc((i.payload||{}).asset&&(i.payload.asset.path)||'')}</div>
<button class="ok" onclick="undoDelete(${i.id})">↩ 撤回（恢复记录并重新入队）</button> <span class="muted">删除人: ${esc(i.reviewer||'')} ${(i.reviewed_at||'').slice(0,16)}</span></div>`).join('')||''}
async function showItem(id){const i=await j('/api/item/'+id);const p=i.payload||{};
$('#view-queue').innerHTML=`<div class="card"><button onclick="loadQueue()">← 返回队列</button><h3>#${i.id} ${esc(i.category)} <span class="badge ${i.status}">${i.status}</span></h3>
<p><b>item_ref:</b> <span class="mono">${esc(i.item_ref)}</span></p>
<pre>${esc(JSON.stringify(p,null,2))}</pre>
<h4>审核（只做判定，不修改原始内容）</h4><input id="rev" placeholder="审核人（必填）" style="width:180px">
<textarea id="rvnote" rows="2" style="width:100%" placeholder="备注（可选）"></textarea><br><br>
<button class="ok" onclick="review(${i.id},'approved')">✓ 认证</button>
<button onclick="review(${i.id},'suspected')">？ 存疑（重新排队靠后）</button>
<button class="danger" onclick="deleteDoc(${i.id})">🗑 标记删除（只删数据库记录，原始文件手动删）</button>
<span class="muted">存疑项排在未审核之后；删除可在队列底部'待实际删除'撤回</span></div>`}
async function review(id,action){const reviewer=document.getElementById('rev').value;if(!reviewer){alert('请填写审核人');return}let result={note:document.getElementById('rvnote').value};try{result=JSON.parse(result.note||'{}')}catch(e){}
await j('/api/item/'+id+'/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,reviewer,result})});loadQueue()}
async function deleteDoc(id){const reviewer=document.getElementById('rev').value;if(!reviewer){alert('请填写审核人');return}
if(!confirm('标记删除：只删除数据库记录，原始资产文件需到"待实际删除"列表手动本地删除。确认？'))return
await j('/api/item/'+id+'/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reviewer,note:document.getElementById('rvnote').value})});loadQueue()}
async function undoDelete(id){if(!confirm('撤回删除：恢复数据库记录并重新加入队列？'))return;await j('/api/item/'+id+'/undo',{method:'POST'});loadQueue()}
async function loadDocs(){const ds=await j('/api/external-docs?limit=500');
$('#view-docs').innerHTML=`<h3>文档管理 — 外源文档（导入的 PDF/MD/表格等，共 ${ds.length} 条）</h3>
<div class="card muted">删除只动数据库（docs + chunks + 向量），<b>本地文件不动</b>；下次增量入库时文件仍在本地会重新入库，文档废弃请手动删除本地文件。GitHub 采集文档不在此列。</div>`+ds.map(d=>`<div class="card" style="border-left:3px solid #d1d5db"><div><b>${esc(d.title||d.source_id)}</b> <span class="badge">${esc(d.source_type)}</span></div>
<div class="mono muted">${esc(d.source_id)}</div>
<div class="muted">${esc(d.asset||'')}${d.verification?` · 验证=${esc(d.verification)}`:''}</div>
<button class="danger" onclick="delExternalDoc('${esc(d.source_id).replace(/'/g,"\\'")}')">🗑 从数据库删除（本地文件保留）</button></div>`).join('')||'<div class="card muted">无外源文档</div>'}
async function delExternalDoc(sid){if(!confirm('从数据库彻底删除该文档（docs+chunks+向量）？本地文件保留，下次增量入库会重新入库。确认？'))return
try{await j('/api/external-docs/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source_id:sid})});loadDocs();toast('✓ 已删除（本地文件保留）')}
catch(e){toast('✗ 删除失败: '+e.message)}}
async function loadConfigs(){const cs=await j('/api/configs');fillCache(cs);
$('#view-configs').innerHTML=`<h3>API 配置中心（key 脱敏；可编辑）</h3>`+cs.map(c=>`<div class="card" id="cfg-${esc(c.name)}"><b>${esc(c.name)}</b> <span class="badge ${c.status==='configured'?'confirmed':'pending'}">${c.status}</span>
<table><tr><th>provider</th><td>${esc(c.provider)}</td></tr>${c.mode&&c.mode!=='custom'?`<tr><th>mode</th><td>${esc(c.mode)}</td></tr>`:''}${c.model?`<tr><th>model</th><td class="mono">${esc(c.model)}</td></tr>`:''}<tr><th>base_url</th><td class="mono">${esc(c.base_url)}</td></tr><tr><th>key</th><td>${c.key_configured?'已配置':'未配置'}</td></tr><tr><th>说明</th><td class="muted">${esc(c.note||'')}</td></tr></table>
<div><button onclick="editConfig('${esc(c.name)}')">编辑配置</button> ${c.name==='embedding'?`<button onclick="testApi('embedding','etest')">测试连通</button><span id="etest" class="muted"></span>`:''}${c.name==='ocr'?`<button onclick="testApi('ocr','otest')">测试连通</button><span id="otest" class="muted"></span>`:''}</div></div>`).join('')}
const CFG_FIELDS={
 embedding:[['provider','select',['openai_compatible','echo']],['base_url','text'],['model','text'],['api_key','password']],
 ocr:[['ocr_provider','select',['ask','api','paddle','none']],['ocr_api_mode','select',['custom','openai']],['ocr_api_base','text'],['ocr_api_model','text'],['ocr_api_key','password']],
 github:[['token','password']]};
function fillCache(cs){window._cfgCache={};cs.forEach(c=>window._cfgCache[c.name]={provider:c.provider,base_url:c.base_url,model:c.model||'',mode:c.mode||'custom',ocr_provider:c.provider,ocr_api_base:c.base_url,ocr_api_model:c.model||'',ocr_api_mode:c.mode||'custom'})}
async function editConfig(name){const old=document.getElementById('form-'+name);if(old)old.remove();
try{fillCache(await j('/api/configs'))}catch(e){} // 打开时拉最新配置，避免缓存/时序回填旧值
const c=CFG_FIELDS[name];const cur=window._cfgCache[name]||{};const html=c.map(([k,t,opts])=>`<div style="margin:4px 0"><label>${esc(k)}: </label>${t==='select'?`<select id="f-${k}">${opts.map(o=>`<option ${(cur[k]||'')===o?'selected':''}>${esc(o)}</option>`).join('')}</select>`:`<input id="f-${k}" type="${t}" placeholder="${t==='password'?'(留空不改)':'...'}" style="width:340px" ${cur[k]&&t!=='password'?`value="${esc(cur[k])}"`:''}>`}</div>`).join('');
document.getElementById('cfg-'+name).insertAdjacentHTML('beforeend',`<div class="card" style="background:#f9fafb" id="form-${name}">${html}<br><button class="ok" onclick="saveConfig('${name}')">保存</button> <button onclick="document.getElementById('form-${name}').remove()">取消</button></div>`)}
async function saveConfig(name){const fields={};CFG_FIELDS[name].forEach(([k])=>{const el=document.getElementById('f-'+k);if(el)fields[k]=el.value});
try{await j('/api/configs/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,fields})});
document.getElementById('form-'+name).remove();await loadConfigs();toast('✓ 已保存（状态已刷新）')}
catch(e){toast('✗ 保存失败: '+e.message)}}
function toast(msg){const t=document.getElementById('toast');t.textContent=msg;t.style.display='block';clearTimeout(t._h);t._h=setTimeout(()=>t.style.display='none',2500)}
async function testApi(name,elId){const el=document.getElementById(elId);el.textContent='测试中…';try{const r=await j('/api/configs/test',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});el.textContent=' ✓ '+r.detail}catch(e){el.textContent=' ✗ '+e.message}}
loadStats();
</script></body></html>
"""


def create_app(config_path: Optional[str] = None, auto_seed: bool = True):
    from fastapi import FastAPI, HTTPException
    from fastapi.staticfiles import StaticFiles

    # AppConfig.load 已自动加载 data/secrets.local.json（密钥注入环境变量）
    cfg = AppConfig.load(config_path, require_keys=False)
    _config_path = str(Path(config_path).resolve()) if config_path else None
    store = ReviewStore(default_review_path(cfg))
    if auto_seed:
        try:
            added = seed_all(cfg, store)
            if any(added.values()):
                print(f"[review] 自动补单：{added}")
        except Exception as e:
            print(f"[review] 自动补单失败（不影响工作台）: {e}")

    app = FastAPI(title="vllm-kb 审核工作台", version="0.1.0",
                  description="人工确认统一入口 + API 配置中心。审核库 data/review.sqlite3 独立于只读检索库。")

    # 原图预览（assets 静态服务；不存在时跳过挂载）
    assets_root = cfg.resolve("data/assets")
    if assets_root.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_root)), name="assets")

    @app.get("/", response_class=None)
    def index():
        from fastapi.responses import HTMLResponse

        return HTMLResponse(_HTML)

    @app.get("/api/categories")
    def categories():
        from vllm_kb.review import CATEGORIES
        return CATEGORIES

    @app.get("/api/external-docs")
    def external_docs(limit: int = 200, offset: int = 0):
        """外源文档列表（导入的 PDF/MD/表格等，排除 GitHub 采集来源）。"""
        from vllm_kb.review import list_external_docs
        return list_external_docs(cfg.resolve(cfg.storage.sqlite_path),
                                  limit=min(limit, 500), offset=offset)

    @app.post("/api/external-docs/delete")
    def external_doc_delete(body: dict):
        """彻底删除外源文档（数据库四层全删：docs + chunks_fts + chunks_meta + 向量），
        本地资产文件不动；下次增量入库时文件仍在本地则重新入库。"""
        from vllm_kb.review import delete_external_doc
        from vllm_kb.vectorstore import build_vector_store

        source_id = (body.get("source_id") or "").strip()
        if not source_id:
            raise HTTPException(400, "source_id 必填")
        kb_path = cfg.resolve(cfg.storage.sqlite_path)
        # 向量清理用可写 store（审核工作台本身可写；检索 API 才强制只读）
        try:
            vector_store = build_vector_store(cfg)
        except Exception:
            vector_store = None
        try:
            r = delete_external_doc(kb_path, source_id, vector_store=vector_store)
        except ValueError as e:
            raise HTTPException(404, str(e))
        return {"ok": True, "note": f"已从数据库删除 {source_id}（{r['chunks_deleted']} 个 chunk，本地文件保留）"}

    @app.get("/api/stats")
    def stats():
        return store.stats()

    @app.get("/api/queue")
    def queue(category: Optional[str] = None, status: Optional[str] = None,
              limit: int = 50, offset: int = 0):
        return store.list_items(category=category, status=status, limit=min(limit, 200), offset=offset)

    @app.get("/api/item/{item_id}")
    def item(item_id: int):
        it = store.get_item(item_id)
        if it is None:
            raise HTTPException(404, "审核项不存在")
        return it

    @app.post("/api/item/{item_id}/review")
    def review(item_id: int, body: dict):
        """标注（不改原始内容）：approved 认证 | suspected 存疑（重新排队靠后）。"""
        action = body.get("action", "")
        reviewer = (body.get("reviewer") or "").strip()
        if not reviewer:
            raise HTTPException(400, "reviewer 必填")
        try:
            ok = store.review(item_id, action, reviewer, result=body.get("result"))
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not ok:
            raise HTTPException(404, "审核项不存在")
        return {"ok": True}

    @app.post("/api/item/{item_id}/delete")
    def delete_doc(item_id: int, body: dict):
        """标记删除：只删 kb.sqlite3 数据库记录，原始资产保留（人工到待删除列表本地删）。"""
        reviewer = (body.get("reviewer") or "").strip()
        if not reviewer:
            raise HTTPException(400, "reviewer 必填")
        try:
            ok = store.mark_deleted(item_id, reviewer, cfg.resolve(cfg.storage.sqlite_path),
                                    note=(body.get("note") or ""))
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not ok:
            raise HTTPException(404, "审核项不存在")
        return {"ok": True, "note": "已删除数据库记录；原始文件请到'待实际删除'列表手动本地删除"}

    @app.post("/api/item/{item_id}/undo")
    def undo_delete(item_id: int):
        """撤回删除：恢复 kb.sqlite3 记录，审核项回到 pending 重新入队。"""
        try:
            ok = store.undo_delete(item_id, cfg.resolve(cfg.storage.sqlite_path))
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not ok:
            raise HTTPException(404, "审核项不存在")
        return {"ok": True, "note": "已恢复数据库记录并重新加入队列"}

    @app.get("/api/configs")
    def configs():
        return api_configs(cfg)

    @app.post("/api/configs/save")
    def save_config(body: dict):
        """保存 API 配置：非密钥字段写 config.json，密钥写 data/secrets.local.json。

        保存后**重载内存 cfg**（AppConfig.load 会重读 config.json + secrets 注入环境变量），
        使后续 /api/configs 与连通性测试立即反映新状态（无需重启即可看到状态变化；
        非密钥字段对已运行的 serve_api 仍需重启生效——那是服务端读取时机问题）。
        """
        nonlocal cfg
        from vllm_kb.review import update_config_json
        from vllm_kb.secrets import save_secret

        name = body.get("name", "")
        fields = body.get("fields") or {}
        if name == "embedding":
            update_config_json(cfg, "embedding",
                               {k: fields[k] for k in ("provider", "base_url", "model")
                                if k in fields}, config_path=_config_path)
            if "api_key" in fields:
                save_secret(cfg, "EMBEDDING_API_KEY", fields.get("api_key") or "")
        elif name == "ocr":
            update_config_json(cfg, "ocr",
                               {k: fields[k] for k in ("ocr_provider", "ocr_api_base",
                                                       "ocr_api_model", "ocr_api_mode")
                                if k in fields},
                               config_path=_config_path)
            if "ocr_api_key" in fields:
                save_secret(cfg, "OCR_API_KEY", fields.get("ocr_api_key") or "")
        elif name == "github":
            if "token" in fields:
                save_secret(cfg, "GITHUB_TOKEN", fields.get("token") or "")
        else:
            raise HTTPException(400, f"未知配置: {name}")
        # 重载内存 cfg（读 config.json + secrets），/api/configs 立即反映
        cfg = AppConfig.load(_config_path, require_keys=False) if _config_path \
            else AppConfig.load(None, require_keys=False)
        return {"ok": True, "note": "已保存。密钥经 secrets 文件；非密钥字段对已运行的 serve_api 需重启生效"}

    @app.post("/api/configs/test")
    def test_config(body: dict):
        """连通性测试：embedding 发真实嵌入请求；OCR 用内置测试图走真实识别链路。"""
        from vllm_kb.review import test_ocr_connectivity

        name = body.get("name", "")
        if name == "embedding":
            try:
                from vllm_kb.embed import EmbeddingClient
                from vllm_kb.config import EmbeddingCfg
                # 兼容：EmbeddingClient 可能按 cfg.embedding 构造
                client = EmbeddingClient(cfg.embedding)
                vecs = client.embed_texts(["ping"])
                if not vecs or not vecs[0]:
                    raise HTTPException(502, "embedding 返回空向量")
                return {"ok": True, "detail": f"连通（dim={len(vecs[0])}）"}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(502, f"embedding API 不可用: {e}")
        if name == "ocr":
            try:
                r = test_ocr_connectivity(cfg)
            except ValueError as e:
                raise HTTPException(400, str(e))
            if not r["ok"]:
                raise HTTPException(502, r["detail"])
            return r
        raise HTTPException(400, "仅支持 embedding / ocr 连通性测试")

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="vllm-kb 审核工作台（队列 + API 配置中心）")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-seed", action="store_true", help="启动时不自动补单")
    ap.add_argument("--seed-only", action="store_true", help="只补单不启动服务")
    args = ap.parse_args()

    cfg = AppConfig.load(args.config, require_keys=False)
    store = ReviewStore(default_review_path(cfg))
    added = seed_all(cfg, store)
    print(f"[review] 补单：{added}")
    if args.seed_only:
        return
    import uvicorn
    from vllm_kb.logging_setup import setup_logging

    setup_logging(cfg, log_name="review_ui")  # 总日志：打屏 + 可选分卷落盘

    app = create_app(args.config, auto_seed=not args.no_seed)
    print(f"[review] 审核工作台 http://{args.host}:{args.port}（审核库 {default_review_path(cfg)}）")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
