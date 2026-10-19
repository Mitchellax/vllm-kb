"""审核工作台（轻量级 Web UI，FastAPI 单页，无构建步骤）。

功能：
1. **审核队列**：6 类人工确认项（verification_pending / case_title_flag / ocr_mismatch /
   low_confidence_ocr / equivalence_candidate / table_join_candidate）统一入口；
   详情页可预览原图（assets 静态服务）+ 补标项正文预览（/api/doc-preview）+ 提交标注
   （确认/拒绝/修改 + 备注）；**表格项按文件粒度**（一次认证整表生效，内容变更自动重审）；
2. **API 配置中心**：集中展示 embedding / OCR / GitHub 等 API 配置状态（key 脱敏），
   embedding 支持连通性测试。

样式对齐 DSH 默认设计语言（近黑主色/三级标签灰/10% 细边框/12px 圆角/暗色跟随系统）。

TODO（暂缓，人工成本优化）：标签打标与 tag_candidate 候选处理的人工成本偏高——
候选按词聚合已做，仍需逐条人工判断。待设计更优流程（规则化自动采纳、批量打标、
按主题聚合视图）后再动，见「标签管理」「文档管理」页内 TODO 横幅与
docs/analysis/business-migration-checklist.md。

用法：
    pip install fastapi uvicorn
    python scripts/review_ui.py [--port 8010] [--no-seed]   # 启动（默认自动 seed 补单，幂等）
    python scripts/review_ui.py --seed-only                 # 只补单不启动

与只读检索 API（serve_api.py）分离端口；审核库 data/review.sqlite3 独立，检索 API 不碰。
"""
import argparse
import json
import sqlite3
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
/* 设计 token 对齐 DSH 默认样式（DeepSeek 设计语言）：
   近黑主色 #0f1115、三级标签灰阶、10% 细边框、12px 圆角、系统字体栈 + 抗锯齿；
   状态色低饱和（tinted badge），暗色跟随系统。 */
:root{
  --rv-bg:#ffffff; --rv-bg-2:#f7f8f9; --rv-panel:#ffffff;
  --rv-label-1:#0f1115; --rv-label-2:#61666b; --rv-label-3:#81858c;
  --rv-border:rgb(0 0 0 / 10%); --rv-border-strong:rgb(0 0 0 / 18%);
  --rv-brand:#0f1115; --rv-brand-contrast:#ffffff;
  --rv-accent:#4d6bfe; --rv-hover:#f2f3f5; --rv-active:#ebecef;
  --rv-green:#0a8f5b; --rv-green-bg:#e6f4ec;
  --rv-amber:#9a6b00; --rv-amber-bg:#faf1dc;
  --rv-violet:#6d4fd4; --rv-violet-bg:#efebfa;
  --rv-red:#c0392f; --rv-red-bg:#fdeceb;
  --rv-radius:12px; --rv-radius-s:8px;
  --rv-shadow:0 1px 2px rgb(15 17 21 / 4%);
}
@media (prefers-color-scheme: dark){
  :root{
    --rv-bg:#151517; --rv-bg-2:#1d1d20; --rv-panel:#1d1d20;
    --rv-label-1:#f9fafb; --rv-label-2:#cfd3d6; --rv-label-3:#adb2b8;
    --rv-border:rgb(255 255 255 / 12%); --rv-border-strong:rgb(255 255 255 / 22%);
    --rv-brand:#f9fafb; --rv-brand-contrast:#0f1115;
    --rv-accent:#7a90ff; --rv-hover:#26262a; --rv-active:#2e2e33;
    --rv-green:#4ecb8f; --rv-green-bg:#17301f;
    --rv-amber:#e3b341; --rv-amber-bg:#332a14;
    --rv-violet:#a794ee; --rv-violet-bg:#241f38;
    --rv-red:#ef7b72; --rv-red-bg:#3a1c19;
    --rv-shadow:none;
  }
}
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei","Helvetica Neue",Helvetica,Arial,sans-serif;
  margin:0;background:var(--rv-bg);color:var(--rv-label-1);font-size:14px;line-height:1.55;
  -webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
header{position:sticky;top:0;z-index:10;background:var(--rv-panel);border-bottom:1px solid var(--rv-border);
  padding:0 24px;display:flex;gap:20px;align-items:center;height:52px}
header strong{font-size:15px;letter-spacing:.2px;margin-right:8px}
header a{color:var(--rv-label-2);text-decoration:none;cursor:pointer;font-size:13.5px;padding:4px 2px;border-bottom:2px solid transparent}
header a.active{color:var(--rv-label-1);font-weight:600;border-bottom-color:var(--rv-label-1)}
header a:hover{color:var(--rv-label-1)}
main{padding:24px;max-width:1080px;margin:0 auto}
h3{font-size:16px;font-weight:600;margin:18px 0 12px}
h4{font-size:14px;font-weight:600;margin:12px 0 8px}
.card{background:var(--rv-panel);border:1px solid var(--rv-border);border-radius:var(--rv-radius);
  padding:16px;margin-bottom:12px;box-shadow:var(--rv-shadow)}
.card.subtle{background:var(--rv-bg-2)}
.badge{display:inline-block;padding:1px 9px;border-radius:999px;font-size:12px;margin-right:6px;
  background:var(--rv-bg-2);color:var(--rv-label-2);border:1px solid var(--rv-border);vertical-align:1px}
.badge.pending{background:var(--rv-amber-bg);color:var(--rv-amber);border-color:transparent}
.badge.approved{background:var(--rv-green-bg);color:var(--rv-green);border-color:transparent}
.badge.suspected{background:var(--rv-violet-bg);color:var(--rv-violet);border-color:transparent}
.badge.deleted{background:var(--rv-red-bg);color:var(--rv-red);border-color:transparent}
button{background:var(--rv-panel);color:var(--rv-label-1);border:1px solid var(--rv-border-strong);
  border-radius:var(--rv-radius-s);padding:6px 14px;cursor:pointer;font-size:13px;transition:background .12s}
button:hover{background:var(--rv-hover)}
button.primary{background:var(--rv-brand);color:var(--rv-brand-contrast);border-color:transparent;font-weight:500}
button.primary:hover{opacity:.88;background:var(--rv-brand)}
button.ok{background:var(--rv-green);color:#fff;border-color:transparent}
button.ok:hover{opacity:.88;background:var(--rv-green)}
button.danger{background:transparent;color:var(--rv-red);border-color:var(--rv-red)}
button.danger:hover{background:var(--rv-red-bg)}
button.sm{padding:2px 8px;font-size:12px}
input,select,textarea{padding:6px 9px;border:1px solid var(--rv-border-strong);border-radius:var(--rv-radius-s);
  font-size:13px;background:var(--rv-panel);color:var(--rv-label-1);font-family:inherit}
input:focus,select:focus,textarea:focus{outline:2px solid var(--rv-accent);outline-offset:-1px;border-color:var(--rv-accent)}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:7px 8px;border-bottom:1px solid var(--rv-border-strong);color:var(--rv-label-3);font-weight:500;font-size:12px}
td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--rv-border)}
tr:hover td{background:var(--rv-hover)}
pre{background:var(--rv-bg-2);border:1px solid var(--rv-border);color:var(--rv-label-1);
  padding:12px;border-radius:var(--rv-radius-s);overflow:auto;font-size:12px;line-height:1.5}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}
.item{display:flex;justify-content:space-between;gap:10px;cursor:pointer;padding:10px 12px;border-radius:var(--rv-radius-s);
  border:1px solid transparent}
.item:hover{background:var(--rv-hover)}
.item.selected{border-color:var(--rv-border-strong);background:var(--rv-active)}
.muted{color:var(--rv-label-3);font-size:12px}
.tagb{display:inline-block;padding:1px 9px;border-radius:999px;font-size:12px;margin:2px 4px 2px 0;cursor:pointer;border:1px solid var(--rv-border)}
.tagb.domain{background:var(--rv-violet-bg);color:var(--rv-violet);border-color:transparent}
.tagb.purpose{background:var(--rv-green-bg);color:var(--rv-green);border-color:transparent}
.tagb.auto{border-style:dashed}
.tagb.manual{background:var(--rv-amber-bg);color:var(--rv-amber);border-color:transparent}
.tagb.excluded{background:var(--rv-red-bg);color:var(--rv-red);text-decoration:line-through}
.tagb.reg{background:var(--rv-bg-2);color:var(--rv-label-2);cursor:default}
.warn{color:var(--rv-amber);font-size:12px}
.todo{background:var(--rv-amber-bg);color:var(--rv-amber);border:1px dashed var(--rv-amber);
  border-radius:var(--rv-radius-s);padding:10px 14px;font-size:12.5px;margin-bottom:12px}
.statgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
.statcard{background:var(--rv-panel);border:1px solid var(--rv-border);border-radius:var(--rv-radius);
  padding:14px 16px;box-shadow:var(--rv-shadow)}
.statcard b{font-size:13.5px}
.actions{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:10px}
.divider{border:0;border-top:1px solid var(--rv-border);margin:14px 0}
#toast{position:fixed;top:64px;right:24px;background:var(--rv-brand);color:var(--rv-brand-contrast);
  padding:9px 16px;border-radius:var(--rv-radius-s);display:none;z-index:99;font-size:13px;box-shadow:0 4px 16px rgb(0 0 0 / 18%)}
#modal{position:fixed;inset:0;background:rgb(15 17 21 / 40%);display:flex;align-items:center;justify-content:center;z-index:98}
#modal[hidden]{display:none}
#modal-card{background:var(--rv-panel);border:1px solid var(--rv-border);border-radius:var(--rv-radius);
  box-shadow:0 8px 32px rgb(0 0 0 / 22%);padding:18px 20px;min-width:340px;max-width:560px}
#modal-card h4{margin:0 0 10px;font-size:14.5px}
#modal-card .row{margin:8px 0;display:flex;align-items:center;gap:8px}
#modal-card .row label{width:64px;color:var(--rv-label-2);font-size:12.5px;flex:none}
#modal-card .actions{margin-top:14px;display:flex;gap:8px;flex-wrap:wrap}
.tagb.editing{outline:2px solid var(--rv-accent);outline-offset:1px}
</style></head><body>
<header><strong>vllm-kb 审核工作台</strong>
<a id="nav-stats" class="active" onclick="tab('stats')">概览</a>
<a id="nav-queue" onclick="tab('queue')">审核队列</a>
<a id="nav-docs" onclick="tab('docs')">文档管理</a>
<a id="nav-tags" onclick="tab('tags')">标签管理</a>
<a id="nav-configs" onclick="tab('configs')">API 配置</a><a id="nav-gaps" onclick="tab('gaps')">知识缺口</a></header>
<div id="toast"></div>
<div id="modal" hidden onclick="if(event.target===this)closeModal()"><div id="modal-card"></div></div>
<main>
<div id="view-stats"></div><div id="view-queue" style="display:none"></div><div id="view-docs" style="display:none"></div><div id="view-tags" style="display:none"></div><div id="view-configs" style="display:none"></div><div id="view-gaps" style="display:none"></div>
</main>
<script>
const $=s=>document.querySelector(s);
async function j(url,opt){const r=await fetch(url,opt);const d=await r.json();if(!r.ok)throw new Error(d.detail||r.status);return d}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function badge(s){return `<span class="badge ${s}">${s}</span>`}
function tab(name){['stats','queue','docs','tags','configs','gaps'].forEach(t=>{document.getElementById('view-'+t).style.display=t===name?'':'none';document.getElementById('nav-'+t).className=t===name?'active':''});if(name==='stats')loadStats();if(name==='queue')loadQueue();if(name==='docs')loadDocs();if(name==='tags')loadTagDict();if(name==='configs')loadConfigs();if(name==='gaps')loadGaps()}
async function loadGaps(){let gs=[];try{gs=await j('/api/gaps')}catch(e){$('#view-gaps').innerHTML=`<h3>知识缺口</h3><div class="card muted">未启用行为遥测（config.confidence.feedback_enabled）或无缺口数据。启用后跑 scripts/build_feedback.py 生成。</div>`;return}
const byType={hard_gap:'强签名零命中',soft_gap:'命中无结论',quality_gap:'弱签名零命中'};
$('#view-gaps').innerHTML=`<h3>知识缺口（${gs.length} 个，行为遥测推断——同签名跨会话反复查无果）</h3>`+gs.map(g=>`<div class="card"><b>${esc(byType[g.gap_type]||g.gap_type)}</b> <span class="badge pending">sessions=${g.session_count}</span> <span class="muted">${esc(g.component||'')}</span><table><tr><th>签名</th><td class="mono">${esc((g.signature_text||'').slice(0,120))}</td></tr><tr><th>样例查询</th><td class="mono">${esc((g.sample_queries||[]).join(' | ').slice(0,200))}</td></tr><tr><th>平均命中</th><td>${g.avg_result_count}</td></tr><tr><th>首现/末现</th><td class="muted">${esc((g.first_seen||'').slice(0,16))} ~ ${esc((g.last_seen||'').slice(0,16))}</td></tr></table></div>`).join('')||'<div class="card muted">暂无缺口</div>'}
async function loadStats(){const s=await j('/api/stats');const cats=Object.entries(s).sort((a,b)=>b[1].pending-a[1].pending);
let tagCard='';try{const td=await j('/api/tag-dict');tagCard=`<div class="statcard"><b>标签词典</b><br><span class="muted">领域类 ${td.stats.domain} 个 · 作用类 ${td.stats.purpose} 个 · 已打标文档 ${td.stats.tagged_docs} 篇</span></div>`}catch(e){}
$('#view-stats').innerHTML=`<h3>待办概览（${cats.reduce((a,[k,v])=>a+v.pending,0)} 待办 / ${cats.reduce((a,[k,v])=>a+v.suspected,0)} 存疑）</h3>`
+(tagCard||'')+`<div class="statgrid">`+(cats.map(([k,v])=>`<div class="statcard"><b>${esc(k)}</b> <span class="badge pending">待办 ${v.pending}</span>${v.suspected?`<span class="badge suspected">存疑 ${v.suspected}</span>`:''} <span class="muted">共 ${v.total}</span></div>`).join('')||'<div class="card">暂无审核项（导入文档后自动补单）</div>')+`</div>`}
async function loadQueue(cat=''){const q=await j('/api/queue'+(cat?'?category='+cat:''));
const del=await j('/api/queue?status=deleted&limit=100');
const opts=(await (async()=>{const r=await fetch('/api/categories');return r.ok?await r.json():[]})());
const fileRows=i=>{const p=i.payload||{};return (p.aggregate==='excel_file'||(p.rows||[]).length>1)?` <span class="badge approved">${p.row_count||(p.rows||[]).length} 行</span>`:''};
// 列表条目标题：tag_candidate 显示候选词本身（payload.title 是"兼容旧单文档"的首文档标题，
// 曾把候选词压住——四篇 Atlas A3 手册的候选在列表里全显示成同一个产品名）；其余类别用文档标题
const itemTitle=i=>{const p=i.payload||{};return i.category==='tag_candidate'?(p.candidate||i.item_ref):(p.title||i.item_ref)};
$('#view-queue').innerHTML=`<h3>审核队列</h3><div class="card subtle"><label>类别: </label><select id="qcat" onchange="loadQueue(this.value)"><option value="">全部</option>${opts.map(c=>`<option ${c===cat?'selected':''}>${esc(c)}</option>`).join('')}</select> <span class="muted">共 ${q.length} 条（未审核在前，存疑在后；表格项按文件粒度，<b>N 行</b>徽标 = 一次认证整表生效）</span></div>`
+`<div id="queuebatch" class="card" style="display:none;border-left:3px solid var(--rv-accent)"><b>已选 <span id="queuebatch-n">0</span> 项补标</b> 认证级别: <select id="qb-level"><option value="tested">tested（已验证）</option><option value="expert">expert（专家认证）</option></select> <button class="primary sm" onclick="queueBatchReview()">✓ 批量认证</button></div>`
+`<div id="queuecand" class="card" style="display:none;border-left:3px solid var(--rv-accent)"><b>已选 <span id="queuecand-n">0</span> 条候选</b> tier（仅采纳可覆盖）: <select id="qc-tier"><option value="">用各候选建议</option><option value="domain">领域</option><option value="purpose">作用</option></select> <button class="primary sm" onclick="queueBatchCandidates('adopt')">✓ 批量采纳</button> <button class="sm" onclick="queueBatchCandidates('ignore')">忽略所选</button> <span class="muted">采纳=入词典+全部提及文档打标；忽略=不再出现（不改任何数据）</span></div>`
+q.map(i=>{const pending=i.status==='pending';
const chk=(pending&&(i.category==='verification_pending'||i.category==='tag_candidate'))?`<input type="checkbox" class="${i.category==='tag_candidate'?'cchk':'qchk'}" data-id="${i.id}" onclick="event.stopPropagation()" onchange="queueSelChanged()" style="margin-right:6px" title="勾选后批量操作"> `:'';
return `<div class="item" onclick="showItem(${i.id})"><div>${chk}<span class="badge ${i.status}">${i.status}</span> <b>${esc(i.category)}</b> ${esc(itemTitle(i))}${fileRows(i)}</div><div class="muted">#${i.id} · ${(i.created_at||'').slice(0,16)}</div></div>`}).join('')||'<div class="card muted">队列为空</div>'
+`<h3 style="margin-top:26px">待实际删除（${del.length}）— 数据库记录已删，原始文件请手动本地删除</h3>`+del.map(i=>`<div class="card" style="border-left:3px solid var(--rv-red)"><div><b>${esc((i.payload||{}).title||i.item_ref)}</b> <span class="mono muted">${esc((i.payload||{}).source_id||i.item_ref)}</span></div>
<div class="muted mono">${esc((i.payload||{}).asset&&(i.payload.asset.path)||'')}</div>
<div class="actions"><button class="ok" onclick="undoDelete(${i.id})">↩ 撤回（恢复记录并重新入队）</button> <span class="muted">删除人: ${esc(i.reviewer||'')} ${(i.reviewed_at||'').slice(0,16)}</span></div></div>`).join('')||''}
async function showItem(id){const i=await j('/api/item/'+id);const p=i.payload||{};
const rows=p.rows||[];
const isFile=i.category==='verification_pending'&&(p.aggregate==='excel_file'||rows.length>1);
let fileHead='';
if(isFile){fileHead=` <span class="badge approved">${p.row_count||rows.length} 行 · 文件粒度</span>
<p class="muted">认证/存疑<b>整体应用到该文件全部 ${p.row_count||rows.length} 行</b>，不逐行审；表格内容变更后会自动重新进入本队列（内容指纹 <span class="mono">file_sha=${esc((p.file_sha||'').slice(0,16))||'无'}</span>）。</p>`}
let docPrev='';
if(i.category==='verification_pending'){const sid=isFile?((rows[0]||{}).source_id||''):(p.source_id||i.item_ref);
 if(sid){try{const d=await j('/api/doc-preview?source_id='+encodeURIComponent(sid));
  docPrev=`<h4>内容预览${isFile?'（该文件首行文档；body 含各 sheet 全表内容）':''}</h4><pre style="max-height:340px">${esc(d.body||'（无正文）')}</pre>${d.truncated?'<div class="muted">内容过长已截断</div>':''}`}
 catch(e){docPrev=`<p class="muted">内容预览不可用：${esc(e.message)}</p>`}}}
$('#view-queue').innerHTML=`<div class="card"><button onclick="loadQueue()">← 返回队列</button><h3>#${i.id} ${esc(i.category)} <span class="badge ${i.status}">${i.status}</span>${fileHead}</h3>
<p><b>item_ref:</b> <span class="mono">${esc(i.item_ref)}</span></p>
${i.category==='tag_candidate'?`<p><b>候选标签:</b> <span class="tagb reg">${esc(p.candidate||'')}</span> 建议tier: ${esc(p.suggested_tier||'自动判定')}（提及文档 <b>${p.doc_count||1}</b> 篇：${esc((p.docs||[p]).map(d=>d.title||d.source_id||'').slice(0,5).join('、')||'')}${(p.doc_count||1)>5?' …':''}）</p>`:''}
${i.category==='low_confidence_ocr'?`<p><b>图片:</b> <span class="mono">${esc((p.sha256||'').slice(0,16)||p.asset_id||'')}</span>
  ${p.asset_path?`<span class="muted mono">· ${esc(p.asset_path)}</span>`:'<span class="muted">（未注册到 asset_registry，无预览）</span>'}
  <br><img src="/${esc(p.asset_path||'')}" alt="原图" style="max-width:100%;max-height:320px;border:1px solid var(--rv-border);border-radius:8px;margin:4px 0" onerror="this.style.display='none'">
  原因: <b>${esc(p.reason||'')}</b>（置信度 ${p.confidence===null||p.confidence===undefined?'无':p.confidence}，来源 ${esc(p.confidence_source||'')}${p.anomaly?`，自报异常 ${esc(p.anomaly)}`:''}）
 ${(p.signatures||[]).length?`<br><b>OCR 签名线索:</b> ${esc((p.signatures||[]).map(s=>(s.kind||'')+':'+(s.text||'')).join('、'))}`:'<br><span class="muted">无签名线索</span>'}
 <br><span class="muted">该图 OCR 文本<b>未进正文</b>（低置信或自报异常），本项只是待核对清单：✓ 认证即忽略，存疑则重新排队。</span></p>`:''}
${isFile?`<h4>文件内行清单（展示前 ${Math.min(rows.length,100)} 行）</h4><table><tr><th>source_id</th><th>标题</th></tr>${rows.slice(0,100).map(r=>`<tr><td class="mono">${esc(r.source_id)}</td><td>${esc(r.title||'')}</td></tr>`).join('')}</table>`:''}
${docPrev}
<h4>原始 payload</h4><pre>${esc(JSON.stringify(p,null,2))}</pre>
<h4>审核（只做判定，不修改原始内容）</h4><input id="rev" placeholder="审核人（必填）" style="width:180px" oninput="saveReviewer(this.value)">
<textarea id="rvnote" rows="2" style="width:100%" placeholder="备注（可选）"></textarea>
<div class="actions">
${i.category==='verification_pending'?`认证级别: <select id="verlevel"><option value="tested">tested（已验证）</option><option value="expert">expert（专家认证）</option></select>`:''}
${i.category==='tag_candidate'?`tier: <select id="adopt-tier"><option value="">自动判定</option><option value="domain">领域</option><option value="purpose">作用</option></select>
<button class="ok" onclick="adoptCandidate(${i.id})">✓ 采纳为标签（入词典+全部提及文档打标，立即生效）</button>`:''}
<button class="primary" onclick="review(${i.id},'approved')">✓ 认证${isFile?`（全部 ${p.row_count||rows.length} 行）`:''}</button>
<button onclick="review(${i.id},'suspected')">？ 存疑（重新排队靠后）</button>
${['tag_candidate','low_confidence_ocr'].includes(i.category)?'':`<button class="danger" onclick="deleteDoc(${i.id})">🗑 标记删除${isFile?'（整表）':''}</button>`}
</div>
<span class="muted">存疑项排在未审核之后；删除可在队列底部'待实际删除'撤回；tag_candidate 按词聚合（同词多文档一条）、low_confidence_ocr 按图片 sha256 聚合，两者 item_ref 不是文档 id，故不提供删除按钮（忽略请用 ✓ 认证）${isFile?'；表格项删除=整表全部行，可在待删除列表撤回':''}</span></div>`
$('#rev').value=loadReviewer()}
async function adoptCandidate(id){const reviewer=document.getElementById('rev').value;if(!reviewer){alert('请填写审核人');return}
try{await j('/api/tag-candidate/'+id+'/adopt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reviewer,tier:document.getElementById('adopt-tier').value||null})});loadQueue();toast('✓ 已采纳（词典已同步，重建图后入图）')}catch(e){toast('✗ '+e.message)}}
async function review(id,action){const reviewer=document.getElementById('rev').value;if(!reviewer){alert('请填写审核人');return}
const result={note:(document.getElementById('rvnote')||{}).value||''};
const vl=document.getElementById('verlevel');if(vl)result.verification=vl.value;  // 补标项认证级别（tested/expert）
await j('/api/item/'+id+'/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,reviewer,result})});loadQueue()}
async function deleteDoc(id){const reviewer=document.getElementById('rev').value;if(!reviewer){alert('请填写审核人');return}
if(!confirm('标记删除：删除数据库记录并登记墓碑（构建不再复活；表格项=整表全部行），原始资产文件需到"待实际删除"列表手动本地删除。确认？'))return
await j('/api/item/'+id+'/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reviewer,note:(document.getElementById('rvnote')||{}).value||''})});loadQueue()}
async function undoDelete(id){if(!confirm('撤回删除：恢复数据库记录、解除墓碑（此后文档可重新入库）并重新加入队列？'))return;await j('/api/item/'+id+'/undo',{method:'POST'});loadQueue()}
let docQuery='';
function clearDocSearch(){docQuery='';loadDocs()}
async function loadDocs(){const inp=document.getElementById('docsearch');if(inp)docQuery=inp.value;
const ds=await j('/api/external-docs?limit=500'+(docQuery?`&q=${encodeURIComponent(docQuery)}`:''));
let wd=[];try{wd=await j('/api/withdrawn')}catch(e){}
let td={};try{const t=await j('/api/tag-dict');t.groups.domain.forEach(x=>td[x.name]='domain');t.groups.purpose.forEach(x=>td[x.name]='purpose')}catch(e){}
const tierOf=n=>td[n]||'domain';
const q=s=>esc(String(s||'')).replace(/'/g,"\\'");
const tagB=(sid,t,cls,act,mark)=>`<span class="tagb ${cls} ${tierOf(t)}" onclick="tagEdit('${q(sid)}','${act}','${q(t)}')">${esc(t)} ${mark}</span>`;
const groupRow=(list,cls,act,mark,sid)=>list.length?`<div style="margin-top:4px"><span class="muted">${cls==='excluded'?'已排除':cls==='manual'?'人工':'自动'}:</span> ${list.map(t=>tagB(sid,t,cls,act,mark)).join('')}</div>`:'';
const docCard=d=>{const t=d.tags||{};const inp='newtag-'+q(d.source_id).replace(/[^a-zA-Z0-9]/g,'_');
const dupPage=d.duplicate&&String(d.source_id).startsWith('html:page--');
return `<div class="card" style="border-left:3px solid ${d.duplicate&&!dupPage?'var(--rv-amber)':'var(--rv-border)'}"><div><input type="checkbox" class="dchk" data-sid="${q(d.source_id)}" onchange="docSelChanged()" style="margin-right:6px" title="勾选后批量操作"><b>${esc(d.title||d.source_id)}</b> <span class="badge">${esc(d.source_type)}</span> ${dupPage?`<span class="muted">统一 page.html 布局（按目录路径消歧，非重名冲突）</span>`:(d.duplicate?`<span class="warn">⚠ 同 stem 重名（入库已按相对路径指纹自动消歧，此处仅提示可合并）</span>`:'')}</div>
<div class="mono muted">${esc(d.source_id)}${d.asset_path?` · ${esc(d.asset_path)}`:''}</div>
${groupRow(t.auto||[],'auto','exclude','✕',d.source_id)}
${groupRow(t.excluded||[],'excluded','restore','↺',d.source_id)}
${groupRow(t.manual||[],'manual','remove','✕',d.source_id)}
<div style="margin-top:6px"><input id="${inp}" placeholder="添加标签" style="width:180px"><button class="sm" onclick="tagAdd('${q(d.source_id)}',document.getElementById('${inp}').value)">添加</button>
<span class="muted">最终: ${(t.final||[]).map(esc).join(' · ')||'（无标签）'}</span></div>
<div class="actions"><button class="danger sm" onclick="delExternalDoc('${q(d.source_id)}')">🗑 从数据库删除（本地文件保留）</button></div></div>`};
// 按 imports 第一层子目录分组，组名带来源 id 前缀区分不同导入来源：
// 组名=来源id\子目录（pdf\support官方文档）；根级=来源id（pdf）；资产回退件=来源id（资产回退）；网络来源=(未分组)
const groupKey=d=>{const src=d.rel_src;if(src==null)return '(未分组)';
 if(d.rel_dir==null)return `${src}（资产回退）`;
 return d.rel_dir?`${src}\\${d.rel_dir}`:src};
const groups={};ds.forEach(d=>{const k=groupKey(d);(groups[k]=groups[k]||[]).push(d)});
const rank=k=>k==='(未分组)'?2:1;
const keys=Object.keys(groups).sort((a,b)=>rank(a)-rank(b)||a.localeCompare(b,'zh'));
const groupHtml=keys.map(k=>{const g=groups[k];
return `<details open style="margin-bottom:10px"><summary style="cursor:pointer;font-weight:600;padding:4px 0">📁 ${esc(k)}（${g.length}）<input type="checkbox" style="margin-left:8px" onclick="event.stopPropagation();docGroupToggle(this,this.checked)" title="全选本组"></summary><div style="padding-left:2px">${g.map(docCard).join('')}</div></details>`}).join('');
$('#view-docs').innerHTML=`<h3>文档管理 — 外源文档（导入的 PDF/MD/表格等${docQuery?`，搜索"${esc(docQuery)}"命中`:`，本页 ${ds.length} 条`}）</h3>
<div class="card subtle">按 <b>imports 第一层子目录</b> 分组，组名 = <b>来源id\\子目录</b>（根级 = 来源 id）——不同导入来源即使子目录同名也各自成组；📁来源id（资产回退）= imports 缺失时从资产层副本收录的文档；(未分组) = 网络来源。勾选多篇（或整组）后用批量栏操作。删除只动数据库（docs + chunks + 向量），<b>本地文件不动</b>；删除会登记<b>墓碑</b>——任何构建路径（增量/资产回退/--rebuild）都不再复活该文档，撤回删除（待实际删除列表）才解除。GitHub 采集文档不在此列。<br>标签：<b>虚线框=自动标签</b>（点 ✕ 排除）、<b>黄底=人工标签</b>、<b>红底删除线=已排除</b>（点 ↺ 恢复）。最终标签 = (自动 − 排除) ∪ 人工，与入库/建图一致。</div>
<div class="card">🔍 <input id="docsearch" value="${esc(docQuery)}" placeholder="搜索 source_id / 标题（子串，回车执行）" style="width:300px" onkeydown="if(event.key==='Enter')loadDocs()"> <button class="sm" onclick="loadDocs()">搜索</button> ${docQuery?'<button class="sm" onclick="clearDocSearch()">清空</button>':''} <span class="muted">库内文档远超单页窗口（本页最多 500 条、按 id 排序），删除后后面的文档会补位——<b>定位文档请用搜索</b></span></div>
<div id="docbatch" class="card" style="display:none;position:sticky;top:52px;z-index:5;border-left:3px solid var(--rv-accent)"><b>已选 <span id="docbatch-n">0</span> 篇</b> <button class="primary sm" onclick="docBatchTagModal()">🏷 批量标签</button> <button class="danger sm" onclick="docBatchDelete()">🗑 批量删除</button> <span class="muted">批量标签支持 添加/删除人工、排除/恢复自动</span></div>
<div class="todo"><b>TODO（暂缓）</b>：规则化自动采纳（词典近义/高频强候选免审）与按主题聚合的打标视图仍待设计——<b>批量操作已上线</b>（多选/按目录：打标、排除、恢复、删除）。</div>`
+(keys.length?groupHtml:'<div class="card muted">无外源文档</div>')
+`<details style="margin-top:16px"><summary style="cursor:pointer;font-weight:600">📥 已回撤（${wd.length}）— 墓碑生效中，任何构建路径（增量/资产回退/--rebuild）都不复活；mirror 来源文件重新出现会自动解除</summary><div class="card">${wd.map(w=>`<div style="margin:3px 0"><span class="mono">${esc(w.source_id)}</span> <span class="muted">${esc((w.deleted_at||'').slice(0,16))} · ${esc(w.reviewer||'')}</span> ${w.undo_item_id?`<button class="sm" onclick="undoWithdraw(${w.undo_item_id})">↻ 撤回</button>`:''}</div>`).join('')||'<span class="muted">（无）</span>'}</div></details>`}
async function tagEdit(sid,action,tag){const reviewer=requireReviewer();if(!reviewer)return;
try{await j('/api/docs/tags/edit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source_id:sid,action,tag,reviewer})});loadDocs();toast('✓ 已更新标签')}catch(e){toast('✗ '+e.message)}}
function tagAdd(sid,t){if(!t||!t.trim())return;tagEdit(sid,'add',t.trim())}
async function delExternalDoc(sid){if(!confirm('从数据库彻底删除该文档（docs+chunks+向量）？本地文件保留；将登记删除墓碑——任何构建路径（增量/资产回退/--rebuild）都不再复活它，撤回删除可恢复。确认？'))return
try{const r=await j('/api/external-docs/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source_id:sid})});loadDocs();
const cc=r.candidates_cleaned||{};toast(`✓ 已删除 + 登记墓碑（本地文件保留）${cc.auto_ignored?`，联动忽略 ${cc.auto_ignored} 条候选`:cc.pruned?`，修剪 ${cc.pruned} 条候选引用`:''}`)}
catch(e){toast('✗ 删除失败: '+e.message)}}
async function undoWithdraw(id){if(!confirm('撤回回撤：恢复数据库记录、解除墓碑（此后文档可重新入库）？'))return;try{await j('/api/item/'+id+'/undo',{method:'POST'});loadDocs();toast('✓ 已撤回（墓碑解除）')}catch(e){toast('✗ '+e.message)}}
async function loadTagDict(){const t=await j('/api/tag-dict');
const q=x=>esc(String(x||'')).replace(/'/g,"\\'");
const groupHtml=(g,title)=>`<details open style="margin-bottom:8px"><summary style="cursor:pointer;font-weight:600;padding:4px 0">${title}（${g.length}）<input type="checkbox" style="margin-left:8px" onclick="event.stopPropagation();tdGroupToggle(this,this.checked)" title="全选本组"></summary><div class="card">${g.map(x=>{const n=q(x.name);
return `<div style="margin:3px 0"><input type="checkbox" class="tchk" data-name="${esc(x.name)}" onchange="tdSelChanged()" style="margin-right:6px" title="勾选后批量操作"><span class="tagb reg">${esc(x.name)} (${x.docs}篇)</span> <button class="sm" onclick="tagDictToggleTier('${n}','${title.includes('domain')?'domain':'purpose'}')" title="点击在领域/作用间切换（1 击，全局生效）">${title.includes('domain')?'domain':'purpose'}</button> <button class="sm" onclick="tagDictForm('${n}', this)">编辑</button></div>`}).join('')||'（空）'}</div></details>`;
$('#view-tags').innerHTML=`<h3>标签词典（config.json tags.registry，全局唯一事实源）</h3>
<div class="todo"><b>TODO（暂缓）</b>：仅剩"按主题聚合的打标视图"——候选批量分拣、设备词表、发端降噪、词典批量操作均已上线（见「文档管理」页与下方批量栏）。</div>
<div class="card subtle">两级分类：领域类=这是什么领域的知识（过滤/圈定范围）；作用类=文档能帮我做什么（能力匹配）。tier 徽标<b>点击即切换</b>（1 击）；勾选多个后用批量栏<b>批量移出/批量改 tier</b>。新增/改名/改 tier 同步 config.json；<b>不热插图</b>——运行 build_graph.py 重建图后入图（Kùzu 单写者约束）。</div>
<div id="tagdictbatch" class="card" style="display:none;position:sticky;top:52px;z-index:5;border-left:3px solid var(--rv-accent)"><b>已选 <span id="tagdictbatch-n">0</span> 个</b> <button class="danger sm" onclick="tagDictBatchDelete()">🗑 批量移出词典</button> <span class="muted">改为</span> <button class="sm" onclick="tagDictBatchTier('domain')">领域</button> <button class="sm" onclick="tagDictBatchTier('purpose')">作用</button> <span class="muted">移出不动已打标文档；改 tier 全局生效</span></div>
${groupHtml(t.groups.domain||[],'主题/领域类 (domain)')}
${groupHtml(t.groups.purpose||[],'具体作用类 (purpose)')}
<div class="card"><b>新增标签</b> <input id="newtag-name" placeholder="标签名" style="width:180px"> tier: <select id="newtag-tier"><option value="">自动判定</option><option value="domain">领域</option><option value="purpose">作用</option></select>
<button class="ok" onclick="tagDictAdd()">加入词典</button></div>`}
async function tagDictAdd(){const name=document.getElementById('newtag-name').value.trim();if(!name){alert('输入标签名');return}
try{await j('/api/tag-dict/add',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,tier:document.getElementById('newtag-tier').value||null})});loadTagDict();toast('✓ 已加入词典（重建图后入图）')}catch(e){toast('✗ '+e.message)}}
// ---------------- 词典批量操作 + 行内 tier 切换 ----------------
function tdCheckedNames(){return Array.from(document.querySelectorAll('#view-tags .tchk:checked')).map(x=>x.dataset.name)}
function tdSelChanged(){const n=tdCheckedNames().length;const bar=document.getElementById('tagdictbatch');if(bar)bar.style.display=n?'block':'none';const c=document.getElementById('tagdictbatch-n');if(c)c.textContent=n}
function tdGroupToggle(cb,checked){const grp=cb.closest('details');if(grp)grp.querySelectorAll('.tchk').forEach(x=>{x.checked=checked});tdSelChanged()}
async function tagDictToggleTier(name,current){const to=current==='domain'?'purpose':'domain';
try{await j('/api/tag-dict/tier',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,tier:to})});loadTagDict();toast(`✓ ${name} → ${to==='domain'?'领域':'作用'}类`)}catch(e){toast('✗ '+e.message)}}
async function tagDictBatchDelete(){const names=tdCheckedNames();if(!names.length){toast('先勾选标签');return}
if(!confirm(`从词典移出选中的 ${names.length} 个标签？只移出词典，已打标文档上的该标签保留。确认？`))return;
try{const r=await j('/api/tag-dict/batch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({names,action:'delete'})});
loadTagDict();toast(`✓ 已移出 ${r.done}/${names.length} 个${r.failed.length?`（首例: ${r.failed[0].error}）`:''}`)}catch(e){toast('✗ '+e.message)}}
async function tagDictBatchTier(tier){const names=tdCheckedNames();if(!names.length){toast('先勾选标签');return}
if(!confirm(`把选中的 ${names.length} 个标签改为 ${tier==='domain'?'领域':'作用'}类？全局生效。确认？`))return;
try{const r=await j('/api/tag-dict/batch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({names,action:'tier',tier})});
loadTagDict();toast(`✓ 已改 ${r.done}/${names.length} 个${r.failed.length?`（首例: ${r.failed[0].error}）`:''}`)}catch(e){toast('✗ '+e.message)}}
function openModal(html){document.getElementById('modal-card').innerHTML=html;document.getElementById('modal').hidden=false}
function closeModal(){document.getElementById('modal').hidden=true;document.querySelectorAll('.tagb.editing').forEach(x=>x.classList.remove('editing'))}
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeModal()})
// 审核人缓存直用：填过一次全站不再询问（消灭每次点标签操作都弹 prompt 的摩擦）
function requireReviewer(){let r=loadReviewer();if(!r){r=(prompt('审核人（首次填写后自动记住，之后不再询问）','')||'').trim();if(r)saveReviewer(r)}return r}
// ---------------- 批量操作（文档管理 + 审核队列补标认证） ----------------
function docCheckedSids(){return Array.from(document.querySelectorAll('#view-docs .dchk:checked')).map(x=>x.dataset.sid)}
function docSelChanged(){const n=docCheckedSids().length;const bar=document.getElementById('docbatch');if(bar)bar.style.display=n?'block':'none';const c=document.getElementById('docbatch-n');if(c)c.textContent=n}
function docGroupToggle(cb,checked){const grp=cb.closest('details');if(grp)grp.querySelectorAll('.dchk').forEach(x=>{x.checked=checked});docSelChanged()}
async function docBatchTagModal(){const sids=docCheckedSids();if(!sids.length){toast('先勾选文档');return}
openModal(`<h4>批量标签（已选 ${sids.length} 篇）</h4>
<div class="row"><label>审核人</label><input id="bt-rev" value="${esc(loadReviewer()||'')}" style="flex:1"></div>
<div class="row"><label>动作</label><select id="bt-action" style="flex:1"><option value="add">添加人工标签</option><option value="remove">删除人工标签</option><option value="exclude">排除自动标签</option><option value="restore">恢复自动标签（按标签名）</option></select></div>
<div class="row"><label>标签</label><input id="bt-tag" placeholder="标签名" style="flex:1"></div>
<div class="actions"><button class="primary" onclick="docBatchTagApply()">应用到 ${sids.length} 篇</button><button onclick="closeModal()">取消</button></div>
<div class="muted" style="margin-top:10px">逐篇独立生效并记审计（reviewer）；部分失败会在结果里逐条列出。</div>`);
setTimeout(()=>{const i=document.getElementById('bt-tag');if(i)i.focus()},0)}
async function docBatchTagApply(){const sids=docCheckedSids();const action=(document.getElementById('bt-action')||{}).value||'add';
const tag=((document.getElementById('bt-tag')||{}).value||'').trim();const reviewer=((document.getElementById('bt-rev')||{}).value||'').trim();
if(!tag){alert('输入标签名');return}
if(!reviewer){alert('填写审核人');return}
saveReviewer(reviewer);
try{const r=await j('/api/docs/tags/batch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source_ids:sids,action,tag,reviewer})});
closeModal();loadDocs();toast(`✓ ${r.updated}/${sids.length} 篇已更新${r.failed.length?`（首例失败: ${r.failed[0].error}）`:''}`)}catch(e){toast('✗ '+e.message)}}
async function docBatchDelete(retrySids){const sids=retrySids||docCheckedSids();if(!sids.length){toast('先勾选文档');return}
if(!retrySids&&!confirm(`从数据库删除选中的 ${sids.length} 篇文档（docs+chunks+向量）？本地文件保留；将登记删除墓碑——任何构建路径（增量/资产回退/--rebuild）都不再复活它们，撤回删除可恢复。确认？`))return
const reviewer=requireReviewer();if(!reviewer)return;
try{const r=await j('/api/external-docs/delete-batch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source_ids:sids})});
loadDocs();closeModal();
const problems=[...r.failed.map(f=>`${f.source_id}：${f.error}`),...(r.still_present||[]).map(s=>`${s}：删除后仍存在于库（服务端校验）`)];
if(problems.length){openModal(`<h4>⚠ 删除未完全成功（${r.deleted}/${sids.length} 篇已删）</h4><div class="card">${problems.map(p=>`<div class="mono" style="margin:2px 0">${esc(p)}</div>`).join('')}</div><div class="muted">失败的条目可重试；"仍存在于库"说明删除未落库（如库被占用），重试通常可解。</div><div class="actions"><button class="danger sm" onclick="docBatchDelete(${JSON.stringify([...r.failed.map(f=>f.source_id),...(r.still_present||[])]).replace(/"/g,'&quot;')})">↻ 重试剩余 ${r.failed.length+(r.still_present||[]).length} 篇</button> <button class="sm" onclick="closeModal()">关闭</button></div>`)}
else{toast(`✓ 已删除 ${r.deleted}/${sids.length} 篇（墓碑已登记）${r.candidates_cleaned&&r.candidates_cleaned.auto_ignored?`，联动忽略 ${r.candidates_cleaned.auto_ignored} 条候选`:''}`)}}catch(e){toast('✗ '+e.message)}}
function queueCheckedIds(sel){return Array.from(document.querySelectorAll(sel+':checked')).map(x=>+x.dataset.id)}
function queueSelChanged(){const vn=queueCheckedIds('#view-queue .qchk').length,cn=queueCheckedIds('#view-queue .cchk').length;
const vb=document.getElementById('queuebatch');if(vb)vb.style.display=vn?'block':'none';const vc=document.getElementById('queuebatch-n');if(vc)vc.textContent=vn;
const cb=document.getElementById('queuecand');if(cb)cb.style.display=cn?'block':'none';const cc=document.getElementById('queuecand-n');if(cc)cc.textContent=cn}
async function queueBatchReview(){const ids=queueCheckedIds('#view-queue .qchk');if(!ids.length){toast('先勾选补标项');return}
const reviewer=requireReviewer();if(!reviewer)return;
const level=((document.getElementById('qb-level')||{}).value)||'tested';
try{const r=await j('/api/items/review-batch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({item_ids:ids,action:'approved',reviewer,result:{verification:level}})});
loadQueue();toast(`✓ 已认证 ${r.reviewed}/${ids.length} 项${r.failed.length?`（首例失败: ${r.failed[0].error}）`:''}`)}catch(e){toast('✗ '+e.message)}}
async function queueBatchCandidates(action){const ids=queueCheckedIds('#view-queue .cchk');if(!ids.length){toast('先勾选候选');return}
const reviewer=requireReviewer();if(!reviewer)return;
const tier=action==='adopt'?(((document.getElementById('qc-tier')||{}).value)||null):null;
try{const r=await j('/api/tag-candidates/batch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({item_ids:ids,action,reviewer,tier})});
loadQueue();toast(`✓ ${action==='adopt'?'已采纳':'已忽略'} ${r.done}/${ids.length} 条${r.failed.length?`（首例失败: ${r.failed[0].error}）`:''}`)}catch(e){toast('✗ '+e.message)}}
function tagDictForm(name,el){document.querySelectorAll('.tagb.editing').forEach(x=>x.classList.remove('editing'));
if(el)el.classList.add('editing');  // 列表里正在编辑的芯片高亮，与对话框呼应
const q=esc(name).replace(/'/g,"\\'");
openModal(`<h4>编辑标签</h4>
<div class="row"><label>当前名</label><span class="tagb reg">${esc(name)}</span></div>
<div class="row"><label>新名</label><input id="td-new" placeholder="留空 = 不改名" style="flex:1"></div>
<div class="row"><label>tier</label><select id="td-tier"><option value="">不变</option><option value="domain">领域 (domain)</option><option value="purpose">作用 (purpose)</option></select></div>
<div class="actions"><button class="primary" onclick="tagDictRename('${q}')">改名</button>
<button onclick="tagDictTier('${q}')">改 tier</button>
<button class="danger" onclick="tagDictDelete('${q}')">删除</button>
<button onclick="closeModal()">取消</button></div>
<div class="muted" style="margin-top:10px">改名=全库替换（docs.tags / doc_tags / config 同步）；不热插图——build_graph.py 重建后入图。删除只移出词典，已打标文档保留该标签。</div>`);
setTimeout(()=>{const i=document.getElementById('td-new');if(i)i.focus()},0)}
async function tagDictRename(name){const nv=document.getElementById('td-new').value.trim();if(!nv){alert('输入新名');return}
try{const r=await j('/api/tag-dict/rename',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({old:name,new:nv})});closeModal();loadTagDict();toast(`✓ 已改名（${r.docs_updated} 篇受影响）`)}catch(e){toast('✗ '+e.message)}}
async function tagDictTier(name){const tv=document.getElementById('td-tier').value;if(!tv){alert('选择 tier');return}
try{await j('/api/tag-dict/tier',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,tier:tv})});closeModal();loadTagDict();toast('✓ tier 已更新（全局生效）')}catch(e){toast('✗ '+e.message)}}
async function tagDictDelete(name){if(!confirm(`从词典删除 ${name}？只移出词典，已打标文档上的该标签保留。`))return
try{await j('/api/tag-dict/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});closeModal();loadTagDict();toast('✓ 已移出词典')}catch(e){toast('✗ '+e.message)}}
async function loadConfigs(){const cs=await j('/api/configs');fillCache(cs);
$('#view-configs').innerHTML=`<h3>API 配置中心（key 脱敏；可编辑）</h3>`+cs.map(c=>`<div class="card" id="cfg-${esc(c.name)}"><b>${esc(c.name)}</b> <span class="badge ${c.status==='configured'?'confirmed':'pending'}">${c.status}</span>
<table><tr><th>provider</th><td>${esc(c.provider)}</td></tr>${c.mode&&c.mode!=='custom'?`<tr><th>mode</th><td>${esc(c.mode)}</td></tr>`:''}${c.model?`<tr><th>model</th><td class="mono">${esc(c.model)}</td></tr>`:''}<tr><th>base_url</th><td class="mono">${esc(c.base_url)}</td></tr><tr><th>key</th><td>${c.key_configured?'已配置':'未配置'}</td></tr><tr><th>说明</th><td class="muted">${esc(c.note||'')}</td></tr></table>
<div><button onclick="editConfig('${esc(c.name)}')">编辑配置</button> ${c.name==='embedding'?`<button onclick="testApi('embedding','etest')">测试连通</button><span id="etest" class="muted"></span>`:''}${c.name==='ocr'?`<button onclick="testApi('ocr','otest')">测试连通</button><span id="otest" class="muted"></span>`:''}${c.name==='code_graph'?`<button onclick="testApi('code_graph','gtest')">测试连通</button><span id="gtest" class="muted"></span>`:''}</div></div>`).join('')}
const CFG_FIELDS={
 embedding:[['provider','select',['openai_compatible','echo']],['base_url','text'],['model','text'],['api_key','password']],
 ocr:[['ocr_provider','select',['ask','api','paddle','none']],['ocr_api_mode','select',['custom','openai']],['ocr_api_base','text'],['ocr_api_model','text'],['ocr_api_key','password'],['ocr_min_confidence','text']],
 github:[['token','password']],
 code_graph:[['enabled','select',['true','false']],['base_url','text'],['path','text']]};
function fillCache(cs){window._cfgCache={};cs.forEach(c=>window._cfgCache[c.name]={provider:c.provider,base_url:c.base_url,model:c.model||'',mode:c.mode||'custom',ocr_provider:c.provider,ocr_api_base:c.base_url,ocr_api_model:c.model||'',ocr_api_mode:c.mode||'custom',ocr_min_confidence:c.min_confidence||'',enabled:c.enabled===true?'true':'false',path:c.path||'/gh-puller/graph'})}
async function editConfig(name){const old=document.getElementById('form-'+name);if(old)old.remove();
try{fillCache(await j('/api/configs'))}catch(e){} // 打开时拉最新配置，避免缓存/时序回填旧值
const c=CFG_FIELDS[name];const cur=window._cfgCache[name]||{};const html=c.map(([k,t,opts])=>`<div style="margin:4px 0"><label>${esc(k)}: </label>${t==='select'?`<select id="f-${k}">${opts.map(o=>`<option ${(cur[k]||'')===o?'selected':''}>${esc(o)}</option>`).join('')}</select>`:`<input id="f-${k}" type="${t}" placeholder="${t==='password'?'(留空不改)':'...'}" style="width:340px" ${cur[k]&&t!=='password'?`value="${esc(cur[k])}"`:''}>`}</div>`).join('');
document.getElementById('cfg-'+name).insertAdjacentHTML('beforeend',`<div class="card subtle" id="form-${name}">${html}<br><button class="ok" onclick="saveConfig('${name}')">保存</button> <button onclick="document.getElementById('form-${name}').remove()">取消</button></div>`)}
async function saveConfig(name){const fields={};CFG_FIELDS[name].forEach(([k])=>{const el=document.getElementById('f-'+k);if(el)fields[k]=el.value});
try{await j('/api/configs/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,fields})});
document.getElementById('form-'+name).remove();await loadConfigs();toast('✓ 已保存（状态已刷新）')}
catch(e){toast('✗ 保存失败: '+e.message)}}
function toast(msg){const t=document.getElementById('toast');t.textContent=msg;t.style.display='block';clearTimeout(t._h);t._h=setTimeout(()=>t.style.display='none',2500)}
// 审核人记忆（localStorage，避免每条审核重复填写）
function saveReviewer(v){try{localStorage.setItem('vllm_kb_reviewer',v||'')}catch(e){}}
function loadReviewer(){try{return localStorage.getItem('vllm_kb_reviewer')||''}catch(e){return ''}}
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
    def external_docs(limit: int = 200, offset: int = 0, q: str = ""):
        """外源文档列表（含标签视图/资产映射/重名告警；排除 GitHub 采集来源）。
        q：source_id/title 子串过滤——73k+ 文档远超单页窗口，必须可搜。"""
        from vllm_kb.review import list_external_docs
        return list_external_docs(cfg.resolve(cfg.storage.sqlite_path),
                                  review_db=default_review_path(cfg),
                                  limit=min(limit, 500), offset=offset, q=q)

    @app.get("/api/withdrawn")
    def withdrawn_list():
        """已回撤文档（删除墓碑）——痕迹视图；mirror 自动回撤的条目可撤回恢复。"""
        from vllm_kb.review import list_withdrawn
        return list_withdrawn(default_review_path(cfg))

    # ---------------- 文档级标签（两层分类：主题/领域 domain、具体作用 purpose） ----------------

    @app.get("/api/docs/tags")
    def docs_tags(source_id: str):
        """文档标签视图：auto（自动快照）/ excluded（排除项，可恢复）/ manual / final。"""
        from vllm_kb.review import doc_tags_view
        return doc_tags_view(cfg.resolve(cfg.storage.sqlite_path), source_id)

    @app.post("/api/docs/tags/edit")
    def docs_tags_edit(body: dict):
        """标签编辑：exclude（排除自动）/ restore（恢复）/ add（人工添加）/ remove（删除人工）。"""
        from vllm_kb.review import update_doc_tags

        source_id = (body.get("source_id") or "").strip()
        action = (body.get("action") or "").strip()
        tag = (body.get("tag") or "").strip()
        reviewer = (body.get("reviewer") or "").strip()
        if not source_id or not reviewer:
            raise HTTPException(400, "source_id/reviewer 必填")
        try:
            r = update_doc_tags(cfg.resolve(cfg.storage.sqlite_path), source_id,
                                action, tag, reviewer, config_path=_config_path)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, **r}

    @app.get("/api/tag-dict")
    def tag_dict_endpoint():
        """标签词典（registry）：按 tier 分组 + 文档计数（config.json 为唯一事实源）。"""
        from vllm_kb.review import tag_dict
        return tag_dict(cfg, cfg.resolve(cfg.storage.sqlite_path), config_path=_config_path)

    @app.post("/api/tag-dict/add")
    def tag_dict_add(body: dict):
        """新增词典标签（同步 config.json；重建图后入图——标准路径，不热插图）。"""
        from vllm_kb.review import add_tag_to_registry

        name = (body.get("name") or "").strip()
        if not name:
            raise HTTPException(400, "name 必填")
        try:
            r = add_tag_to_registry(cfg, name, body.get("tier"), config_path=_config_path)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, **r}

    @app.post("/api/tag-dict/rename")
    def tag_dict_rename(body: dict):
        """词典改名：registry + 全库替换 docs.tags / doc_tags 排除+人工项（同步 config）。"""
        from vllm_kb.review import rename_tag

        old = (body.get("old") or "").strip()
        new = (body.get("new") or "").strip()
        if not old or not new:
            raise HTTPException(400, "old/new 必填")
        try:
            r = rename_tag(cfg, old, new, cfg.resolve(cfg.storage.sqlite_path),
                           config_path=_config_path)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, **r}

    @app.post("/api/tag-dict/tier")
    def tag_dict_tier(body: dict):
        """修改标签层级（domain/purpose，全局生效）。"""
        from vllm_kb.review import set_tag_tier

        name = (body.get("name") or "").strip()
        tier = (body.get("tier") or "").strip()
        if not name:
            raise HTTPException(400, "name 必填")
        try:
            r = set_tag_tier(cfg, name, tier, config_path=_config_path)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, **r}

    @app.post("/api/tag-dict/delete")
    def tag_dict_delete(body: dict):
        """词典删除：仅移出词典（不动已打标文档）。"""
        from vllm_kb.review import delete_tag

        name = (body.get("name") or "").strip()
        if not name:
            raise HTTPException(400, "name 必填")
        try:
            r = delete_tag(cfg, name, config_path=_config_path)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, **r}

    @app.post("/api/tag-dict/batch")
    def tag_dict_batch(body: dict):
        """词典批量操作：delete=批量移出（已打标文档上的标签保留）| tier=批量改层级（全局生效）。

        逐项独立生效；不存在/非法项进 failed 明细。词典级操作不涉及文档审计
        （与单条 delete/tier 端点一致，不要求 reviewer）。
        """
        from vllm_kb.review import delete_tag, set_tag_tier

        action = (body.get("action") or "").strip()
        names = [s.strip() for s in (body.get("names") or []) if s and s.strip()]
        tier = (body.get("tier") or "").strip()
        if action not in ("delete", "tier"):
            raise HTTPException(400, "action 须为 delete|tier")
        if not names:
            raise HTTPException(400, "names 必填")
        if action == "tier" and tier not in ("domain", "purpose"):
            raise HTTPException(400, "tier 须为 domain|purpose")
        if len(names) > 200:
            raise HTTPException(400, "一次最多 200 个")
        done, failed = 0, []
        for name in names:
            try:
                ok = (delete_tag(cfg, name, config_path=_config_path) if action == "delete"
                      else set_tag_tier(cfg, name, tier, config_path=_config_path))
                if not ok:
                    failed.append({"name": name, "error": "词典中不存在"})
                    continue
                done += 1
            except ValueError as e:
                failed.append({"name": name, "error": str(e)})
            except Exception as e:
                failed.append({"name": name, "error": f"{type(e).__name__}: {e}"})
        return {"ok": True, "done": done, "failed": failed}

    @app.post("/api/tag-candidate/{item_id}/adopt")
    def tag_candidate_adopt(item_id: int, body: dict):
        """采纳候选：入词典（config）+ 写入该文档 manual（立即生效）。"""
        from vllm_kb.review import adopt_tag_candidate

        reviewer = (body.get("reviewer") or "").strip()
        if not reviewer:
            raise HTTPException(400, "reviewer 必填")
        try:
            r = adopt_tag_candidate(cfg, store, item_id, reviewer, tier=body.get("tier"),
                                    config_path=_config_path)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, **r}

    @app.post("/api/tag-candidates/batch")
    def tag_candidates_batch(body: dict):
        """候选批量分拣（消化积压）：adopt=采纳（入词典+全部提及文档打标）| ignore=忽略。

        逐项独立生效记审计；仅 pending 项可分拣（已审项报进 failed 明细）；
        tier 仅 adopt 时可整批覆盖（缺省用各候选建议 tier）。
        """
        from vllm_kb.review import adopt_tag_candidate

        action = (body.get("action") or "").strip()
        reviewer = (body.get("reviewer") or "").strip()
        tier = body.get("tier") or None
        item_ids = []
        for x in (body.get("item_ids") or []):
            try:
                item_ids.append(int(x))
            except (TypeError, ValueError):
                continue
        if action not in ("adopt", "ignore"):
            raise HTTPException(400, "action 须为 adopt|ignore")
        if not item_ids or not reviewer:
            raise HTTPException(400, "item_ids/reviewer 必填")
        if len(item_ids) > 200:
            raise HTTPException(400, "一次最多 200 项")
        done, failed = 0, []
        for iid in item_ids:
            try:
                it = store.get_item(iid)
                if it is None or it["category"] != "tag_candidate":
                    failed.append({"item_id": iid, "error": "候选项不存在"})
                    continue
                if it["status"] != "pending":
                    failed.append({"item_id": iid, "error": f"状态 {it['status']} 非 pending"})
                    continue
                if action == "adopt":
                    adopt_tag_candidate(cfg, store, iid, reviewer, tier=tier,
                                        config_path=_config_path)
                else:
                    store.review(iid, "approved", reviewer)
                done += 1
            except ValueError as e:
                failed.append({"item_id": iid, "error": str(e)})
            except Exception as e:
                failed.append({"item_id": iid, "error": f"{type(e).__name__}: {e}"})
        return {"ok": True, "done": done, "failed": failed}

    @app.post("/api/external-docs/delete")
    def external_doc_delete(body: dict):
        """彻底删除外源文档（数据库四层全删：docs + chunks_fts + chunks_meta + 向量），
        本地资产文件不动；登记删除墓碑——任何构建路径（增量/资产回退/--rebuild）都
        不再复活该 sid，撤回删除（待实际删除列表）解除。
        删除后联动修剪 tag_candidate 队列里的悬空引用（快照空则自动忽略该候选）。"""
        from vllm_kb.review import (delete_external_doc, prune_tag_candidates_for_doc,
                                    record_tombstones)
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
        review_db = default_review_path(cfg)
        cleaned = prune_tag_candidates_for_doc(review_db, source_id)
        # 墓碑：一等删除——增量/资产回退/--rebuild 都不再复活（撤回删除时解除）
        record_tombstones(review_db, [source_id])
        return {"ok": True, "candidates_cleaned": cleaned,
                "note": f"已从数据库删除 {source_id}（{r['chunks_deleted']} 个 chunk，本地文件保留）"}

    @app.post("/api/docs/tags/batch")
    def docs_tags_batch(body: dict):
        """批量标签编辑（文档管理多选/按目录全选）：逐篇复用单文档逻辑，记明细成败。

        不做跨文档整体事务——单文档内 doc_tags+docs.tags 已是原子；部分成功返回
        {updated, failed:[{source_id,error}]}，前端提示"N 成功 / M 失败"。
        reviewer 必填（审计一致性）；上限 200 篇防误操作放大。
        """
        from vllm_kb.review import update_doc_tags

        action = (body.get("action") or "").strip()
        tag = (body.get("tag") or "").strip()
        reviewer = (body.get("reviewer") or "").strip()
        ids = [s.strip() for s in (body.get("source_ids") or []) if s and s.strip()]
        if not ids or not reviewer:
            raise HTTPException(400, "source_ids/reviewer 必填")
        if len(ids) > 200:
            raise HTTPException(400, "一次最多 200 篇")
        import sqlite3

        kb = cfg.resolve(cfg.storage.sqlite_path)
        # 存在性预检：update_doc_tags 对不存在的 sid 会静默创建孤儿覆盖层
        # （单文档 UI 入口只列真实文档所以无感；批量必须先验，否则成功数虚高）
        conn = sqlite3.connect(str(kb))
        try:
            existing = {r[0] for r in conn.execute(
                f"SELECT source_id FROM docs WHERE source_id IN ({','.join('?' * len(ids))})",
                ids).fetchall()}
        finally:
            conn.close()
        updated, failed = 0, [{"source_id": s, "error": "kb.sqlite3 中不存在"}
                              for s in ids if s not in existing]
        for sid in ids:
            if sid not in existing:
                continue
            try:
                update_doc_tags(kb, sid, action, tag, reviewer, config_path=_config_path)
                updated += 1
            except ValueError as e:
                failed.append({"source_id": sid, "error": str(e)})
            except Exception as e:
                failed.append({"source_id": sid, "error": f"{type(e).__name__}: {e}"})
        return {"ok": True, "updated": updated, "failed": failed}

    @app.post("/api/external-docs/delete-batch")
    def external_docs_delete_batch(body: dict):
        """批量删除外源文档（数据库四层全删：docs + chunks_fts + chunks_meta + 向量，
        本地文件不动）。向量 store **只建一次**（逐篇重建是主要浪费）。
        删除后联动修剪 tag_candidate 悬空引用（逐篇）。"""
        from vllm_kb.review import (delete_external_doc, prune_tag_candidates_for_doc,
                                    record_tombstones)
        from vllm_kb.vectorstore import build_vector_store

        ids = [s.strip() for s in (body.get("source_ids") or []) if s and s.strip()]
        if not ids:
            raise HTTPException(400, "source_ids 必填")
        if len(ids) > 200:
            raise HTTPException(400, "一次最多 200 篇")
        kb_path = cfg.resolve(cfg.storage.sqlite_path)
        try:
            vector_store = build_vector_store(cfg)
        except Exception:
            vector_store = None
        review_db = default_review_path(cfg)
        deleted, failed = 0, []
        cleaned = {"pruned": 0, "auto_ignored": 0}
        for sid in ids:
            try:
                delete_external_doc(kb_path, sid, vector_store=vector_store)
                deleted += 1
                c = prune_tag_candidates_for_doc(review_db, sid)
                cleaned["pruned"] += c["pruned"]
                cleaned["auto_ignored"] += c["auto_ignored"]
            except ValueError as e:
                failed.append({"source_id": sid, "error": str(e)})
            except Exception as e:
                failed.append({"source_id": sid, "error": f"{type(e).__name__}: {e}"})
        # 墓碑：请求删除的 sid 全部登记（含"已不存在"的——防未来任何路径复活；
        # 失败项可重试，重试成功后墓碑语义自然成立）
        record_tombstones(review_db, ids)
        # 服务端真值校验（2026-10 用户报告前端/库 gap）：循环结束后回查库，
        # 报告"声称已删但仍存在"的 sid——前端据此持久告警，不再依赖瞬时 toast。
        still_present: list[str] = []
        if deleted:
            conn = sqlite3.connect(str(kb_path))
            try:
                qmarks = ",".join("?" * len(ids))
                want = set(ids)
                still_present = [r[0] for r in conn.execute(
                    f"SELECT source_id FROM docs WHERE source_id IN ({qmarks})", ids).fetchall()
                    if r[0] in want]
            finally:
                conn.close()
        return {"ok": True, "deleted": deleted, "failed": failed,
                "still_present": still_present, "candidates_cleaned": cleaned}

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

    @app.get("/api/doc-preview")
    def doc_preview(source_id: str, limit: int = 3000):
        """文档正文预览（审核详情用）：chunks 按 seq 拼装截断返回——审核人先看到内容再判定。

        只读、不落审计；表格文件项预览首行文档（body 含各 sheet 全表内容）。
        docs 表**不存正文**（正文在 chunks_fts.text，chunks_meta.seq 定序）。
        库不存在/无该文档/表结构不全时降级为空正文（不 500，前端显示降级提示）。
        """
        import sqlite3

        kb = cfg.resolve(cfg.storage.sqlite_path)
        if not kb.exists():
            return {"source_id": source_id, "title": "", "body": "", "truncated": False}
        conn = sqlite3.connect(f"file:{kb.as_posix()}?mode=ro", uri=True)
        try:
            try:
                row = conn.execute("SELECT title FROM docs WHERE source_id=?",
                                   (source_id,)).fetchone()
                texts = [r[0] for r in conn.execute(
                    "SELECT f.text FROM chunks_meta m JOIN chunks_fts f "
                    "ON f.chunk_id = m.chunk_id WHERE m.doc_id=? ORDER BY m.seq",
                    (source_id,)).fetchall()]
            except sqlite3.OperationalError:
                return {"source_id": source_id, "title": "", "body": "", "truncated": False}
        finally:
            conn.close()
        if row is None:
            raise HTTPException(404, f"文档不存在: {source_id}")
        body = "\n\n".join(t for t in texts if t)
        n = max(200, min(int(limit), 20000))
        return {"source_id": source_id, "title": row[0] or "",
                "body": body[:n], "truncated": len(body) > n}

    @app.post("/api/item/{item_id}/review")
    def review(item_id: int, body: dict):
        """标注（不改原始内容）：approved 认证 | suspected 存疑（重新排队靠后）。

        **补标项（verification_pending）特殊**：认证要真的改可靠度——写覆盖层
        （doc_verification）+ 立即刷新 docs.extra.verification，否则点了认证 w_rel 仍停在
        unverified 的 0.5（见 vllm_kb.review.review_verification）。
        """
        action = body.get("action", "")
        reviewer = (body.get("reviewer") or "").strip()
        if not reviewer:
            raise HTTPException(400, "reviewer 必填")
        try:
            it = store.get_item(item_id)
            if it is None:
                raise HTTPException(404, "审核项不存在")
            if it["category"] == "verification_pending":
                from vllm_kb.review import review_verification

                r = review_verification(cfg, store, item_id, action, reviewer,
                                        result=body.get("result"))
                return {"ok": True, **r}
            ok = store.review(item_id, action, reviewer, result=body.get("result"))
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not ok:
            raise HTTPException(404, "审核项不存在")
        return {"ok": True}

    @app.post("/api/items/review-batch")
    def items_review_batch(body: dict):
        """批量审核（审核队列多选）：补标项走 review_verification（写覆盖层+立即刷新
        docs.extra.verification，否则 w_rel 仍停在 unverified 0.5），其余走 store.review。
        逐项记成败不整体回滚；reviewer 必填；上限 200 项。"""
        action = body.get("action", "")
        reviewer = (body.get("reviewer") or "").strip()
        raw_ids = body.get("item_ids") or []
        item_ids = []
        for x in raw_ids:
            try:
                item_ids.append(int(x))
            except (TypeError, ValueError):
                continue
        if not item_ids or not reviewer:
            raise HTTPException(400, "item_ids/reviewer 必填")
        if len(item_ids) > 200:
            raise HTTPException(400, "一次最多 200 项")
        reviewed, failed = 0, []
        for iid in item_ids:
            try:
                it = store.get_item(iid)
                if it is None:
                    failed.append({"item_id": iid, "error": "审核项不存在"})
                    continue
                if it["category"] == "verification_pending":
                    from vllm_kb.review import review_verification

                    review_verification(cfg, store, iid, action, reviewer,
                                        result=body.get("result"))
                else:
                    if not store.review(iid, action, reviewer, result=body.get("result")):
                        failed.append({"item_id": iid, "error": "审核项不存在"})
                        continue
                reviewed += 1
            except ValueError as e:
                failed.append({"item_id": iid, "error": str(e)})
            except Exception as e:
                failed.append({"item_id": iid, "error": f"{type(e).__name__}: {e}"})
        return {"ok": True, "reviewed": reviewed, "failed": failed}

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

    @app.get("/api/gaps")
    def gaps():
        """知识缺口列表（行为遥测推断：同签名跨会话反复查无果）。

        读 telemetry 库的 knowledge_gaps 表（build_feedback.py 产出）。
        feedback 未启用或表不存在时返回空列表。
        """
        import sqlite3

        from pathlib import Path

        db = cfg.resolve(cfg.confidence.telemetry_path)
        if not Path(str(db)).exists():
            return []
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        try:
            try:
                rows = conn.execute(
                    "SELECT * FROM knowledge_gaps ORDER BY session_count DESC"
                ).fetchall()
            except sqlite3.OperationalError:
                return []  # 表不存在
            return [dict(r) for r in rows]
        finally:
            conn.close()

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
        # ValueError（未知配置段/缺来源等）转 400 JSON——不接会变纯文本 500，
        # 前端 j() 的 r.json() 解析失败报 "Unexpected token"（code_graph 曾中招）
        try:
            if name == "embedding":
                update_config_json(cfg, "embedding",
                                   {k: fields[k] for k in ("provider", "base_url", "model")
                                    if k in fields}, config_path=_config_path)
                if "api_key" in fields:
                    save_secret(cfg, "EMBEDDING_API_KEY", fields.get("api_key") or "")
            elif name == "ocr":
                update_config_json(cfg, "ocr",
                                   {k: fields[k] for k in ("ocr_provider", "ocr_api_base",
                                                           "ocr_api_model", "ocr_api_mode",
                                                           "ocr_min_confidence")
                                    if k in fields},
                                   config_path=_config_path)
                if "ocr_api_key" in fields:
                    save_secret(cfg, "OCR_API_KEY", fields.get("ocr_api_key") or "")
            elif name == "github":
                if "token" in fields:
                    save_secret(cfg, "GITHUB_TOKEN", fields.get("token") or "")
            elif name == "code_graph":
                upd = {k: fields[k] for k in ("enabled", "base_url", "path") if k in fields}
                # enabled 是 bool，前端传 "true"/"false" 字符串
                if "enabled" in upd:
                    upd["enabled"] = upd["enabled"] in ("true", "True", "1", True)
                update_config_json(cfg, "code_graph", upd, config_path=_config_path)
            else:
                raise HTTPException(400, f"未知配置: {name}")
        except ValueError as e:
            raise HTTPException(400, str(e))
        # 重载内存 cfg（读 config.json + secrets），/api/configs 立即反映
        cfg = AppConfig.load(_config_path, require_keys=False) if _config_path \
            else AppConfig.load(None, require_keys=False)
        return {"ok": True, "note": "已保存。密钥经 secrets 文件；非密钥字段对已运行的 serve_api 需重启生效"}

    @app.post("/api/configs/test")
    def test_config(body: dict):
        """连通性测试：embedding 发真实嵌入请求；OCR 用内置测试图走真实识别链路。"""
        from vllm_kb.review import probe_ocr_connectivity

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
                r = probe_ocr_connectivity(cfg)
            except ValueError as e:
                raise HTTPException(400, str(e))
            if not r["ok"]:
                raise HTTPException(502, r["detail"])
            return r
        if name == "code_graph":
            try:
                from vllm_kb.code_graph import CodeGraphClient

                r = CodeGraphClient(cfg.code_graph).health()
                if r["status"] == "ok":
                    return {"ok": True, "detail": "连通（gh-puller 可达）"}
                raise HTTPException(502, f"不可达: {r.get('status')} {r.get('error', '')}")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(502, f"代码图谱服务不可用: {e}")
        raise HTTPException(400, "仅支持 embedding / ocr / code_graph 连通性测试")

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
