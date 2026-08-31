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
.tagb{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;margin:2px 4px 2px 0;cursor:pointer;border:1px solid transparent}
.tagb.domain{background:#dbeafe;color:#1e40af}.tagb.purpose{background:#dcfce7;color:#166534}
.tagb.auto{border-style:dashed}.tagb.manual{background:#fef3c7;color:#92400e}
.tagb.excluded{background:#fee2e2;color:#991b1b;text-decoration:line-through}
.tagb.reg{background:#f3f4f6;color:#374151;cursor:default}
.warn{color:#b45309;font-size:12px}
</style></head><body>
<header><strong>vllm-kb 审核工作台</strong>
<a id="nav-stats" class="active" onclick="tab('stats')">概览</a>
<a id="nav-queue" onclick="tab('queue')">审核队列</a>
<a id="nav-docs" onclick="tab('docs')">文档管理</a>
<a id="nav-tags" onclick="tab('tags')">标签管理</a>
<a id="nav-configs" onclick="tab('configs')">API 配置</a></header>
<div id="toast" style="position:fixed;top:12px;right:20px;background:#111827;color:#fff;padding:8px 14px;border-radius:6px;display:none;z-index:99;font-size:13px"></div>
<main>
<div id="view-stats"></div><div id="view-queue" style="display:none"></div><div id="view-docs" style="display:none"></div><div id="view-tags" style="display:none"></div><div id="view-configs" style="display:none"></div>
</main>
<script>
const $=s=>document.querySelector(s);
async function j(url,opt){const r=await fetch(url,opt);const d=await r.json();if(!r.ok)throw new Error(d.detail||r.status);return d}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function badge(s){return `<span class="badge ${s}">${s}</span>`}
function tab(name){['stats','queue','docs','tags','configs'].forEach(t=>{document.getElementById('view-'+t).style.display=t===name?'':'none';document.getElementById('nav-'+t).className=t===name?'active':''});if(name==='stats')loadStats();if(name==='queue')loadQueue();if(name==='docs')loadDocs();if(name==='tags')loadTagDict();if(name==='configs')loadConfigs()}
async function loadStats(){const s=await j('/api/stats');const cats=Object.entries(s).sort((a,b)=>b[1].pending-a[1].pending);
let tagCard='';try{const td=await j('/api/tag-dict');tagCard=`<div class="card"><b>标签词典</b> 领域类 ${td.stats.domain} 个 · 作用类 ${td.stats.purpose} 个 · 已打标文档 ${td.stats.tagged_docs} 篇</div>`}catch(e){}
$('#view-stats').innerHTML=`<h3>待办概览（${cats.reduce((a,[k,v])=>a+v.pending,0)} 待办 / ${cats.reduce((a,[k,v])=>a+v.suspected,0)} 存疑）</h3>`+tagCard+cats.map(([k,v])=>`<div class="card"><b>${esc(k)}</b> <span class="badge pending">待办 ${v.pending}</span>${v.suspected?`<span class="badge suspected">存疑 ${v.suspected}</span>`:''} <span class="muted">共 ${v.total}</span></div>`).join('')||'<div class="card">暂无审核项（导入文档后自动补单）</div>'}
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
${i.category==='tag_candidate'?`<p><b>候选标签:</b> <span class="tagb reg">${esc(p.candidate||'')}</span> 建议tier: ${esc(p.suggested_tier||'自动判定')}（提及文档 <b>${p.doc_count||1}</b> 篇：${esc((p.docs||[p]).map(d=>d.title||d.source_id||'').slice(0,5).join('、')||'')}${(p.doc_count||1)>5?' …':''}）</p>`:''}
<pre>${esc(JSON.stringify(p,null,2))}</pre>
<h4>审核（只做判定，不修改原始内容）</h4><input id="rev" placeholder="审核人（必填）" style="width:180px" oninput="saveReviewer(this.value)">
<textarea id="rvnote" rows="2" style="width:100%" placeholder="备注（可选）"></textarea><br><br>
${i.category==='tag_candidate'?`tier: <select id="adopt-tier"><option value="">自动判定</option><option value="domain">领域</option><option value="purpose">作用</option></select>
<button class="ok" onclick="adoptCandidate(${i.id})">✓ 采纳为标签（入词典+全部提及文档打标，立即生效）</button><br><br>`:''}
<button class="ok" onclick="review(${i.id},'approved')">✓ 认证</button>
<button onclick="review(${i.id},'suspected')">？ 存疑（重新排队靠后）</button>
<button class="danger" onclick="deleteDoc(${i.id})">🗑 标记删除（只删数据库记录，原始文件手动删）</button>
<span class="muted">存疑项排在未审核之后；删除可在队列底部'待实际删除'撤回；tag_candidate 按词聚合（同词多文档一条），忽略/采纳后不再出现</span></div>`
$('#rev').value=loadReviewer()}
async function adoptCandidate(id){const reviewer=document.getElementById('rev').value;if(!reviewer){alert('请填写审核人');return}
try{await j('/api/tag-candidate/'+id+'/adopt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reviewer,tier:document.getElementById('adopt-tier').value||null})});loadQueue();toast('✓ 已采纳（词典已同步，重建图后入图）')}catch(e){toast('✗ '+e.message)}}
async function review(id,action){const reviewer=document.getElementById('rev').value;if(!reviewer){alert('请填写审核人');return}let result={note:document.getElementById('rvnote').value};try{result=JSON.parse(result.note||'{}')}catch(e){}
await j('/api/item/'+id+'/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,reviewer,result})});loadQueue()}
async function deleteDoc(id){const reviewer=document.getElementById('rev').value;if(!reviewer){alert('请填写审核人');return}
if(!confirm('标记删除：只删除数据库记录，原始资产文件需到"待实际删除"列表手动本地删除。确认？'))return
await j('/api/item/'+id+'/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reviewer,note:document.getElementById('rvnote').value})});loadQueue()}
async function undoDelete(id){if(!confirm('撤回删除：恢复数据库记录并重新加入队列？'))return;await j('/api/item/'+id+'/undo',{method:'POST'});loadQueue()}
async function loadDocs(){const ds=await j('/api/external-docs?limit=500');
let td={};try{const t=await j('/api/tag-dict');t.groups.domain.forEach(x=>td[x.name]='domain');t.groups.purpose.forEach(x=>td[x.name]='purpose')}catch(e){}
const tierOf=n=>td[n]||'domain';
const q=s=>esc(String(s||'')).replace(/'/g,"\\'");
const tagB=(sid,t,cls,act,mark)=>`<span class="tagb ${cls} ${tierOf(t)}" onclick="tagEdit('${q(sid)}','${act}','${q(t)}')">${esc(t)} ${mark}</span>`;
const groupRow=(list,cls,act,mark,sid)=>list.length?`<div style="margin-top:4px"><span class="muted">${cls==='excluded'?'已排除':cls==='manual'?'人工':'自动'}:</span> ${list.map(t=>tagB(sid,t,cls,act,mark)).join('')}</div>`:'';
$('#view-docs').innerHTML=`<h3>文档管理 — 外源文档（导入的 PDF/MD/表格等，共 ${ds.length} 条）</h3>
<div class="card muted">删除只动数据库（docs + chunks + 向量），<b>本地文件不动</b>；下次增量入库时文件仍在本地会重新入库，文档废弃请手动删除本地文件。GitHub 采集文档不在此列。<br>标签：<b>虚线框=自动标签</b>（点 ✕ 排除）、<b>黄底=人工标签</b>、<b>红底删除线=已排除</b>（点 ↺ 恢复）。最终标签 = (自动 − 排除) ∪ 人工，与入库/建图一致。</div>`+ds.map(d=>{const t=d.tags||{};
return `<div class="card" style="border-left:3px solid ${d.duplicate?'#b45309':'#d1d5db'}"><div><b>${esc(d.title||d.source_id)}</b> <span class="badge">${esc(d.source_type)}</span> ${d.duplicate?`<span class="warn">⚠ 同 stem 重名（人工处理，不自动消歧）</span>`:''}</div>
<div class="mono muted">${esc(d.source_id)}${d.asset_path?` · ${esc(d.asset_path)}`:''}</div>
${groupRow(t.auto||[],'auto','exclude','✕',d.source_id)}
${groupRow(t.excluded||[],'excluded','restore','↺',d.source_id)}
${groupRow(t.manual||[],'manual','remove','✕',d.source_id)}
<div style="margin-top:6px"><input id="newtag-${q(d.source_id).replace(/[^a-zA-Z0-9]/g,'_')}" placeholder="添加标签" style="width:180px"><button onclick="tagAdd('${q(d.source_id)}',document.getElementById('newtag-${q(d.source_id).replace(/[^a-zA-Z0-9]/g,'_')}').value)">添加</button>
<span class="muted">最终: ${(t.final||[]).map(esc).join(' · ')||'（无标签）'}</span></div>
<div style="margin-top:4px"><button class="danger" onclick="delExternalDoc('${q(d.source_id)}')">🗑 从数据库删除（本地文件保留）</button></div></div>`}).join('')||'<div class="card muted">无外源文档</div>'}
async function tagEdit(sid,action,tag){const reviewer=prompt('审核人（必填）',loadReviewer());if(!reviewer)return;
if(reviewer!==loadReviewer())saveReviewer(reviewer);
try{await j('/api/docs/tags/edit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source_id:sid,action,tag,reviewer})});loadDocs();toast('✓ 已更新标签')}catch(e){toast('✗ '+e.message)}}
function tagAdd(sid,t){if(!t||!t.trim())return;tagEdit(sid,'add',t.trim())}
async function delExternalDoc(sid){if(!confirm('从数据库彻底删除该文档（docs+chunks+向量）？本地文件保留，下次增量入库会重新入库。确认？'))return
try{await j('/api/external-docs/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source_id:sid})});loadDocs();toast('✓ 已删除（本地文件保留）')}
catch(e){toast('✗ 删除失败: '+e.message)}}
async function loadTagDict(){const t=await j('/api/tag-dict');
const groupHtml=(g,title)=>`<h4 style="margin:10px 0 4px">${title}（${g.length}）</h4><div class="card">${g.map(x=>`<span class="tagb reg">${esc(x.name)} (${x.docs}篇)</span> <button style="padding:2px 8px" onclick="tagDictForm('${esc(x.name).replace(/'/g,"\\'")}')">编辑</button>`).join('')||'（空）'}</div>`;
$('#view-tags').innerHTML=`<h3>标签词典（config.json tags.registry，全局唯一事实源）</h3>
<div class="card muted">两级分类：领域类=这是什么领域的知识（过滤/圈定范围）；作用类=文档能帮我做什么（能力匹配）。新增/改名/改 tier 同步 config.json；<b>不热插图</b>——运行 build_graph.py 重建图后入图（Kùzu 单写者约束）。</div>
${groupHtml(t.groups.domain||[],'主题/领域类 (domain)')}
${groupHtml(t.groups.purpose||[],'具体作用类 (purpose)')}
<div class="card"><b>新增标签</b> <input id="newtag-name" placeholder="标签名" style="width:180px"> tier: <select id="newtag-tier"><option value="">自动判定</option><option value="domain">领域</option><option value="purpose">作用</option></select>
<button class="ok" onclick="tagDictAdd()">加入词典</button></div>`}
async function tagDictAdd(){const name=document.getElementById('newtag-name').value.trim();if(!name){alert('输入标签名');return}
try{await j('/api/tag-dict/add',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,tier:document.getElementById('newtag-tier').value||null})});loadTagDict();toast('✓ 已加入词典（重建图后入图）')}catch(e){toast('✗ '+e.message)}}
function tagDictForm(name){const old=document.getElementById('tdform');if(old)old.remove();
const div=document.createElement('div');div.id='tdform';div.className='card';div.style.background='#f9fafb';
div.innerHTML=`<b>编辑: ${esc(name)}</b><br>新名: <input id="td-new" placeholder="${esc(name)}" style="width:140px">
tier: <select id="td-tier"><option value="">不变</option><option value="domain">领域</option><option value="purpose">作用</option></select>
<button class="ok" onclick="tagDictRename('${esc(name).replace(/'/g,"\\'")}')">改名</button>
<button onclick="tagDictTier('${esc(name).replace(/'/g,"\\'")}')">改 tier</button>
<button class="danger" onclick="tagDictDelete('${esc(name).replace(/'/g,"\\'")}')">删除</button>`;
document.getElementById('view-tags').appendChild(div)}
async function tagDictRename(name){const nv=document.getElementById('td-new').value.trim();if(!nv){alert('输入新名');return}
try{const r=await j('/api/tag-dict/rename',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({old:name,new:nv})});loadTagDict();toast(`✓ 已改名（${r.docs_updated} 篇受影响）`)}catch(e){toast('✗ '+e.message)}}
async function tagDictTier(name){const tv=document.getElementById('td-tier').value;if(!tv){alert('选择 tier');return}
try{await j('/api/tag-dict/tier',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,tier:tv})});loadTagDict();toast('✓ tier 已更新（全局生效）')}catch(e){toast('✗ '+e.message)}}
async function tagDictDelete(name){if(!confirm(`从词典删除 ${name}？只移出词典，已打标文档上的该标签保留。`))return
try{await j('/api/tag-dict/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});loadTagDict();toast('✓ 已移出词典')}catch(e){toast('✗ '+e.message)}}
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
    def external_docs(limit: int = 200, offset: int = 0):
        """外源文档列表（含标签视图/资产映射/重名告警；排除 GitHub 采集来源）。"""
        from vllm_kb.review import list_external_docs
        return list_external_docs(cfg.resolve(cfg.storage.sqlite_path),
                                  review_db=default_review_path(cfg),
                                  limit=min(limit, 500), offset=offset)

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
