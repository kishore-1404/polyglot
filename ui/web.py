"""ui/web.py — FastAPI + WebSocket web UI.

text_ready → show transcript + response immediately (TTS pending badge)
turn        → update latency, remove pending badge, show status badges
control     → receive {"type":"control","action":"mute_toggle"} from browser
"""
import asyncio, json, threading
from typing import Set, Optional, Callable

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Polyglot Live</title>
<style>
:root{
  --bg:#0d1117;--bg2:#161b22;--bg3:#1f2937;--border:#30363d;
  --text:#e6edf3;--dim:#8b949e;
  --green:#3fb950;--yellow:#d29922;--red:#f85149;--cyan:#58a6ff;--purple:#bc8cff;--mag:#e040fb;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
     display:flex;flex-direction:column;height:100vh;overflow:hidden}

header{display:flex;align-items:center;gap:12px;padding:9px 16px;
       background:var(--bg2);border-bottom:1px solid var(--border);flex-wrap:wrap}
header h1{font-size:1rem;font-weight:700;color:var(--cyan);flex:1}
#status-badge{display:flex;align-items:center;gap:7px;font-size:.85rem}
#dot{width:9px;height:9px;border-radius:50%;background:var(--dim);transition:all .2s}
#dot.recording {background:var(--red);   box-shadow:0 0 7px var(--red);  animation:pulse .8s infinite}
#dot.processing{background:var(--yellow);box-shadow:0 0 7px var(--yellow)}
#dot.speaking  {background:var(--green); box-shadow:0 0 7px var(--green)}
#dot.muted     {background:var(--mag);   box-shadow:0 0 7px var(--mag)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

/* Mute button */
#mute-btn{
  padding:5px 12px;border-radius:5px;border:1px solid var(--border);
  background:var(--bg3);color:var(--text);cursor:pointer;font-size:.82rem;
  transition:all .2s;white-space:nowrap;
}
#mute-btn:hover{border-color:var(--cyan);color:var(--cyan)}
#mute-btn.muted{background:rgba(224,64,251,.15);border-color:var(--mag);color:var(--mag)}
.langs{font-size:.78rem;color:var(--dim)}

main{display:flex;flex:1;overflow:hidden}

/* Conversation */
#conv{flex:2;display:flex;flex-direction:column;border-right:1px solid var(--border)}
.ptitle{padding:7px 14px;font-size:.72rem;color:var(--dim);border-bottom:1px solid var(--border);
        text-transform:uppercase;letter-spacing:.1em}
#cscroll{flex:1;overflow-y:auto;padding:10px 14px;display:flex;flex-direction:column;gap:8px}
#cscroll::-webkit-scrollbar{width:3px}
#cscroll::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.turn{animation:fadein .15s ease}
@keyframes fadein{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.msg{display:flex;gap:7px;align-items:flex-start;margin-bottom:2px}
.role{font-size:.68rem;font-weight:700;opacity:.55;min-width:26px;padding-top:2px;text-transform:uppercase}
.flag{font-size:1rem}
.txt{font-size:.88rem;line-height:1.5}
.you-txt{color:var(--text)}
.bot-txt{font-weight:500}
.lang-en .bot-txt{color:var(--cyan)}
.lang-hi .bot-txt{color:var(--green)}
.lang-es .bot-txt{color:#f0b429}
.badge{display:inline-flex;align-items:center;gap:3px;font-size:.7rem;padding:1px 5px;
       border-radius:3px;margin-left:5px;vertical-align:middle}
.badge-pending{background:rgba(88,166,255,.1);color:var(--cyan)}
.badge-inter  {background:rgba(210,153,34,.12);color:var(--yellow)}
.badge-notts  {background:rgba(248,81,73,.12); color:var(--red)}
.badge-notrans{background:rgba(139,148,158,.12);color:var(--dim)}

/* Right panel */
#right{flex:1;min-width:290px;display:flex;flex-direction:column;overflow-y:auto}

/* Current latency */
#lat-cur{padding:12px 14px;border-bottom:1px solid var(--border)}
.lrow{display:flex;flex-direction:column;gap:2px;margin-bottom:8px}
.llab{display:flex;justify-content:space-between;font-size:.73rem}
.llab span:first-child{color:var(--dim)}
.lval{font-weight:700;font-variant-numeric:tabular-nums}
.lbg{background:var(--bg3);border-radius:3px;height:6px;position:relative}
.lfill{height:100%;border-radius:3px;transition:width .35s ease,background .25s}
.tmark{position:absolute;top:-3px;bottom:-3px;width:1px;background:rgba(255,255,255,.15)}
.ldiv{border:none;border-top:1px solid var(--border);margin:5px 0}
#e2e-badge{text-align:center;font-size:.78rem;font-weight:700;padding:4px 8px;
           border-radius:5px;margin-top:3px;transition:all .3s}
#e2e-badge.stretch{background:rgba(63,185,80,.15);color:var(--green)}
#e2e-badge.pass   {background:rgba(210,153,34,.15);color:var(--yellow)}
#e2e-badge.slow   {background:rgba(248,81,73,.15); color:var(--red)}
#e2e-badge.pending{background:var(--bg3);color:var(--dim)}
.no-data{color:var(--dim);font-size:.82rem;font-style:italic;padding:6px 0}

/* History */
#lat-hist{padding:10px 14px}
.hist-hdr{font-size:.7rem;color:var(--dim);text-transform:uppercase;
          letter-spacing:.08em;margin-bottom:6px}
#htbl{width:100%;border-collapse:collapse;font-size:.75rem;font-variant-numeric:tabular-nums}
#htbl th{color:var(--dim);text-align:right;padding:1px 5px;font-weight:400}
#htbl th:first-child{text-align:left}
#htbl td{text-align:right;padding:2px 5px;border-bottom:1px solid var(--border)}
#htbl td:first-child{text-align:left;color:var(--dim)}
</style>
</head>
<body>
<header>
  <h1>🌍 Polyglot Live</h1>
  <div id="status-badge"><div id="dot" class="listening"></div><span id="stxt">Listening…</span></div>
  <button id="mute-btn" onclick="toggleMute()">🎙️  Mute</button>
  <div class="langs">🇬🇧 EN &nbsp;·&nbsp; 🇮🇳 HI &nbsp;·&nbsp; 🇪🇸 ES</div>
</header>
<main>
  <div id="conv">
    <div class="ptitle">Conversation</div>
    <div id="cscroll"><p class="no-data">Waiting for speech…</p></div>
  </div>
  <div id="right">
    <div class="ptitle">Demo Mode Controls</div>
    <div id="demo-ctrls" style="padding:12px 14px;border-bottom:1px solid var(--border)">
      <div style="display:flex;flex-direction:column;gap:8px">
        <div style="display:flex;gap:8px;align-items:center">
          <select id="demo-scenario-select" style="flex:1;background:var(--bg3);color:var(--text);border:1px solid var(--border);padding:6px;border-radius:4px;font-size:.82rem">
            <option value="1">Scenario 1: Customer Support (Order Status)</option>
            <option value="2">Scenario 2: Travel Booking</option>
            <option value="4">Scenario 4: Rapid Switching</option>
          </select>
        </div>
        <div style="display:flex;align-items:center;gap:6px;font-size:.8rem">
          <input type="checkbox" id="demo-auto-checkbox" checked>
          <label for="demo-auto-checkbox">Auto-Advance turns</label>
        </div>
        <div style="display:flex;gap:6px">
          <button id="demo-start-btn" onclick="startDemo()" style="flex:1;padding:6px;background:var(--green);color:white;border:none;border-radius:4px;cursor:pointer;font-size:.82rem;font-weight:700">▶ Start Demo</button>
          <button id="demo-stop-btn" onclick="stopDemo()" disabled style="flex:1;padding:6px;background:var(--red);color:white;border:none;border-radius:4px;cursor:pointer;font-size:.82rem;font-weight:700;opacity:0.5">■ Stop Demo</button>
        </div>
        <button id="demo-next-btn" onclick="nextDemoTurn()" disabled style="padding:8px;background:var(--cyan);color:white;border:none;border-radius:4px;cursor:pointer;font-size:.82rem;font-weight:700;opacity:0.5">⏭ Next Turn</button>
        
        <div id="demo-state-panel" style="display:none;background:var(--bg2);border:1px solid var(--border);border-radius:4px;padding:8px;margin-top:6px;font-size:.78rem">
          <div style="font-weight:700;color:var(--yellow);margin-bottom:4px" id="demo-state-title">Demo Active</div>
          <div style="color:var(--dim);margin-bottom:4px" id="demo-turn-progress">Turn 0 / 0</div>
          <div style="border-top:1px solid var(--border);margin:4px 0"></div>
          <div style="font-weight:600">Next Script Turn:</div>
          <div style="font-style:italic;color:var(--cyan);margin-top:2px" id="demo-next-text">—</div>
        </div>
      </div>
    </div>
    <div class="ptitle">Latency — current turn</div>
    <div id="lat-cur">
      <p class="no-data" id="lph">No turns yet.</p>

      <div id="lrows" style="display:none">
        <div class="lrow">
          <div class="llab"><span>LLM TTFT</span><span class="lval" id="v-ttft">—</span></div>
          <div class="lbg"><div class="lfill" id="b-ttft" style="width:0;background:var(--cyan)"></div>
            <div class="tmark" style="left:66.6%"></div></div></div>
        <div class="lrow">
          <div class="llab"><span>LLM Total</span><span class="lval" id="v-llm">—</span></div>
          <div class="lbg"><div class="lfill" id="b-llm" style="width:0;background:var(--purple)"></div>
            <div class="tmark" style="left:66.6%"></div></div></div>
        <div class="lrow">
          <div class="llab"><span>TTS Synth</span><span class="lval" id="v-tts">—</span></div>
          <div class="lbg"><div class="lfill" id="b-tts" style="width:0;background:var(--green)"></div></div></div>
        <hr class="ldiv">
        <div class="lrow">
          <div class="llab"><span><b>E2E</b></span><span class="lval" id="v-e2e">—</span></div>
          <div class="lbg" style="height:10px">
            <div class="lfill" id="b-e2e" style="width:0;height:10px;background:var(--cyan)"></div>
            <div class="tmark" style="left:66.6%" title="800ms"></div>
            <div class="tmark" style="left:100%" title="1200ms"></div></div></div>
        <div id="e2e-badge" class="pending">—</div>
      </div>
    </div>
    <div id="lat-hist">
      <div class="hist-hdr">Latency History</div>
      <p class="no-data" id="hph">No turns yet.</p>
      <table id="htbl" style="display:none">
        <thead><tr><th>#</th><th>TTFT</th><th>LLM</th><th>TTS</th><th>E2E</th><th></th></tr></thead>
        <tbody id="hbody"></tbody>
      </table>
    </div>
  </div>
</main>
<script>
const FLAGS={en:'🇬🇧',hi:'🇮🇳',es:'🇪🇸'};
const SL={listening:'🎙️  Listening…',recording:'🔴  Recording…',
          processing:'🧠  Processing…',speaking:'🔊  Speaking…',muted:'🔇  Muted'};
const MAX=1200, conv=document.getElementById('cscroll');
const dot=document.getElementById('dot'), stxt=document.getElementById('stxt');
const lrows=document.getElementById('lrows'), lph=document.getElementById('lph');
const htbl=document.getElementById('htbl'), hbody=document.getElementById('hbody');
const hph=document.getElementById('hph');
let muted=false, pendingTag=null, turnN=0;

const ws=new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
ws.onmessage=e=>{
  const m=JSON.parse(e.data);
  if(m.type==='state')       handleState(m);
  if(m.type==='text_ready')  handleTextReady(m);
  if(m.type==='turn')        handleTurn(m);
  if(m.type==='latency_update') handleLatencyUpdate(m);
  if(m.type==='mute_state')  syncMuteBtn(m.muted);
  if(m.type==='demo_state')  handleDemoState(m.value);
  if(m.type==='log')         console.log('[pipeline]', m.message);
};

ws.onclose=()=>{stxt.textContent='⚠️  Disconnected';dot.className=''};

function handleState(m){dot.className=m.value;stxt.textContent=SL[m.value]||m.value}

function handleLatencyUpdate(m){
  if(m.latency) updateLat(m.latency, m.latency.status==='pending');
}

function handleTextReady(m){
  const nd=conv.querySelector('.no-data'); if(nd) nd.remove();
  const {lang,transcript,response}=m;
  const flag=FLAGS[lang]||'🌍';
  const id=`turn-${++turnN}`;
  const div=document.createElement('div');
  div.className=`turn lang-${lang}`;
  div.id=id;
  // You row
  const youTxt = transcript
    ? `<span class="txt you-txt">${esc(transcript)}</span>`
    : `<span class="txt you-txt" style="color:var(--dim);font-style:italic">(no transcript) <span style="color:var(--yellow)">⚠</span></span>`;
  div.innerHTML+=`<div class="msg"><span class="role">You</span><span class="flag">${flag}</span>${youTxt}</div>`;
  // Bot row with pending badge
  div.innerHTML+=`<div class="msg"><span class="role">Bot</span><span class="flag">${flag}</span>
    <span class="txt bot-txt">${esc(response)}</span>
    <span class="badge badge-pending" id="ptag-${turnN}">🔊 speaking</span></div>`;
  conv.appendChild(div);
  conv.scrollTop=conv.scrollHeight;
  pendingTag={n:turnN};
  // Partial latency
  if(m.latency) {
    const isPending = (m.latency.total_ms === null);
    updateLat(m.latency, isPending);
  }
}

function handleTurn(m){
  // Remove pending badge
  if(pendingTag){
    const tag=document.getElementById(`ptag-${pendingTag.n}`);
    if(tag) tag.remove();
    // Add status badges
    const bot=document.querySelector(`#turn-${pendingTag.n} .bot-txt`);
    if(bot){
      if(m.interrupted) bot.insertAdjacentHTML('afterend','<span class="badge badge-inter">⚡ interrupted</span>');
      if(m.tts_ok===false) bot.insertAdjacentHTML('afterend','<span class="badge badge-notts">🔇 no audio</span>');
    }
    pendingTag=null;
  }
  if(m.latency) updateLat(m.latency, false);
  addHistRow(m);
}

function updateLat(lat, pending){
  lph.style.display='none'; lrows.style.display='block';
  setBar('ttft',lat.llm_ttft_ms, MAX,'var(--cyan)');
  setBar('llm', lat.llm_total_ms,MAX,'var(--purple)');
  setBar('tts', lat.tts_ms,      MAX,'var(--green)');
  if(pending){
    document.getElementById('v-tts').textContent='…';
    setBar('e2e',null,MAX,'var(--cyan)');
    const b=document.getElementById('e2e-badge');
    b.textContent='⏳ TTS in progress…'; b.className='pending';
  } else {
    setBar('e2e',lat.total_ms,MAX,scol(lat.status));
    const b=document.getElementById('e2e-badge');
    b.className=lat.status||'pending';
    b.textContent=badgeLabel(lat);
  }
}

function addHistRow(m){
  const lat=m.latency; if(!lat||lat.status==='pending') return;
  hph.style.display='none'; htbl.style.display='table';
  const n=hbody.rows.length+1;
  const tr=hbody.insertRow(0);
  const flags={};  // collect badges for row
  let badges='';
  if(m.interrupted) badges+='<span title="interrupted" style="color:var(--yellow)">⚡</span>';
  if(m.tts_ok===false) badges+='<span title="no audio" style="color:var(--red)">🔇</span>';
  if(!m.transcript) badges+='<span title="no transcript" style="color:var(--dim)">⚠</span>';
  tr.innerHTML=`<td>${n}</td><td>${fms(lat.llm_ttft_ms)}</td><td>${fms(lat.llm_total_ms)}</td>
    <td>${fms(lat.tts_ms)}</td><td>${fe2e(lat.total_ms,lat.status)}</td><td>${badges}</td>`;
  while(hbody.rows.length>10) hbody.deleteRow(hbody.rows.length-1);
}

function setBar(id,ms,max,col){
  const v=document.getElementById(`v-${id}`), b=document.getElementById(`b-${id}`);
  if(!v||!b) return;
  if(ms==null){v.textContent='…';b.style.width='0';return;}
  v.textContent=`${Math.round(ms)}ms`;
  b.style.width=`${Math.min(ms/max*100,100)}%`;
  b.style.background=col;
}
function badgeLabel(lat){
  const ms=lat.total_ms; if(!ms) return '—';
  const over=ms>1200?` (+${Math.round(ms-1200)}ms over)`:'';
  return {stretch:`✅ ${Math.round(ms)}ms — <800ms ✓`,
          pass:`✅ ${Math.round(ms)}ms — <1.2s ✓`,
          slow:`⚠️  ${Math.round(ms)}ms${over}`}[lat.status]||`${Math.round(ms)}ms`;
}
function fms(ms){
  if(ms==null) return '<span style="color:var(--dim)">—</span>';
  const c=ms<800?'var(--green)':ms<1200?'var(--yellow)':'var(--red)';
  return `<span style="color:${c}">${Math.round(ms)}ms</span>`;
}
function fe2e(ms,s){
  if(ms==null) return '<span style="color:var(--dim)">—</span>';
  const c=scol(s); const over=ms>1200?` <span style="color:var(--red);font-size:.7em">(+${Math.round(ms-1200)})</span>`:'';
  return `<span style="color:${c}">${Math.round(ms)}ms</span>${over}`;
}
function scol(s){return{stretch:'var(--green)',pass:'var(--yellow)',slow:'var(--red)'}[s]||'var(--cyan)'}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}

// Mute
function toggleMute(){ws.send(JSON.stringify({type:'control',action:'mute_toggle'}))}
function syncMuteBtn(isMuted){
  muted=isMuted;
  const btn=document.getElementById('mute-btn');
  btn.textContent=isMuted?'🎙️  Unmute':'🎙️  Mute';
  btn.className=isMuted?'muted':'';
}

// Demo Mode
function startDemo(){
  const scSelect = document.getElementById('demo-scenario-select');
  const autoChk = document.getElementById('demo-auto-checkbox');
  ws.send(JSON.stringify({
    type: 'demo',
    action: 'start',
    scenario_id: parseInt(scSelect.value),
    auto_advance: autoChk.checked
  }));
}
function stopDemo(){
  ws.send(JSON.stringify({
    type: 'demo',
    action: 'stop'
  }));
}
function nextDemoTurn(){
  ws.send(JSON.stringify({
    type: 'demo',
    action: 'next'
  }));
}
function handleDemoState(state){
  const startBtn = document.getElementById('demo-start-btn');
  const stopBtn = document.getElementById('demo-stop-btn');
  const nextBtn = document.getElementById('demo-next-btn');
  const statePanel = document.getElementById('demo-state-panel');
  const stateTitle = document.getElementById('demo-state-title');
  const turnProgress = document.getElementById('demo-turn-progress');
  const nextText = document.getElementById('demo-next-text');

  if (state.active) {
    startBtn.disabled = true;
    startBtn.style.opacity = 0.5;
    stopBtn.disabled = false;
    stopBtn.style.opacity = 1.0;
    
    nextBtn.disabled = state.auto_advance || !state.next_turn_text;
    nextBtn.style.opacity = nextBtn.disabled ? 0.5 : 1.0;

    statePanel.style.display = 'block';
    stateTitle.textContent = `Demo: ${state.scenario_name}`;
    turnProgress.textContent = `Turn ${state.turn_idx} / ${state.total_turns}`;
    nextText.textContent = state.next_turn_text ? `"${state.next_turn_text}" [${state.next_turn_lang.toUpperCase()}]` : '(no more turns)';
  } else {
    startBtn.disabled = false;
    startBtn.style.opacity = 1.0;
    stopBtn.disabled = true;
    stopBtn.style.opacity = 0.5;
    nextBtn.disabled = true;
    nextBtn.style.opacity = 0.5;
    statePanel.style.display = 'none';
  }
}
</script>
</body></html>"""

# ── FastAPI ───────────────────────────────────────────────────────────────────
app      = FastAPI(title="Polyglot Live")
_clients: Set[WebSocket] = set()
_alock   = asyncio.Lock()
_loop: Optional[asyncio.AbstractEventLoop] = None
_control_cb: Optional[Callable] = None
_demo_cb: Optional[Callable] = None


def set_control_callback(fn: Callable):
    """Call this from main.py: set_control_callback(pipeline.handle_control)"""
    global _control_cb
    _control_cb = fn


def set_demo_callback(fn: Callable):
    """Call this from main.py: set_demo_callback(pipeline.handle_demo_control)"""
    global _demo_cb
    _demo_cb = fn


@app.get("/", response_class=HTMLResponse)
async def index():
    return _HTML


@app.websocket("/ws")
async def ws_ep(ws: WebSocket):
    await ws.accept()
    async with _alock:
        _clients.add(ws)
    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "control" and _control_cb:
                    # Run sync callback in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, _control_cb, msg.get("action"))
                elif msg.get("type") == "demo" and _demo_cb:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, _demo_cb, msg)
            except Exception:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        async with _alock:
            _clients.discard(ws)



async def _bcast(msg: dict):
    data = json.dumps(msg)
    async with _alock:
        dead = set()
        for c in _clients:
            try:
                await c.send_text(data)
            except Exception:
                dead.add(c)
        _clients.difference_update(dead)


def broadcast(msg: dict):
    if _loop and not _loop.is_closed():
        asyncio.run_coroutine_threadsafe(_bcast(msg), _loop)


# ── Pipeline callbacks ────────────────────────────────────────────────────────

def on_state(state: str):
    broadcast({"type": "state", "value": state})
    if state == "muted":
        broadcast({"type": "mute_state", "muted": True})
    elif state == "listening":
        broadcast({"type": "mute_state", "muted": False})


def on_text_ready(turn: dict):
    broadcast({"type": "text_ready", **turn})


def on_latency_update(latency: dict):
    broadcast({"type": "latency_update", "latency": latency})


def on_turn(turn: dict):
    broadcast({"type": "turn", **turn})


def on_log(msg: str):
    broadcast({"type": "log", "message": msg})


def on_demo_state(state: dict):
    broadcast({"type": "demo_state", "value": state})



def run_server(port: int = 7860):
    global _loop

    def _start():
        global _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        config = uvicorn.Config(app, host="0.0.0.0", port=port,
                                log_level="warning", loop="none")
        uvicorn.Server(config).serve_with_loop(_loop)

    # Fallback for older uvicorn versions
    def _start_compat():
        global _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
        server = uvicorn.Server(config)
        _loop.run_until_complete(server.serve())

    t = threading.Thread(target=_start_compat, daemon=True)
    t.start()
    return t
