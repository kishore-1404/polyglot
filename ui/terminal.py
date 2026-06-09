"""ui/terminal.py — Rich terminal UI.

on_text_ready → shows text immediately (TTS pending badge)
on_turn       → updates latency panel, removes pending badge, shows status badges
"""
import threading, time
from collections import deque
from typing import Optional

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from core.pipeline import State

_FLAG  = {"en": "🇬🇧", "hi": "🇮🇳", "es": "🇪🇸"}
_COLOR = {"en": "cyan",  "hi": "green", "es": "yellow"}

_STATUS_MAP = {
    State.LISTENING:  ("🎙️ ", "Listening…",   "dim white"),
    State.RECORDING:  ("🔴 ", "Recording…",   "bold red"),
    State.PROCESSING: ("🧠 ", "Processing…",  "bold yellow"),
    State.SPEAKING:   ("🔊 ", "Speaking…",    "bold cyan"),
    State.MUTED:      ("🔇 ", "Muted",         "bold magenta"),
}

MAX_TURNS = 10
MAX_HIST  = 8


class TerminalUI:
    def __init__(self, cfg: dict):
        self.cfg       = cfg
        self.console   = Console()
        self._lock     = threading.Lock()
        self._state    = State.LISTENING
        self._turns    = deque(maxlen=MAX_TURNS)
        self._lat_now  = None
        self._lat_hist = deque(maxlen=MAX_HIST)
        self._logs     = deque(maxlen=5)

    # ── Pipeline callbacks ────────────────────────────────────────────────────

    def on_state(self, state: str):
        with self._lock:
            self._state = state

    def on_text_ready(self, turn: dict):
        """LLM done → show text NOW with TTS-pending badge."""
        with self._lock:
            self._turns.append({**turn, "_pending": True})
            self._lat_now = turn.get("latency")

    def on_latency_update(self, latency: dict):
        """Real-time latency update (e.g. when audio starts playing)."""
        with self._lock:
            self._lat_now = latency

    def on_turn(self, turn: dict):
        """TTS done → update latency, replace pending turn, show final badges."""
        with self._lock:
            if self._turns and self._turns[-1].get("_pending"):
                self._turns[-1] = turn
            else:
                self._turns.append(turn)
            self._lat_now = turn.get("latency")
            lat = turn.get("latency", {})
            if lat.get("status") not in (None, "pending"):
                self._lat_hist.append(lat)

    def on_log(self, msg: str):
        with self._lock:
            self._logs.append(msg)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _header(self) -> Panel:
        icon, label, color = _STATUS_MAP.get(self._state, ("●", self._state, "white"))
        g = Table.grid(expand=True)
        g.add_column(justify="left"); g.add_column(justify="right")
        g.add_row(Text(f"{icon} {label}", style=color),
                  Text("[dim]Polyglot Live  ·  🇬🇧 EN  🇮🇳 HI  🇪🇸 ES  ·  Ctrl+C quit[/dim]"))
        return Panel(g, padding=(0, 1))

    def _conv(self) -> Panel:
        tbl = Table(box=None, padding=(0, 1), expand=True, show_header=False)
        tbl.add_column(width=5); tbl.add_column(width=4); tbl.add_column(ratio=1)

        if not self._turns:
            tbl.add_row("", "", Text("Waiting for speech…", style="dim italic"))
        else:
            for t in self._turns:
                lang    = t.get("lang", "en")
                color   = _COLOR.get(lang, "white")
                flag    = _FLAG.get(lang, "🌍")
                pending = t.get("_pending", False)
                inter   = t.get("interrupted", False)
                tts_ok  = t.get("tts_ok")
                transcript = t.get("transcript", "")

                # ── You row ─────────────────────────────────────────────────
                you_text = Text(transcript if transcript else "(no transcript)",
                                style="white" if transcript else "dim italic")
                if not transcript:
                    you_text.append(" ⚠", style="yellow")
                tbl.add_row(Text("You", style="bold white"), Text(flag), you_text)

                # ── Bot row ─────────────────────────────────────────────────
                bot_line = Text(t.get("response", ""), style=color)
                if pending:
                    bot_line.append("  🔊…", style="dim")
                else:
                    if inter:
                        bot_line.append("  ⚡ interrupted", style="dim yellow")
                    if tts_ok is False:
                        bot_line.append("  🔇 no audio", style="dim red")
                tbl.add_row(Text("Bot", style=f"bold {color}"), Text(flag), bot_line)
                tbl.add_row("", "", "")

        return Panel(tbl, title="[bold]Conversation[/bold]",
                     border_style="blue", padding=(0, 1))

    def _ms(self, ms, kind="") -> str:
        if ms is None:
            return "[dim]…[/dim]"
        ms = float(ms)
        if kind == "e2e" and ms > 1200:
            over = round(ms - 1200)
            return f"[red]{ms:.0f}ms (+{over} over)[/red]"
        c = "green" if ms < 800 else ("yellow" if ms < 1200 else "red")
        return f"[{c}]{ms:.0f}ms[/{c}]"

    def _bar(self, ms, maxv=1200.0, w=22) -> str:
        if ms is None:
            return "[dim]" + "░" * w + "[/dim]"
        f = min(int(float(ms) / maxv * w), w)
        c = "green" if float(ms) < 800 else ("yellow" if float(ms) < 1200 else "red")
        return f"[{c}]{'█'*f}[/{c}][dim]{'░'*(w-f)}[/dim]"

    def _latency(self) -> Panel:
        rows = []
        lat  = self._lat_now

        # Current turn
        rows.append(Text("── Current ──", style="dim"))
        if lat:
            tbl = Table(box=None, padding=(0, 1), show_header=False)
            tbl.add_column(width=10); tbl.add_column(width=24); tbl.add_column(width=18)
            for label, key, kind in [
                ("LLM TTFT",  "llm_ttft_ms",  ""),
                ("LLM Total", "llm_total_ms", ""),
                ("TTS Synth", "tts_ms",        ""),
                ("E2E",       "total_ms",      "e2e"),
            ]:
                v = lat.get(key)
                tbl.add_row(Text(label, style="dim"),
                            Text.from_markup(self._bar(v)),
                            Text.from_markup(self._ms(v, kind)))
            rows.append(tbl)

            status = lat.get("status", "pending")
            badge = {
                "pending": "[dim]⏳ TTS in progress[/dim]",
                "stretch": "[green]✅ <800ms  stretch goal ✓[/green]",
                "pass":    "[yellow]✅ <1.2s   target met[/yellow]",
                "slow":    f"[red]⚠️  above target[/red]",
            }.get(status, "")
            rows.append(Text.from_markup(badge))
        else:
            rows.append(Text("No turns yet.", style="dim italic"))

        # History
        if self._lat_hist:
            rows.append(Rule(style="dim"))
            rows.append(Text("── History ──", style="dim"))
            ht = Table(box=None, padding=(0, 0), show_header=True,
                       header_style="dim", show_edge=False)
            ht.add_column("#",    width=3, justify="right")
            ht.add_column("TTFT", width=8, justify="right")
            ht.add_column("LLM",  width=8, justify="right")
            ht.add_column("TTS",  width=8, justify="right")
            ht.add_column("E2E",  width=18, justify="left")
            for i, h in enumerate(self._lat_hist, 1):
                ht.add_row(
                    Text(str(i), style="dim"),
                    Text.from_markup(self._ms(h.get("llm_ttft_ms"))),
                    Text.from_markup(self._ms(h.get("llm_total_ms"))),
                    Text.from_markup(self._ms(h.get("tts_ms"))),
                    Text.from_markup(self._ms(h.get("total_ms"), "e2e")),
                )
            rows.append(ht)

        return Panel(Group(*rows), title="[bold]Latency[/bold]",
                     border_style="green", padding=(0, 1))

    def _log_panel(self) -> Panel:
        lines = list(self._logs) or ["—"]
        return Panel(Text("\n".join(lines[-4:]), style="dim"),
                     title="[bold]Log[/bold]", border_style="dim", padding=(0, 1))

    def _build(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="hdr", size=3),
            Layout(name="main", ratio=1),
            Layout(name="log", size=5),
        )
        layout["main"].split_row(
            Layout(name="conv", ratio=2),
            Layout(name="lat",  ratio=1),
        )
        with self._lock:
            layout["hdr"].update(self._header())
            layout["conv"].update(self._conv())
            layout["lat"].update(self._latency())
            layout["log"].update(self._log_panel())
        return layout

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self, pipeline_run_fn):
        self.console.print(Panel.fit(
            "[bold cyan]🌍 POLYGLOT LIVE[/bold cyan]  —  Multilingual Voice Companion\n"
            "[dim]English · Hindi · Spanish  ·  Ctrl+C to quit[/dim]",
            border_style="cyan"))
        t = threading.Thread(target=pipeline_run_fn, daemon=True)
        t.start()
        try:
            with Live(self._build(), refresh_per_second=10, screen=False) as live:
                while t.is_alive():
                    live.update(self._build())
                    time.sleep(0.1)
        except KeyboardInterrupt:
            pass
