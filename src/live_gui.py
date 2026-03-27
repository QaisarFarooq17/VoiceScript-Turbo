import argparse
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from queue import Empty, Queue
from tkinter import ttk
from typing import Optional

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import sounddevice as sd
from src.audio_utils import boost_quiet_audio, describe_selected_device, pick_sample_rate

# ── Design Tokens ──────────────────────────────────────────────────────────────
_C = {
    "bg_deep":    "#0d1118",   # deepest background (text area, plot bg)
    "bg_base":    "#131a24",   # main window background
    "bg_card":    "#1a2231",   # card / panel background
    "bg_row":     "#202a3b",   # section-header / row background
    "border":     "#364666",   # border / divider color
    "accent":     "#7bbcff",   # accent blue
    "green":      "#32c778",   # start / active / success
    "green_dk":   "#28a965",   # green hover / dark variant
    "red":        "#ff4c63",   # stop / error
    "red_dk":     "#dc3c52",   # red hover / dark variant
    "yellow":     "#f5b640",   # loading / warning
    "text_pri":   "#eaf1ff",   # primary text
    "text_sec":   "#b4c1d8",   # secondary / label text
    "text_muted": "#445576",   # very muted (idle dot)
    "plot":       "#5cb2ff",   # spectrum line & fill
}

_FONT_UI = "DejaVu Sans"
_FONT_MONO = "DejaVu Sans Mono"

LANGUAGE_OPTIONS = [
    ("Auto Detect", ""),
    ("English",    "en"),
    ("Urdu",       "ur"),
    ("Hindi",      "hi"),
    ("Arabic",     "ar"),
    ("Turkish",    "tr"),
    ("French",     "fr"),
    ("German",     "de"),
    ("Spanish",    "es"),
    ("Italian",    "it"),
    ("Portuguese", "pt"),
    ("Chinese",    "zh"),
    ("Japanese",   "ja"),
    ("Korean",     "ko"),
    ("Russian",    "ru"),
]


class LiveTranscriptionGUI:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = tk.Tk()
        self.root.title("VoiceScript — Live Transcription")
        self.root.geometry("1680x1080")
        self.root.minsize(1360, 920)
        self.root.configure(bg=_C["bg_base"])
        try:
            self.root.tk.call("tk", "scaling", 1.6)
        except Exception:
            pass

        self.is_running = False
        self.audio_queue: Queue[np.ndarray] = Queue()
        self.text_queue: Queue[str] = Queue()
        self.level_queue: Queue[np.ndarray] = Queue(maxsize=5)
        self.stop_event = threading.Event()
        self._fill_collection = None
        self._pulse_on = False

        self.model: Optional[WhisperModel] = None
        self.stream: Optional[sd.InputStream] = None
        self.transcriber_thread: Optional[threading.Thread] = None

        self.sample_rate = 0
        self.chunk_samples = 0

        self._configure_style()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._load_devices()

    # ── TTK Style Configuration ────────────────────────────────────────────────

    def _configure_style(self) -> None:
        s = ttk.Style(self.root)
        s.theme_use("clam")

        # Ensure dropdown popup list text is also large.
        self.root.option_add("*TCombobox*Listbox.font", (_FONT_UI, 22))

        # Combobox — dark field
        s.configure(
            "Dark.TCombobox",
            fieldbackground=_C["bg_row"],
            background=_C["bg_row"],
            foreground=_C["text_pri"],
            arrowcolor=_C["accent"],
            bordercolor=_C["border"],
            lightcolor=_C["border"],
            darkcolor=_C["border"],
            selectbackground=_C["accent"],
            selectforeground="#ffffff",
            padding=(16, 12),
            font=(_FONT_UI, 24),
        )
        s.map(
            "Dark.TCombobox",
            fieldbackground=[("readonly", _C["bg_row"])],
            foreground=[("readonly", _C["text_pri"])],
            selectbackground=[("readonly", _C["accent"])],
            selectforeground=[("readonly", "#ffffff")],
        )

        # Checkbutton
        s.configure(
            "Dark.TCheckbutton",
            background=_C["bg_card"],
            foreground=_C["text_pri"],
            font=(_FONT_UI, 19),
        )
        s.map(
            "Dark.TCheckbutton",
            background=[("active", _C["bg_card"])],
            foreground=[("active", _C["accent"])],
            indicatorcolor=[("selected", _C["accent"]), ("!selected", _C["border"])],
        )

    # ── Widget Helpers ─────────────────────────────────────────────────────────

    def _mk_btn(
        self,
        parent: tk.Widget,
        text: str,
        command,
        bg: str,
        bg_hover: str,
        fg: str = "#ffffff",
        state: str = "normal",
    ) -> tk.Button:
        """Flat, colored tk.Button with hover highlight."""
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=bg_hover,
            activeforeground=fg,
            relief="flat",
            bd=0,
            padx=24,
            pady=14,
            font=(_FONT_UI, 20, "bold"),
            cursor="hand2",
            state=state,
        )
        # Store original colors on the button object for hover restoration
        btn._base_bg = bg        # type: ignore[attr-defined]
        btn._hover_bg = bg_hover  # type: ignore[attr-defined]

        def _enter(e: tk.Event) -> None:
            if str(btn["state"]) == "normal":
                btn.configure(bg=btn._hover_bg)  # type: ignore[attr-defined]

        def _leave(e: tk.Event) -> None:
            if str(btn["state"]) == "normal":
                btn.configure(bg=btn._base_bg)   # type: ignore[attr-defined]

        btn.bind("<Enter>", _enter)
        btn.bind("<Leave>", _leave)
        return btn

    def _cap_label(
        self,
        parent: tk.Widget,
        text: str,
        bg: str = "",
    ) -> tk.Label:
        """Small ALL-CAPS field label."""
        return tk.Label(
            parent,
            text=text,
            bg=bg or _C["bg_card"],
            fg=_C["text_sec"],
            font=(_FONT_UI, 16, "bold"),
        )

    # ── Top-level Layout ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self._build_header()
        tk.Frame(self.root, bg=_C["border"], height=1).pack(fill="x")  # hairline
        main = tk.Frame(self.root, bg=_C["bg_base"])
        main.pack(fill="both", expand=True, padx=26, pady=20)
        self._build_controls(main)
        self._build_spectrum(main)
        self._build_transcript(main)

    def _build_header(self) -> None:
        hbar = tk.Frame(self.root, bg=_C["bg_deep"], height=92)
        hbar.pack(fill="x")
        hbar.pack_propagate(False)

        # Left: app icon + name
        left = tk.Frame(hbar, bg=_C["bg_deep"])
        left.pack(side="left", padx=18, fill="y")
        tk.Label(
            left, text="VoiceScript", bg=_C["bg_deep"],
            fg=_C["text_pri"],
            font=(_FONT_UI, 34, "bold"),
        ).pack(side="left", padx=(0, 10))
        tk.Label(
            left, text="Live Transcription",
            bg=_C["bg_deep"], fg=_C["text_sec"],
            font=(_FONT_UI, 22),
        ).pack(side="left", pady=(6, 0))

        # Right: pulsing status dot + status text
        right = tk.Frame(hbar, bg=_C["bg_deep"])
        right.pack(side="right", padx=18, fill="y")
        self._dot_canvas = tk.Canvas(
            right, width=12, height=12,
            bg=_C["bg_deep"], highlightthickness=0,
        )
        self._dot_canvas.pack(side="left", padx=(0, 7))
        self._dot_oval = self._dot_canvas.create_oval(
            1, 1, 11, 11, fill=_C["text_muted"], outline="",
        )
        self.status_var = tk.StringVar(value="Idle")
        tk.Label(
            right, textvariable=self.status_var,
            bg=_C["bg_deep"], fg=_C["text_sec"],
            font=(_FONT_UI, 20, "italic"),
        ).pack(side="left")

    # ── Panels ─────────────────────────────────────────────────────────────────

    def _bordered_card(
        self, parent: tk.Widget, expand: bool = False, pady: tuple = (0, 10)
    ) -> tk.Frame:
        """1-px border card."""
        outer = tk.Frame(parent, bg=_C["border"])
        outer.pack(fill="both" if expand else "x", expand=expand, pady=pady)
        inner = tk.Frame(outer, bg=_C["bg_card"])
        inner.pack(fill="both" if expand else "x", expand=expand, padx=1, pady=1)
        return inner

    def _section_bar(self, parent: tk.Widget, title: str) -> tk.Frame:
        """Dark section-header strip inside a card."""
        bar = tk.Frame(parent, bg=_C["bg_row"], padx=20, pady=12)
        bar.pack(fill="x")
        tk.Label(
            bar, text=title,
            bg=_C["bg_row"], fg=_C["text_sec"],
            font=(_FONT_UI, 16, "bold"),
        ).pack(side="left")
        return bar

    def _build_controls(self, parent: tk.Widget) -> None:
        card = self._bordered_card(parent)
        self._section_bar(card, "CONFIGURATION")

        body = tk.Frame(card, bg=_C["bg_card"], padx=20, pady=20)
        body.pack(fill="x")

        # ── Row A: dropdowns + toggle + boost ──────────────────────
        rowA = tk.Frame(body, bg=_C["bg_card"])
        rowA.pack(fill="x", pady=(0, 10))

        self._cap_label(rowA, "INPUT DEVICE").pack(side="left")
        self.device_var = tk.StringVar(value="default")
        self.device_combo = ttk.Combobox(
            rowA, textvariable=self.device_var,
            width=38, state="readonly", style="Dark.TCombobox",
            font=(_FONT_UI, 24),
        )
        self.device_combo.pack(side="left", padx=(6, 22))

        self._cap_label(rowA, "LANGUAGE").pack(side="left")
        self.language_var = tk.StringVar(
            value=self._display_for_code(self.args.language)
        )
        self.language_combo = ttk.Combobox(
            rowA,
            textvariable=self.language_var,
            width=20,
            state="readonly",
            style="Dark.TCombobox",
            font=(_FONT_UI, 24),
            values=[self._display_for_code(c) for _, c in LANGUAGE_OPTIONS],
        )
        self.language_combo.pack(side="left", padx=(6, 22))

        self.whisper_mode_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            rowA,
            text="Whisper Mode",
            variable=self.whisper_mode_var,
            style="Dark.TCheckbutton",
        ).pack(side="left", padx=(0, 22))

        self._cap_label(rowA, "MIC BOOST").pack(side="left")
        boost_wrap = tk.Frame(rowA, bg=_C["bg_card"])
        boost_wrap.pack(side="left", padx=(6, 0))
        self.boost_var = tk.DoubleVar(value=24.0)
        tk.Scale(
            boost_wrap,
            from_=1.0, to=35.0, variable=self.boost_var,
            orient="horizontal", length=260,
            bg=_C["bg_card"], fg=_C["text_sec"],
            troughcolor=_C["border"], activebackground=_C["accent"],
            highlightthickness=0, relief="flat",
            showvalue=False, sliderlength=30,
        ).pack(side="left")
        self.boost_label_var = tk.StringVar(value="24.0×")
        tk.Label(
            boost_wrap, textvariable=self.boost_label_var,
            bg=_C["bg_card"], fg=_C["accent"],
            font=(_FONT_MONO, 19, "bold"), width=6,
        ).pack(side="left", padx=(4, 0))

        # ── Row B: action buttons ───────────────────────────────────
        rowB = tk.Frame(body, bg=_C["bg_card"])
        rowB.pack(fill="x")

        self.start_btn = self._mk_btn(
            rowB, "▶  Start Listening", self.start_listening,
            _C["green"], _C["green_dk"],
        )
        self.start_btn.pack(side="left", padx=(0, 8))

        self.stop_btn = self._mk_btn(
            rowB, "■  Stop", self.stop_listening,
            _C["bg_row"], _C["border"],
            fg=_C["text_sec"], state="disabled",
        )
        self.stop_btn.pack(side="left", padx=(0, 26))

        # Vertical divider
        tk.Frame(rowB, bg=_C["border"], width=1, height=30).pack(
            side="left", padx=(0, 26)
        )

        self.copy_btn = self._mk_btn(
            rowB, "⎘  Copy Text", self.copy_text,
            _C["bg_row"], _C["border"], fg=_C["text_pri"],
        )
        self.copy_btn.pack(side="left", padx=(0, 8))

        self.clear_btn = self._mk_btn(
            rowB, "✕  Clear", self.clear_text,
            _C["bg_row"], _C["border"], fg=_C["text_pri"],
        )
        self.clear_btn.pack(side="left")

    def _build_spectrum(self, parent: tk.Widget) -> None:
        card = self._bordered_card(parent)
        self._section_bar(card, "FREQUENCY SPECTRUM")

        wrap = tk.Frame(card, bg=_C["bg_card"], padx=12, pady=12)
        wrap.pack(fill="x")

        self.figure = Figure(figsize=(12, 3.8), dpi=130)
        self.figure.patch.set_facecolor(_C["bg_card"])
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(_C["bg_deep"])
        self.ax.grid(alpha=0.12, linestyle="-", color=_C["border"])
        self.ax.set_ylim(0, 1)
        self.ax.set_xlim(0, 8000)
        self.ax.set_xlabel("Frequency (Hz)", color=_C["text_sec"], fontsize=18)
        self.ax.set_ylabel("Magnitude",      color=_C["text_sec"], fontsize=18)
        self.ax.tick_params(colors=_C["text_sec"], labelsize=15)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(_C["border"])
        self.figure.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.25)

        self.line, = self.ax.plot(
            [], [], lw=1.5, color=_C["plot"], alpha=0.9, zorder=3
        )

        self.canvas = FigureCanvasTkAgg(self.figure, master=wrap)
        self.canvas.get_tk_widget().pack(fill="x")
        self.canvas.get_tk_widget().configure(
            bg=_C["bg_card"], highlightthickness=0
        )

    def _build_transcript(self, parent: tk.Widget) -> None:
        card = self._bordered_card(parent, expand=True, pady=(0, 0))

        # Header bar with live word-count on the right
        bar = self._section_bar(card, "LIVE TRANSCRIPT")
        self._wc_var = tk.StringVar(value="0 words")
        tk.Label(
            bar, textvariable=self._wc_var,
            bg=_C["bg_row"], fg=_C["accent"],
            font=(_FONT_UI, 16),
        ).pack(side="right")

        body = tk.Frame(card, bg=_C["bg_card"])
        body.pack(fill="both", expand=True, padx=2, pady=2)

        self.text_box = tk.Text(
            body,
            wrap="word",
            font=(_FONT_MONO, 30),
            relief="flat",
            padx=20, pady=18,
            bg=_C["bg_deep"],
            fg=_C["text_pri"],
            insertbackground=_C["accent"],
            selectbackground=_C["accent"],
            selectforeground="#ffffff",
            spacing1=3, spacing3=3,
            bd=0,
        )
        scroll = tk.Scrollbar(
            body, orient="vertical",
            command=self.text_box.yview,
            bg=_C["bg_card"], troughcolor=_C["bg_deep"],
            relief="flat", bd=0, width=16,
        )
        self.text_box.configure(yscrollcommand=scroll.set)
        self.text_box.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    # ── Status Helpers ─────────────────────────────────────────────────────────

    def _set_status(self, text: str, dot_color: str = "") -> None:
        self.status_var.set(text)
        if dot_color:
            self._dot_canvas.itemconfig(self._dot_oval, fill=dot_color)

    def _pulse_dot(self) -> None:
        """Alternate the status dot between two greens while listening."""
        if not self.is_running:
            return
        self._pulse_on = not self._pulse_on
        fill = _C["green"] if self._pulse_on else _C["green_dk"]
        self._dot_canvas.itemconfig(self._dot_oval, fill=fill)
        self.root.after(600, self._pulse_dot)

    def _arm_start_state(self) -> None:
        """UI state when listening is active."""
        self.start_btn.configure(
            state="disabled", bg=_C["text_muted"], fg=_C["text_sec"]
        )
        self.stop_btn.configure(
            state="normal", bg=_C["red"],
            activebackground=_C["red_dk"], fg="#ffffff",
        )
        self.stop_btn._base_bg = _C["red"]    # type: ignore[attr-defined]
        self.stop_btn._hover_bg = _C["red_dk"]  # type: ignore[attr-defined]

    def _arm_idle_state(self) -> None:
        """UI state when idle / stopped."""
        self.start_btn.configure(
            state="normal", bg=_C["green"], fg="#ffffff"
        )
        self.stop_btn.configure(
            state="disabled", bg=_C["bg_row"], fg=_C["text_sec"]
        )
        self.stop_btn._base_bg = _C["bg_row"]   # type: ignore[attr-defined]
        self.stop_btn._hover_bg = _C["border"]   # type: ignore[attr-defined]

    # ── Data Helpers ───────────────────────────────────────────────────────────

    def _display_for_code(self, code: str) -> str:
        normalized = (code or "").strip().lower()
        for name, lang_code in LANGUAGE_OPTIONS:
            if lang_code == normalized:
                return f"{name} ({lang_code})" if lang_code else "Auto Detect"
        return "Auto Detect"

    def _code_for_display(self, display_value: str) -> Optional[str]:
        value = (display_value or "").strip()
        for name, lang_code in LANGUAGE_OPTIONS:
            candidate = f"{name} ({lang_code})" if lang_code else "Auto Detect"
            if value == candidate:
                return lang_code or None
        return None

    def _load_devices(self) -> None:
        devices = sd.query_devices()
        input_options = ["default"]
        for idx, d in enumerate(devices):
            if int(d.get("max_input_channels", 0)) > 0:
                input_options.append(f"{idx}: {d['name']}")
        self.device_combo["values"] = input_options
        self.device_combo.current(0)

    def get_selected_device(self) -> Optional[int]:
        value = self.device_var.get().strip()
        if not value or value == "default":
            return None
        try:
            return int(value.split(":", 1)[0])
        except Exception:
            return None

    # ── Pipeline Control ───────────────────────────────────────────────────────

    def start_listening(self) -> None:
        if self.is_running:
            return
        self._arm_start_state()
        self._set_status("Loading model…", _C["yellow"])
        self.is_running = True
        self.stop_event.clear()
        threading.Thread(target=self._start_pipeline, daemon=True).start()

    def _start_pipeline(self) -> None:
        try:
            if self.model is None:
                self.model = WhisperModel(
                    self.args.model,
                    device=self.args.device,
                    compute_type=self.args.compute_type,
                )

            input_device = self.get_selected_device()
            self.sample_rate = pick_sample_rate(input_device, self.args.sample_rate)
            self.chunk_samples = int(self.sample_rate * self.args.chunk_seconds)
            device_idx, device_name = describe_selected_device(input_device)

            self._set_status(
                f"Listening — device {device_idx} ({device_name}) @ {self.sample_rate} Hz",
                _C["green"],
            )
            self.root.after(0, self._pulse_dot)

            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
                blocksize=self.chunk_samples,
                device=input_device,
            )
            self.stream.start()

            self.transcriber_thread = threading.Thread(
                target=self._transcribe_loop, daemon=True
            )
            self.transcriber_thread.start()

        except Exception as exc:
            self._set_status(f"Error: {exc}", _C["red"])
            self.stop_event.set()
            self.is_running = False
            self._arm_idle_state()

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time: object, status: object
    ) -> None:
        del frames, time
        if status:
            self._set_status(f"Audio: {status}", _C["yellow"])
        chunk = indata.copy()
        self.audio_queue.put(chunk)
        if not self.level_queue.full():
            self.level_queue.put(chunk)

    def _transcribe_loop(self) -> None:
        output_path = Path(self.args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        chunks: list[np.ndarray] = []

        while not self.stop_event.is_set():
            try:
                chunks.append(self.audio_queue.get(timeout=0.2))
            except Empty:
                continue

            sample_count = sum(c.shape[0] for c in chunks)
            if sample_count < self.chunk_samples:
                continue

            audio = np.concatenate(chunks, axis=0).reshape(-1)
            chunks = []

            max_gain = float(self.boost_var.get())
            whisper_mode = bool(self.whisper_mode_var.get())
            target_rms = 0.10 if whisper_mode else 0.075
            silence_threshold = 0.0012 if whisper_mode else 0.0028

            boosted_audio, _, peak, _ = boost_quiet_audio(
                audio,
                target_rms=target_rms,
                max_gain=max_gain,
                silence_threshold=silence_threshold,
            )

            if peak < silence_threshold:
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, boosted_audio, self.sample_rate)
                wav_path = Path(tmp.name)

            try:
                assert self.model is not None
                language = self._code_for_display(self.language_var.get())
                segments, _ = self.model.transcribe(
                    str(wav_path),
                    language=language,
                    beam_size=self.args.beam_size,
                    vad_filter=self.args.vad,
                    task="transcribe",
                )
                text = " ".join(seg.text.strip() for seg in segments).strip()
                if text:
                    self.text_queue.put(text)
                    with output_path.open("a", encoding="utf-8") as f:
                        f.write(text + "\n")
            finally:
                wav_path.unlink(missing_ok=True)

    def stop_listening(self) -> None:
        if not self.is_running:
            return
        self.stop_event.set()
        self.is_running = False
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        self.stream = None
        self._set_status("Stopped", _C["text_muted"])
        self._arm_idle_state()

    def copy_text(self) -> None:
        content = self.text_box.get("1.0", "end").strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self._set_status("Transcript copied to clipboard ✓", _C["accent"])

    def clear_text(self) -> None:
        self.text_box.delete("1.0", "end")
        self._wc_var.set("0 words")
        self._set_status("Transcript cleared", _C["text_muted"])

    # ── Periodic UI Updates ────────────────────────────────────────────────────

    def _update_text(self) -> None:
        drained = False
        while True:
            try:
                text = self.text_queue.get_nowait()
                self.text_box.insert("end", text + "\n")
                self.text_box.see("end")
                drained = True
            except Empty:
                break
        if drained:
            content = self.text_box.get("1.0", "end").strip()
            wc = len(content.split()) if content else 0
            self._wc_var.set(f"{wc:,} words")
        self.root.after(120, self._update_text)

    def _update_graph(self) -> None:
        latest = None
        while True:
            try:
                latest = self.level_queue.get_nowait()
            except Empty:
                break

        if latest is not None:
            signal = latest.reshape(-1)
            window = np.hanning(len(signal))
            spec = np.abs(np.fft.rfft(signal * window))
            freqs = np.fft.rfftfreq(len(signal), d=1.0 / max(self.sample_rate, 1))
            if spec.size > 0:
                spec = spec / (np.max(spec) + 1e-9)
            self.line.set_data(freqs, spec)
            # Filled area under the curve for depth
            if self._fill_collection is not None:
                self._fill_collection.remove()
            self._fill_collection = self.ax.fill_between(
                freqs, spec, alpha=0.15, color=_C["plot"], zorder=2
            )
            self.ax.set_xlim(0, min(8000, max(2000, self.sample_rate // 2)))
            self.ax.set_ylim(0, 1)
            self.canvas.draw_idle()

        self.boost_label_var.set(f"{self.boost_var.get():.1f}×")
        self.root.after(120, self._update_graph)

    def on_close(self) -> None:
        self.stop_listening()
        self.root.destroy()

    def run(self) -> None:
        self._update_text()
        self._update_graph()
        self.root.mainloop()


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live GUI microphone transcription")
    parser.add_argument("--model",         default="large-v3-turbo")
    parser.add_argument("--device",        default="cuda",
                        choices=["cuda", "cpu", "auto"])
    parser.add_argument("--compute-type",  default="float16")
    parser.add_argument("--language",      default="en")
    parser.add_argument("--sample-rate",   type=int,   default=0)
    parser.add_argument("--chunk-seconds", type=float, default=3.0)
    parser.add_argument("--beam-size",     type=int,   default=1)
    parser.add_argument("--vad",           action="store_true")
    parser.add_argument("--output",        default="outputs/live_transcript.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = LiveTranscriptionGUI(args)
    app.run()


if __name__ == "__main__":
    main()
