# 🎙️ VoiceScript-Turbo
![VoiceScript-Turbo](https://github.com/user-attachments/assets/e7927d2c-85af-4daf-a3b0-f65191c1a581)

> **"Necessity is the mother of invention"**
> 
> My ring finger got injured during cricket which motivated me to build this application to type for me.... and not stop working in any case as a PhD researcher in Computer Science.

**VoiceScript-Turbo** is a high-performance, offline speech-to-text tool designed for local Linux machines. By leveraging the **Whisper Large-v3-Turbo** model via `faster-whisper`, it provides near-instant, private transcription to ensure your workflow remains uninterrupted.

---

## ✨ Key Features

* 🚀 **High-Speed Transcription**: Optimized for the `large-v3-turbo` model for ultra-fast, live processing.
* 🔒 **100% Local & Private**: Runs entirely on your hardware—no data leaves your machine.
* 📈 **Live Spectrum Visualizer**: Real-time frequency spectrum display using Matplotlib to monitor audio input levels.
* 🎙️ **Advanced Audio Control**: Built-in adjustable Mic Boost and a "Whisper Mode" toggle to handle varying acoustic environments.
* 🌍 **Multilingual Support**: Supports Auto-Detection and specific presets for English, Urdu, Hindi, Arabic, French, German, Spanish, and more.
* 🛠️ **Research-Ready**: Includes Voice Activity Detection (VAD) filtering and beam size configuration for high-accuracy results.
* 📄 **Auto-Export**: Automatically saves your live transcriptions to a local text file for later use.

---

## 🌍 Language Support

VoiceScript-Turbo uses the **Whisper Large-v3-Turbo** model, which natively supports **99 different languages**, including multi-lingual translation tasks.

To keep the interface clean, the application features a built-in dropdown menu with **14 specific presets** plus an Auto-Detect option.

### Featured GUI Languages:
1. **Auto Detect** (Detects any of the 99 supported languages automatically)
2. **English** 🇬🇧/🇺🇸
3. **Urdu** 🇵🇰
4. **Hindi** 🇮🇳
5. **Arabic** 🇸🇦
6. **Turkish** 🇹🇷
7. **French** 🇫🇷
8. **German** 🇩🇪
9. **Spanish** 🇪🇸
10. **Italian** 🇮🇹
11. **Portuguese** 🇵🇹
12. **Chinese** 🇨🇳
13. **Japanese** 🇯🇵
14. **Korean** 🇰🇷

> **Note:** If you need to transcribe a language not in this list (like Russian or Dutch), simply select **Auto Detect**. The underlying Whisper model will automatically identify and process it for you!

---

## 🛠️ Technical Stack

* **Transcription Engine**: `faster-whisper`
* **GUI Framework**: Python `Tkinter` with a custom dark-themed interface
* **Audio Handling**: `sounddevice` and `soundfile`
* **Visualization**: `matplotlib` (FFT Frequency Spectrum)

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.8+ and the necessary system audio libraries installed:
```bash
sudo apt-get install libportaudio2
```

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/QaisarFarooq17/VoiceScript-Turbo.git
   cd VoiceScript-Turbo
   ```
2. Install dependencies using the provided requirements file:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
Launch the GUI with your preferred settings using the wrapper script:
```bash
./run_live_gui.sh
```
Or optionally via Python directly:
```bash
python -m src.live_gui
```
*Use `--device cpu` if you do not have a compatible NVIDIA GPU.*

---

## ⚙️ Configuration Options

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--model` | Whisper model size to use | `large-v3-turbo` |
| `--device` | Device for computation (`cuda`, `cpu`, `auto`) | `cuda` |
| `--compute-type` | Weight quantization (e.g., `float16`, `int8`) | `float16` |
| `--language` | Default language code | `en` |
| `--vad` | Enable Voice Activity Detection | `False` |

---

## 💡 About the Project
As a PhD researcher in Computer Science, time and data privacy are paramount. This tool was born out of a physical necessity following a sports injury, but it evolved into a robust solution for anyone needing professional-grade, local transcription without the latency or privacy risks of cloud-based APIs.

---

## 🙌 Acknowledgments
I would like to extend a special thanks to **Muhammad Rashid** and **Hakim Ziani**. The core idea were first sparked and refined during our insightful discussions together over cups of coffee in the bar.
