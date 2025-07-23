## 🗣️ Voice Recorder with Gemini AI

**An AI-powered language learning assistant built with Streamlit + Google Gemini API.**
Record your voice, get intelligent audio feedback, and improve your pronunciation and grammar — all in your browser.

---

### 🚀 Demo

Try it live: _\[https://language-learning-agent.streamlit.app/]_
Or run locally (instructions below).

---

### 🧠 Features

- ✅ Record your voice directly in the browser
- ✅ Upload pre-recorded audio for analysis
- ✅ Get AI-generated audio feedback via Gemini
- ✅ Supports streaming real-time response
- ✅ Clean modular architecture (easy to extend!)
- ✅ Runs locally or on [Streamlit Community Cloud](https://streamlit.io/cloud)

---

### 🗂️ Project Structure

```
language-learning-agent/
├── app.py                     # Streamlit main app
├── config/
│   └── settings.py            # Model and API settings
├── services/
│   ├── gemini.py              # Gemini processing logic
│   └── audio_processing.py    # Audio conversion utils
├── ui/
│   ├── sidebar.py             # Sidebar for API input
│   └── recorder.py            # Recorder and audio UI
├── .env                       # Local API key storage (not used in cloud)
├── packages.txt               # OS dependencies for Streamlit Cloud
├── requirements.txt           # Python dependencies
└── README.md
```

---

### 🛠️ Getting Started

#### 🔧 Requirements

- Python 3.9+
- Google Gemini API Key — [get one here](https://aistudio.google.com/app/apikey)

---

#### 📦 Install Dependencies

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/language-learning-agent.git
cd language-learning-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install system packages (Linux only):

```bash
sudo apt-get update
sudo apt-get install ffmpeg libsndfile1
```

---

#### 🔐 Set up API Key

Create a `.env` file:

```
GEMINI_API_KEY=your-google-gemini-api-key
```

Or enter it in the **sidebar** after running the app.

---

### ▶️ Run the App

```bash
streamlit run app.py
```

---

### ☁️ Deploying to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub and deploy
4. In **Settings > Secrets**, add:

```
GEMINI_API_KEY=your-google-api-key
```

5. Ensure `requirements.txt` and `packages.txt` exist

---

### 🔍 Example Use Cases

- Language learning and pronunciation practice
- AI-powered language teaching tools
- Voice-enabled chatbots with intelligent feedback
- Feedback tools for ESL (English as Second Language) students

---

### 🧩 Extending the App

- Add **text transcription**
- Add **grammar or vocabulary score**
- Use Gemini's **text and audio output**
- Save user sessions and learning progress

---
