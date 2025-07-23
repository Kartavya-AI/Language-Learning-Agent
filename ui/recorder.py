import streamlit as st
from streamlit_mic_recorder import mic_recorder

def handle_recording_ui():
    st.markdown("### 🎙️ Record your voice")
    audio = mic_recorder(start_prompt="🎤 Start", stop_prompt="⏹️ Stop", just_once=False, use_container_width=True)

    if audio:
        st.audio(audio["bytes"], format=audio["format"])
        st.download_button("💾 Download Recording", data=audio["bytes"], file_name=f"recording.{audio['format']}", mime=f"audio/{audio['format']}")
        return audio

    return None
