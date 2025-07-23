import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import io
import time

# --- Configuration ---
SAMPLE_RATE = 16000  # Sample rate in Hz
CHANNELS = 1         # Number of audio channels (1 for mono)
DTYPE = 'int16'      # Data type for recording

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Sound Recorder",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Simple Sound Recorder")
st.markdown("Click **Start Recording** to begin, and **Stop Recording** when you're done. The recorded audio will appear below.")

# --- Session State Initialization ---
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'recorded_chunks' not in st.session_state:
    st.session_state.recorded_chunks = []

# --- Helper function to convert numpy array to WAV bytes ---
def numpy_to_wav_bytes(audio_array, sample_rate, channels):
    """Converts a NumPy array to WAV file bytes."""
    byte_io = io.BytesIO()
    with wave.open(byte_io, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(np.dtype(DTYPE).itemsize)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_array.tobytes())
    return byte_io.getvalue()

# --- Recording Controls and Logic ---
col1, col2, col3 = st.columns([1,1,2])

with col1:
    if not st.session_state.is_recording:
        if st.button("🎤 Start Recording", use_container_width=True, type="primary"):
            st.session_state.is_recording = True
            st.session_state.recorded_chunks = []  # Clear previous recording
            st.session_state.audio_data = None      # Clear previous audio player
            st.rerun()
    else:
        if st.button("⏹️ Stop Recording", use_container_width=True, type="secondary"):
            st.session_state.is_recording = False
            st.rerun()

# --- Recording Loop ---
if st.session_state.is_recording:
    with col2:
        st.info("🔴 Recording...")
    
    try:
        # Record a 1-second chunk and append it to our list of chunks
        chunk = sd.rec(
            int(1 * SAMPLE_RATE), 
            samplerate=SAMPLE_RATE, 
            channels=CHANNELS, 
            dtype=DTYPE
        )
        sd.wait() # Wait for the recording chunk to complete
        st.session_state.recorded_chunks.append(chunk)
        
        # Force a rerun to continue the "loop" until the user clicks stop
        time.sleep(0.1) 
        st.rerun()

    except Exception as e:
        st.error(f"An error occurred during recording: {e}")
        st.session_state.is_recording = False
        st.rerun()


# --- Process and Display Audio ---
if not st.session_state.is_recording and st.session_state.recorded_chunks:
    st.success("✅ Recording finished!")
    
    # Combine all recorded chunks into a single NumPy array
    full_recording = np.concatenate(st.session_state.recorded_chunks, axis=0)
    
    # Convert the NumPy array to WAV bytes for st.audio
    wav_bytes = numpy_to_wav_bytes(full_recording, SAMPLE_RATE, CHANNELS)
    
    # Store it in session state to persist across reruns
    st.session_state.audio_data = wav_bytes
    
    # Clear the chunks list as it's no longer needed
    st.session_state.recorded_chunks = []

if st.session_state.audio_data:
    st.markdown("### 🎧 Listen to your recording:")
    st.audio(st.session_state.audio_data, format='audio/wav')
    
    st.download_button(
        label="💾 Download Recording",
        data=st.session_state.audio_data,
        file_name="my_recording.wav",
        mime="audio/wav"
    )

# --- Footer ---
st.markdown("---")
st.markdown("A simple app to quickly check your microphone input.")
