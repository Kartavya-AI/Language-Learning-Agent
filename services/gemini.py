import asyncio
import io
import wave
from google import genai
from google.genai import types
from services.audio_processing import convert_audio_to_pcm16

async def process_audio_async(client, audio_bytes, model, config):
    pcm_audio = convert_audio_to_pcm16(audio_bytes)
    if not pcm_audio:
        return None

    response_audio = io.BytesIO()
    wf = wave.open(response_audio, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)

    async with client.aio.live.connect(model=model, config=config) as session:
        await session.send_realtime_input(audio=types.Blob(data=pcm_audio, mime_type="audio/pcm;rate=16000"))
        async for response in session.receive():
            if response.data:
                wf.writeframes(response.data)

    wf.close()
    response_audio.seek(0)
    return response_audio.getvalue()

def run_gemini_processing_sync(audio_bytes, api_key, model, config):
    client = genai.Client(api_key=api_key)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(process_audio_async(client, audio_bytes, model, config))
    finally:
        loop.close()
