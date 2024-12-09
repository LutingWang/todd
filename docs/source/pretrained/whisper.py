from transformers import AutomaticSpeechRecognitionPipeline, pipeline

from todd.utils import get_audio

audio, _ = get_audio(
    'https://github.com/SWivid/F5-TTS/raw/refs/heads/main/'
    'src/f5_tts/infer/examples/basic/basic_ref_zh.wav',
)

pipe: AutomaticSpeechRecognitionPipeline = pipeline(
    'automatic-speech-recognition',
    model='pretrained/whisper/whisper-large-v3-turbo',
    torch_dtype='auto',
    device_map='auto',
)

result = pipe(audio)
print(result)

result = pipe(audio, generate_kwargs=dict(language='zh'))
print(result)

result = pipe(audio, generate_kwargs=dict(task='translate', language='en'))
print(result)

result = pipe(audio, return_timestamps=True)
print(result)

result = pipe(audio, return_timestamps='word')
print(result)
