import einops
from transformers import AutomaticSpeechRecognitionPipeline, pipeline

from todd.utils import get_audio

audio, _ = get_audio(
    'https://github.com/SWivid/F5-TTS/raw/refs/heads/main/'
    'src/f5_tts/infer/examples/basic/basic_ref_zh.wav',
)
audio_array = audio.numpy()
audio_array = einops.rearrange(audio_array, '1 t -> t')

p: AutomaticSpeechRecognitionPipeline = pipeline(
    'automatic-speech-recognition',
    'pretrained/whisper/whisper-large-v3-turbo',
    torch_dtype='auto',
    device_map='auto',
)

result = p(audio_array)
print(result)

result = p(audio_array, generate_kwargs=dict(language='zh'))
print(result)

result = p(audio_array, generate_kwargs=dict(task='translate', language='en'))
print(result)

result = p(audio_array, return_timestamps=True)
print(result)

result = p(audio_array, return_timestamps='word')
print(result)
