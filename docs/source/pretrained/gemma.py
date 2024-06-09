import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gemma import GemmaForCausalLM, GemmaTokenizerFast

from todd.patches.torch import get_device

PRETRAINED = "pretrained/gemma/gemma-1.1-2b-it"

tokenizer: GemmaTokenizerFast = AutoTokenizer.from_pretrained(PRETRAINED)
model: GemmaForCausalLM = AutoModelForCausalLM.from_pretrained(
    PRETRAINED,
    device_map=get_device(),
    torch_dtype=torch.float16,
    revision="float16",
)

WORD_TOKEN = '<word>'  # nosec B105
TEMPLATE = (
    "Imagine an instance of the given word and describe its appearance in a "
    "single sentence. Be creative and avoid describing the scene. The word is"
    f": {WORD_TOKEN}"
)

prompt = tokenizer.apply_chat_template(
    [dict(role='user', content=TEMPLATE)],
    tokenize=False,
    add_generation_prompt=True,
)
prefix, suffix = prompt.split(WORD_TOKEN)
prefix_ids = tokenizer.encode(
    prefix,
    return_tensors='pt',
    add_special_tokens=False,
).to('cuda')
suffix_ids = tokenizer.encode(
    suffix,
    return_tensors='pt',
    add_special_tokens=False,
).to('cuda')
prefix_outputs: CausalLMOutputWithPast = model(prefix_ids, use_cache=True)
prefix_cache = prefix_outputs.past_key_values

WORD = 'car'

word_ids = tokenizer.encode(
    WORD,
    return_tensors='pt',
    add_special_tokens=False,
).to('cuda')

input_ids = torch.cat([prefix_ids, word_ids, suffix_ids], dim=-1)
output_ids = model.generate(
    input_ids,
    past_key_values=prefix_cache,
    use_cache=True,
    max_new_tokens=50,
    do_sample=True,
)
output_ids = output_ids[0, input_ids.shape[-1]:]
output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
print(output_text)
