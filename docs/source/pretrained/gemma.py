import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gemma import GemmaForCausalLM, GemmaTokenizerFast

import todd

PRETRAINED = 'pretrained/gemma/gemma-1.1-2b-it'

tokenizer: GemmaTokenizerFast = AutoTokenizer.from_pretrained(PRETRAINED)
model: GemmaForCausalLM = AutoModelForCausalLM.from_pretrained(
    PRETRAINED,
    device_map='auto',
    torch_dtype='auto',
    revision='float16',
)

WORD_TOKEN = '<word>'  # nosec B105
TEMPLATE = (
    "Imagine an instance of the given word and describe its appearance in a "
    "single sentence. Be creative and avoid describing the scene. The word is"
    f": {WORD_TOKEN}"
)

conversation = [dict(role='user', content=TEMPLATE)]
inputs = tokenizer.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=False,
)
prefix, suffix = inputs.split(WORD_TOKEN)
prefix_ids: torch.Tensor = tokenizer.encode(
    prefix,
    add_special_tokens=False,
    return_tensors='pt',
)
suffix_ids: torch.Tensor = tokenizer.encode(
    suffix,
    add_special_tokens=False,
    return_tensors='pt',
)
if todd.Store.cuda:  # pylint: disable=using-constant-test
    prefix_ids = prefix_ids.cuda()
    suffix_ids = suffix_ids.cuda()
prefix_outputs: CausalLMOutputWithPast = model(prefix_ids, use_cache=True)
prefix_cache = prefix_outputs.past_key_values

WORD = 'car'

word_ids: torch.Tensor = tokenizer.encode(
    WORD,
    add_special_tokens=False,
    return_tensors='pt',
)
if todd.Store.cuda:  # pylint: disable=using-constant-test
    word_ids = word_ids.cuda()

input_ids = torch.cat([prefix_ids, word_ids, suffix_ids], -1)
output_ids = model.generate(
    input_ids,
    past_key_values=prefix_cache,
    use_cache=True,
    max_new_tokens=50,
    do_sample=True,
)

_, input_length = input_ids.shape
output_ids = output_ids[0, input_length:]
output_text = tokenizer.decode(output_ids, True)
print(output_text)
