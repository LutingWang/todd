import einops
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gemma import GemmaForCausalLM, GemmaTokenizerFast

from todd.patches.torch import get_device
from todd.utils import TensorTreeUtil

WORD_TOKEN = '<word>'  # nosec B105
PRETRAINED = "pretrained/gemma/gemma-1.1-2b-it"

tokenizer: GemmaTokenizerFast = AutoTokenizer.from_pretrained(PRETRAINED)
model: GemmaForCausalLM = AutoModelForCausalLM.from_pretrained(
    PRETRAINED,
    device_map=get_device(),
    torch_dtype=torch.float16,
    revision="float16",
)

prompt = tokenizer.apply_chat_template(
    [
        dict(
            role='user',
            content=(
                "Imagine an instance of the given word and describe its "
                "appearance in a single sentence. Avoid describing the scene. "
                f"The word is: {WORD_TOKEN}"
            ),
        ),
    ],
    tokenize=False,
    add_generation_prompt=True,
)
prefix, suffix = prompt.split(WORD_TOKEN)
prefix_inputs = tokenizer(
    prefix,
    return_tensors='pt',
    add_special_tokens=False,
).to('cuda')
suffix_inputs = tokenizer(
    suffix,
    return_tensors='pt',
    add_special_tokens=False,
).to('cuda')
prefix_outputs: CausalLMOutputWithPast = model(**prefix_inputs, use_cache=True)
prefix_cache = prefix_outputs.past_key_values

words = ['person', 'car']

word_inputs = tokenizer(
    words,
    return_tensors='pt',
    add_special_tokens=False,
).to('cuda')
assert prefix_inputs.keys() == word_inputs.keys() == suffix_inputs.keys()

inputs: dict[str, torch.Tensor] = TensorTreeUtil.map(
    lambda *i: torch.cat(i, dim=-1),
    TensorTreeUtil.map(
        lambda i: einops.repeat(i, '1 l -> b l', b=len(words)),
        prefix_inputs,
    ),
    word_inputs,
    TensorTreeUtil.map(
        lambda i: einops.repeat(i, '1 l -> b l', b=len(words)),
        suffix_inputs,
    ),
)

output_ids = model.generate(
    **inputs,
    past_key_values=TensorTreeUtil.map(
        lambda i: einops.repeat(i, '1 h l d -> b h l d', b=len(words)),
        prefix_cache,
    ),
    use_cache=True,
    max_new_tokens=50,
)
output_ids = output_ids[:, inputs['input_ids'].shape[-1]:]
output_texts = [
    tokenizer.decode(oi, skip_special_tokens=True) for oi in output_ids
]
for output_text in output_texts:
    print(output_text)
