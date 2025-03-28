import torch
from transformers import T5EncoderModel, T5Tokenizer
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
)

import todd

PRETRAINED = 'pretrained/t5/t5-large'
tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(PRETRAINED)
model = T5EncoderModel.from_pretrained(
    PRETRAINED,
    device_map='auto',
    torch_dtype='auto',
)

text = 'Studies have been shown that owning a dog is good for you'  # noqa: E501 pylint: disable=invalid-name
tokens = tokenizer(text, return_tensors='pt')
print(tokenizer.convert_ids_to_tokens(tokens.input_ids[0]))
# ['▁Studies', '▁have', '▁been', '▁shown', '▁that', '▁own', 'ing', '▁', 'a',
#  '▁dog', '▁is', '▁good', '▁for', '▁you', '</s>']

if todd.Store.cuda:  # pylint: disable=using-constant-test
    tokens = tokens.to('cuda')

with torch.no_grad():
    outputs: BaseModelOutputWithPastAndCrossAttentions = model(**tokens)

print(outputs.last_hidden_state[0, -1])
# tensor([ 0.0900, -0.0016, -0.0112,  ...,  0.0356, -0.0397,  0.0150],
#        device='cuda:0')
