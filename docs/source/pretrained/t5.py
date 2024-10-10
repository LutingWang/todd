import torch
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

import todd

tokenizer: T5Tokenizer = AutoTokenizer.from_pretrained(
    'pretrained/t5/t5-large',
)
model = T5EncoderModel.from_pretrained('pretrained/t5/t5-large')
tokens = tokenizer(
    'Studies have been shown that owning a dog is good for you',
    return_tensors='pt',
)
print(tokenizer.convert_ids_to_tokens(tokens['input_ids'][0]))
# ['▁Studies', '▁have', '▁been', '▁shown', '▁that', '▁own', 'ing', '▁', 'a',
#  '▁dog', '▁is', '▁good', '▁for', '▁you', '</s>']
if todd.Store.cuda:  # pylint: disable=using-constant-test
    model = model.cuda()
    tokens = tokens.to('cuda')
with torch.no_grad():
    outputs: BaseModelOutput = model(**tokens)
print(outputs.last_hidden_state[0, -1])
# tensor([ 0.0900, -0.0016, -0.0112,  ...,  0.0356, -0.0397,  0.0150],
#        device='cuda:0')
