import torch
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

tokenizer: T5Tokenizer = AutoTokenizer.from_pretrained(
    'pretrained/t5/t5-large',
)
model = T5EncoderModel.from_pretrained('pretrained/t5/t5-large')
tokens = tokenizer(
    'Studies have been shown that owning a dog is good for you',
    return_tensors='pt',
)
with torch.no_grad():
    outputs: BaseModelOutput = model(**tokens)
print(outputs.last_hidden_state)
