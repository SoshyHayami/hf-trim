import torch
from .T5Trimmer import T5Trimmer

class LongT5Trimmer(T5Trimmer):
    def __init__(self, model, config, tokenizer):
        super().__init__(model, config, tokenizer)

    def initialize_new_model(self):
        # arch = self.config.architectures[0]
        arch = self.model.__class__.__name__
        if arch=='LongT5Model':
            from transformers import LongT5Model
            model = LongT5Model(self.config)
        elif arch=='LongT5ForConditionalGeneration':
            from transformers import LongT5ForConditionalGeneration
            model = LongT5ForConditionalGeneration(self.config)
        elif arch=='LongT5EncoderModel':
            from transformers import LongT5EncoderModel
            model = LongT5EncoderModel(self.config)
        else:
            raise NotImplementedError("ERROR: LongT5Trimmer does not support this architecture!")

        self.trimmed_model = model
