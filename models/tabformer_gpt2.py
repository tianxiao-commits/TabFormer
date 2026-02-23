import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from .gpt_native import NativeGPTLMHeadModel


class TabFormerGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config, vocab, native_gpt=False):
        """
        Args:
            config: Model configuration
            vocab: Vocabulary object
            native_gpt: If True, use native PyTorch GPT with RoPE and RMSNorm.
                       If False, use HuggingFace GPT2 (default for backward compatibility)
        """
        self.native_gpt = native_gpt
        self.vocab = vocab

        if native_gpt:
            # Use native PyTorch GPT implementation (torch.compile friendly)
            # Skip parent __init__ to avoid HuggingFace model initialization
            nn.Module.__init__(self)
            self.config = config
            self.transformer = NativeGPTLMHeadModel(config)
        else:
            # Use HuggingFace GPT2 implementation
            super().__init__(config)
            self.vocab = vocab

    def forward(
            self,
            input_ids=None,
            past=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            use_cache=True,
    ):
        if self.native_gpt:
            # Native GPT forward pass
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
            )
            lm_logits = outputs['logits']
            hidden_states = outputs['hidden_states']
            transformer_outputs = (hidden_states,)
        else:
            # HuggingFace GPT2 forward pass
            transformer_outputs = self.transformer(
                input_ids,
                past=past,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
            )
            hidden_states = transformer_outputs[0]
            lm_logits = self.lm_head(hidden_states)

        # lm_logits : [bsz x seq_len x vsz]
        # labels    : [bsz x seq_len]
        # When flatten is set to True:
        # seq_len = num_transactions * (num_columns + 2)  --> plus 2 because each transaction has BOS and EOS padding

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_labels = labels[..., 1:-1].contiguous()  # Remove first and last label: [BOS] and [EOS] tokens
            shift_logits = lm_logits[..., :-2, :].contiguous()  # Line up logits accordingly

            seq_len = shift_logits.size(1)
            total_lm_loss = 0
            field_names = self.vocab.get_field_keys(remove_target=True, ignore_special=True)

            for field_idx, field_name in enumerate(field_names):
                col_ids = list(range(field_idx, seq_len, len(field_names)))
                global_ids_field = self.vocab.get_field_ids(field_name)
                lm_logits_field = shift_logits[:, col_ids, :][:, :, global_ids_field]  # bsz * 10 * K
                lm_labels_field = shift_labels[:, col_ids]
                lm_labels_local_field = self.vocab.get_from_global_ids(global_ids=lm_labels_field,
                                                                       what_to_get='local_ids')

                loss_fct = CrossEntropyLoss()
                lm_loss_field = loss_fct(lm_logits_field.view(-1, len(global_ids_field)),
                                         lm_labels_local_field.view(-1))
                total_lm_loss += lm_loss_field

            outputs = (total_lm_loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)
