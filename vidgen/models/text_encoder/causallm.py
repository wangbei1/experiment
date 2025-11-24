import torch
import torch.nn as nn
import functools
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from vidgen.registry import MODELS


def setup_lm_fsdp_sync(model: nn.Module) -> FSDP:
    # LM FSDP always use FULL_SHARD among the node.
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in list(model.layers),
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=next(model.parameters()).dtype,
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model

@MODELS.register_module("causallm")
class CausalLMEncoder:
    def __init__(
        self,
        from_pretrained=None,
        model_max_length=120,
        device="cuda",
        dtype=torch.float,
        cache_dir=None,
        shardformer=True,
        local_files_only=False,
    ):
        assert from_pretrained is not None, "Please specify the path to the Llama model"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            from_pretrained, 
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            pad_token='[PAD]')
        self.model = AutoModelForCausalLM.from_pretrained(
            from_pretrained,
            cache_dir=cache_dir,
            local_files_only=local_files_only
            ).get_decoder().eval()
        if shardformer:
            self.model = setup_lm_fsdp_sync(self.model)

        self.model_max_length = model_max_length
        self.output_dim = self.model.config.hidden_size
        self.device = device

    def encode(self, text):
        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.model_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        
        input_ids = text_tokens_and_mask["input_ids"]
        attention_mask = text_tokens_and_mask["attention_mask"]
        input_ids[attention_mask==0] = 0
        
        with torch.no_grad():
            text_encoder_embs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state.detach()
            
        text_encoder_embs = text_encoder_embs[:, None]
            
        return dict(y=text_encoder_embs, mask=attention_mask)