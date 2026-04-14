# common imports
from types import MethodType
from typing import Callable
import torch
import torch.nn as nn
from accelerate.hooks import add_hook_to_module
from transformers import PreTrainedModel

# simdiff methods

from simdiff.utils import TEXT_TOKEN, IGNORE_TOKEN, get_attr_by_name

# model types
from transformers import LlavaNextVideoForConditionalGeneration
from simdiff.models.nvila.llava_arch import _embed
try:
    from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
    from simdiff.models.llava_video.modeling_llava_video import prepare_inputs_labels_for_multimodal_get_patch_type
    SKIP_LLAVA_NEXT = False
except ModuleNotFoundError:
    SKIP_LLAVA_NEXT = True
    print("Skipping import from LLAVA-NEXT")

try:
    from llava.model import LlavaLlamaModel
    SKIP_NVILA = False
except:
    SKIP_NVILA = True
    print("Skipping import from VILA")

# replace methods
from simdiff.models.llava_next_video.modeling_llava_next_video import _merge_input_ids_with_image_features_get_token_type

from simdiff.models.minicpmv.modeling_minicpmv import get_vllm_embedding
from simdiff.models.qwen2.modeling_qwen2 import Qwen2Model_merge_then_fastv_cost_given_forward, Qwen2DecoderLayer_merge_then_prune_by_cost_forward, Qwen2SdpaAttention_merge_then_prune_by_cost_forward
def apply_simdiff(model, cost, similarity_lower_bound, ratio_lower_bound,merge_args=None,padding=-1):
    """
    Apply SimDiff to the model

    Args:
        model: the model to apply SimDiff to
        cost: the cost of the SimDiff
        similarity_lower_bound: the similarity lower bound of the SimDiff
        ratio_lower_bound: the ratio lower bound of the SimDiff
    """
    # from pdb import set_trace; set_trace()

    # LlavaNextVideo Model
    if isinstance(model, LlavaNextVideoForConditionalGeneration):
        model._merge_input_ids_with_image_features = MethodType(_merge_input_ids_with_image_features_get_token_type, model)

        llm_forward = Qwen2Model_merge_then_fastv_cost_given_forward
        decoder_forward = Qwen2DecoderLayer_merge_then_prune_by_cost_forward
        attention_forward = Qwen2SdpaAttention_merge_then_prune_by_cost_forward
        llm_key = "model"
        decoder_key = "layers"
        attention_key = "self_attn"

    # LlavaVideo Model
    elif (not SKIP_LLAVA_NEXT) and isinstance(model, LlavaQwenForCausalLM):
        model.prepare_inputs_labels_for_multimodal = MethodType(prepare_inputs_labels_for_multimodal_get_patch_type, model)

        llm_forward = Qwen2Model_merge_then_fastv_cost_given_forward
        decoder_forward = Qwen2DecoderLayer_merge_then_prune_by_cost_forward
        attention_forward = Qwen2SdpaAttention_merge_then_prune_by_cost_forward
        llm_key = "model"
        decoder_key = "layers"
        attention_key = "self_attn"

    # MiniCPM Model
    elif model.config.architectures[0] == "MiniCPMV":

        model.get_vllm_embedding = MethodType(get_vllm_embedding, model)
        llm_forward = Qwen2Model_merge_then_fastv_cost_given_forward
        decoder_forward = Qwen2DecoderLayer_merge_then_prune_by_cost_forward
        attention_forward = Qwen2SdpaAttention_merge_then_prune_by_cost_forward
        llm_key = "llm.model"
        decoder_key = "layers"
        attention_key = "self_attn"

    # NVILA Model
    elif (not SKIP_NVILA) and isinstance(model, LlavaLlamaModel):
        model._embed = MethodType(_embed, model)
        llm_forward = Qwen2Model_merge_then_fastv_cost_given_forward
        decoder_forward = Qwen2DecoderLayer_merge_then_prune_by_cost_forward
        attention_forward = Qwen2SdpaAttention_merge_then_prune_by_cost_forward
        llm_key = "llm.model"
        decoder_key = "layers"
        attention_key = "self_attn"

    elif isinstance(model, Qwen2_5_VLForConditionalGeneration):
        model.forward = MethodType(forward_qwen2_5_vl, model)

        llm_forward = Qwen2_5_VLModel_merge_then_fastv_cost_given_forward
        decoder_forward = Qwen2_5_VLDecoderLayer_merge_then_prune_by_cost_forward
        attention_forward = Qwen2_5_VLSdpaAttention_merge_then_prune_by_cost_forward
        llm_key = "model"
        decoder_key = "layers"
        attention_key = "self_attn"

    else:
        raise NotImplementedError

    replace_simdiff_forward(
        model,
        cost=cost,
        similarity_lower_bound=similarity_lower_bound,
        ratio_lower_bound=ratio_lower_bound,
        llm_forward=llm_forward,
        decoder_forward=decoder_forward,
        attention_forward=attention_forward,
        llm_key=llm_key,
        decoder_key=decoder_key,
        attention_key=attention_key,
        merge_args=merge_args,
        padding=padding
    )


def get_token_type(model):
    # LlavaNextVideo Model
    if isinstance(model, LlavaNextVideoForConditionalGeneration):
        model._merge_input_ids_with_image_features = MethodType(_merge_input_ids_with_image_features_get_token_type, model)

    # LlavaVideo Model
    elif (not SKIP_LLAVA_NEXT) and isinstance(model, LlavaQwenForCausalLM):
        model.prepare_inputs_labels_for_multimodal = MethodType(prepare_inputs_labels_for_multimodal_get_patch_type, model)

    # MiniCPM Model
    elif model.config.architectures[0] == "MiniCPMV":
        model.get_vllm_embedding = MethodType(get_vllm_embedding, model)

    # NVILA Model
    elif (not SKIP_NVILA) and isinstance(model, LlavaLlamaModel):
        model._embed = MethodType(_embed, model)
    else:
        raise NotImplementedError


def replace_simdiff_forward(
    module: torch.nn.Module,
    cost: float,
    similarity_lower_bound: float,
    ratio_lower_bound: float,
    llm_forward: Callable,
    decoder_forward: Callable,
    attention_forward: Callable,
    llm_key: str = "model",
    decoder_key: str = "layers",
    attention_key: str = "self_attn",
    merge_args: dict = None,
    padding: int =-1,
):
    """
    Replace the forward method of the model with the simdiff forward method.
    Make simdiff a property of the model.

    The keys are accessed in an hierarchical manner: llm_key -> decoder_key -> attention_key. Each key can have multiple hierarchies, e.g. "llm.model", which will be accessed by module.llm.model
    """

    if merge_args['merge_type']=='org':
        from simdiff.main import SimDiff
        merge_args.pop('merge_type')
        simdiff = SimDiff(cost, similarity_lower_bound, ratio_lower_bound,padding,**merge_args)
    elif merge_args['merge_type']=='new_topk':
        from simdiff.new_topk import SimDiff
        merge_args.pop('merge_type')
        simdiff = SimDiff(cost, similarity_lower_bound, ratio_lower_bound,padding,**merge_args)
    elif merge_args['merge_type']=='random':
        from simdiff.ran import SimDiff
        merge_args.pop('merge_type')
        simdiff = SimDiff(cost, similarity_lower_bound, ratio_lower_bound,padding,**merge_args)
    elif merge_args['merge_type']=='st_topk':
        from simdiff.st_topk import SimDiff
        merge_args.pop('merge_type')
        simdiff = SimDiff(cost, similarity_lower_bound, ratio_lower_bound,padding,**merge_args)


    module.simdiff = simdiff

    llm = get_attr_by_name(module, llm_key)
    assert isinstance(llm, PreTrainedModel), f"{llm_key} is not a PreTrainedModel"

    llm.simdiff = simdiff
    llm.forward = MethodType(llm_forward, llm)

    # import pdb; pdb.set_trace()

    decoder_layers = get_attr_by_name(llm, decoder_key)
    for i, decoder_layer in enumerate(decoder_layers):
        assert isinstance(decoder_layer, nn.Module), f"{decoder_key}[{i}] is not a nn.Module"

        decoder_layer.simdiff = simdiff
        decoder_layer.forward = MethodType(decoder_forward, decoder_layer)

        # ensure accelerate hooks are not removed
        if hasattr(decoder_layer, "_hf_hook"):
            decoder_layer._old_forward = MethodType(decoder_forward, decoder_layer)
            add_hook_to_module(decoder_layer, decoder_layer._hf_hook)

        qwen2_attention_instance = get_attr_by_name(decoder_layer, attention_key)
        assert isinstance(qwen2_attention_instance, nn.Module), f"{decoder_key}[{i}].self_attn is not a nn.Module"

        # replace the forward method of the attention layer
        qwen2_attention_instance.simdiff = simdiff
        qwen2_attention_instance.forward = MethodType(attention_forward, qwen2_attention_instance)
