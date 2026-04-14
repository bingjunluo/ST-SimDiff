import math
import os
from datetime import timedelta
from typing import List, Optional, Tuple, Union
from einops import rearrange
import torch.nn as nn
import numpy as np
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
import re
import yaml
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
    process_images
)
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_llama import LlavaConfig


# eval_logger = logging.getLogger("lmms-eval")
# import sys;sys.path.append("llava-video")
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav
from prepare_inputs import prepare
# try:
#     from llavavid.model.builder import load_pretrained_model
#     from llavavid.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
#     from llavavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
#     from llavavid.conversation import conv_templates, SeparatorStyle
#     from llavavid.mm_utils import tokenizer_image_token_qwen_merge, preprocess_qwen, preprocess_llama3
# except ImportError:
#     import llava
#     import pdb;pdb.set_trace()
#     if "llava-video-old" in llava.__file__:
#         from llava.model.language_model.llava_llama import LlavaConfig
#         from llava.model.language_model.llava_qwen import LlavaQwenConfig
#         from llava.model.builder import load_pretrained_model
#         from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
#         from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
#         from llava.conversation import conv_templates, SeparatorStyle

#         AutoConfig.register("llava_llama", LlavaConfig)
#         AutoConfig.register("llava_qwen", LlavaQwenConfig)
#     else:
#         eval_logger.debug("LLaVA-Video is not installed. Please install LLaVA-Video to use this model.")

# from llavavid.model.language_model.llava_qwen import LlavaQwenConfig
# from llavavid.model.language_model.llava_llama import LlavaConfig

# AutoConfig.register("llava_qwen", LlavaQwenConfig)
# AutoConfig.register("llava_llama", LlavaConfig)


AutoConfig.register("llava_llama", LlavaConfig)



@register_model("llava_vid")
class LlavaVid(lmms):
    """
    LlavaVid Model
    """

    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        truncation: Optional[bool] = True,
        torch_dtype: Optional[Union[str, torch.dtype]] = "float16",
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation=(
            "sdpa" if torch.__version__ >= "2.1.2" else "eager"
        ),  # inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
        device_map="cuda:0",
        conv_template="vicuna_v1",
        use_cache=True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        max_frames_num: int = 20,
        video_fps: int = 1,
        mm_resampler_type: str = "spatial_pool",
        mm_spatial_pool_stride: int = 2,
        mm_spatial_pool_out_channels: int = 1024,
        mm_spatial_pool_mode: str = "average",
        mm_resampler_location: str = "before",
        mm_newline_position: str = "grid",
        overwrite: bool = True,
        video_decode_backend: str = "decord",
        delay_load: bool = False,
        tie_weights: bool = True,
        force_sample: bool = False,
        add_time_instruction: bool = False,
        add_faster_video: bool = False,
        faster_token_stride: int = 10,
        prepare_config: str = None,
        **kwargs,
    ) -> None:
        super().__init__()
        eval_logger.info(prepare_config)
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and (device_map == "auto" or device_map == "balanced_low_0"):
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        
        self.pool_config=prepare_config.split('|') if prepare_config!='' else None
        self.pretrained = pretrained
        self.model_name = get_model_name_from_path(pretrained)
        self.video_decode_backend = video_decode_backend
        # self._config = AutoConfig.from_pretrained(self.pretrained)
        self.overwrite = overwrite
        self.mm_resampler_type = mm_resampler_type
        self.mm_spatial_pool_stride = int(mm_spatial_pool_stride)
        self.mm_spatial_pool_out_channels = int(mm_spatial_pool_out_channels)
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.max_frames_num = int(max_frames_num)
        eval_logger.warning(self.max_frames_num)
        self.fps = int(video_fps)
        self.mm_resampler_location = mm_resampler_location
        self.delay_load = delay_load
        self.force_sample = force_sample
        self.add_time_instruction = add_time_instruction
        print("force sample:", self.force_sample)
        # self.add_faster_video = add_faster_video
        # self.faster_token_stride = faster_token_stride
        self.torch_dtype = torch_dtype
        self.overwrite=overwrite
        if self.overwrite == True and self.pool_config==None:
            overwrite_config = {}
            # overwrite_config["mm_resampler_type"] = self.mm_resampler_type
            overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
            overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
            overwrite_config["mm_pooling_position"] = self.mm_resampler_location
            overwrite_config["mm_newline_position"] = mm_newline_position
            overwrite_config["delay_load"] = self.delay_load
            # overwrite_config["attn_implementation"] = attn_implementation

            if "vicuna" in self.pretrained.lower() or "yi" in self.pretrained.lower():
                cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)

                if "224" in cfg_pretrained.mm_vision_tower:
                    # suppose the length of text tokens is around 1000, from bo's report
                    least_token_number = self.max_frames_num * (16 // self.mm_spatial_pool_stride) ** 2 + 1000
                else:
                    least_token_number = self.max_frames_num * (24 // self.mm_spatial_pool_stride) ** 2 + 1000

                scaling_factor = math.ceil(least_token_number / 4096)
                if scaling_factor >= 2:
                    eval_logger.info(f"Scaling factor: {scaling_factor}")
                    # print(float(scaling_factor))
                    overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                    overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                    overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
                pretrained, None, self.model_name, device_map=self.device_map, torch_dtype=self.torch_dtype, overwrite_config=overwrite_config, attn_implementation=attn_implementation
            )
        else:
            
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, self.model_name,device_map="cuda")
        self._model.config.image_aspect_ratio = "resize"
        self._config = self._model.config

        # import pdb;pdb.set_trace()

        if self._tokenizer.pad_token_id is None:
            if "qwen" in self._tokenizer.name_or_path.lower():
                print("Setting pad token to bos token for qwen model.")
                self._tokenizer.pad_token_id = 151643

        self.model.eval()
        if tie_weights:
            self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context

        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

        

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def load_image(self, image_path):
        frame_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
        frame_files.sort()  # Ensure the frames are sorted if they are named sequentially

        # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
        num_frames_to_sample = 10

        total_frames = len(frame_files)

        sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

        # Read and store the sampled frames
        video = []
        for idx in sampled_indices:
            frame_path = frame_files[idx]
            try:
                with Image.open(frame_path) as img:
                    # Convert the PIL image to a numpy array if needed
                    # frame = np.array(img.convert('RGB'))
                    frame = img.convert("RGB")
                    video.append(frame)
            except IOError:
                print(f"Failed to read frame at path: {frame_path}")
        return video

    def get_seq_frames(self,total_num_frames, desired_num_frames):
        """
        Calculate the indices of frames to extract from a video.

        Parameters:
        total_num_frames (int): Total number of frames in the video.
        desired_num_frames (int): Desired number of frames to extract.

        Returns:
        list: List of indices of frames to extract.
        """

        # Calculate the size of each segment from which a frame will be extracted
        seg_size = float(total_num_frames - 1) / desired_num_frames

        seq = []
        for i in range(desired_num_frames):
            # Calculate the start and end indices of each segment
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))

            # Append the middle index of the segment to the list
            seq.append((start + end) // 2)
        return seq
    def load_video(self,video_path, num_clips=1, num_frms=4):
        """
        Load video frames from a video file.

        Parameters:
        video_path (str): Path to the video file.
        num_clips (int): Number of clips to extract from the video. Defaults to 1.
        num_frms (int): Number of frames to extract from each clip. Defaults to 4.

        Returns:
        list: List of PIL.Image.Image objects representing video frames.
        """

        # Load video frame from a directory


        # Load video with VideoReader
        vr = VideoReader(video_path, ctx=cpu(0))
        total_num_frames = len(vr)

        # Currently, this function supports only 1 clip
        assert num_clips == 1

        # Calculate desired number of frames to extract
        desired_num_frames = min(total_num_frames, num_frms)

        # Get indices of frames to extract
        frame_idx = self.get_seq_frames(total_num_frames, desired_num_frames)

        # Extract frames as numpy array
        img_array = vr.get_batch(frame_idx).asnumpy()  # (T H W C)
        clip_imgs = [Image.fromarray(img_array[i]) for i in range(desired_num_frames)]

        # Get original sizes of video frame
        original_size = (img_array.shape[-2], img_array.shape[-3])  # (W, H)
        original_sizes = (original_size,) * desired_num_frames

        return clip_imgs, original_sizes

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            videos = []
            for visual in visuals:
                video, frame_time, video_time,sizes = self.load_video(visual, num_frms=self.max_frames_num)
                video = self._image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda()
                if self.torch_dtype == "bfloat16":
                    video = video.bfloat16()
                else:
                    video = video.half()
                videos.append(video)

            qs = contexts
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            contxt_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], continuation)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().cuda()

            labels = input_ids.clone()
            # Context part no need to calculate for loss
            labels[0, : contxt_id.shape[1]] = -100

            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, images=videos, modalities="video")

            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : input_ids.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list
    def get_option_prompt(self,candidates, version="default"):
        option_prompt = ""
        options = []
        for idx, candidate in enumerate(candidates):
            choice = chr(ord("A") + idx)
            if version == "v4":
                option_prompt += f"({choice}) {candidate}\n"
            else:
                option_prompt += f"({choice}):{candidate} "
            options.append(choice)
        options = "(" + ",".join(options) + ")"
        return option_prompt, options
    def get_multiple_choice_prompt(self, conv_mode, question, candidates):
        if conv_mode == "multiple_choice_allvideo_v4":
            prompt = "You are a helpful expert in video analysis. Select the best option to answer the question. USER: <image>\nThe input consists of a sequence of key frames from a video.\nQuestion: %s\nOptions:\n%sOnly give the best option. \nASSISTANT:\nAnswer: Best option:("
            option_prompt, options = self.get_option_prompt(candidates, version="v4")
            prompt = prompt % (question, option_prompt)
        elif conv_mode == "multiple_choice_allvideo_34b_v4":
            prompt = "<|im_start|>system\n You are a helpful expert in video analysis. Select the best option to answer the question. <|im_end|>\n<|im_start|>user\n <image>\nThe input consists of a sequence of key frames from a video. Question: %s\nOptions:\n%sOnly give the best option. <|im_end|>\n<|im_start|>assistant\nAnswer: Best option:("
            option_prompt, options = self.get_option_prompt(candidates, version="v4")
            prompt = prompt % (question, option_prompt)
        else:
            raise ValueError(f"Unknown conv_mode: {conv_mode}")
        return prompt
    def get_vision_tower(self):
        return self.model.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.model.get_model().get_vision_tower()(images)
        image_features = self.model.get_model().mm_projector(image_features)
        return image_features

    def temporal_aggregation(self, image_features, temporal_aggregation):
        T, N, D = image_features.shape

        if temporal_aggregation == "concat":
            ## temporal cat
            image_features = image_features.view(T * N, D)
        elif temporal_aggregation == "spatial_1d_max_pool":
            ## horizontal max pool + temporal cat
            pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
            image_features = rearrange(image_features, 't n d -> t d n')
            image_features = pool2(image_features)
            image_features = rearrange(image_features, 't d n -> t n d', t=T)
            image_features = image_features.view(-1, D)
        elif temporal_aggregation == "spatial_1d_avg_pool":
            ## horizontal avg pool + temporal cat
            pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
            image_features = rearrange(image_features, 't n d -> t d n')
            image_features = pool2(image_features)
            image_features = rearrange(image_features, 't d n -> t n d', t=T)
            image_features = image_features.view(-1, D)
        elif temporal_aggregation == "spatial_2d_max_pool":
            ## spatial max pool + temporal cat
            n0 = n1 = int(math.sqrt(N))
            pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            image_features = rearrange(image_features, 't (n0 n1) d -> d t n0 n1', n0=n0, n1=n1)
            image_features = pool2(image_features)
            image_features = rearrange(image_features, 'd t n0 n1 -> (t n0 n1) d')
        elif temporal_aggregation == "spatial_2d_avg_pool":
            ## spatial avg pool + temporal cat
            n0 = n1 = int(math.sqrt(N))
            pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            image_features = rearrange(image_features, 't (n0 n1) d -> d t n0 n1', n0=n0, n1=n1)
            image_features = pool2(image_features)
            image_features = rearrange(image_features, 'd t n0 n1 -> (t n0 n1) d')
        elif temporal_aggregation == "spatial_temporal_pool":
            ## spatial pool + temporal pool
            pooling_size = (16, 12, 12)
            n0 = n1 = int(math.sqrt(N))
            pool3 = nn.AdaptiveAvgPool3d(pooling_size)
            image_features = rearrange(image_features, 't (n0 n1) d -> d t n0 n1', n0=n0, n1=n1)
            image_features = pool3(image_features)
            image_features = rearrange(image_features, 'd t n0 n1 -> (t n0 n1) d')
        elif temporal_aggregation == "temporal_global_pool":
            ## temporal pool
            image_features = torch.mean(image_features, dim=0)
        else:
            raise ValueError(f'Unknown temporal aggregation method: {temporal_aggregation}')

        image_features = image_features.unsqueeze(0)
        return image_features
    
    def prepare_slowfast(self, image_features, temporal_aggregation):
        T, N, D = image_features.shape

        # Example: temporal_aggregation = "slowfast-slow_10frms_spatial_1d_max_pool-fast_4x4"
        slowfast_match = re.match(r'^slowfast-slow_(\d+)frms_(\w+)-fast_(\d+)x(\d+)$', temporal_aggregation)
        if not slowfast_match:
            raise ValueError(f'Failed to parse the temporal aggregation for slowfast: {temporal_aggregation}')
        num_slowpath = int(slowfast_match.group(1))
        slowpath_temporal_aggregation = slowfast_match.group(2)
        fastpath_output_size = (
            int(slowfast_match.group(3)),
            int(slowfast_match.group(4)),
        )

        # Prepare slow pathway
        slowpath_idx = torch.linspace(0, T, num_slowpath + 1)
        slowpath_idx = slowpath_idx.to(torch.int32).tolist()
        slowpath_idx.pop()
        slowpath_features = self.temporal_aggregation(
            image_features[slowpath_idx],
            slowpath_temporal_aggregation,
        )

        # Prepare fast pathway
        fastpath_features = image_features  # [T N D]
        pool2 = nn.AdaptiveAvgPool2d(fastpath_output_size)
        n0 = n1 = int(math.sqrt(N))
        fastpath_features = rearrange(fastpath_features, 't (n0 n1) d -> d t n0 n1', n0=n0, n1=n1)  # [T N D] -> [D T N0 N1]
        fastpath_features = pool2(fastpath_features)
        fastpath_features = rearrange(fastpath_features, 'd t n0 n1 -> (t n0 n1) d')  # [D T N0/2 N1/2] -> [-1 D]
        fastpath_features = fastpath_features.unsqueeze(0)

        slowfast_features=[]
        tmp=0
        for i in range(len(fastpath_features)):
            if i in slowpath_idx:
                slowfast_features.append(slowpath_features[tmp])
                tmp+=1
            else:
                slowfast_features.append(fastpath_features[i])
        #slowfast_features=torch.cat(slowfast_features).unsqueeze(0)
        
        slowfast_features = torch.cat((slowpath_features, fastpath_features), dim=1)

        return slowfast_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, temporal_aggregation=None,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')

            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                
        else:

            image_features = self.encode_images(images)

        if temporal_aggregation and \
           temporal_aggregation.lower() != 'none' and \
           temporal_aggregation.lower() != 'false':
            if temporal_aggregation.startswith('slowfast'):
                image_features = self.prepare_slowfast(image_features, temporal_aggregation)
            else:
                image_features = self.temporal_aggregation(image_features, temporal_aggregation)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.model.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.model.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        temp=0
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # if self.task_dict[task][split][doc_id]["duration"] != "short":
            # # if doc_id != 112:
            #     # import pdb;pdb.set_trace()
            #     res.append("A")
            #     pbar.update(1)
            #     continue
            # encode, pad, and truncate contexts for this batch
            # import pdb;pdb.set_trace()
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            # visuals = [visuals]
            # visuals = self.flatten(visuals)
            videos = []
            try:
                # for visual in visuals:
                if len(visuals) == 1:
                    if self.video_decode_backend == "decord":
                        video, sizes = self.load_video(visuals[0], num_frms=self.max_frames_num)
                    elif self.video_decode_backend == "pyav":
                        video, frame_time, video_time = read_video_pyav(visuals[0], self.max_frames_num, self.fps, force_sample=self.force_sample)
                    elif self.video_decode_backend == "image":
                        video = self.load_image(visuals[0])
                else:
                    if task == "seedbench":
                        video = visuals
                        frame_time = "1.00s"
                        video_time = 1
                    elif "mvbench" in task:
                        # video = visuals
                        # Reference: https://github.com/jayleicn/TVQA/blob/dfb0e5fe4582efca574dfddfeafd1008db3b33ef/data/README.md?plain=1#L50C34-L50C60
                        fps = 3
                        video_time = len(visuals) / fps
                        sampled_indices = np.linspace(0, len(visuals) - 1, self.max_frames_num, dtype=int)
                        frame_idx = sampled_indices.tolist()
                        frame_time = [i / fps for i in frame_idx]
                        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
                        video = [visuals[i] for i in frame_idx]

                #video = self._image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda()
                video=process_images(video, self._image_processor, self._config).to(dtype=torch.float16, device="cuda", non_blocking=True)

                videos.append(video)

            except Exception as e:
                # import pdb;pdb.set_trace()
                eval_logger.info(f"{e}")
                eval_logger.info(f"Video {visuals} can not load, check the source")
                video_path = "\n".join(visuals)
                res.append(f"Video {video_path} can not load, check the source")
                pbar.update(1)
                continue

            qs = contexts
            # import pdb;pdb.set_trace()
            if self.add_time_instruction:
                time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video)} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
                qs = f"{time_instruciton}\n{qs}"
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN * len(videos) + "\n" + qs

            # This is much safer for llama3, as we now have some object type in it
            
            if self.conv_template=='multiple_choice':
                conv = conv_templates["vicuna_v1"].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)

                prompt = conv.get_prompt()
                prompt_list=prompt.split('\n')
                prompt_list[0]='You are a helpful expert in video analysis. Select the best option to answer the question. USER: <image>\nThe input consists of a sequence of key frames from a video.'
                prompt_list[1]='Question: '+prompt_list[1]+'\nOptions:'
                prompt_list[-1]='Only give the best option. '
                prompt_list.append('ASSISTANT:')
                prompt_list.append('Answer: Best option:(')
                prompt='\n'.join(prompt_list)
                prompt=re.sub('A\.', '(A)', prompt)
                prompt=re.sub('B\.', '(B)', prompt)
                prompt=re.sub('C\.', '(C)', prompt)
                prompt=re.sub('D\.', '(D)', prompt)
                prompt=re.sub('E\.', '(E)', prompt)
                
            elif self.conv_template=='vcgbench':
                conv = conv_templates["vicuna_v1"].copy()
                
                prompt = "USER: <image>\nThe input consists of a sequence of key frames from a video. Answer concisely with overall content and context of the video, highlighting any significant events, characters, or objects that appear throughout the video. Question: %s \nASSISTANT:\nAnswer: In the video,"
                prompt = prompt % contexts

            else:
                if "llama_3" in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()

                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)

                prompt = conv.get_prompt()
            eval_logger.info(prompt)
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            if "llama_3" in self.conv_template:
                pad_token_ids = 0  # lmms-lab/llama3-llava-8b is trained on this pad token id. You may need to customize this for other models.
            attention_masks = input_ids.ne(pad_token_ids).long().cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]

            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            cur_prompt = qs


            gen_kwargs["sizes"]=sizes
            gen_kwargs["temperature"] = 0

            gen_kwargs["top_p"] = None

            gen_kwargs["num_beams"] = 1
            # import pdb;pdb.set_trace()
            #torch.save(input_ids,f'/root/autodl-tmp/lmms-eval/tensor/llava_input{temp}.pt')
            
            #torch.save(videos[0],f'/root/autodl-tmp/lmms-eval/tensor/llava_image{temp}.pt')
            if self.pool_config!=None:
                (
                    inputs,
                    position_ids,
                    attention_mask,
                    _,
                    inputs_embeds,
                    _
                ) = prepare(self.model,self.config,self.device,self.pool_config).prepare_inputs_labels_for_multimodal(
                    input_ids,
                    None,
                    None,
                    None,
                    None,
                    videos[0],
                    temporal_aggregation='slowfast-slow_10frms_spatial_1d_max_pool-fast_4x4',
                )
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        direct_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        do_sample=True if gen_kwargs["temperature"] > 0 else False,
                        temperature=gen_kwargs["temperature"],
                        top_p=gen_kwargs["top_p"],
                        num_beams=gen_kwargs["num_beams"],
                        max_new_tokens=128,
                    )
                    # output_ids_2 = self.model.generate(inputs=input_ids, images=videos, attention_mask=attention_masks, modalities="video", do_sample=False, max_new_tokens=50,stopping_criteria=[stopping_criteria])
                    # output_ids = self.model.generate(inputs=input_ids, images=videos, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=50,use_cache=True)
                #torch.save(output_ids,f'/root/autodl-tmp/lmms-eval/tensor/llava_output{temp}.pt')
                #output_ids=torch.load(f'/root/autodl-tmp/lmms-eval/tensor/sf_output{temp}.pt')
            else:

                # These steps are not in LLaVA's original code, but are necessary for generation to work
                # TODO: attention to this major generation step...

                output_ids = self.model.generate(
                    inputs=input_ids,
                    images=videos,
                    attention_mask=attention_masks,
                    modalities="video",
                    use_cache=self.use_cache,
                    stopping_criteria=[stopping_criteria],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=128,
 
                )
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            

            temp+=1
            eval_logger.debug(f"Answer: {outputs}")
            # import pdb;pdb.set_trace()
            
            res.append(outputs)
            pbar.update(1)
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVAVid")
