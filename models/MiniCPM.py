

from functools import partial
from pathlib import Path
import random
import pandas as pd
from transformers import AutoModel, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM, PeftModel, AutoPeftModel
import torch
import torch.nn as nn
import copy
import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer

from constants import TRANSFORMS
from models.BaseModel import BaseModel
import pickle
from torchtune.modules import RMSNorm

from models.utils import xavier_init
from mamba_ssm import Mamba, Mamba2

# Dataset code copied and modified from https://github.com/OpenBMB/MiniCPM-V


llama3_chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}"

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        args,
        raw_data,
        transform,
        tokenizer,
        slice_config,
        llm_type="minicpm",
        patch_size=14,
        query_nums=64,
        batch_vision=False,
    ):
        super(SupervisedDataset, self).__init__()
        self.args = args
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.transform = transform
        self.slice_config = slice_config
        self.llm_type = llm_type
        self.patch_size = patch_size
        self.query_nums=query_nums
        self.batch_vision = batch_vision

        # load data
        data_path = Path(args['data_dir']) / args['dataset']
        descriptor = pd.read_csv(data_path / 'data.csv')

        all_data = {}
        for idx, row in descriptor.iterrows():
            img_path = data_path / 'images' / row['name']
            ref_img_path = data_path/ 'images'/ row['reference'] if 'reference' in row else None
            label = row[args['label_name']] * args['label_scale']
            
            all_data[str(img_path.resolve())] = (row['prompt'] if 'prompt' in row else None, img_path, ref_img_path, label)

        self.agi_data = all_data
        

    @staticmethod
    def load_data(args, model): # pass model here, ugly 😕
        # return tuple of SupervisedDataset, train, val, test
        tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
        train_raw_data = json.load(open(args['data_path'], "r"))
        # train_raw_data = random.sample(train_raw_data, len(train_raw_data) // 100)  # debug
        # val_raw_data = json.load(open(args['eval_data_path'], "r"))

        # use eval_data_res_path, same structure with val_raw_data
        val_raw_data = json.load(open(args['eval_data_res_path'], "r"))
        # val_raw_data = random.sample(val_raw_data, len(val_raw_data) // 100)  # debug

        for i in range(len(val_raw_data)):
            val_raw_data[i]['conversations'][1]['content'] = val_raw_data[i]['response_0']
            val_raw_data[i]['conversations'][3]['content'] = val_raw_data[i]['response_1']
        
        if args['dataset'] == 'aigciqa-30k':
            # aigciqa-30k has test set
            test_raw_data = json.load(open(args['test_data_path'], "r"))
            for i in range(len(test_raw_data)):
                test_raw_data[i]['conversations'][1]['content'] = test_raw_data[i]['response_0']
                test_raw_data[i]['conversations'][3]['content'] = test_raw_data[i]['response_1']
        else:
            test_raw_data = val_raw_data

        # model.model is MiniCPMIQA
        model = model.model

        if hasattr(model.config, "slice_config"):
            model.config.slice_config.max_slice_nums = args['max_slice_nums']
            slice_config = model.config.slice_config.to_dict()
        else:
            model.config.max_slice_nums = args['max_slice_nums']
            slice_config = model.config.to_dict()

        if hasattr(model.config, "batch_vision_input"):
            batch_vision = model.config.batch_vision_input
        else:
            batch_vision = False

        train_dataset = SupervisedDataset(
            args=args,
            raw_data=train_raw_data,
            tokenizer=tokenizer,
            transform=model.transform,
            slice_config=slice_config,
            llm_type=args['llm_type'],
            patch_size=model.config.patch_size,
            query_nums=model.config.query_num,
            batch_vision=batch_vision
        )

        val_dataset = SupervisedDataset(
            args=args,
            raw_data=val_raw_data,
            tokenizer=tokenizer,
            transform=model.transform,
            slice_config=slice_config,
            llm_type=args['llm_type'],
            patch_size=model.config.patch_size,
            query_nums=model.config.query_num,
            batch_vision=batch_vision
        )

        if args['dataset'] == 'aigciqa-30k':
            test_dataset = SupervisedDataset(
                args=args,
                raw_data=test_raw_data,
                tokenizer=tokenizer,
                transform=model.transform,
                slice_config=slice_config,
                llm_type=args['llm_type'],
                patch_size=model.config.patch_size,
                query_nums=model.config.query_num,
                batch_vision=batch_vision
            )
        else:
            test_dataset = val_dataset

        return train_dataset, val_dataset, test_dataset

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        image = Image.open(self.raw_data[i]["image"]).convert("RGB")

        self.raw_data[i]["conversations"] = self.raw_data[i]["conversations"][:3]

        ret = preprocess(
            image,
            self.raw_data[i]["conversations"],
            self.tokenizer,
            self.transform,
            query_nums=self.query_nums,
            slice_config=self.slice_config,
            llm_type=self.llm_type,
            patch_size=self.patch_size,
            batch_vision=self.batch_vision,
        )
        ret = dict(
            input_ids=ret["input_ids"],
            position_ids=ret["position_ids"],
            labels=ret["target"],
            attention_mask=torch.ones_like(ret["input_ids"], dtype=torch.bool),
            pixel_values=ret["pixel_values"],
            tgt_sizes=ret["tgt_sizes"],
            image_bound=ret["image_bound"],
        )

        # return ret, image_path and label, image_path is used for cache the output of the model
        img_path = self.raw_data[i]['image']
        if img_path in self.agi_data:
            lbl = self.agi_data[img_path][3]
        else:
            lbl = self.agi_data[img_path.replace('/home', '/mnt/ExtendHdd')][3]
        return ret, img_path, torch.tensor(lbl, dtype=torch.float32) # x, y

def data_collator(examples, padding_value=0, max_length=2048):
    def trim_and_pad(seq, batch_first, padding_value):
        return pad_sequence([s[:max_length] for s in seq], batch_first=True, padding_value=padding_value)

    input_ids = trim_and_pad(
        [example["input_ids"] for example in examples],
        batch_first=True,
        padding_value=padding_value,
    )
    position_ids = trim_and_pad(
        [example["position_ids"] for example in examples],
        batch_first=True,
        padding_value=padding_value,
    )
    targets = trim_and_pad(
        [example["labels"] for example in examples],
        batch_first=True,
        padding_value=-100,
    )
    attention_mask = trim_and_pad(
        [example["attention_mask"] for example in examples],
        batch_first=True,
        padding_value=padding_value,
    )
    pixel_values = [example["pixel_values"] for example in examples]
    image_bound = [example["image_bound"] for example in examples]
    tgt_sizes = [example["tgt_sizes"] for example in examples]
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": targets,
        "attention_mask": attention_mask,
        "image_bound": image_bound,
        "tgt_sizes": tgt_sizes,
        "pixel_values": pixel_values,
    }


def conversation_to_ids(conversation, tokenizer, llm_type=None):
    """
    for single image multi-turn conversation
    conversation: [{'role': 'user', 'content': 'Describe this image'},
                   {'role': 'assistant', 'content': 'This is a cat.'}]
    """
    if llm_type == "llama3":
        input_ids, context, raw_msg = conversation_to_ids_llama3(
            conversation, tokenizer
        )
    else:
        input_ids, context, raw_msg = conversation_to_ids_minicpm(
            conversation, tokenizer
        )

    ids = torch.from_numpy(np.hstack(input_ids, dtype=np.int32))
    context = torch.from_numpy(np.hstack(context, dtype=np.int8))

    # build target, what is this target
    target = torch.full_like(ids, -100, dtype=torch.int32)
    for i in range(1, len(ids)):
        if context[i] == 0:
            target[i - 1] = ids[i]
        if context[i] == 1 and context[i - 1] == 0:
            if hasattr(tokenizer, "eot_id"):
                target[i - 1] = tokenizer.eot_id
            else:
                target[i - 1] = tokenizer.eos_id

    # build image bound
    image_start_tokens = torch.where(ids == tokenizer.im_start_id)[0]
    image_start_tokens += 1
    image_end_tokens = torch.where(ids == tokenizer.im_end_id)[0]
    if len(image_start_tokens) != len(image_end_tokens):
        print("image start token != image end tokens")
        
    if len(image_start_tokens) > 0:
        image_bound = torch.hstack(
            [image_start_tokens.unsqueeze(-1), image_end_tokens.unsqueeze(-1)]
        )
    else:
        image_bound = []

    position_ids = torch.arange(ids.size(0)).long()
    return {
        "input_ids": ids,
        "target": target,
        "image_bound": image_bound,
        "raw_msg": raw_msg,
        "position_ids": position_ids
    }


def conversation_to_ids_minicpm(conversation, tokenizer):
    raw_msg = ""
    input_ids = []
    context = []
    for idx, msg in enumerate(conversation):
        role = msg["role"]
        message = msg["content"]
        assert role in ["user", "assistant"]
        if role == "user":
            prefix = "<用户>"
        else:
            prefix = "<AI>"
        # append eos
        if idx == len(conversation) - 1:
            message = message + tokenizer.eos_token
        prefix_ids = tokenizer.encode(prefix)[1:]  # remove bos
        message_ids = tokenizer.encode(message)[1:]

        input_ids.append(prefix_ids)
        input_ids.append(message_ids)

        context.append(np.ones((len(prefix_ids),), dtype=np.int8))
        if role == "assistant":
            context.append(np.zeros((len(message_ids),), dtype=np.int8))
        else:
            context.append(np.ones((len(message_ids),), dtype=np.int8))

        raw_msg += prefix + message

    return input_ids, context, raw_msg


def conversation_to_ids_llama3(conversation, tokenizer):
    raw_msg = ""
    input_ids = []
    context = []
    raw_msg = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False, chat_template=llama3_chat_template,
    )
    input_ids = tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=False, chat_template=llama3_chat_template,
    )
    input_ids = np.array(input_ids)

    start_header_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    )[0]
    assistant_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("assistant")
    )[0]
    end_header_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    )[0]
    eot_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("<|eot_id|>"))[0]

    context = np.ones_like(input_ids, dtype=np.int8)

    for assistant_idx in assistant_idxs:
        if assistant_idx in set((start_header_idxs + end_header_idxs) / 2):
            st = assistant_idx + 3  # assistant<|end_header_id|>\n\n
            for eot_idx in eot_idxs:
                if eot_idx > st:
                    context[st: eot_idx + 1] = 0
                    break

    input_ids = np.hstack(input_ids)
    context = np.hstack(context)

    return input_ids, context, raw_msg


def preprocess(
    image,
    conversation,
    tokenizer,
    transform,
    query_nums=64,
    slice_config=None,
    llm_type=None,
    patch_size=14,
    batch_vision=False,
):
    """
    single image preprocess, the image will be placed at the top of the conversation
    """
    conversation = copy.deepcopy(conversation)
    # assert len(conversation) > 1, "conversation length must large than 2"
    assert conversation[0]["role"] == "user", "the first role must be user"

    if slice_config is not None:
        assert isinstance(slice_config, Dict)
        assert "patch_size" in slice_config
        assert "max_slice_nums" in slice_config
        assert "scale_resolution" in slice_config
    default_image_placeholder = (
        tokenizer.im_start + tokenizer.unk_token * query_nums + tokenizer.im_end
    )
    if slice_config:
        images = []
        source_image, patches, best_grid = slice_image(
            image,
            slice_config["max_slice_nums"],
            slice_config["scale_resolution"],
            slice_config["patch_size"],
        )
        images.append(source_image)
        image_placeholder = default_image_placeholder
        if len(patches) > 0:
            for i in range(len(patches)):
                for j in range(len(patches[0])):
                    images.append(patches[i][j])

            image_placeholder += get_grid_placeholder(
                tokenizer, best_grid, query_nums)
        images = [transform(i) for i in images]
    else:
        images = [transform(image)]
        image_placeholder = default_image_placeholder
    if "<image>" in conversation[0]["content"]:
        conversation[0]["content"] = conversation[0]["content"].replace(
            "<image>", image_placeholder
        )
    else:
        conversation[0]["content"] = (
            image_placeholder + "\n" + conversation[0]["content"]
        )

    input_dict = conversation_to_ids(conversation, tokenizer, llm_type)

    if batch_vision:
        tgt_sizes = []
        reshape_images = []
        for image in images:
            H, W = image.shape[1:]
            reshape_image = reshape_by_patch(image, patch_size)
            reshape_images.append(reshape_image)
            tgt_sizes.append([H // patch_size, W // patch_size])
        if tgt_sizes:
            tgt_sizes = torch.Tensor(tgt_sizes).type(torch.int32)

        input_dict["pixel_values"] = reshape_images
        input_dict["tgt_sizes"] = tgt_sizes

    else:
        input_dict["pixel_values"] = images
        input_dict["tgt_sizes"] = []

    return input_dict


def slice_image(
    image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False
):
    original_size = image.size
    original_width, original_height = original_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / \
        (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    source_image = None
    best_grid = None
    patches = []

    if multiple <= 1 or never_split:
        # dont need to slice, upsample
        best_size = find_best_resize(
            original_size, scale_resolution, patch_size, allow_upscale=True
        )
        source_image = image.resize(best_size, Image.Resampling.BICUBIC)
    else:
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        # source image, down-sampling and ensure divided by patch_size
        best_resize = find_best_resize(
            original_size, scale_resolution, patch_size)
        source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
        candidate_grids = []

        # find best grid
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        refine_size = get_refine_size(
            original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
        )

        refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)
        patches = split_to_patches(refine_image, best_grid)

    return source_image, patches, best_grid


def ensure_divide(length, patch_size):
    return max(round(length / patch_size) * patch_size, patch_size)


def find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / height
        height = int(scale_resolution / math.sqrt(r))
        width = int(height * r)
    best_width = ensure_divide(width, patch_size)
    best_height = ensure_divide(height, patch_size)
    return (best_width, best_height)


def get_refine_size(
    original_size, grid, scale_resolution, patch_size, allow_upscale=False
):
    width, height = original_size
    grid_x, grid_y = grid

    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)

    grid_width = refine_width / grid_x
    grid_height = refine_height / grid_y

    best_grid_size = find_best_resize(
        (grid_width, grid_height),
        scale_resolution,
        patch_size,
        allow_upscale=allow_upscale,
    )

    refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)

    return refine_size


def split_to_patches(image, grid):
    patches = []
    width, height = image.size
    grid_x = int(width / grid[0])
    grid_y = int(height / grid[1])

    for i in range(0, height, grid_y):
        images = []
        for j in range(0, width, grid_x):
            box = (j, i, j + grid_x, i + grid_y)
            patch = image.crop(box)
            images.append(patch)
        patches.append(images)

    return patches


def get_grid_placeholder(tokenizer, grid, query_num):
    image_placeholder = (
        tokenizer.im_start + tokenizer.unk_token * query_num + tokenizer.im_end
    )

    cols = grid[0]
    rows = grid[1]
    slices = []
    for i in range(rows):
        lines = []
        for j in range(cols):
            lines.append(image_placeholder)
        slices.append("".join(lines))
    slice_placeholder = tokenizer.slice_start + \
        "\n".join(slices) + tokenizer.slice_end
    return slice_placeholder


def reshape_by_patch(image_tensor, patch_size):
    """
    :param image_tensor: shape [3, H, W]
    :param patch_size:
    :return: [3, patch_size, HW/patch_size]
    """
    patches = torch.nn.functional.unfold(
        image_tensor, (patch_size, patch_size), stride=(patch_size, patch_size)
    )

    patches = patches.reshape(image_tensor.size(0), patch_size, patch_size, -1)
    patches = patches.permute(0, 1, 3, 2).reshape(
        image_tensor.size(0), patch_size, -1)
    return patches

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

class xLSTMPooler(nn.Module):
    def __init__(self, args, emb_dim=512, context_length=512, num_blocks=4):
        super().__init__()
        self.args = args
        self.xlstm_cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=context_length,
            num_blocks=num_blocks,
            embedding_dim=emb_dim,
            slstm_at=[1],
        )
        self.xlstm_stack = xLSTMBlockStack(self.xlstm_cfg)

    def forward(self, x):
        return self.xlstm_stack(x)

'''
should use SupervisedDataset
'''
class MiniCPMIQA(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.use_cache = False
        self.inference_cache = {}   # store the inference result to disk, this cache stores the img_path to logits pickle mapping
        # pretrained_checkpoint defined in yaml file, lora finetuned        model_name = "openbmb/MiniCPM-Llama3-V-2_5"
        
        model_name = "openbmb/MiniCPM-Llama3-V-2_5"
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).eval()    # fp16 to reduce memory usage
        self.model = PeftModel.from_pretrained(self.model, self.args['pretrained_checkpoint'], trust_remote_code=True, torch_dtype=torch.float16).eval()
    
        for param in self.model.parameters():
            param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(self.model.config.vocab_size, 1024),  # [128256, 1024], 128256 is too much.
            nn.Linear(1024, 1)
        )
        self.reset_parameters()


    def reset_parameters(self):
        self.fc.apply(xavier_init)

    def collate_fn(self, batch):
        ret, img_path, y = map(list, zip(*batch))
        ret = data_collator(ret, max_length=self.args['model_max_length'])
        return (ret, img_path), torch.tensor(y)

    def get_dataset_class(self):
        return SupervisedDataset

    def forward_logits(self, x, y):
        # x is the inputs
        # inference
        x, img_paths = x # img_path is list of input images path, [batch_size]

        img_path = img_paths[0] # force batch_size to 1
        if self.use_cache:
            if img_path in self.inference_cache:
                logits_file = self.inference_cache[img_path]
                with open(logits_file, 'rb') as f:
                    logits = pickle.load(f)
                logits.to(x['input_ids'].device)
            else:
                with self.model._enable_peft_forward_hooks(**x):
                    outputs = self.model.base_model(data=x, use_cache=False)
                    outputs.logits.requires_grad = False
                logits = outputs.logits
                # create cache file for this img_path
                cache_file = f"{self.args['cache_dir']}/{self.args['model']}-{self.args['run_name']}{img_path.replace('/', '-')}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(logits.clone().detach().cpu(), f)
                self.inference_cache[img_path] = cache_file  # put on cpu to save vram
        else:   
            # cache cost too much ram or disk...
            with self.model._enable_peft_forward_hooks(**x):
                outputs = self.model.base_model(data=x, use_cache=True)
            logits = outputs.logits
        return logits

    def forward(self, x, y):
        logits = self.forward_logits(x, y)

        out = self.fc(logits.mean(dim=1))
        return out.squeeze(dim=1)
    
# follow https://github.com/NX-AI/xlstm/blob/main/environment_pt220cu121.yaml for xlstm env preparation
# 
class MiniCPMIQA_xLSTMHeader(MiniCPMIQA):
    def __init__(self, args):
        super().__init__(args)
        # context length changed due to sequentce length limit
        # 2024/10/07 try 6 layers and ReLU in fc
        self.xlstm_pooler = xLSTMPooler(args, 512, context_length=self.args['model_max_length'], num_blocks=4) # 477 emmm
        self.W = nn.Linear(self.model.config.vocab_size, 512)
        self.fc = nn.Sequential(
            nn.Linear(512, 128),  # [128, 1024]
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        if not (self.args['stage'] == 'test' or self.args['stage'] == 'predict'):
            self._reset_parameters()
        
    def _reset_parameters(self):
        self.W.apply(xavier_init)
        self.fc.apply(xavier_init)

    def forward(self, x, y):
        logits = self.forward_logits(x, y)  # [bs, 477, 128256]
        dim_reduced_logits = self.W(logits)
        pooler_res = self.xlstm_pooler(dim_reduced_logits)

        if self.args['pooler'] == 'mean':
            # mean pooling
            pooled_result = pooler_res.mean(dim=1)
        elif self.args['pooler'] == 'max':
            # max pooling
            pooled_result, _ = pooler_res.max(dim=1)
        elif self.args['pooler'] == 'firstlastmean':
            # mean 1st and last res
            pooled_result = (pooler_res[:, 0, :] + pooler_res[:, -1, :]) / 2
        elif self.args['pooler'] == 'last':
            pooled_result = pooler_res[:, -1, :]
        else:
            pooled_result = pooler_res.mean(dim=1)

        out = self.fc(pooled_result)
        return out.squeeze(dim=1)
    
class MiniCPMIQA_LSTMHeader(MiniCPMIQA):
    def __init__(self, args):
        super().__init__(args)
        self.W = nn.Linear(self.model.config.vocab_size, 512)
        self.lstm = nn.LSTM(512, 512, num_layers=4, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(1024, 256),  # [128, 1024]
            nn.Dropout(args['dropout']),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        if not (self.args['stage'] == 'test' or self.args['stage'] == 'predict'):
            self._reset_parameters()
        
    def _reset_parameters(self):
        self.W.apply(xavier_init)
        self.fc.apply(xavier_init)

    def forward(self, x, y):
        logits = self.forward_logits(x, y)  # [bs, 477, 128256]
        logits = self.W(logits)
        res, hidden_state = self.lstm(logits)  # [bs, 477, 512*2]
        # mean 1st and last res
        res_mean = (res[:, 0, :] + res[:, -1, :]) / 2
        out = self.fc(res_mean)
        return out.squeeze(dim=1)
    
class MiniCPMIQA_GRUHeader(MiniCPMIQA):
    def __init__(self, args):
        super().__init__(args)
        self.W = nn.Linear(self.model.config.vocab_size, 512)
        self.gru = nn.GRU(512, 512, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(1024, 256),  # [128, 1024]
            nn.Dropout(args['dropout']),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        if not (self.args['stage'] == 'test' or self.args['stage'] == 'predict'):
            self._reset_parameters()
        
    def _reset_parameters(self):
        self.W.apply(xavier_init)
        self.fc.apply(xavier_init)

    def forward(self, x, y):
        logits = self.forward_logits(x, y)  # [bs, 477, 128256]
        logits = self.W(logits)
        res, hidden_state = self.gru(logits)  # [bs, 477, 512*2]
        # mean 1st and last res
        res_mean = (res[:, 0, :] + res[:, -1, :]) / 2
        out = self.fc(res_mean)
        return out.squeeze(dim=1)

class GRUResidual(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_rms_norm = False
        self.args = args
        self.hidden_size = 512
        self.gru_layers = 1
        self.W_gru = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_weight = nn.Linear(self.hidden_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=self.gru_layers, batch_first=True, bidirectional=True)
        self.gru_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.relu = nn.ReLU()
        if self.use_rms_norm:
            self.norm_before = RMSNorm(self.hidden_size)
        else:
            self.norm_before = nn.LayerNorm(self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self._reset_parameters()

    def _reset_parameters(self):
        self.W_gru.apply(xavier_init)
        self.W_weight.apply(xavier_init)
        self.gru_linear.apply(xavier_init)
        self.out_proj.apply(xavier_init)

    def forward(self, x, y):
        normed_x = self.norm_before(x)

        gru_x = self.W_gru(normed_x)
        res, hidden_state = self.gru(gru_x)
        res_out = self.gru_linear(res)

        weight_x = self.W_weight(normed_x)
        weight_x = self.relu(weight_x)

        out = res_out * weight_x
        out = self.out_proj(out)
        
        # residual
        out = out + x
        return out  # [bs, seq_len, hidden_size]

    
class MiniCPMIQA_GRUHeaderResidual(MiniCPMIQA):
    def __init__(self, args):
        super().__init__(args)
        self.llama_model_hidden_size = 4096
        self.use_rms_norm = False
        self.hidden_size = 512
        self.original_proj = nn.Sequential(
            nn.Linear(self.model.config.vocab_size, self.hidden_size),
        )
        self.n_gru_residual_layers = 6
        self.gru_residual_layers = nn.ModuleList([
            GRUResidual(args) for _ in range(self.n_gru_residual_layers)
        ])
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),  # [128, 1024]
            nn.Linear(self.hidden_size // 2, 1)
        )

        if self.use_rms_norm:
            self.norm_after = RMSNorm(self.hidden_size)
        else:
            self.norm_after = nn.LayerNorm(self.hidden_size)

        self._reset_parameters()
        
    def _reset_parameters(self):
        self.original_proj.apply(xavier_init)
        self.fc.apply(xavier_init)

    def forward(self, x, y):    
        logits = self.forward_logits(x, y)  # [bs, 477, 128256]
        residual_logits = self.original_proj(logits)
        
        for i in range(self.n_gru_residual_layers):
            residual_logits = self.gru_residual_layers[i](residual_logits, y)
        
        # post norm
        out = self.norm_after(residual_logits)

        # meam pooling
        out = out.mean(dim=1)

        # regression
        out = self.fc(out)

        return out.squeeze(dim=1)
    
class MiniCPMIQA_Transformer(MiniCPMIQA):
    def __init__(self, args):
        super().__init__(args)
        self.llama_model_hidden_size = 4096
        self.use_rms_norm = False
        self.hidden_size = 512
        self.original_proj = nn.Sequential(
            nn.Linear(self.model.config.vocab_size, self.hidden_size),
        )
        self.trm_layer = nn.TransformerEncoderLayer(self.hidden_size, 2, dropout=self.args['dropout'])
        self.trm_layers = nn.TransformerEncoder(self.trm_layer, 6)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),  # [128, 1024]
            nn.Linear(self.hidden_size // 2, 1)
        )

        self.norm_after = nn.LayerNorm(self.hidden_size)
        self._reset_parameters()
        
    def _reset_parameters(self):
        self.original_proj.apply(xavier_init)
        self.fc.apply(xavier_init)

    def forward(self, x, y):    
        logits = self.forward_logits(x, y)  # [bs, 477, 128256]
        residual_logits = self.original_proj(logits)
        
        residual_logits = self.trm_layers(residual_logits)
        
        # post norm
        out = self.norm_after(residual_logits)

        # meam pooling
        out = out.mean(dim=1)

        # regression
        out = self.fc(out)

        return out.squeeze(dim=1)
    
class MiniCPMIQA_mambaHeader(MiniCPMIQA):
    def __init__(self, args):
        super().__init__(args)
        
        self.use_rms_norm = False
        self.hidden_size = 512
        self.original_proj = nn.Sequential(
            nn.Linear(self.model.config.vocab_size, self.hidden_size),
        )
        self.mamba_layers = nn.ModuleList([
            Mamba2(d_model=self.hidden_size, d_state=16)] * 4)   # 4 layers
        self.mamba_layers = nn.Sequential(*self.mamba_layers)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),  # [128, 1024]
            nn.Linear(self.hidden_size // 2, 1)
        )
        self.norm_after = nn.LayerNorm(self.hidden_size)

        self._reset_parameters()
        
    def _reset_parameters(self):
        self.original_proj.apply(xavier_init)
        self.fc.apply(xavier_init)

    def forward(self, x, y):    
        logits = self.forward_logits(x, y)  # [bs, 477, 128256]
        residual_logits = self.original_proj(logits)
        # residual_logits = logits
        
        residual_logits = self.mamba_layers(residual_logits)
        
        # post norm
        out = self.norm_after(residual_logits)

        # meam pooling
        out = out.mean(dim=1)

        # regression
        out = self.fc(out)

        return out.squeeze(dim=1)


class MiniCPMIQA_nofinetune(BaseModel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        model_name = "openbmb/MiniCPM-Llama3-V-2_5"
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Linear(self.model.config.vocab_size, 1024),  # [128256, 1024]
            nn.Linear(1024, 1)
        )

        if not (self.args['stage'] == 'test' or self.args['stage'] == 'predict'):
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
    def collate_fn(self, batch):
        ret, y = map(list, zip(*batch))
        ret = data_collator(ret, max_length=self.args['model_max_length'])
        return ret, torch.tensor(y)
    
    def get_dataset_class(self):
        return SupervisedDataset

    def forward(self, x, y):
        # x is the inputs
        # inference
        outputs = self.model.base_model(data=x, use_cache=True) 

        logits = outputs.logits # [B, seq_len, vocab_size]

        out = self.fc(logits.mean(dim=1))
        return out.squeeze(dim=1)
