import math
import torch
import torch.nn as nn  
import torch.nn.functional as F  
from einops import rearrange
import torch.nn as nn
import re
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
from loguru import logger as eval_logger
from collections import OrderedDict, defaultdict, deque
from itertools import chain
import torch.distributed as dist
import time

import numpy as np  
import matplotlib.pyplot as plt  
from matplotlib.patches import Rectangle  

# from llava.model.language_model.llava_llama import LlavaLlamaModel, LlavaLlamaForCausalLM
# try:
#     from vila.model.language_model.llava_llama import LlavaLlamaModel as VilaLlavaLlamaModel
# except:
#     VilaLlavaLlamaModel = LlavaLlamaModel
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

# Pool Function
def L2NormAvgPool2d(image_features,kernel_size=2,stride=2,temp=1.,p=2, padding=0):
    T, N, D = image_features.shape
    n0 = n1 = int(math.sqrt(N))
    pool2 = SoftmaxWeightedL2NormPool2d(kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=padding, temp=temp, p=p)
    image_features = rearrange(image_features, 't (n0 n1) d -> t d n0 n1', n0=n0, n1=n1)
    image_features = pool2(image_features)
    image_features = rearrange(image_features, 't d n0 n1 -> t (n0 n1) d')
    return image_features

def uniformly_sample_frames(frames, num_samples=4):
    """
    Uniformly samples frames from a batch of frames.

    Args:
    - frames (tensor): A tensor of shape [N, C, H, W].
    - num_samples (int): The number of frames to sample.

    Returns:
    - sampled_frames (tensor): A tensor of sampled frames with shape [num_samples, C, H, W].
    """
    N = frames.shape[0]
    if N < num_samples:
        raise ValueError(f"Cannot sample {num_samples} frames from a tensor with only {N} frames.")

    # Compute indices for uniform sampling
    indices = torch.linspace(0, N, steps=num_samples + 1)[1:] - 1
    indices = torch.round(indices).long()

    # Ensure indices are within valid range
    indices = torch.clamp(indices, 0, N - 1)

    # Sample frames
    sampled_frames = frames[indices]
    return sampled_frames, indices


def resize_images(images, size):
    """
    Resizes a batch of images to the specified size using optimal interpolation methods.

    Args:
    - images (tensor): A tensor of shape [N, C, H, W].
    - size (tuple): The desired size (height, width) for the images.

    Returns:
    - resized_images (tensor): A tensor of resized images with shape [N, C, new_H, new_W].
    """
    original_height, original_width = images.shape[2], images.shape[3]
    new_height, new_width = size

    if new_height < original_height or new_width < original_width:
        # Downsampling - use area interpolation
        resized_images = nn.functional.interpolate(
            images, size=size, mode='area'
        )
    else:
        # Upsampling - use bilinear interpolation
        resized_images = nn.functional.interpolate(
            images, size=size, mode='bilinear', align_corners=False
        )
    return resized_images

def find_best_grid(num_images):
    """
    Finds the grid dimensions (rows and columns) that best fit the number of images,
    minimizing the difference between rows and columns.

    Args:
    - num_images (int): The number of images to arrange in the grid.

    Returns:
    - best_rows (int): The optimal number of grid rows.
    - best_cols (int): The optimal number of grid columns.
    """
    factors = []
    for i in range(1, int(math.sqrt(num_images)) + 1):
        if num_images % i == 0:
            factors.append((i, num_images // i))

    # Find the factor pair with minimal difference between rows and columns
    best_rows, best_cols = min(factors, key=lambda x: abs(x[0] - x[1]))

    # Ensure that rows >= cols for consistency (optional)
    if best_rows < best_cols:
        best_rows, best_cols = best_cols, best_rows

    return best_rows, best_cols

def create_image_grid(images, final_height=336, final_width=336, shuffling=False):
    """
    Combines a batch of images into a grid.

    Args:
    - images (tensor): A tensor of shape [N, C, H, W].
    - final_height (int): Desired height of the final grid image.
    - final_width (int): Desired width of the final grid image.
    - shuffling (bool): whether to shuffle the selected frames.

    Returns:
    - grid_image (tensor): Combined image grid tensor with shape [C, final_height, final_width].
    """
    num_images = images.shape[0]
    channels = images.shape[1]

    # Find the best grid dimensions
    grid_rows, grid_cols = find_best_grid(num_images)

    # Compute new size for each image in the grid
    img_height = math.ceil(final_height / grid_rows)
    img_width = math.ceil(final_width / grid_cols)

    # Resize images using optimal interpolation
    resized_images = resize_images(images, size=(img_height, img_width))  # [N, C, img_height, img_width]

    if shuffling:
        shuffled_indices = torch.randperm(resized_images.size(0))
        resized_images = resized_images[shuffled_indices]

    # Reshape and rearrange images to form the grid
    # Resized images shape: [N, C, img_height, img_width]
    # First, reshape to [grid_rows, grid_cols, C, img_height, img_width]
    grid_images = resized_images.view(grid_rows, grid_cols, channels, img_height, img_width)

    # Permute dimensions to bring channels to the front
    grid_images = grid_images.permute(2, 0, 3, 1, 4)  # [C, grid_rows, img_height, grid_cols, img_width]

    # Reshape to combine rows and columns
    grid_image = grid_images.contiguous().view(
        channels,
        grid_rows * img_height,
        grid_cols * img_width
    )

    # Ensure the final image has the exact desired dimensions
    grid_image = grid_image[:, :final_height, :final_width]

    return grid_image  # Shape: [C, final_height, final_width]

class SoftmaxWeightedL2NormPool2d(torch.nn.Module):  
    def __init__(self, kernel_size, temp, stride=None, padding=0, p=2):  
        """  
        基于 L2 范数的 Softmax 加权池化操作  
        :param kernel_size: 池化窗口大小 (h, w)  
        :param stride: 步幅 (h, w)，默认为 kernel_size  
        :param padding: 填充大小，默认为 0  
        """  
        super(SoftmaxWeightedL2NormPool2d, self).__init__()  
        self.kernel_size = kernel_size  
        self.stride = stride if stride is not None else kernel_size  
        self.padding = padding  
        self.temp = temp
        self.p = p

    def forward(self, x:torch.Tensor):  
        """  
        前向传播  
        :param x: 输入张量，形状为 [batch_size, channels, height, width]  
        :return: 池化后的张量，形状为 [batch_size, channels, pooled_height, pooled_width]  
        """  
        # 1. 对输入进行填充  
        if self.padding > 0:  
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))  

        # 2. 计算每个位置的 L2 范数  
        # L2 范数是特征维度上的平方和的平方根  
        l2_norms = torch.norm(x, p=self.p, dim=1, keepdim=True) / (x.shape[1] ** (1 / self.p))  # 形状为 [batch_size, 1, height, width]  

        # 3. 使用 unfold 提取滑动窗口  
        batch_size, channels, height, width = x.shape  
        kernel_h, kernel_w = self.kernel_size  
        stride_h, stride_w = self.stride  

        # unfold 操作将输入张量展平为滑动窗口  
        unfolded_x = F.unfold(x, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))  # [batch_size, channels * kernel_h * kernel_w, num_windows]  
        unfolded_x = unfolded_x.view(batch_size, channels, kernel_h * kernel_w, -1)  # [batch_size, channels, kernel_h * kernel_w, num_windows]  

        unfolded_l2 = F.unfold(l2_norms, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))  # [batch_size, kernel_h * kernel_w, num_windows]  
        unfolded_l2 = unfolded_l2.view(batch_size, kernel_h * kernel_w, -1)  # [batch_size, kernel_h * kernel_w, num_windows]  

        # 4. 计算 Softmax 权重  
        softmax_weights = F.softmax(unfolded_l2 * self.temp, dim=1)  # [batch_size, kernel_h * kernel_w, num_windows]  

        # 5. 对窗口内的特征进行加权平均  
        # 将权重扩展到与特征相同的形状  
        softmax_weights = softmax_weights.unsqueeze(1)  # [batch_size, 1, kernel_h * kernel_w, num_windows]  
        weighted_x = unfolded_x * softmax_weights  # [batch_size, channels, kernel_h * kernel_w, num_windows]  
        pooled = weighted_x.sum(dim=2)  # [batch_size, channels, num_windows]  

        # # 6. 将结果 reshape 成池化后的形状  
        out_height = (height - kernel_h) // stride_h + 1  
        out_width = (width - kernel_w) // stride_w + 1  
        pooled = rearrange(pooled, "t d (n1 n2) -> t d n1 n2", n1=out_height, n2=out_width)

        return pooled


class prepare():
    def __init__(self,model,model_config,device,pool_config,processor=None):
        self.model=model
        self.config=model_config
        self.device=device
        self.pool_config=pool_config
        self.processor=processor

    def encode_images(self, images):
        if self.model.__class__.__name__ in ['LlavaLlamaModel', 'LlavaLlamaForCausalLM', 'LlavaQwenForCausalLM']:
            # image_features, image_embeddings = self.model.get_model().get_vision_tower()(images)
            image_features = self.model.get_model().get_vision_tower()(images)
            image_features = self.model.get_model().mm_projector(image_features)
            return image_features#, image_embeddings
        else:
            image_features = self.model.forward_image(images.cuda())
            return image_features

    def only_pool_features(self,images):
        tmp_features=[]
        for arg in self.pool_config:
            tmp_features.append(eval(f'self.prepare_unit(images, {arg})'))
        image_features=torch.cat(tmp_features,dim=1)
        return image_features
    
    def prepare_unit(self, frames, num_frames, contexts=None, pool_func=L2NormAvgPool2d, grid_size=1, grid_size_list=[], grid_freq_list=[], num_sampled_tokens=-1, sample_type='linear', vila=False, **pool_kwargs):
        try:
            sampled_frames, indices = uniformly_sample_frames(frames, num_samples=num_frames)
        except:
            sampled_frames = frames
            indices = torch.arange(len(sampled_frames))

        if grid_size > 1:
            grided_frames = []
            num_grids = sampled_frames.size(0) // grid_size  # Adjusted to accommodate the new chunk size
            sampled_frames=sampled_frames.unsqueeze(1)
            new_indices = []
            for i in range(num_grids):
                # Extract a tensor of shape [chunk_size,1,3,336,336]
                start_idx = i * grid_size

                frames_part = sampled_frames[start_idx:start_idx + grid_size, 0, :, :, :]  # Adjusted the slice to use chunk_size

                # Pass frames_part through your create_image_grid() function
                img = create_image_grid(frames_part,final_height=frames_part.shape[2],final_width=frames_part.shape[3])  # Outputs a tensor of shape [1,3,336,336]

                grided_frames.append(img.unsqueeze(0))
                new_indices.append(indices[start_idx + grid_size - 1])
            if len(grided_frames) > 0:
                grided_frames=torch.cat(grided_frames)
            else:
                return [], []
            indices = new_indices
        else:
            grided_frames = sampled_frames

        # image_features, image_embeds = self.encode_images(grided_frames)

        # start_time = time.time()
        image_features = self.encode_images(grided_frames)
        # end_time = time.time()
        # print('encode image time: ', end_time - start_time)

        if len(grid_size_list) > 0:
            for feature_grid_size, feature_grid_freq in zip(grid_size_list, grid_freq_list):
                N = image_features.shape[0]
                grid_num_samples = N // feature_grid_freq
                grid_frame_inds = torch.linspace(0, N, steps=grid_num_samples + 1)[1:] - 1
                grid_frame_inds = torch.round(grid_frame_inds).long()
                grid_frame_inds = torch.clamp(grid_frame_inds, 0, N - 1)

                grid_image_features = image_features[grid_frame_inds]

                num_grids = grid_image_features.size(0) // feature_grid_size  # Adjusted to accommodate the new chunk size
                grid_image_features=grid_image_features.unsqueeze(1)

                for i in range(num_grids):
                    # Extract a tensor of shape [chunk_size,1,3,336,336]
                    start_idx = i * feature_grid_size

                    frames_part = grid_image_features[start_idx:start_idx + feature_grid_size, 0, :, :]  # Adjusted the slice to use chunk_size
                    frames_part = rearrange(frames_part, 't (n1 n2) d -> t d n1 n2', n1=int(np.sqrt(frames_part.shape[1])), n2=int(np.sqrt(frames_part.shape[1])))
                    # Pass frames_part through your create_image_grid() function
                    img = create_image_grid(frames_part,final_height=frames_part.shape[2],final_width=frames_part.shape[3])  # Outputs a tensor of shape [1,3,336,336]
                    img = rearrange(img, 'd n1 n2 -> (n1 n2) d')

                    image_features[grid_frame_inds[start_idx + feature_grid_size - 1]] = img.unsqueeze(0)

        pooled_image_features = pool_func(image_features, **pool_kwargs)

        return pooled_image_features, indices
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, temporal_aggregation=None, contexts=None
    ):


        # image_features = self.encode_images(images)

        """ if temporal_aggregation and \
            temporal_aggregation.lower() != 'none' and \
            temporal_aggregation.lower() != 'false':
            if temporal_aggregation.startswith('slowfast'):
                image_features = self.prepare_slowfast(image_features, temporal_aggregation)
            else:
                image_features = self.temporal_aggregation(image_features, temporal_aggregation) """
        
        # cur_input_ids = input_ids[0]
        # image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        # cur_input_ids_noim = []
        # for i in range(len(image_token_indices) - 1):
        #     cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
        # cur_input_embeds = self.model.get_model().embed_tokens(torch.cat(cur_input_ids_noim))

        # vision_tower_params_dict = self.model.get_model().get_vision_tower().vision_tower.state_dict()
        # clip_vision_model_dict = self.clip_vision_model.half().state_dict()
        # equal_list = [(p.data == q.data).all() for p, q in zip(self.clip_vision_model.parameters(), )]
        # assert(all(equal_list) == True)

        # context_input = self.model.get_model().get_vision_tower().tokenizer(contexts.split('\n')[1:-1], padding=True, return_tensors="pt").to(self.device)
        # context_features = self.model.get_model().get_vision_tower().text_tower(**context_input).last_hidden_state.mean(dim=1)

        tmp_features=[]
        tmp_indices=[]
        for arg in self.pool_config:
            tokens, indices = eval(f'self.prepare_unit(images, {arg})')
            tmp_features.extend(tokens)
            tmp_indices.extend(indices)
            # tmp_features.append(eval(f'self.prepare_unit(images, text_embeds=None, {arg})'))

        tmp_indices = torch.stack(tmp_indices, dim=0)
        tmp_indices_reverse = tmp_indices.flip(0).tolist()
        # ind = tmp_indices.argsort()
        # tmp_features = [tmp_features[i] for i in ind]
        image_features = []
        for frame_ind in tmp_indices.unique(sorted=True):
            ind = len(tmp_indices_reverse) - 1 - tmp_indices_reverse.index(frame_ind)
            image_features.append(tmp_features[ind])
        image_features=torch.cat(image_features,dim=0).unsqueeze(0)
        # image_features_lens = torch.mean(image_features ** 2, dim=-1).detach().flatten()
        # max_lens = torch.quantile(image_features_lens.flatten().float(), 0.95).item()
        # image_features[0, image_features_lens > max_lens] = image_features[0, image_features_lens > max_lens] / image_features_lens[image_features_lens > max_lens].unsqueeze(1) * max_lens
        # image_features = image_features[:, image_features_lens < torch.quantile(image_features_lens.flatten().float(), 0.95).item()]


        self.model.config.video_token_shape = image_features.shape
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
    def __truncate_sequence(
        self, inputs: List[torch.Tensor], labels: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if any(len(input) > self.model.tokenizer.model_max_length for input in inputs):

            inputs = [input[: self.model.tokenizer.model_max_length] for input in inputs]
            labels = [label[: self.model.tokenizer.model_max_length] for label in labels]
        return inputs, labels
    def __batchify_sequence(
        self, inputs: List[torch.Tensor], labels: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(inputs)
        device = inputs[0].device
        hidden_size = inputs[0].shape[1]
        max_length = max(inputs[k].shape[0] for k in range(batch_size))
        attention_mask = torch.ones((batch_size, max_length), dtype=torch.bool, device=device)

        inputs_p, labels_p = [], []
        for k in range(batch_size):
            size_pk = max_length - inputs[k].shape[0]
            inputs_pk = torch.zeros((size_pk, hidden_size), dtype=inputs[k].dtype, device=device)
            labels_pk = torch.full((size_pk,), IGNORE_INDEX, dtype=labels[k].dtype, device=device)
            if self.model.tokenizer.padding_side == "right":
                attention_mask[k, inputs[k].shape[0] :] = False
                inputs_pk = torch.cat([inputs[k], inputs_pk], dim=0)
                labels_pk = torch.cat([labels[k], labels_pk], dim=0)
            else:
                attention_mask[k, : -inputs[k].shape[0]] = False
                inputs_pk = torch.cat([inputs_pk, inputs[k]], dim=0)
                labels_pk = torch.cat([labels_pk, labels[k]], dim=0)
            inputs_p.append(inputs_pk)
            labels_p.append(labels_pk)

        inputs = torch.stack(inputs_p, dim=0)
        labels = torch.stack(labels_p, dim=0)
        return inputs, labels, attention_mask
    def __embed_media_tokens(
        self,
        media: Dict[str, List[torch.Tensor]],
        media_config: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[torch.Tensor]]:
        embeds = defaultdict(deque)
        for name in media:
            if self.model.training:
                # Gather metainfo of media objects from all ranks
                info = [{"shape": tensor.shape, "dtype": tensor.dtype} for tensor in media.get(name, [])]
                infos = list(chain(*dist.all_gather(info)))

                # The entire batch does not contain any media objects of this type.
                if not infos:
                    continue

                # Create a dummy tensor to ensure the encoder is called, otherwise the training will hang.
                if not media.get(name):
                    dummy = torch.zeros(infos[0]["shape"], dtype=infos[0]["dtype"], device=self.model.device)
                    embeds["dummy"].extend(self.model.encoders[name]([dummy], media_config[name]))
                    continue

            end_token_embeds=self.model.encoders[name].embed_tokens(self.model.encoders[name].end_tokens)
            tmp_features=[]
            for arg in self.pool_config:
                image_feature = eval(f'self.prepare_unit(media["video"][0], vila=True, {arg})')
                end_embeds = torch.stack([end_token_embeds] * image_feature.shape[0], dim=0)
                image_feature = torch.cat([image_feature, end_embeds], dim=1)
                tmp_features.append(image_feature.flatten(0, 1))
            image_features=torch.cat(tmp_features,dim=0)
            image_features=image_features

            self.model.config.video_token_shape = image_features.shape

            # end_embeds = torch.stack([end_token_embeds] * image_features.shape[0], dim=0)
            # for i in range(len(tmp_features)):
            #     image_feature = tmp_features[i]
                
                
            #     tmp_features

            # image_features=image_features.flatten(0,1)

            # self.config.video_token_shape = image_features.shape

            embeds[name] = deque([image_features])
        return embeds
    def _embed(
        self,
        input_ids: torch.Tensor,
        media: Dict[str, List[torch.Tensor]],
        media_config: Dict[str, Dict[str, Any]],
        labels: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = labels if labels is not None else torch.full_like(input_ids, IGNORE_INDEX)
        attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)

        # Extract text and media embeddings
        text_embeds = self.model.llm.model.embed_tokens(input_ids)
        media_embeds = self.__embed_media_tokens(media, media_config)
        


        # This is a workaround to make sure the dummy embeddings are consumed
        while media_embeds.get("dummy"):
            dummy_embed = media_embeds["dummy"].popleft()
            text_embeds += torch.sum(dummy_embed) * 0

        # Remove padding
        batch_size = labels.shape[0]
        text_embeds = [text_embeds[k][attention_mask[k]] for k in range(batch_size)]
        labels = [labels[k][attention_mask[k]] for k in range(batch_size)]

        # Build inverse mapping from token ID to media name
        media_tokens = {}
        for name, token_id in self.model.tokenizer.media_token_ids.items():
            media_tokens[token_id] = name

        # Fuse text and media embeddings
        inputs_m, labels_m = [], []
        for k in range(batch_size):
            inputs_mk, labels_mk = [], []
            pos = 0
            while pos < len(labels[k]):
                if input_ids[k][pos].item() in media_tokens:
                    end = pos + 1
                    name = media_tokens[input_ids[k][pos].item()]
                    input = media_embeds[name].popleft()
                    label = torch.full([input.shape[0]], IGNORE_INDEX, device=labels[k].device, dtype=labels[k].dtype)
                else:
                    end = pos
                    while end < len(labels[k]) and input_ids[k][end].item() not in media_tokens:
                        end += 1
                    input = text_embeds[k][pos:end]
                    label = labels[k][pos:end]
                inputs_mk.append(input)
                labels_mk.append(label)
                pos = end
            inputs_m.append(torch.cat(inputs_mk, dim=0))
            labels_m.append(torch.cat(labels_mk, dim=0))
        inputs, labels = inputs_m, labels_m

        # Check if all media embeddings are consumed
        for name in media_embeds:
            if media_embeds[name]:
                raise ValueError(f"Not all {name} embeddings are consumed!")

        # Truncate sequences to `model_max_length` as media embeddings are inserted
        inputs, labels = self.__truncate_sequence(inputs, labels)

        # Pad sequences to the longest one in the batch
        return self.__batchify_sequence(inputs, labels)
