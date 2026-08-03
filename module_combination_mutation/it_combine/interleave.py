import torch
from it_combine.strategy import ImageTextCombineStrategy

class InterleaveCombineStrategy(ImageTextCombineStrategy):
    def __init__(self):
        super().__init__("interleave")
        self.IMAGE_TOKEN_INDEX = -200
        self.IGNORE_INDEX = -100

    def combine(self, image_features: torch.Tensor, text_embeddings: torch.Tensor, input_ids: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        labels = torch.full_like(input_ids, self.IGNORE_INDEX)
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask]
                     for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
                     ]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == self.IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                # text_embeddings 为 (seq_len, batch_size, hidden)，取第 batch_idx 个 batch 再按 mask 过滤
                cur_input_embeds_1 = text_embeddings[:, batch_idx, :][attention_mask[batch_idx]]
                cur_input_embeds = cur_input_embeds_1
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == self.IMAGE_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = text_embeddings[:, batch_idx, :][attention_mask[batch_idx]]
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
                    cur_new_labels.append(
                        torch.full((cur_image_features.shape[0],), self.IGNORE_INDEX, device=cur_labels.device,
                                   dtype=cur_labels.dtype))

            cur_new_input_embeds = [x for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        # tokenizer_model_max_length = getattr(self.config, 'language_max_sequence_length', None)
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        #     new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), self.IGNORE_INDEX, dtype=new_labels[0].dtype,
                                       device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            new_input_embeds_padded.append(torch.cat((
                cur_new_embed,
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device)
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                            device=position_ids.device)

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

        causal_attention_mask = torch.triu(
            torch.ones(new_input_embeds.shape[0], 1, new_input_embeds.shape[1], new_input_embeds.shape[1],
                       device=new_input_embeds.device),
            diagonal=1
        ).bool()
        attention_mask = ~attention_mask
        expanded_attention_mask = attention_mask[:, None, None, :].expand(
            new_input_embeds.shape[0], 1, new_input_embeds.shape[1], new_input_embeds.shape[1]
        )
        # 用逻辑或替代 masked_fill(..., True)，避免 NPU 上 masked_fill 布尔填充报错
        attention_mask = causal_attention_mask | expanded_attention_mask

        return new_input_embeds