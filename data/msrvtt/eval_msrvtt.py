import argparse
import io
import json
import math
import os
import numpy as np
import torch
import tqdm
import av
from transformers import AutoProcessor


def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1, 2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k


def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.

    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    if len(indices) == 0:
        # to debug
        indices = [0]
        print("No indices to decode, might be an empty video please check")
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def validate_msrvtt(model, tokenizer, image_processor, root, metadata,
                    num_frames=1, prefix='summarize:', mode='InternVL-G', recall_k_list=[1, 5, 10],
                    use_dsl=True, eval_batch_size=32):
    metadata = json.load(open(metadata))

    video_features = []
    text_features = []
    all_scores = []

    # # compute text features
    # # print('Computing text features', flush=True)
    # # for data in tqdm.tqdm(metadata):
    # #     caption = prefix + data['caption']
    # #     input_ids = tokenizer(caption, return_tensors='pt', max_length=80,
    # #                     truncation=True, padding='max_length').input_ids.cuda()
    # #     with torch.no_grad():
    # #         feat = model.encode_text(input_ids)
    # #     text_features.append(feat.cpu())
    # # text_features = torch.cat(text_features)

    # # compute video features
    # print('Computing video features', flush=True)
    # for data in tqdm.tqdm(metadata):
    #     caption = prefix + data['caption']
    #     input_ids = tokenizer(caption, return_tensors='pt', max_length=80,
    #                     truncation=True, padding='max_length').input_ids.cuda()
        
    #     video_id = data['video']
    #     video_path = os.path.join(root, video_id)
    #     # video_data = mmengine.get(video_path)
    #     # video_data = io.BytesIO(video_data)
    #     video_reader = decord.VideoReader(video_path)

    #     # uniformly sample frames
    #     interval = math.ceil(len(video_reader) / num_frames)
    #     frames_id = np.arange(0, len(video_reader), interval) + interval // 2
    #     assert len(frames_id) == num_frames and frames_id[-1] < len(video_reader)

    #     frames = video_reader.get_batch(frames_id).asnumpy()

    #     pixel_values = image_processor(images=frames, return_tensors='pt').pixel_values
    #     with torch.no_grad():
    #         pixel_values = pixel_values.to(torch.bfloat16).cuda()
    #         logits_per_image, logits_per_text = model(
    #             image=pixel_values, text=input_ids, mode='InternVL-G')
    #         all_scores.append(logits_per_text)
    #         # feat = model.encode_image(pixel_values, mode=mode)
    #         # feat = feat.mean(dim=0, keepdim=True)
    #     # video_features.append(feat.cpu())
    # scores = torch.cat(all_scores)
    
    
    for item in tqdm.tqdm(metadata, desc='Computing feature similarity'):
        video_id = item['video']
        caption = prefix + item['caption']
        video_path = os.path.join(root, video_id)
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        sample_fps = 0.5
        ori_fps = container.streams.video[0].average_rate
        indices = np.arange(0, total_frames, int(ori_fps/sample_fps))
        # indices = [0]
        frames = read_video_pyav(container, indices)
        inputs = image_processor(text=[caption], images=frames, padding="max_length", return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        inputs['pixel_values'] = [inputs['pixel_values']]
        with torch.no_grad():
            outputs = model(**inputs)
            text_embeds = outputs.text_embeds
            video_embeds = outputs.video_embeds
            # video_embeds = outputs.image_embeds.mean(dim=0, keepdim=True)
            # video_embeds = outputs.perceiver_resampler_output.last_hidden_state[0]
            # video_embeds = video_embeds.reshape(-1, video_embeds.shape[-1]).mean(dim=0, keepdim=True)
        text_features.append(text_embeds)
        video_features.append(video_embeds)
    text_features = torch.cat(text_features)
    video_features = torch.cat(video_features)
    # scores = text_features @ video_features.T * model.logit_scale.exp() + model.logit_bias
    scores = text_features @ video_features.T
    
    
    
    
        
    
    # video_features = torch.cat(video_features)

    # print('Computing metrics', flush=True)
    # texts_emb = text_features / text_features.norm(dim=-1, keepdim=True)
    # images_emb = video_features / video_features.norm(dim=-1, keepdim=True)

    # get the score for each text and image pair
    # scores = texts_emb @ images_emb.t()
    

    # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), torch.arange(len(scores))] = True

    scores_T = scores.T
    positive_pairs_T = positive_pairs.T

    if use_dsl:
        scores = scores * scores.softmax(dim=0)
        scores_T = scores_T * scores_T.softmax(dim=0)

    metrics = {}
    for recall_k in recall_k_list:
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        # for each image, that number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        # it over the dataset.
        metrics[f't2v_retrieval_recall@{recall_k}'] = (
                    batchify(recall_at_k, scores, positive_pairs, eval_batch_size, scores.device,
                             k=recall_k) > 0).float().mean().item()
        metrics[f'v2t_retrieval_recall@{recall_k}'] = (
                    batchify(recall_at_k, scores_T, positive_pairs_T, eval_batch_size, scores.device,
                             k=recall_k) > 0).float().mean().item()

    print(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='validate MSR-VTT', add_help=False)
    parser.add_argument('--video-root', type=str)
    parser.add_argument('--metadata', type=str)
    parser.add_argument('--mode', type=str, default='InternVL-C',choices=['InternVL-C', 'InternVL-G'])
    parser.add_argument('--num-frames', type=int, default=1)
    args = parser.parse_args()

    # model = AutoModel.from_pretrained(
    #     'OpenGVLab/InternVL-14B-224px',
    #     torch_dtype=torch.bfloat16,
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True,
    #     device_map="auto").eval()

    # image_processor = CLIPImageProcessor.from_pretrained('OpenGVLab/InternVL-14B-224px')

    from mantis.models.siglip_video import SiglipVideoModel
    from transformers import CLIPModel, CLIPProcessor
    print("Loading model")
    model = SiglipVideoModel.from_pretrained("Mantis-VL/siglip-video_16384_2fps_128").to("cuda:2").eval()
    processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    # model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda:2").eval()
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    # tokenizer.pad_token_id = 0  # set pad_token_id to 0

    print("Validating")
    metrics = validate_msrvtt(model, processor.tokenizer, processor,
                              root=args.video_root,
                              metadata=args.metadata,
                              mode=args.mode,
                              num_frames=args.num_frames,)