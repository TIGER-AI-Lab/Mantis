
import dataclasses
from dataclasses import dataclass, field
import torch
import os
import wandb
import math
import torch.distributed as dist
from transformers import Trainer as HFTrainer, TrainingArguments, BitsAndBytesConfig
from accelerate import init_empty_weights
from transformers.hf_argparser import HfArgumentParser
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
# from mantis.models.intern_vl_25_8b.conversation import conv_templates
from mantis.models.conversation import conv_templates
from mantis.train.data import (
    load_data_from_config, 
    set_ignore_index, set_default_image_token, 
    set_default_image_token_id,
    set_default_video_token,
    set_default_video_token_id,
    ClassificationDataset,
)
from pathlib import Path
from typing import Optional
from pathlib import Path
from torch.utils.data.distributed import DistributedSampler

os.environ["WANDB_RESUME"] = "allow"
os.environ["WANDB_RUN_ID"] = wandb.util.generate_id()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)

class RingAttentionSampler(DistributedSampler):
    
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        per_device_batch_size: int = None
    ) -> None:
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) * self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) * self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples // self.num_replicas
        self.per_device_batch_size = per_device_batch_size
        if self.total_size % self.per_device_batch_size != 0:
            self.total_size = self.total_size - (self.total_size % self.per_device_batch_size) + self.per_device_batch_size
        
    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size, f"len(indices)={len(indices)}, self.total_size={self.total_size}"
        
        if self.per_device_batch_size:
            indices = torch.tensor(indices).reshape(-1, self.per_device_batch_size)
            indices = indices.repeat_interleave(self.num_replicas, dim=0).reshape(-1).tolist()
        else:
            indices = torch.tensor(indices).repeat_interleave(self.num_replicas).tolist()
        assert len(indices) == self.num_samples, f"len(indices)={len(indices)}, self.num_samples={self.num_samples}"
        
        print("Rank:", dist.get_rank(), "Sampler rank:", self.rank, "Num replicas:", self.num_replicas, "Indices:", indices[:10])

        return iter(indices)
    
class Trainer(HFTrainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if hasattr(self.args, "num_replicas") and self.args.num_replicas > 0:
            # for ring flash attention
            print("Using distributed sampler because num_replicas > 1 for ring flash attention")
            assert hasattr(self.args, "sampler_rank"), "sampler_rank is required for distributed training, please set it in the training script"
            print("Rank:", dist.get_rank(), "Sampler rank:", self.args.sampler_rank, "Num replicas:", self.args.num_replicas)
            return RingAttentionSampler(
                self.train_dataset, 
                num_replicas=self.args.num_replicas, rank=self.args.sampler_rank, 
                seed=self.args.seed, shuffle=False, 
                per_device_batch_size=self.args.per_device_train_batch_size
            )
        else:
            return DistributedSampler(self.train_dataset, seed=self.args.seed, shuffle=False)
        # return super()._get_train_sampler()

@dataclass
class DataArguments:
    max_seq_len: Optional[int] = field(
        metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer "
                          "than this will be truncated.", "default": 1024, "required": False},
        default=1024,
    )
    data_config_file: Optional[str] = field(
        metadata={"help": "Pretrained config name or path if not the same as model_name", "default": None, "required": False},
        default=None,
    )
    dataset_balancing: Optional[bool] = field(
        metadata={"help": "Whether to balance the dataset", "default": False, "required": False},
        default=False,
    )
    use_video_encoder: Optional[bool] = field(
        metadata={"help": "Whether to use video encoder", "default": True, "required": False},
        default=True,
    )
    load_video_frames: Optional[bool] = field(
        metadata={"help": "Whether to load video frames", "default": False, "required": False},
        default=False,
    )
    packing_type: Optional[str] = field(
        metadata={"help": "The packing type", "default": "cross_attn", "required": False},
        default="cross_attn",
    )
    max_self_attn_len: Optional[int] = field(
        metadata={"help": "The maximum self attention length", "default": 1024, "required": False},
        default=1024,
    )
    max_cross_attn_kv_len: Optional[int] = field(
        metadata={"help": "The maximum cross attention key value length", "default": 32768, "required": False},
        default=32768,
    )
    num_tokens_per_image: Optional[int] = field(
        metadata={"help": "The number of tokens per image", "default": 256, "required": False},
        default=256,
    )

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models", "default": "Qwen/Qwen2-VL-7B-Instruct", "required": False},
        default="OpenGVLab/InternVL2_5-8B",
    )
    lora_enabled: Optional[bool] = field(
        metadata={"help": "Whether to use LoRA", "default": False, "required": False},
        default=False,
    )
    qlora_enabled: Optional[bool] = field(
        metadata={"help": "Whether to use QLoRA", "default": False, "required": False},
        default=False,
    )
    dora_enabled: Optional[bool] = field(
        metadata={"help": "Whether to use Dora", "default": False, "required": False},
        default=True,
    )
    lora_r: Optional[int] = field(
        metadata={"help": "LoRA r", "default": 8, "required": False},
        default=8,
    )
    lora_alpha: Optional[float] = field(
        metadata={"help": "LoRA alpha", "default": 8, "required": False},
        default=8,
    )
    lora_dropout: Optional[float] = field(
        metadata={"help": "LoRA dropout", "default": 0.1, "required": False},
        default=0.1,
    )
    lora_bias: Optional[str] = field(
        metadata={"help": "LoRA bias", "default": 'none', "required": False},
        default='none',
    )
    attn_implementation: Optional[str] = field(
        metadata={"help": "The attention implementation to use (eager|flash_attention_2|sdpa)", "default": "flash_attention_2", "required": False},
        default="flash_attention_2",
    )
    use_ring_flash_attn: Optional[bool] = field(
        metadata={"help": "Whether to use ring flash attention", "default": True, "required": False},
        default=True,
    )
    ring_attn_group_size: Optional[int] = field(
        metadata={"help": "The ring attention group", "default": None, "required": False},
        default=None,
    )
    conv_template: Optional[str] = field(
        metadata={"help": "The conversation template to use", "default": None, "required": False},
        default=None,
    )
    num_labels: Optional[int] = field(
        metadata={"help": "The number of labels", "default": None, "required": False},
        default=None,
    )
    problem_type: Optional[str] = field(
        metadata={"help": "The problem type", "default": "generation", "required": False, "choices": ["regression", "single_label_classification", "multi_label_classification", "generation"]},
        default="generation",
    )
    enable_cross_attention: Optional[bool] = field(
        metadata={"help": "Whether to enable cross attention", "default": True, "required": False},
        default=True,
    )
    do_pretrain: Optional[bool] = field(
        metadata={"help": "Whether to do pretraining", "default": False, "required": False},
        default=False,
    )

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['vision_model', 'mlp1']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def load_model(model_args, training_args):
    print("Loading model...")
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32
    from transformers import AutoTokenizer, AutoModel
    from mantis.models.intern_vl_25_8b import InternVLChatModel, InternVLChatConfig, InternLM2Tokenizer, InternVLChatProcessor
    tokenizer = InternLM2Tokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, use_fast=False)
    processor = InternVLChatProcessor(tokenizer, enable_cross_attention=model_args.enable_cross_attention)
    
    if model_args.qlora_enabled:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    assert model_args.problem_type in ["regression", "single_label_classification", "multi_label_classification", "generation"]
    config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
    if model_args.enable_cross_attention:
        config.enable_cross_attention = True
        config.llm_config.enable_cross_attention = True
    else:
        config.enable_cross_attention = False
        config.llm_config.enable_cross_attention = False
    use_flash_attn = True if model_args.attn_implementation == "flash_attention_2" else False
    use_ring_flash_attn = model_args.use_ring_flash_attn
    
    model_init_kwargs = {
        "torch_dtype": torch_dtype,
        "attn_implementation": model_args.attn_implementation,
        "use_ring_flash_attn": use_ring_flash_attn,
        "use_flash_attn": use_flash_attn,
    }
    if use_ring_flash_attn:
        import torch.distributed as dist
        world_size = dist.get_world_size()
        # create new groups, ring_attn_group_size gpus one group
        if model_args.ring_attn_group_size is None:
            model_args.ring_attn_group_size = world_size
        model_args.ring_attn_group_size = min(model_args.ring_attn_group_size, world_size)
        
        num_groups = dist.get_world_size() // model_args.ring_attn_group_size
        group_list = []
        for i in range(num_groups):
            group_list.append(
                dist.new_group(ranks=list(range(i * model_args.ring_attn_group_size, (i + 1) * model_args.ring_attn_group_size))))
        dist.barrier()
        model_init_kwargs["group_list"] = group_list
        # training_args.num_replicas = num_groups
        # training_args.sampler_rank = dist.get_rank() // model_args.ring_attn_group_size
        training_args.num_replicas = model_args.ring_attn_group_size
        training_args.sampler_rank = dist.get_rank() % model_args.ring_attn_group_size
        training_args.gradient_accumulation_steps *= training_args.num_replicas
        training_args.per_device_train_batch_size *= training_args.num_replicas
        
    if model_args.problem_type == "generation":
        if model_args.enable_cross_attention and model_args.do_pretrain:
            initial_emsemble_model_path = Path(training_args.output_dir).parent / "initial_model"
            if not os.path.exists(initial_emsemble_model_path):
                print("Creating initial model...")
                model = InternVLChatModel._from_config(config, **model_init_kwargs)
                pretrained_model = AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, torch_dtype=torch_dtype)
                model.load_state_dict(pretrained_model.state_dict(), strict=False)
                for layer in model.language_model.model.layers:
                    layer.cross_attention.load_state_dict(layer.attention.state_dict(), strict=True)
                    layer.cross_attention_norm.load_state_dict(layer.attention_norm.state_dict(), strict=True)
                    # print(layer.cross_attn_attn_gate)
                    # gate_state_dict = {'cross_attn_attn_gate': torch.zeros(1, device=model.device, dtype=torch_dtype)}
                    # layer.load_state_dict(gate_state_dict, strict=False, assign=True)
                model.save_pretrained(initial_emsemble_model_path)
                tokenizer.save_pretrained(initial_emsemble_model_path)
                del pretrained_model
                print("Saved initial model to:", initial_emsemble_model_path)
                print("Please re-run the script to load the initial model")
            else:
                model = InternVLChatModel.from_pretrained(initial_emsemble_model_path, config=config, trust_remote_code=True, **model_init_kwargs)
            model = model.to(training_args.device)
        
            # keep the vision backbone frozen all the time
            for name, param in model.named_parameters():
                tune_key_words = ["cross_attention", "cross_attn"]
                if any([x in name for x in tune_key_words]):
                    param.requires_grad = True
                    print("Enabling gradient for", name)
                else:
                    param.requires_grad = False
            assert training_args.gradient_checkpointing == False, "Gradient checkpointing is not supported for partial training cross attention layers for now"
        else:
            model = InternVLChatModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True, **model_init_kwargs)
    else:
        raise NotImplementedError("Only generation is supported for now")
    # copied from intern_vl training script
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    for name, param in model.named_parameters():
        not_to_tune_key_words = ["vision_model"]
        if any([x in name for x in not_to_tune_key_words]):
            param.requires_grad = False
    
    if bnb_config:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    print("Successfully loaded model from:", model_args.model_name_or_path)
    model.img_context_token_id = processor.img_context_token_id
    
    if model_args.lora_enabled or model_args.qlora_enabled:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=model_args.lora_dropout,
            use_dora=model_args.dora_enabled,
            init_lora_weights="gaussian"
        )
        model = get_peft_model(model, lora_config)
    
    set_ignore_index(-100)
    # set_default_image_token_id(model.config.image_token_id)
    # set_default_video_token_id(model.config.video_token_id)
    set_default_image_token("<image>") # this will be transformed to <|vision_start|><|image_pad|><|vision_end|> in the conversation template of qwen2_vl
    set_default_video_token("<video>") # this will be transformed to <|vision_start|><|video_pad|><|vision_end|> in the conversation template of qwen2_vl
    
    return model, processor
    

def main(
    training_args: TrainingArguments,
    data_args: DataArguments,
    model_args: ModelArguments,
):
    training_args.output_dir = Path(training_args.output_dir) / model_args.model_name_or_path.split("/")[-1] / training_args.run_name
    
    training_args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args.output_dir = str(training_args.output_dir)
    training_args.remove_unused_columns = False
    data_args.is_master_worker = training_args.local_rank in [-1, 0]
    
    if not training_args.resume_from_checkpoint:
        training_args.resume_from_checkpoint = True
    if training_args.resume_from_checkpoint == True:
        # search for the latest checkpoint
        all_checkpoints = list(Path(training_args.output_dir).glob("checkpoint-*"))
        all_checkpoints = [x for x in all_checkpoints if (x / "trainer_state.json").exists() and not x.name.endswith("final")]
        if len(all_checkpoints) == 0:
            training_args.resume_from_checkpoint = None
            print("No checkpoint found, starting from scratch")
        else:
            all_checkpoints = [str(x) for x in all_checkpoints]
            latest_checkpoint = max(all_checkpoints, key=os.path.getctime)
            training_args.resume_from_checkpoint = latest_checkpoint
            print("Resuming from checkpoint", latest_checkpoint)
    
    model, processor = load_model(model_args, training_args)
    
    if model_args.conv_template:
        data_args.conv_format = conv_templates[model_args.conv_template] 
    else:
        data_args.conv_format = conv_templates["internvl2_5"]
    print("Using conversation template:", data_args.conv_format)
    if data_args.data_config_file is not None:
        train_dataset, val_dataset, test_dataset, collate_fn = load_data_from_config(data_args, processor)
    else:
        raise ValueError("Data config file is required")
    
    if model_args.problem_type != "generation":
        assert all([isinstance(x, ClassificationDataset) for x in train_dataset.datasets])
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        tokenizer=processor
    )
    if trainer.is_world_process_zero():
        print("Training arguments:")
        print(training_args)
        print("Data arguments:")
        print(data_args)
        print("Model arguments:")
        print(model_args)
    if training_args.do_train:
        print("Training model...")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        # save
        final_checkpoint_dir = os.path.join(training_args.output_dir, 'checkpoint-final')
        if model_args.lora_enabled:
            state_dict = get_peft_state_maybe_zero_3(
                model.named_parameters(), model_args.lora_bias
            )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                model.named_parameters()
            )
            if training_args.local_rank == 0 or training_args.local_rank == -1:
                model.config.save_pretrained(final_checkpoint_dir)
                model.save_pretrained(final_checkpoint_dir, state_dict=state_dict)
                torch.save(non_lora_state_dict, os.path.join(final_checkpoint_dir, 'non_lora_trainables.bin'))
        else:
            trainer.save_model(output_dir=final_checkpoint_dir)
        processor.save_pretrained(final_checkpoint_dir)
    if training_args.do_predict:
        print("Predicting...")
        trainer.predict(test_dataset)


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, DataArguments, ModelArguments))
    training_args, data_args, model_args = parser.parse_args_into_dataclasses()

    main(training_args, data_args, model_args)