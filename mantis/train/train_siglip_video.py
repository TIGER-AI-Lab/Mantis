import dataclasses
from dataclasses import dataclass, field
import torch
import os
import wandb
import regex as re
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig
from transformers.hf_argparser import HfArgumentParser
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
from mantis.models.siglip_video import SiglipVideoModel, SiglipVideoConfig
from mantis.train.data import load_data_from_config, SiglipVideoDataset
from transformers import AutoProcessor, AutoConfig, SiglipModel
from pathlib import Path
from typing import Optional
from pathlib import Path

os.environ["WANDB_RESUME"] = "allow"
os.environ["WANDB_RUN_ID"] = wandb.util.generate_id()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)

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
        metadata={"help": "Whether to balance the dataset", "default": True, "required": False},
        default=False,
    )

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models", "default": "Qwen/Qwen2-VL-7B-Instruct", "required": False},
        default="google/siglip-so400m-patch14-384",
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
        metadata={"help": "The attention implementation to use", "default": "flash_attention_2", "required": False},
        default="flash_attention_2",
    )
    do_pretrain: Optional[bool] = field(
        metadata={"help": "Whether to pretrain the model", "default": False, "required": False},
        default=True,
    )
    initial_logit_scale: Optional[float] = field(
        metadata={"help": "Initial logit scale", "default": 10.0, "required": False},
        default=10.0,
    )
    initial_logit_bias: Optional[float] = field(
        metadata={"help": "Initial logit bias", "default": -10.0, "required": False},
        default=-10.0,
    )
    
    

def load_model(model_args, training_args):
    print("Loading model...")
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    
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
        
        
    if model_args.do_pretrain:
        if os.path.exists(training_args.output_dir + "/initial_model"):
            print("Loading initial model from", training_args.output_dir + "/initial_model")
            model = SiglipVideoModel.from_pretrained(
                training_args.output_dir + "/initial_model",
                attn_implementation=model_args.attn_implementation,
            ).to(training_args.device)
        else:
            print("Assembling initial model from pretrained models")
            config = SiglipVideoConfig.from_pretrained(model_args.model_name_or_path, attn_implementation=model_args.attn_implementation)
            model = SiglipVideoModel._from_config(config, torch_dtype=torch_dtype).to(training_args.device)
            pretrained_model = SiglipModel.from_pretrained(model_args.model_name_or_path)
            model.load_state_dict(pretrained_model.state_dict(), strict=False)
            model.save_pretrained(training_args.output_dir + "/initial_model")
    else:
        model = SiglipVideoModel.from_pretrained(
            model_args.model_name_or_path,
            attn_implementation=model_args.attn_implementation,
        ).to(training_args.device)
    
    if model_args.do_pretrain:
        # pre-train only the resampler
        for name, param in model.named_parameters():
            if "resampler" in name:
                param.requires_grad = True
                print("Training", name)
            else:
                param.requires_grad = False
        
    for name, param in model.named_parameters():
        if "logit" in name:
            param.requires_grad = True
            print("Training", name)
            
    if bnb_config:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
        
    
    print("Successfully loaded model from:", model_args.model_name_or_path)
    
    if model_args.lora_enabled or model_args.qlora_enabled:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules='.*(visual|model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
            lora_dropout=model_args.lora_dropout,
            use_dora=model_args.dora_enabled,
            init_lora_weights="gaussian"
        )
        model = get_peft_model(model, lora_config)
    
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
    
    
    if data_args.data_config_file is not None:
        train_dataset, val_dataset, test_dataset, collate_fn = load_data_from_config(data_args, processor)
    else:
        raise ValueError("Data config file is required")
    
    assert all([isinstance(x, SiglipVideoDataset) for x in train_dataset.datasets])
    
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