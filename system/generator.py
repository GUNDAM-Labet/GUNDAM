import torch
import logging

from typing import List, Dict
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, AutoConfig, 
    TrainingArguments, Trainer
)
from utils import trim_batch_data
from config import ConfigGenerator
from manager import BaseManager

logging.basicConfig(level=logging.INFO)

def chunks(lst: List, n: int):  # yield successive n-sized chunks from lst
    for i in range(0, len(lst), n):
        yield lst[i: i + n]

class BaseGenerator():
    def __init__(self, model_name, model_path, from_config: bool, config_name: str, is_autoreg: bool,
                        batch_size: int, use_fp16: bool):
        self.model_name = model_name
        self.model_path = model_path
        self.from_config = from_config
        self.config_name = config_name
        
        self.use_fp16 = use_fp16
        self.is_autoreg = is_autoreg
        self.batch_size = batch_size
        self.max_context_len = 4096
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"use device: {self.device}, load model from {model_path}")

        self.model, self.tokenizer = None, None
        
    
    def load(self):
        raise NotImplementedError
    
    def set(self):
        raise NotImplementedError

    def act(self, input_text) -> Dict:
        raise NotImplementedError

    def tune(self):
        raise NotImplementedError


class GPTGenerator(BaseGenerator):
    def __init__(self, model_name, model_path, from_config: bool = False, config_name: str = None, is_autoreg: bool = True,
                        batch_size: int = 32, use_fp16: bool = False):
        super().__init__(model_name=model_name, model_path=model_path, from_config=from_config, config_name=config_name, 
                            is_autoreg=is_autoreg, batch_size=batch_size, use_fp16=use_fp16)
        
        self.cfg = ConfigGenerator(
            model_name=self.model_name, model_path=self.model_path, from_config=self.from_config, config_name=self.config_name
        )

    def load(self):
        if self.from_config:
            tokenizer, model = self._load_from_config(self.config_name, self.model_path)
        else:
            tokenizer, model = self._load_from_pretrained(self.config_name, self.model_name, self.model_path)
        
        self.tokenizer = tokenizer
        self.model = model.to(self.device).eval()

        if self.is_autoreg:
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
            
            if len(self.tokenizer) != self.model.config.vocab_size:
                print(f"tokenizer vocab size {len(self.tokenizer)} not match model vocab size {self.model.config.vocab_size}")
                self.model.resize_token_embeddings(len(self.tokenizer))
        
        if hasattr(self.model.config, "n_positions"):
            self.max_context_len = self.model.config.n_positions
        elif hasattr(self.model.config, "n_ctx"):
            self.max_context_len = self.model.config.n_ctx
        else:
            self.max_context_len = 4096

    def set(self, decode_method: str = None, num_generate: int = None, num_return_sequence: int = None, 
                add_score: bool = None, temperature: float = None, max_new_tokens: int = None, 
                num_batch: int = None, max_source_len: int = None, max_target_len: int = None):
        self.cfg.set(
            decode_method=decode_method, add_score=add_score, num_generate=num_generate, max_new_tokens=max_new_tokens,
            num_batch=num_batch, num_return_sequence=num_return_sequence, temperature=temperature, 
            max_source_len=max_source_len, max_target_len=max_target_len
        )

    def _load_from_config(self, config_name: str, model_path: str):
        assert self.is_autoreg, "config only works for autoreg now"
        config = AutoConfig.from_pretrained(config_name)
        model = AutoModelForCausalLM.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained(config_name)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        if self.use_fp16:
            model = model.half()
        return (tokenizer, model)
    
    def _load_from_pretrained(self, config_name: str, model_name: str, model_path: str):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except OSError: # for cases where only the model is saved, not the tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config_name)
        
        if self.is_autoreg:
            model = AutoModelForCausalLM.from_pretrained(model_path)
            if self.use_fp16:
                model = model.half()
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        return (tokenizer, model)

    def act(self, input_text) -> Dict:
        kwargs = self.cfg.get()
        generations = []
        scores = []
        batch_idx = 0
        
        with torch.no_grad():
            for input_text_batch in list(chunks(input_text, self.batch_size)):
                if self.cfg.num_batch:
                    if batch_idx >= self.cfg.num_batch:
                        break
                
                with torch.cuda.amp.autocast():
                    input_ids = self.tokenizer(
                        input_text_batch,
                        return_tensors="pt",
                        truncation=True,
                        padding="max_length"
                    ).to(self.device)

                    input_ids, attention_mask = trim_batch_data(**input_ids, pad_token_id=self.tokenizer.pad_token_id)
                    if input_ids.shape[1] >= self.max_context_len - self.cfg.max_new_tokens:
                        input_ids = input_ids[:,:self.max_context_len-self.cfg.max_new_tokens]
                        attention_mask = attention_mask[:,:self.max_context_len-self.cfg.max_new_tokens]
                    batch_generations = self.model.generate(
                        input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=self.cfg.max_new_tokens, **kwargs,
                    )
                
                if self.cfg.add_score:
                    batch_scores = batch_generations["sequence_scores"] 
                    batch_generations = batch_generations["sequence"]
                batch_generations = self.tokenizer.batch_decode(batch_generations, skip_special_tokens=False, clean_up_tokenization_spaces=False)

                if (self.num_return_sequence and self.num_return_sequence > 1) or (self.cfg.num_generate and self.cfg.num_generate > 1):
                    self.num_return_sequence = self.num_return_sequence if self.num_return_sequence else self.cfg.num_generate
                    for i in range(0, len(batch_generations), self.num_return_sequence):
                        generations.append(batch_generations[i : i + self.num_return_sequence])
                
                if self.cfg.add_score:
                    scores.append(batch_scores.detach().cpu().tolist())
                
                batch_idx += 1
        
        if self.cfg.add_score:
            return (generations, scores)
        else:
            return generations
    
    def tune(self, data: BaseManager, num_epoch=1):
        training_args = TrainingArguments(do_train=True, do_eval=False, output_dir=self.model, overwrite_output_dir=True,
                            num_train_epochs=num_epoch, fp16=self.use_fp16, logging_steps=128, save_steps=1024, 
                            per_device_train_batch_size=self.batch_size, warmup_steps=128, weight_decay=0.01, 
                            logging_dir=self.model_path, logging_strategy="steps", report_to="wandb")
        trainer = Trainer(model=self.model, args=training_args, train_dataset=data, 
                            data_collator=lambda data: {"input_ids": torch.stack([f[0] for f in data]),
                                                        "attention_mask": torch.stack([f[1] for f in data]), 
                                                        "label": torch.stack([f[2] for f in data])})
        trainer.train()
        self.tokenizer.save_pretrained(self.model_path)