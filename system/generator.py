import torch
import logging

from typing import Iterator, List, Dict
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, AutoConfig, 
    TrainingArguments, Trainer
)
from torch.utils.data import IterableDataset
from utils import ConfigGenerator, IO_SEP_TOKEN, PAD_TOKEN

logging.basicConfig(level=logging.INFO)
    

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

    @staticmethod
    def chunks(lst: List, n: int):  # yield successive n-sized chunks from lst
        for i in range(0, len(lst), n   ):
            yield lst[i: i + n]

    @staticmethod
    def trim_batch_data(input_ids, pad_token_id, attention_mask=None): # remove columns by pad_token_id
        keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
        if attention_mask is None:
            return input_ids[:, keep_column_mask]    
        else:
            return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])            


class GPTGenerator(BaseGenerator):
    def __init__(self, model_name, model_path, from_config: bool = False, config_name: str = None, is_autoreg: bool = True,
                        batch_size: int = 32, use_fp16: bool = False):
        super().__init__(model_name=model_name, model_path=model_path, from_config=from_config, config_name=config_name, 
                            is_autoreg=is_autoreg, batch_size=batch_size, use_fp16=use_fp16)
        self.cfg = ConfigGenerator()

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
            for input_text_batch in list(self.chunks(input_text, self.batch_size)):
                if self.cfg.num_batch:
                    if batch_idx >= self.cfg.num_batch:
                        break
                
                with torch.cuda.amp.autocast():
                    inputs = self.tokenizer(
                        input_text_batch,
                        return_tensors="pt",
                        truncation=True,
                        padding="max_length"
                    ).to(self.device)
                    
                    input_ids, attention_mask = self.trim_batch_data(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], pad_token_id=self.tokenizer.pad_token_id)
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
    
    def tune(self, data: Dict, num_epoch=1):

        class Dataset4Tune(IterableDataset):
            def __init__(self, data, tokenizer, cfg):
                self.data = data
                self.tokenizer = tokenizer
                self.cfg = cfg
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx) -> Iterator:
                source = self.data[idx].strip().rstrip("\n")
                target = self.data[idx].strip().rstrip("\n")

                source_idx = self.tokenizer(source, padding="do_not_pad", truncation=True, 
                                    max_length=self.cfg.max_source_len, return_tensors="pt")["input_ids"].squeeze(0)
                target_idx = self.tokenizer(target, padding="do_not_pad", truncation=True,
                                    max_length=self.cfg.max_target_len, return_tensor="pt")["input_ids"].squeeze(0)
                io_sep_token_id = self.tokenizer(IO_SEP_TOKEN)["input_ids"]
                io_sep_token_id = torch.Tensor(io_sep_token_id)
                eos_token_id = torch.Tensor([self.tokenizer.eos_token_id])
                if self.tokenizer.pad_token_id:
                    pad_token_id = self.tokenizer.pad_token_id
                else:
                    self.tokenizer.add_tokens([PAD_TOKEN], special_tokens=True)
                    pad_token_id = self.tokenizer(PAD_TOKEN)["input_ids"][0]
                print(f"===== PAD TOKEN ID: {pad_token_id} ======")
                x = torch.cat([source_idx, io_sep_token_id, target_idx, eos_token_id], dim=0)
                input_span = len(source_idx)    
                # labels are everything after input span, not standard language modeling, it's a seq2seq setup (similar strategy used to train COMET with GPT-2)
                y = torch.cat([torch.Tensor([-100] * (input_span)), x[input_span:]], dim = 0)   
                attention_mask = torch.tensor([1] * len(x))
                assert x.shape == y.shape, f"x.shape {x.shape} != y.shape {y.shape}"
                
                max_input_len = self.cfg.max_source_len + self.cfg.max_target_len + 2
                pad_len = max_input_len - len(x)    # pad tensors to max_input_len
                x = torch.nn.functional.pad(x, (0, pad_len), value=pad_token_id)
                attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len))
                y = torch.nn.functional.pad(y, (0, pad_len), value=pad_token_id)
                return (x.long(), attention_mask.long(), y.long())

        train_dataset = Dataset4Tune(data=data, tokenizer=self.tokenizer, cfg=self.cfg)
        training_args = TrainingArguments(do_train=True, do_eval=False, output_dir=self.model, overwrite_output_dir=True,
                            num_train_epochs=num_epoch, fp16=self.use_fp16, logging_steps=128, save_steps=1024, 
                            per_device_train_batch_size=self.batch_size, warmup_steps=128, weight_decay=0.01, 
                            logging_dir=self.model_path, logging_strategy="steps", report_to="wandb")
        trainer = Trainer(model=self.model, args=training_args, train_dataset=train_dataset, 
                            data_collator=lambda data: {"input_ids": torch.stack([f[0] for f in data]),
                                                        "attention_mask": torch.stack([f[1] for f in data]), 
                                                        "label": torch.stack([f[2] for f in data])})
        trainer.train()
        self.tokenizer.save_pretrained(self.model_path)
    


# ===== DEBUG =====
if __name__ == "__main__":
    generator = GPTGenerator(model_name="EleutherAI/gpt-neo-1.3B", model_path="EleutherAI/gpt-neo-1.3B")
    generator.load()