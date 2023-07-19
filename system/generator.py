import torch
import logging
import os

from typing import List, Dict
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, AutoConfig, 
    TrainingArguments, Trainer
)
from torch.utils.data import Dataset
from utils import ConfigGenerator, IO_SEP_TOKEN, PAD_TOKEN

logging.basicConfig(level=logging.INFO)
    

class BaseGenerator():
    def __init__(self, model_name: str, save_path: str, is_autoreg: bool, batch_size: int, use_fp16: bool, from_save: bool):
        self.model_name = model_name
        current_path = os.path.abspath(os.getcwd())
        if save_path is None:
            save_path = os.path.join(os.path.dirname(current_path), "model/")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        
        self.save_path = save_path
        self.from_save = from_save
        self.logging_path = os.path.join(os.path.dirname(current_path), "logg/")
        
        self.use_fp16 = use_fp16
        self.is_autoreg = is_autoreg
        self.batch_size = batch_size
        self.max_context_len = 4096
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if from_save:
            logging.info(f"use device: {self.device}, load model from {save_path}")
        else:
            logging.info(f"use device: {self.device}, load model from online")

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
    def __init__(self, model_name: str = "gpt2-medium", save_path: str = None, is_autoreg: bool = True, batch_size: int = 32, 
                        use_fp16: bool = False, from_save: bool = True):
        super().__init__(model_name=model_name, save_path=save_path, is_autoreg=is_autoreg, batch_size=batch_size, 
                            use_fp16=use_fp16, from_save=from_save)
        self.cfg = ConfigGenerator()

    def load(self):
        save_dir = os.path.join(self.save_path, f"{self.model_name}/")
        if self.from_save and os.path.exists(save_dir):
            tokenizer, model = self._load_from_save(save_dir)
        else:
            tokenizer, model = self._load_from_pretrained()
        
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

    def config(self, decode_method: str = None, num_generate: int = None, num_return_sequence: int = None, 
                add_score: bool = None, temperature: float = None, max_new_tokens: int = None, 
                num_batch: int = None, max_source_len: int = None, max_target_len: int = None):
        self.cfg.config(
            decode_method=decode_method, add_score=add_score, num_generate=num_generate, max_new_tokens=max_new_tokens,
            num_batch=num_batch, num_return_sequence=num_return_sequence, temperature=temperature, 
            max_source_len=max_source_len, max_target_len=max_target_len
        )

    def _load_from_save(self, save_dir: str):
        config = AutoConfig.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        state_dict = torch.load(save_dir)
        model.load_state_dict(state_dict)
        if self.use_fp16:
            model = model.half()
        return (tokenizer, model)
    
    def _load_from_pretrained(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.is_autoreg:
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            if self.use_fp16:
                model = model.half()
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        return (tokenizer, model)

    def act(self, input_text) -> Dict:
        kwargs = self.cfg.get()
        generations = []
        scores = []
        batch_idx = 0
        
        with torch.no_grad():
            for input_text_batch in list(self.chunks(input_text, self.batch_size)):
                if self.cfg.num_batch is not None:
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


                if (self.cfg.num_return_sequence and self.cfg.num_return_sequence > 1) or (self.cfg.num_generate and self.cfg.num_generate > 1):
                    self.cfg.num_return_sequence = self.cfg.num_return_sequence if self.cfg.num_return_sequence else self.cfg.num_generate
                    for i in range(0, len(batch_generations), self.cfg.num_return_sequence):
                        generations.extend(batch_generations[i : i + self.cfg.num_return_sequence])
                    batch_generations = generations
                
                if self.cfg.add_score:
                    scores.extend(batch_scores.detach().cpu().tolist())
                
                batch_idx += 1

        if self.cfg.add_score:
            return (batch_generations, scores)
        else:
            return batch_generations
    
    def tune(self, train: Dict, valid: Dict, num_epoch=1):

        class Dataset4Tune(Dataset):
            def __init__(self, data: Dict, tokenizer, cfg):
                self.data = data
                self.tokenizer = tokenizer
                self.cfg = cfg

                self.uids = list(self.data.keys())

                # config tokenizer
                io_sep_token_id = self.tokenizer(IO_SEP_TOKEN)["input_ids"]
                self.io_sep_token_id = torch.Tensor(io_sep_token_id)
                self.eos_token_id = torch.Tensor([self.tokenizer.eos_token_id])
                if self.tokenizer.pad_token_id:
                    self.pad_token_id = self.tokenizer.pad_token_id
                else:
                    self.tokenizer.add_tokens([PAD_TOKEN], special_tokens=True)
                    self.pad_token_id = self.tokenizer(PAD_TOKEN)["input_ids"][0]
            
            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                uid = self.uids[idx]
                source_input = self.data[uid].source_input.strip().rstrip("\n")
                target_output = self.data[uid].target_output.strip().rstrip("\n")
                source_idx = self.tokenizer(source_input, padding="do_not_pad", truncation=True, 
                                    max_length=self.cfg.max_source_len, return_tensors="pt")["input_ids"].squeeze(0)
                target_idx = self.tokenizer(target_output, padding="do_not_pad", truncation=True,
                                    max_length=self.cfg.max_target_len, return_tensors="pt")["input_ids"].squeeze(0)
                
                x = torch.cat([source_idx, self.io_sep_token_id, target_idx, self.eos_token_id], dim=0)
                input_span = len(source_idx)    
                
                # labels are everything after input span, not standard language modeling, it's a seq2seq setup (similar strategy used to train COMET with GPT-2)
                y = torch.cat([torch.Tensor([-100] * (input_span)), x[input_span:]], dim = 0)   
                attention_mask = torch.tensor([1] * len(x))
                assert x.shape == y.shape, f"x.shape {x.shape} != y.shape {y.shape}"
                
                max_input_len = self.cfg.max_source_len + self.cfg.max_target_len + 2
                pad_len = max_input_len - len(x)    # pad tensors to max_input_len
                
                x = torch.nn.functional.pad(x, (0, pad_len), value=self.pad_token_id)
                attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len))
                y = torch.nn.functional.pad(y, (0, pad_len), value=self.pad_token_id)
                return (x.long(), attention_mask.long(), y.long())

        train = Dataset4Tune(data=train, tokenizer=self.tokenizer, cfg=self.cfg)
        valid = Dataset4Tune(data=valid, tokenizer=self.tokenizer, cfg=self.cfg)
        training_args = TrainingArguments(do_train=True, do_eval=True, output_dir=self.save_path, overwrite_output_dir=True,
                            num_train_epochs=num_epoch, fp16=self.use_fp16, logging_steps=128, save_steps=1024, 
                            per_device_train_batch_size=self.batch_size, warmup_steps=128, weight_decay=0.01, 
                            logging_dir=self.logging_path, logging_strategy="steps", evaluation_strategy="steps", 
                            eval_steps=1024, report_to="wandb")
        trainer = Trainer(model=self.model, args=training_args, train_dataset=train, eval_dataset=valid,
                            data_collator=lambda data: {"input_ids": torch.stack([f[0] for f in data]),
                                                        "attention_mask": torch.stack([f[1] for f in data]), 
                                                        "labels": torch.stack([f[2] for f in data])})
        trainer.train()

        save_dir = os.path.join(self.save_path, f"{self.model_name}/")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        trainer.save_model(save_dir)
        self.tokenizer.save_pretrained(save_dir)
    
    def interact(self):
        input_text = input("> ")
        max_new_tokens = int(input("set max_new_tokens>"))
        self.config(max_new_tokens=max_new_tokens)

        while input_text != "exit":
            if self.is_autoreg:
                input_text = input_text #+ AutoregLMDataset.IO_SEP
            input_text = self.tokenizer.encode(input_text, return_tensors='pt').cuda()
            print(f"> input text: {input_text}")
            outputs = self.act(input_text=input_text)
            print(f"> output tokens: {outputs}")
            print(
                f"> output text: {self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)}"
            )
            print("=============================================")
            input_text = input("> ")