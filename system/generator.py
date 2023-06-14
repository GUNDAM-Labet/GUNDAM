import torch
import logging

from typing import List, Dict
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, AutoConfig
)
from data import trim_batch_data

logging.basicConfig(level=logging.INFO)


def chunks(lst: List, n: int):  # yield successive n-sized chunks from lst
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


class Generator:
    def __init__(self, model_name, model_path, use_config: bool = False, config_name: str = None, 
                    is_autoreg: bool = True, batch_size: int = 32, use_fp16: bool = False, 
                    decode_method: str = "beam", add_score: bool = False, 
                    num_generate: int = 5, num_return_sequence: int = None
                ):
        self.is_autoreg = is_autoreg
        self.use_fp16 = use_fp16
        if use_config:
            tokenizer, model = self.load_from_config(config_name, model_path)
        else:
            tokenizer, model = self.load_from_pretrained(config_name, model_name, model_path)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"use device: {self.device}, load model from {model_path}")
        self.tokenizer = tokenizer
        self.model = model.to(self.device).eval()

        if self.is_autoreg:
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
            
            if len(self.tokenizer) != self.model.config.vocab_size:
                print(f"tokenizer vocab size {len(self.tokenizer)} not match model vocab size {self.model.config.vocab_size}")
                self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.batch_size = batch_size
        if hasattr(self.model.config, "n_positions"):
            self.max_context_len = self.model.config.n_positions
        elif hasattr(self.model.config, "n_ctx"):
            self.max_context_len = self.model.config.n_ctx
        else:
            self.max_context_len = 4096

        self.add_score = add_score
        self.num_generate = num_generate
        self.num_return_sequence = num_return_sequence
        self.decode_method = decode_method


    def set_config(self, decode_method: str = None, num_generate: int = None, num_return_sequence: int = None, 
                        add_score: bool = None, temperature: float = None):
        self.decode_method = decode_method if decode_method else self.decode_method
        self.num_generate = num_generate if num_generate else self.num_generate
        self.num_return_sequence if num_return_sequence else self.num_return_sequence
        self.add_score = add_score if add_score else self.add_score

        # config model
        method_to_kwargs = {
            "beam": {
                "num_beams": self.num_generate,
                "early_stopping": True,
                "num_return_sequences": self.num_return_sequence if self.num_return_sequence else self.num_generate
            },
            "greedy": {
                "do_sample": False
            },
            "sample": {
                "do_sample": True,
                "num_return_sequences": self.num_return_sequence if self.num_return_sequence else self.num_generate
            },
            "greedy_add_score": {
                "return_dict_in_gen": True,
                "output_score": True
            }
        }
        common_kwargs = {
            "temperature": temperature
        }
        if self.add_score and self.decode_method == "greedy":
            self.decode_method = "greedy_add_score"
        kwargs = method_to_kwargs[self.decode_method]
        kwargs.update(common_kwargs)
        return kwargs


    def load_from_config(self, config_name: str, model_path: str):
        assert self.is_autoreg, "config only works for autoreg now"
        config = AutoConfig.from_pretrained(config_name)
        model = AutoModelForCausalLM.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained(config_name)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        if self.use_fp16:
            model = model.half()
        return (tokenizer, model)
    
    
    def load_from_pretrained(self, config_name: str, model_name: str, model_path: str):
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


    def act(self, input_text, max_new_tokens: int = 150, num_batch: int = None, temperature: float = 1.0) -> Dict:
        kwargs = self.set_config(temperature=temperature)
        generations = []
        scores = []
        batch_idx = 0
        
        with torch.no_grad():
            for input_text_batch in list(chunks(input_text, self.batch_size)):
                if num_batch:
                    if batch_idx >= num_batch:
                        break
                
                with torch.cuda.amp.autocast():
                    input_ids = self.tokenizer(
                        input_text_batch,
                        return_tensors="pt",
                        truncation=True,
                        padding="max_length"
                    ).to(self.device)

                    input_ids, attention_mask = trim_batch_data(**input_ids, pad_token_id=self.tokenizer.pad_token_id)
                    if input_ids.shape[1] >= self.max_context_len - max_new_tokens:
                        input_ids = input_ids[:,:self.max_context_len-max_new_tokens]
                        attention_mask = attention_mask[:,:self.max_context_len-max_new_tokens]
                    batch_generations = self.model.generate(
                        input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, **kwargs,
                    )
                
                if self.add_score:
                    batch_scores = batch_generations["sequence_scores"] 
                    batch_generations = batch_generations["sequence"]
                batch_generations = self.tokenizer.batch_decode(batch_generations, skip_special_tokens=False, clean_up_tokenization_spaces=False)

                if (self.num_return_sequence and self.num_return_sequence > 1) or (self.num_generate and self.num_generate > 1):
                    self.num_return_sequence = self.num_return_sequence if self.num_return_sequence else self.num_generate
                    for i in range(0, len(batch_generations), self.num_return_sequence):
                        generations.append(batch_generations[i : i + self.num_return_sequence])
                
                if self.add_score:
                    scores.append(batch_scores.detach().cpu().tolist())
                
                batch_idx += 1
        
        if self.add_score:
            return (generations, scores)
        else:
            return generations
            

