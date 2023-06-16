

class ConfigGenerator:
    def __init__(self, decode_method: str = "beam", add_score: bool = False, num_generate: int = 5, 
                    max_source_len: int = 64, max_target_len: int = 10, max_new_tokens: int = 150, 
                    num_batch: int = None, num_return_sequence: int = None, temperature: float = None):
        self.decode_method = decode_method
        self.add_score = add_score
        self.num_generate = num_generate
        self.num_return_sequence = num_return_sequence
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.num_batch = num_batch
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def set(self, decode_method=None, add_score=None, num_generate=None, max_new_tokens=None, num_batch=None, 
                num_return_sequence=None, temperature=None, max_source_len=None, max_target_len=None):
        self.decode_method = decode_method if decode_method else self.decode_method
        self.add_score = add_score if add_score else self.add_score
        self.num_generate = num_generate if num_generate else self.num_generate
        self.num_return_sequence = num_return_sequence if num_return_sequence else self.num_return_sequence
        self.temperature = temperature if temperature else self.temperature
        self.max_new_tokens = max_new_tokens if max_new_tokens else self.max_new_tokens
        self.num_batch = num_batch if num_batch else self.num_batch
        self.max_source_len = max_source_len if max_source_len else self.max_source_len
        self.max_target_len if max_target_len else self.max_target_len
    
    def get(self):
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
            "temperature": self.temperature
        }
        if self.add_score and self.decode_method == "greedy":
            self.decode_method = "greedy_add_score"
        kwargs = method_to_kwargs[self.decode_method]
        kwargs.update(common_kwargs)
        return kwargs

