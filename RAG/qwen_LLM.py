from typing import Optional, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from langchain.llms.base import LLM
import torch

class Qwen_LLM(LLM):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    device: str = None  
        
    def __init__(self, mode_name_or_path: str, device: Optional[str] = None):
        super().__init__()
        print("Loading model from the local path...")
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            mode_name_or_path,
            torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
            device_map={'': self.device} if self.device == 'cuda' else None,
            trust_remote_code=True
        )
        self.model.generation_config = GenerationConfig.from_pretrained(mode_name_or_path, trust_remote_code=True)
        if self.device == 'cuda':
            self.model.to(self.device)
        print("Model loading completed.")
        
    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[Any] = None,
              **kwargs: Any) -> str:

        messages = [{"role": "user", "content": prompt}]
        
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([input_ids], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response

    @property
    def _llm_type(self) -> str:
        return "Qwen2.5_LLM"
    
    def eval(self):
        self.model.eval()
