import PIL
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch

class MinicpmV2_5ImageAnalyzer():
    def __init__(self, args):
        self.args = args
        self.model_name = "openbmb/MiniCPM-Llama3-V-2_5"
        self.model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = self.model.to(device=f"cuda:{args['gpu']}")
        self.model.eval()

    def analyze(self, prompt, img):
        res = self.model.chat(image=img,
               msgs=[{'role': 'user', 'content': prompt}],
               context=None,
               tokenizer=self.tokenizer,
               sampling=False)
        return res
        pass



