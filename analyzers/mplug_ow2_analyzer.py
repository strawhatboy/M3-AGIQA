# need pip install "numpy<2.0.0"

import torch
from PIL import Image
from transformers import TextStreamer

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

image_file = '' # Image Path
model_path = 'MAGAer13/mplug-owl2-llama2-7b'
query = "Describe the image."



class mPLUGOwl2Analyzer:
    def __init__(self, args):
        self.args = args
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device=f"cuda:{args['gpu']}")
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len
        pass

    def analyze(self, prompt, img):
        img = img.convert('RGB')
        
        max_edge = max(img.size) # We recommand you to resize to squared image for BEST performance.
        img = img.resize((max_edge, max_edge))

        image_tensor = process_images([img], self.image_processor)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float32)
        
        conv = conv_templates["mplug_owl2"].copy()
        roles = conv.roles
        inp = DEFAULT_IMAGE_TOKEN + prompt
        conv.append_message(roles[0], inp)
        conv.append_message(roles[1], None)
        sys_prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(sys_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        temperature = 0.7
        max_new_tokens = 512

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        return outputs
