import PIL
from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import copy

# offline inference
# export HF_HUB_OFFLINE=1

FINETUNED=True
MODEL_NAME="openbmb/MiniCPM-Llama3-V-2_5"

FINETUNED_CKPT="strawhat/minicpm2.5-aigciqa-20k-ft"
eval_data_json="./data/somedataset/train.json"  # val.json or test.json
OUTPUT_FILE="./data_processed/somedataset/{aspect}_res.json"

if FINETUNED:
    model_name = MODEL_NAME
    model = AutoModel.from_pretrained(model_name, 
                                                    trust_remote_code=True).to(dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_CKPT, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, FINETUNED_CKPT, torch_dtype=torch.float16,
                                                    device_map="auto", trust_remote_code=True).eval().cuda()
    output_file = OUTPUT_FILE
else:
    model_name = MODEL_NAME
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True).eval().cuda()
    model.eval()
    output_file = OUTPUT_FILE

# load eval data
import json
import torch
from tqdm import tqdm


PROMPT_QUALITY = "Based on your analysis, could you please rate the image's overall quality? Choose one word from the following list: [bad, poor, fair, good, excellent], where the words range from low to high quality."

PROMPT_ALIGNMENT = "Based on your analysis, could you please rate the image's alignment with its original prompt? Choose one word from the following list: [bad, poor, fair, good, excellent], where the words indicate the degree of alignment from low to high."

PROMPT_AUTHENTICITY = "Based on your analysis, could you please rate the image's authenticity? Choose one word from the following list: [bad, poor, fair, good, excellent], where the words indicate the degree of authenticity from low to high."


with open(eval_data_json, "r") as f:
    eval_data = json.load(f)

results_quality = []
results_alignment = []
results_authenticity = []
for d in tqdm(eval_data):
    img = PIL.Image.open(d["image"]).convert('RGB')
    msgs = [d['conversations'][0]]
    msgs[0]['content'] = msgs[0]['content'][8:]
    # to_be_append = d
    res = model.chat(image=img,
               msgs=msgs,
               context=None,
               tokenizer=tokenizer,
               sampling=False)
    d['response_0'] = res

    # second round
    msgs.append({'role': 'assistant', 'content': res})
    idx = 2
    original_response = ''
    while idx < len(d['conversations']):
        # now check quality
        # if 'quality' in d['conversations'][idx]['content']:
        # now check align
            if len(msgs) > 2:
                 msgs = msgs[:2]
            msgs.append(d['conversations'][idx])  # ask for the result
            res_1 = model.chat(image=img,
                        msgs=msgs,
                        context=None,
                        tokenizer=tokenizer,
                        sampling=False)
            d['response_1'] = res_1

            to_append = copy.deepcopy(d)
            to_append['conversations'] = [0] * 4
            to_append['conversations'][0] = d['conversations'][0]
            to_append['conversations'][1] = d['conversations'][1]
            to_append['conversations'][2] = d['conversations'][idx]
            to_append['conversations'][3] = d['conversations'][idx+1]

            # original_response = d['conversations'][idx+1]['content']
            if 'quality' in d['conversations'][idx]['content']:
                 results_quality.append(copy.deepcopy(d))
            elif 'alignment' in d['conversations'][idx]['content']:
                 results_alignment.append(copy.deepcopy(d))
            elif 'authenticity' in d['conversations'][idx]['content']:
                 results_authenticity.append(copy.deepcopy(d))
            idx = idx + 2
    

if len(results_quality) > 0:
    with open(output_file.format(aspect='quality'), "w") as f:
        json.dump(results_quality, f, indent=4)

if len(results_alignment) > 0:
    with open(output_file.format(aspect='alignment'), "w") as f:
        json.dump(results_alignment, f, indent=4)
        
if len(results_authenticity) > 0:
    with open(output_file.format(aspect='authenticity'), "w") as f:
        json.dump(results_authenticity, f, indent=4)



