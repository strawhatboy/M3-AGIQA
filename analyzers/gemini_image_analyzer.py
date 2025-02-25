from pathlib import Path
from time import sleep
import google.generativeai as genai
import os
from pprint import pprint
from PIL import Image
import pandas as pd
from tqdm import tqdm
import random

# genai.configure(api_key=os.environ['API_KEY'])

PROMPT = 'Please detailedly help to analyze the image and provide semantic information inside, check if they matches the original prompt for generating it: "{prompt}". And provide the scores on 3 dimentions: image quality, alignment between prompt and image, whether it can be recognized as AI-generated. scores are between 0 and 1. you may output the analysis and scores in json format, with keys "semantic_info", "image_quality", "prompt_alignment", "ai_generated_recognition", "image_quality_score", "prompt_alignment_score", "ai_generated_probability", "overall_conclusion"'

PROMPT = '''Analyze the provided image and extract semantic information present within it. 

1. **Semantic Information:**  Describe the objects, characters, actions, and overall scene depicted in the image. Be as detailed as possible, including relationships between elements and potential narrative implications. 

2. **Image Quality:** Assess the technical quality of the image based on factors like resolution, clarity, composition, and visual appeal. Assign a score between 0 and 1, where 1 represents excellent quality.

3. **Prompt Alignment:** Evaluate how well the image reflects the original prompt used for its generation: "{prompt}".  Assign a score between 0 and 1, where 1 represents perfect alignment.

4. **AI-Generated Recognition:** Analyze the image for telltale signs of AI generation, such as unnatural textures, distorted features, or inconsistencies in lighting and perspective. Assign a probability score between 0 and 1, where 1 represents high confidence of AI origin.

Output your analysis in JSON format with the following keys:

* `"semantic_info"`:  (string) Detailed description of the image content.
* `"image_quality"`: (string) analysis about the image's quality.
* `"prompt_alignment"`: (string) analysis about prompt alignment.
* `"ai_generated_recognition"`: (string) analysis about ai generated recognition.
* `"image_quality_score"`: (float) Score between 0 and 1
* `"prompt_alignment_score"`: (float) Score between 0 and 1.
* `"ai_generated_probability"`: (float) Score between 0 and 1.
* `"overall_conclusion"`: (string) A concise summary of your analysis, highlighting key findings and your confidence in its AI origin. '''

PROMPT = '''Analyze this image generated from the text prompt: "{prompt}". it's score is {mos_q}, scores are between {min} and {max}.

Provide your assessment in JSON format with the following keys:

* **quality_explanation**: Describe the overall quality of the image. Consider aspects like composition, clarity, realism (if applicable), and any noticeable flaws.  
* **alignment_explanation**: Explain how well the image reflects the elements and intent of the provided prompt. Be specific about which aspects of the prompt are successfully conveyed and which might be missing or misinterpreted. 

Do not include the numerical scores in your response.  

**Example:**

```json
{{
 "quality_explanation": "The image is highly detailed and possesses good lighting, creating a realistic scene. However, there's a slight blur in the background that detracts from the overall sharpness.",
 "alignment_explanation": "The image accurately depicts the objects described in the prompt and maintains the correct spatial relationships. The color scheme aligns with the prompt's description as well. However, the action requested in the prompt is not present in the generated image."
}}```'''

PROMPT = '''Analyze this image generated from the text prompt: "{prompt}"

The score is {mos_q}. Scores are between {min} and {max}, with higher scores being better.

Provide your assessment in JSON format with the following keys:

quality_explanation: Describe the overall quality of the image, considering aspects such as composition, clarity, and any noticeable flaws.
alignment_explanation: Explain how well the image reflects the elements and intent of the provided prompt. Be specific about which aspects of the prompt are successfully conveyed and which might be missing or misinterpreted.
authenticity_explanation: Discuss how closely the image resembles real artworks. Highlight any parts of the image that appear non-real or artificial.
Do not include the numerical scores in your response.

**Example:**
```json
{{
  "quality_explanation": "The overall quality of the image is quite high. The composition is balanced, with the boat centered and leading the viewer's eye towards the bridge and beyond. The clarity is excellent, with sharp details on the boat, water, and surrounding cliffs. The lighting is dramatic and enhances the overall atmosphere of the scene. However, there are some noticeable flaws, such as the overly saturated colors, which can detract from the natural feel of the image.",
  "alignment_explanation": "The image closely aligns with the prompt 'bridge over a body of water with a boat in the water.' The bridge is prominently featured, and the boat is clearly visible in the water. The scene captures the essential elements of the prompt well. However, some additional details, such as the type of bridge or the style of the boat, could have been more specific to better reflect the intent of the prompt.",
  "authenticity_explanation": "The image has a surreal, almost dreamlike quality, which makes it less authentic as a representation of a real-world scene. The colors are highly saturated and the lighting effects are dramatic, which enhances the artistic feel but reduces the realism. The boat and the bridge look more like artistic renditions rather than actual structures, and the overall scene feels more like a digital artwork or a scene from a fantasy world rather than a photograph of a real place."
}}```'''

# align only
PROMPT = '''Analyze this image generated from the text prompt: "{prompt}"

The alignment score between the image and text prompt is {mos_a}. Scores are between {min} and {max}, with higher scores being better.

Provide explain how well the image reflects the elements and intent of the provided prompt. Be specific about which aspects of the prompt are successfully conveyed and which might be missing or misinterpreted.
Do not include the numerical scores in your response.

**Example:**
The image closely aligns with the prompt 'bridge over a body of water with a boat in the water.' The bridge is prominently featured, and the boat is clearly visible in the water. The scene captures the essential elements of the prompt well. However, some additional details, such as the type of bridge or the style of the boat, could have been more specific to better reflect the intent of the prompt.
'''
# for model in genai.list_models():
#     pprint(model.name)

class GeminiImageAnalyzer:
    def __init__(self):
        genai.list_models()
        self.model = genai.GenerativeModel('models/gemini-1.5-flash')

    def analyze(self, prompt, img, mos_q=0, mos_a=0, mos_r=0, min_value=0, max_value=5):
        try:
            res = self.model.generate_content([PROMPT.format(prompt=prompt, mos_a=mos_a, min=min_value, max=max_value), img])
            pprint(res)
            return res.candidates[0].content.parts[0].text
        except Exception as e:
            pprint(e)
            return 'error'
        

ERROR_PATIENCE=10

if __name__ == '__main__':
    # load api keys
    with open('./api.key', 'r') as f:
        api_keys = f.readlines()

    # remove empty lines and strip each line
    api_keys = [key.strip() for key in api_keys if key.strip() != '']

    current_api_key_idx = 0
    # set api_key
    genai.configure(api_key=api_keys[current_api_key_idx])
    print(f'Using API_KEY {api_keys[current_api_key_idx]}...')
    # loop through the image dataset
    data_path = Path("../data/AIGCIQA2023")
    odescriptor = pd.read_csv(data_path / 'data.csv')
    image_dir = data_path / 'images'
    output_csv = data_path / 'gemini_pro_analysis_explaination_align.csv'
    error_patience = ERROR_PATIENCE

    success_rate = 0

    n_epoch = 0
    while success_rate < 1.0:
        if output_csv.exists():
            # load output_csv, and skip the ones that are already analyzed
            origin_df = pd.read_csv(output_csv)
            # select the ones that are not analyzed, in origin_df, success=False and those that are not in origin_df
            xdescriptor = odescriptor[odescriptor['name'].isin(origin_df[origin_df['success'] == False]['name'])]
            # select also the ones that are not in origin_df
            ydescriptor = odescriptor[~odescriptor['name'].isin(origin_df['name'])]
            descriptor = pd.concat([xdescriptor, ydescriptor], axis=0)

            success_rate = len(origin_df[origin_df['success'] == True]) / len(odescriptor)
            pprint(f'epoch {n_epoch}: {success_rate}')
        else:
            descriptor = odescriptor

        data = []
        analyzer = GeminiImageAnalyzer()
        n_total = 0
        n_success = 0
        the_list = list(descriptor.iterrows())
        random.shuffle(the_list)
        bar = tqdm(list(enumerate(the_list)))    # shuffle to avoid the same failed processed image
        
        for idx, (index, row) in bar:

            img_path = image_dir / row['name']
            print(f'Processing {img_path}... with prompt: {row["prompt"]} and alignment score {row["mos_correspondence"]}')
            img = Image.open(img_path)
            prompt = row['prompt']
            # AIGCIQA2023: mos_quality,mos_authenticity,mos_correspondence, min=0, max=100
            res = analyzer.analyze(prompt, img, mos_a=row['mos_correspondence'], min_value=0, max_value=100)
            success = res != 'error'
            if not success:
                error_patience -= 1
                if error_patience == 0:
                    # switch to another API_KEY
                    print(f'Switching API_KEY from {api_keys[current_api_key_idx]}...', end='')
                    current_api_key_idx = (current_api_key_idx + 1) % len(api_keys)
                    print(f'to {api_keys[current_api_key_idx]}')
                    genai.configure(api_key=api_keys[current_api_key_idx])
                    # reset error_patience
                    error_patience = ERROR_PATIENCE
                    pass
            else:
                error_patience = ERROR_PATIENCE

            data.append({'name': row['name'], 'prompt': prompt, 'result': res, 'success': success})
            print(res)

            n_total += 1
            n_success += success
            success_rate = n_success / n_total

            bar.set_postfix({'success_rate': success_rate})
            
            sleep(2)
            # print(idx)
            if idx > 20:
                break
        n_epoch += 1

        if output_csv.exists():
            success_rate = (n_success + len(origin_df[origin_df['success'] == True])) / len(odescriptor)
        else: 
            success_rate = n_success / len(odescriptor)

        print('flushing result...', end='')
        # merge the result with the original dataframe every 100 data
        df = pd.DataFrame(data)
        if len(df) == 0:
            continue
        df_left = odescriptor[odescriptor['name'].isin(df['name'])]
        if output_csv.exists():
            origin_df = origin_df[origin_df['success'] == True]
            # concat origin_df (success=True) and df (processed success=False) and df_left (not processed success=False)
            df = pd.concat([origin_df, df, df_left], axis=0)
        df.to_csv(output_csv, index=False)
        print(f'ok, flushed {len(df)} results')
