# M3-AGIQA: Multimodal, Multi-Round, Multi-Aspect AI-Generated Image Quality Assessment

Repository of paper *M3-AGIQA: Multimodal, Multi-Round, Multi-Aspect AI-Generated Image Quality Assessment*. 

M3-AGIQA integrates multimodal large language models (MLLMs) to evaluate AI-Generated Images across multiple dimensions, leveraging distilled quality captioning capabilities from online MLLMs to local models.

Performance comparison on quality aspect of AIGCIQA2023 dataset:
<img src="./radar_plot.png" width="400" alt="Performance comparison on quality aspect of AIGCIQA2023 dataset" />

## Datasets & Checkpoints
> Disclaimer: The datasets uploaded are from external papers and are not owned by the repository owner. They are hosted on Hugging Face or Google Drive for easier access.

[AGIQA-3k](https://github.com/lcysyzxdxc/AGIQA-3k-Database), [AIGCIQA2023](https://github.com/wangjiarui153/AIGCIQA2023), [AIGCIQA-20k](https://www.modelscope.cn/datasets/lcysyzxdxc/AIGCQA-30K-Image), They can be also downloaded from [GoogleDrive](https://drive.google.com/drive/folders/1-Y75CJgRpgdAVpUAC0y3tapl2xpzg8-x?usp=sharing) 
> For dataset AIGCIQA-20k, you may download the [original dataset](https://www.modelscope.cn/datasets/lcysyzxdxc/AIGCQA-30K-Image) and then put descriptors in `aigciqa-20k.7z` with it.

Also they're avaiable on Huggingface: [AGIQA-3k](https://huggingface.co/datasets/strawhat/agiqa-3k), [AIGCIQA2023](https://huggingface.co/datasets/strawhat/aigciqa2023) (Only metadata), [AIGCIQA-20k](https://huggingface.co/datasets/strawhat/aigciqa-20k).

Fine-tuned adapters:
- AGIQA-3k: [quality](https://huggingface.co/strawhat/minicpm2.5-agiqa-3k-ft), [correspondence](strawhat/minicpm2.5-agiqa-3k-corr-ft)
- AIGCIQA2023: [quality](https://huggingface.co/strawhat/minicpm2.5-aigciqa2023-ft), [correspondence](strawhat/minicpm2.5-aigciqa2023-corr-ft), [authenticity](strawhat/minicpm2.5-aigciqa2023-auth-ft),
- AIGCIQA-20k: [quality](https://huggingface.co/strawhat/minicpm2.5-aigciqa-20k-ft)

Checkpoints can be loaded as follows (`strawhat/minicpm2.5-aigciqa-20k-ft` as an example):
```py
from transformers import AutoModel
from peft import PeftModel

model = AutoModel.from_pretrained("openbmb/MiniCPM-Llama3-V-2_5", trust_remote_code=True, torch_dtype=torch.float16).eval()
model = PeftModel.from_pretrained(model, "strawhat/minicpm2.5-aigciqa-20k-ft", trust_remote_code=True, torch_dtype=torch.float16).eval()
```

## Train & predict
An xlstm environment is required to run the training script, notice that the xlstm library need to be run on the first GPU (cuda:0).
```bash
conda env create -f ./environment_xlstm.yml
conda activate xlstm

# training, need at least 20GB vram
python main.py --config ./cfg/minicpm-xlstm-agiqa-3k.yaml --run_name testrun

# predicting
# scores will be stored in ./predictions/<model_name>/<run_name>.json
python main.py --config ./cfg/minicpm-xlstm-agiqa-3k.yaml --stage predict --ckpt_path <best_checkpoint_path> --run_name testrun_predictions
```

## Dataset Adaption
### With Training
1. Prepare dataset for training with structure:
```bash
|--my_dataset
    |--images   # all the images
    |--data.csv # images and their MOS score
    |--train.json   # json file with 2 conversations, answers can be empty since no fine-tuning needed. Check examples in ./data_processed 
    |--val.json     
```
2. Call `inference.py` with proper parameters `FINETUNED_CKPT` (`strawhat/minicpm2.5-aigciqa-20k-ft` is recommended for better cross dataset performance), `eval_data_json`, and `OUTPUT_FILE` to get the response by fine-tuned MLLM;
3. Create your own configuration file in `./cfg` (`strawhat/minicpm2.5-aigciqa-20k-ft` would require `model_max_length` set to `768` and `max_slice_nums` set to `4`);
4. Launch the training and predicting command as previous section `Train & predict` describes.

### Without Training
1. Prepare and call `inference.py` as step 1&2 in section `w/ training`, you may only need `test.json` which represents your whole dataset due to no training process.
2. Run command for `predicting` to predict the scores.

## Fine-tune & Train from scratch
Our experiments showed that with fine-tuned MLLM and additional training, the result is promising, while if you prefer to train your own MLLM from scratch, additional steps would be taken into account:
1. Produce intermediate image quality descriptions in your preferred aspect (quality, correspondence, authenticity, or any other) manually or by online MLLM api. In our case, you may try the Gemini Flash API from Google, by running `./analyzers/gemini_image_analyzer.py` with a `./analyzers/api.key` file including your Gemini API keys;
2. Prepare fine-tuning environment according to [MiniCPM](https://github.com/OpenBMB/MiniCPM-o);
3. Copy scripts in `./finetune` in this project to MiniCPM environment and apply your modifications and then run `./finetune_lora.sh` to fine-tune the local MLLM;
> 40GB vram recommended.
4. Do the rest training & predicting steps.

## Citation
If you find our work useful, please cite it as follows:
```bibtex
@article{cui2025m3agiqa,
    title={M3-AGIQA: Multimodal, Multi-Round, Multi-Aspect AI-Generated Image Quality Assessment}, 
    author={Chuan Cui and Kejiang Chen and Zhihua Wei and Wen Shen and Weiming Zhang and Nenghai Yu},
    journal={arXiv preprint arXiv:2502.13763},
    year={2025}
}
```
