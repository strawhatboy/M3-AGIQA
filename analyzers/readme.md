# Analyzers

Analyze the image with MLLM or online APIs

to run the analyzer: (with `agiqa-3k` dataset in `./data` folder)
```bash
python analyzer.py --dataset agiqa-3k --provider minicpmv2.5 --gpu 2 --prompt ./prompt_mosq_n_mosa.template --data_path /home/user/data -o minicpm
```

then a file in `./data/agiqa-3k` named `minicpm_analysis_output.csv` will be created. which can be process later by other purpose, such as fine-tuning a MLLM.

