# input data directory, with data.csv and images folder
# load prompt
# call analyzer with prompt and image
# has patience for error

#  python analyzer.py --dataset agiqa-3k --provider minicpmv2.5 --gpu 2 --prompt ./prompt_mosq_n_mosa.template --data_path /home/user/data

import asyncio
from argparse import ArgumentParser
from loguru import logger
from prettytable import PrettyTable
import pandas as pd
import random
from pathlib import Path
import aiofiles
from tqdm import tqdm
from mplug_ow2_analyzer import mPLUGOwl2Analyzer
from gemini_image_analyzer import GeminiImageAnalyzer
from minicpmv2_5_analyzer import MinicpmV2_5ImageAnalyzer
from PIL import Image

def parse_args(print_args=True):
    p = ArgumentParser()
    p.add_argument('-d', '--dataset', type=str, default='agiqa-1k', help='dataset name, in ./data directory')
    p.add_argument('-dp', '--data_path', type=str, default='../data', help='data path contains the datasets')
    # p.add_argument('-p', '--patience', type=int, default=3, help='patience for error')
    p.add_argument('-p', '--prompt', type=str, default='./prompt.template', help='text file contains the prompt template')
    p.add_argument('-pr', '--provider', type=str, default='google', help='provider name, google, minicpmv2.5, etc')
    p.add_argument('-es', '--epoch_size', type=int, default=20, help='record processed in one epoch')
    p.add_argument('-o', '--output_postfix', type=str, default='_analysis_output.csv', help='output csv file, will be in dataset folder')
    p.add_argument('-s', '--seed', type=int, default=42, help='random seed')
    p.add_argument('-g', '--gpu', type=str, default='0', help='gpu id')

    args = vars(p.parse_args())
    if print_args:
        tab = PrettyTable(['Argument', 'Value'], align='l')
        for key, value in args.items():
            tab.add_row([key, value])
        print(tab)
    return args

def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

async def load_prompt(prompt_path):
    async with aiofiles.open(prompt_path, 'r') as f:
        prompt = await f.read()
    return prompt

def analyzer_selector(args):
    provider = args['provider']
    if provider == 'google':
        return GeminiImageAnalyzer(args)
    elif provider == 'minicpmv2.5':
        return MinicpmV2_5ImageAnalyzer(args)
    elif provider == 'mplug_owl2':
        return mPLUGOwl2Analyzer(args)
    else:
        raise ValueError(f'Provider {provider} not supported')


async def main():
    args = parse_args()
    set_seed(args['seed'])
    prompt_template = await load_prompt(args['prompt'])
    path_dataset = Path(args['data_path']) / args['dataset']
    output_file = path_dataset / f"{args['provider']}_{args['output_postfix']}"
    descriptor = pd.read_csv(path_dataset / 'data.csv')
    success_rate = 0
    if not output_file.exists():
        # create the file with 'success' set to False
        df_need_processing = descriptor.copy()
        df_processing = df_need_processing
        df_need_processing['success'] = False
        # keep only name, prompt, and success columns
        # df_need_processing = df_need_processing[['name', 'prompt', 'success']]
        # save
        df_need_processing.to_csv(output_file, index=False)
    else:
        # load the file and check the success rate
        df_processing = pd.read_csv(output_file)
        success_rate = len(df_processing[df_processing['success'] == True]) / len(descriptor)
        logger.info(f'success rate: {success_rate}')
        df_need_processing = df_processing[df_processing['success'] == False]

    analyzer = analyzer_selector(args)
    n_epoch = 0
    while success_rate < 1.0:
        n_total = 0
        n_success = 0
        df_need_processing = df_need_processing[df_need_processing['success'] == False] # filter out the ones that are already processed
        the_list = list(df_need_processing.iterrows())
        random.shuffle(the_list)
        bar = tqdm(list(enumerate(the_list)))    # shuffle to avoid the same failed processed image
        for idx, (index, row) in bar:
            img_path = path_dataset / 'images' / row['name']
            logger.info(f'Processing {img_path}...')
            img = Image.open(img_path)
            format_dict = {}
            for key in row.keys():
                if key.startswith('mos'):
                    format_dict[key] = row[key]
            prompt = prompt_template.format(**format_dict, prompt=row['prompt'], min=0, max=5)
            res = analyzer.analyze(prompt, img)
            print(res)
            success = res != 'error'
            if success:
                n_success += 1
            n_total += 1
            # update the output file
            df_processing.loc[index, 'success'] = success
            df_need_processing.loc[index, 'success'] = success
            df_processing.loc[index, 'result'] = res
            success_rate = n_success / n_total
            bar.set_postfix({'success_rate': success_rate})
            if n_total % args['epoch_size'] == 0:
                success_rate = len(df_processing[df_processing['success'] == True]) / len(descriptor)
                logger.info(f'epoch {n_epoch}: {success_rate}')
                df_processing.to_csv(output_file, index=False)
                n_epoch += 1
                break


if __name__ == '__main__':
    asyncio.run(main())


