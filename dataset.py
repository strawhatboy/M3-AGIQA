import json
import torch
import torch.utils.data as data
from torchvision.transforms import v2
from torchvision.datasets import VisionDataset
from models.utils import map_y_to_quality
from utils import set_seed
from pathlib import Path
import pandas as pd
from PIL import Image
from transformers import AutoTokenizer

# 7:1:2 for all datasets

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def to_image_transform():
    return v2.Compose([
        v2.ToImage(),
    ])

def load_and_split_data(args):
    data_path = Path(args['data_dir']) / args['dataset']
    descriptor = pd.read_csv(data_path / 'data.csv')
    
    if args.get('train_data', None) is not None:
        print('using pre-split data')
        train_data = pd.read_csv(data_path / args['train_data'])
        val_data = pd.read_csv(data_path / args['val_data'])
        
        # if test_data file exists, use it, otherwise use val_data as test_data
        if args.get('test_data', None) is not None and (data_path / args['test_data']).is_file():
            print('using pre-split test data')
            test_data = pd.read_csv(data_path / args['test_data'])
        else:
            test_data = val_data.copy()
            
        # convert to tuples
        train_data = [(row['prompt'] if 'prompt' in row else None, 
                       Path(args['data_dir']) / row['dataset'] / 'images' / row['name'] if 'dataset' in row else data_path / 'images' / row['name'], 
                       Path(args['data_dir']) / row['dataset'] / 'images' / row['reference'] if 'dataset' in row and 'reference' in row else data_path / 'images' / row['reference'] if 'reference' in row else None, 
                       row[args['label_name']]) for idx, row in train_data.iterrows()]
        val_data = [(row['prompt'] if 'prompt' in row else None, 
                     Path(args['data_dir']) / row['dataset'] / 'images' / row['name'] if 'dataset' in row else data_path / 'images' / row['name'], 
                     Path(args['data_dir']) / row['dataset'] / 'images' / row['reference'] if 'dataset' in row and 'reference' in row else data_path / 'images' / row['reference'] if 'reference' in row else None, 
                     row[args['label_name']]) for idx, row in val_data.iterrows()]
        test_data = [(row['prompt'] if 'prompt' in row else None, 
                      Path(args['data_dir']) / row['dataset'] / 'images' / row['name'] if 'dataset' in row else data_path / 'images' / row['name'], 
                      Path(args['data_dir']) / row['dataset'] / 'images' / row['reference'] if 'dataset' in row and 'reference' in row else data_path / 'images' / row['reference'] if 'reference' in row else None, 
                      row[args['label_name']]) for idx, row in test_data.iterrows()]
    else:

        all_data = []
        for idx, row in descriptor.iterrows():
            # Use dataset-specific path if 'dataset' column exists, otherwise use original path
            if 'dataset' in row:
                img_path = Path(args['data_dir']) / row['dataset'] / 'images' / row['name']
                ref_img_path = Path(args['data_dir']) / row['dataset'] / 'images' / row['reference'] if 'reference' in row else None
            else:
                img_path = data_path / 'images' / row['name']
                ref_img_path = data_path / 'images' / row['reference'] if 'reference' in row else None
            
            label = row[args['label_name']]
            
            all_data.append((row['prompt'] if 'prompt' in row else None, img_path, ref_img_path, label))

        # split to train val test
        set_seed(args['seed'])
        train_size = int(args['train_ratio'] * len(all_data))
        val_size = int(args['val_ratio'] * len(all_data))
        test_size = len(all_data) - train_size - val_size
        train_data, val_data, test_data = data.random_split(all_data, [train_size, val_size, test_size])
    return train_data, val_data, test_data

def get_patches(image: torch.Tensor, kernel_size, stride):
    # image: C x H x W
    C, H, W = image.shape
    patches = image.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
    patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, C, kernel_size, kernel_size)
    # patches: N x C x kernel_size x kernel_size
    return patches

def get_patches_from_batch_img(batch_img: torch.Tensor, kernel_size, stride):
    # batch: B x C x H x W
    B, C, H, W = batch_img.shape
    patches = batch_img.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C, kernel_size, kernel_size)
    # patches: B x N x C x kernel_size x kernel_size
    return patches

class Dataset(VisionDataset):
    def __init__(self, args, data, transforms=None):
        super().__init__(transforms=transforms)
        self.args = args
        self.data = data
        self.transforms = transforms
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        prompt, img, ref_img, label = self.data[idx]
        img = Image.open(img)
        img = img.convert("RGB")
        img = self.transforms(img)
        if ref_img is not None:
            ref_img = Image.open(ref_img)
            ref_img = ref_img.convert("RGB")
            ref_img = self.transforms(ref_img)
        return (img, ref_img if ref_img is not None else torch.Tensor()), torch.tensor(label, dtype=torch.float32)

    @staticmethod
    def load_data(args):
        train_data, val_data, test_data, data_transforms = load_and_split_data(args)

        # return train val test, use transforms for train data
        return Dataset(args, train_data, data_transforms), Dataset(args, val_data), Dataset(args, test_data)

class AGIDataset(VisionDataset):
    # can be used for T2I/I2I IQA tasks
    # prompt, img, ref_img, labels in self.data
    # prompt: str, img: str, ref_img: str, labels: List[float]

    # when getitem
    # prompt -> tokenizer -> token_ids
    # prompt -> tokenizer -> attention_mask
    # img -> load image -> transforms -> tensor
    # ref_img -> load image -> transforms -> tensor
    # labels -> tensor

    def __init__(self, args, data, transforms=None):
        # tokenizer can be used to tokenize the prompt
        super().__init__()
        self.transforms = transforms
        if transforms is None:
            self.transforms = to_image_transform()
        
        self.args = args
        self.data = data
        if args['use_tokenizer']:
            self.tokenizer = AutoTokenizer.from_pretrained(args['language_model'], trust_remote_code=True)

    def __len__(self):
        return len(self.data)

    def save_data(self, path):
        # save the data to a csv file
        if len(self.data) == 0:
            return
        
        dict_data = [{'prompt': prompt, 'img': img, 'ref_img': ref_img, 'label': label} for prompt, img, ref_img, label in self.data]
        df = pd.DataFrame(dict_data)
        df.to_csv(path, index=False)
    
    def __getitem__(self, idx):
        prompt, img_path, ref_img, labels = self.data[idx]
        img = Image.open(img_path)
        img = img.convert("RGB")
        # if self.args['enable_transforms']:
        img = self.transforms(img)
        if ref_img is not None:
            ref_img = Image.open(ref_img)
            ref_img = ref_img.convert("RGB")
            # if self.args['enable_transforms']:
            ref_img = self.transforms(ref_img)
        if prompt is None:
            prompt = ""
        if self.args['use_tokenizer']:
            output = self.tokenizer(prompt, return_tensors='pt', padding='max_length', max_length=self.args['max_length'], truncation=True)
            input_ids, attention_mask = output['input_ids'].squeeze(), output['attention_mask'].squeeze()
        else:
            input_ids, attention_mask = torch.tensor([]), torch.tensor([])
        return (str(img_path), prompt, input_ids, attention_mask, img, ref_img if ref_img is not None else torch.Tensor()), torch.tensor(labels, dtype=torch.float32) * self.args['label_scale']

    @staticmethod
    def load_data(args, model):
        train_data, val_data, test_data = load_and_split_data(args)
        return AGIDataset(args, train_data, model.trans), AGIDataset(args, val_data, model.trans_test), AGIDataset(args, test_data, model.trans_test)
    
class AIGCIQA20kDataset(AGIDataset):
    @staticmethod
    def load_data(args, model):
        
        data_path = Path(args['data_dir']) / args['dataset']
        descriptor = pd.read_csv(data_path / 'data.csv')
        train_descriptor = pd.read_csv(data_path / 'info_train.csv', index_col='name')
        val_descriptor = pd.read_csv(data_path / 'info_val.csv', index_col='name')
        test_descriptor = pd.read_csv(data_path / 'info_test.csv', index_col='name')

        train_data, val_data, test_data = [], [], []
        for idx, row in descriptor.iterrows():
            img_path = data_path / 'images' / row['name']
            ref_img_path = data_path/ 'images'/ row['reference'] if 'reference' in row else None
            label = row[args['label_name']]

            if row['name'] in train_descriptor.index:
                train_data.append((row['prompt'] if 'prompt' in row else None, img_path, ref_img_path, label))
            elif row['name'] in val_descriptor.index:
                val_data.append((row['prompt'] if 'prompt' in row else None, img_path, ref_img_path, label))
            elif row['name'] in test_descriptor.index:
                test_data.append((row['prompt'] if 'prompt' in row else None, img_path, ref_img_path, label))
            else:
                raise ValueError(f"Image {row['name']} not found in train/val/test split")


        return AIGCIQA20kDataset(args, train_data, model.trans), AIGCIQA20kDataset(args, val_data, model.trans_test), AIGCIQA20kDataset(args, test_data, model.trans_test)


class PtEngineDataset(data.Dataset):
    def __init__(self, args, raw_data):
        super().__init__()
        self.args = args
        
        # load data
        data_path = Path(args['data_dir']) / args['dataset']
        descriptor = pd.read_csv(data_path / 'data.csv')

        all_data = {}
        for idx, row in descriptor.iterrows():
            img_path = data_path / 'images' / row['name']
            ref_img_path = data_path/ 'images'/ row['reference'] if 'reference' in row else None
            label = row[args['label_name']]
            
            all_data[str(img_path.resolve())] = (row['prompt'] if 'prompt' in row else None, img_path, ref_img_path, label)

        self.agi_data = all_data
        self.raw_data = raw_data
        
    def __len__(self):
        return len(self.raw_data)
        
    def __getitem__(self, idx):
        # construct the InferRequest and RequestConfig
        request_data = self.raw_data[idx]
        request_data['messages'] = request_data['messages'][0:3]    # exclude the final result
        # need to check if the 1st msessage has <image> token
        # if not, then we need to add it
        
        img_path = self.raw_data[idx]['images'][0]
        if img_path in self.agi_data:
            lbl = self.agi_data[img_path][3]
        else:
            lbl = self.agi_data[str(Path(img_path).resolve())][3]
        
        return request_data, img_path, torch.tensor(lbl, dtype=torch.float32) * self.args['label_scale']
    
    @staticmethod
    def load_data(args, model):
        # data_path
        # 
        train_raw_data = json.load(open(args['data_path'], "r"))
        # train_raw_data = random.sample(train_raw_data, len(train_raw_data) // 100)  # debug
        # val_raw_data = json.load(open(args['eval_data_path'], "r"))

        # use eval_data_res_path, same structure with val_raw_data
        val_raw_data = json.load(open(args['eval_data_path'], "r"))
        
        # if args['dataset'] == 'aigciqa-30k':
        #     # aigciqa-30k has test set
        #     test_raw_data = json.load(open(args['test_data_path'], "r"))
        #     for i in range(len(test_raw_data)):
        #         test_raw_data[i]['messages'][1]['content'] = test_raw_data[i]['response_0']
        #         test_raw_data[i]['messages'][3]['content'] = test_raw_data[i]['response_1']
        #         del test_raw_data[i]['response_0']
        #         del test_raw_data[i]['response_1']
        # else:
        test_raw_data = json.load(open(args['test_data_path'], "r"))

        for i in range(len(test_raw_data)):
            test_raw_data[i]['messages'][1]['content'] = test_raw_data[i]['response_0']
            test_raw_data[i]['messages'][3]['content'] = test_raw_data[i]['response_1']
            del test_raw_data[i]['response_0']
            del test_raw_data[i]['response_1']
            
        return PtEngineDataset(args, train_raw_data), PtEngineDataset(args, val_raw_data), PtEngineDataset(args, test_raw_data)
        
