import argparse
import csv
import json
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import PIL
import torch
import torch.nn as nn
from PIL import Image
from peft import AutoPeftModelForCausalLM, PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

# Import models and utils from the project
from models.MiniCPM import MiniCPMIQA_xLSTMHeader, SupervisedDataset, preprocess, conversation_to_ids
from wrapper import Wrapper
from huggingface_hub import hf_hub_download

# Constants for prompt templates
PROMPT_QUALITY = "Based on your analysis, could you please rate the image's overall quality? Choose one word from the following list: [bad, poor, fair, good, excellent], where the words range from low to high quality."
PROMPT_CORRESPONDENCE = "Based on your analysis, could you please rate the image's alignment with its original prompt? Choose one word from the following list: [bad, poor, fair, good, excellent], where the words indicate the degree of alignment from low to high."
PROMPT_AUTHENTICITY = "Based on your analysis, could you please rate the image's authenticity? Choose one word from the following list: [bad, poor, fair, good, excellent], where the words indicate the degree of authenticity from low to high."

# Mapping of ratings to scores
RATING_TO_SCORE = {
    "bad": 0.0,
    "poor": 1.0,
    "fair": 2.0,
    "good": 3.0,
    "excellent": 4.0,
}

# map score to rating
def score_to_rating(score: float) -> str:
    """Convert a score to a rating string."""
    if score < 1.0:
        return "bad"
    elif score < 2.0:
        return "poor"
    elif score < 3.0:
        return "fair"
    elif score < 4.0:
        return "good"
    else:
        return "excellent"

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference for AGIQA scoring')
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to either an image file or a CSV file containing image paths and prompts')
    parser.add_argument('--prompt', type=str, 
                        help='Prompt for the image (required when input is an image file)')
    parser.add_argument('--aspect', type=str, required=True, choices=['quality', 'correspondence', 'authenticity'],
                        help='Aspect to evaluate: quality, correspondence, or authenticity')
    parser.add_argument('--output', type=str, 
                        help='Path to save output results (default is input path with _results appended)')
    parser.add_argument('--mllm_model_name', type=str, default="openbmb/MiniCPM-Llama3-V-2_5",
                        help='Base MLLM model name')
    parser.add_argument('--mllm_checkpoint', type=str, default=None,
                        help='Finetuned MLLM checkpoint path for description generation (auto-selected based on aspect if not provided)')
    parser.add_argument('--predictor_checkpoint', type=str, 
                        default="strawhat/m3-agiqa-predictor-checkpoints",
                        help='Predictor checkpoint path for score prediction')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Config file path for the predictor model (auto-selected based on aspect if not provided)')
    return parser.parse_args()

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    config['loss_func'] = 'mse'
    config['pooler'] = 'mean'
    return config

def load_mllm_model(model_name: str, checkpoint_path: str) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """Load the MLLM model for description generation."""
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, checkpoint_path, torch_dtype=torch.float16,
                                      device_map="auto", trust_remote_code=True).eval().cuda()
    return model, tokenizer

def get_predictor_model_checkpoints(checkpoint_path: str, aspect: str) -> str:
    if aspect == 'quality':
        return 'aigciqa-30k/quality/best-epoch=11-val_SRCC=0.89.ckpt'
    elif aspect == 'correspondence':
        return 'agiqa-3k/alignment/best-epoch=14-val_SRCC=0.87.ckpt'
    elif aspect == 'authenticity':
        # return 'aigciqa2023/authenticity/best-epoch=19-val_SRCC=0.83.ckpt'
        return 'aigciqa2023/authenticity/best-epoch=15-val_SRCC=0.83.ckpt'

def load_predictor_model(config: Dict, checkpoint_path: str, aspect: str, mllm_model: torch.nn.Module) -> torch.nn.Module:
    """Load the predictor model for score prediction."""
    # Modify config for predictor
    config['stage'] = 'predict'
    config['model'] = 'minicpm-xlstm'
    # config['no_load_model'] = False  # Prevent loading MLLM again in Wrapper
    
    # Create the model wrapper
    wrapper = Wrapper(config)
    
    # Reuse the already loaded MLLM model
    wrapper.model.model = mllm_model
    
    # Construct checkpoint path based on aspect
    if aspect == "correspondence":
        aspect_folder = "alignment" 
    else:
        aspect_folder = aspect
        
    # download the checkpoint from Hugging Face Hub
    
    ckpt_filename = get_predictor_model_checkpoints(checkpoint_path, aspect)
    print(f"Using checkpoint: {ckpt_filename}")

    ckpt_file = hf_hub_download(repo_id=checkpoint_path, filename=ckpt_filename)
    
    # Load the checkpoint
    checkpoint = torch.load(ckpt_file, map_location='cpu')
    wrapper.load_state_dict(checkpoint['state_dict'], strict=False)
    wrapper.eval()
    wrapper.cuda()
    
    return wrapper

def get_aspect_prompt(aspect: str) -> str:
    """Get the appropriate prompt for the aspect."""
    if aspect == "quality":
        return PROMPT_QUALITY
    elif aspect == "correspondence":
        return PROMPT_CORRESPONDENCE
    elif aspect == "authenticity":
        return PROMPT_AUTHENTICITY
    else:
        raise ValueError(f"Invalid aspect: {aspect}. Choose from quality, correspondence, authenticity.")

def prepare_data_for_predictor(image_path: str, description: str, prompt: str, aspect: str, config: Dict, mllm_model: torch.nn.Module) -> Dict:
    """Prepare data in the format expected by the predictor model."""
    
    # Create the detailed analysis prompt based on aspect and original generation prompt
    if aspect == "quality":
        analysis_prompt = f"Please closely examine this AI-generated image and provide a detailed analysis of its overall quality. The original prompt for the image was: '{prompt}'. Consider aspects like visual clarity, composition, color balance, and technical execution."
    elif aspect == "correspondence":
        analysis_prompt = f"Please closely examine this AI-generated image and provide a detailed analysis of its alignment with the original prompt. The original prompt for the image was: '{prompt}'. How well does the image match what was requested in the prompt?"
    elif aspect == "authenticity": 
        analysis_prompt = f"Please closely examine this AI-generated image and provide a detailed analysis of its authenticity. The original prompt for the image was: '{prompt}'. How closely does the image resemble real artworks or scenes? Highlight any parts of the image that appear non-real or artificial."
    else:
        raise ValueError(f"Invalid aspect: {aspect}")
    
    # Create conversation structure exactly like training data (3 conversations)
    conversation = [
        {
            "role": "user",
            "content": f"{analysis_prompt}"
        },
        {
            "role": "assistant", 
            "content": description
        },
        {
            "role": "user",
            "content": get_aspect_prompt(aspect)
        }
    ]
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    
    # Get tokenizer for preprocessing
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
    
    # Use the already loaded MLLM model for transform function
    # Extract the base model from PeftModel to access transform
    base_model = mllm_model.base_model if hasattr(mllm_model, 'base_model') else mllm_model
    
    # Get slice config from the actual model config, similar to how it's done in SupervisedDataset.load_data
    # For consistency with model.chat(), we might want to use the exact same slice config
    # that the model would use internally, or disable slicing for better consistency
    if hasattr(base_model.config, "slice_config"):
        # Use the model's built-in slice config
        slice_config = base_model.config.slice_config.to_dict()
        # Update max_slice_nums from our config
        slice_config['max_slice_nums'] = config.get('max_slice_nums', slice_config.get('max_slice_nums', 4))
    else:
        # Create slice config from model config if slice_config doesn't exist
        slice_config = {
            "patch_size": getattr(base_model.config, 'patch_size', config.get('patch_size', 14)),
            "max_slice_nums": config.get('max_slice_nums', 4),
            "scale_resolution": getattr(base_model.config, 'scale_resolution', 448)
        }
    
    # Check batch_vision config similar to SupervisedDataset.load_data
    if hasattr(base_model.config, "batch_vision_input"):
        batch_vision = base_model.config.batch_vision_input
    else:
        batch_vision = False
    
    # Preprocess the data using the same pipeline as training
    processed_data = preprocess(
        image=image,
        conversation=conversation,
        tokenizer=tokenizer,
        transform=base_model.transform,
        query_nums=config.get('query_nums', 96),
        slice_config=slice_config,
        llm_type=config.get('llm_type', 'llama3'),
        patch_size=slice_config.get('patch_size', 14),
        batch_vision=batch_vision
    )
    
    # Format the single sample data exactly like the dataset __getitem__ method
    sample_data = {
        "input_ids": processed_data["input_ids"],
        "position_ids": processed_data["position_ids"], 
        "labels": processed_data["target"],
        "attention_mask": torch.ones_like(processed_data["input_ids"], dtype=torch.bool),
        "pixel_values": processed_data["pixel_values"],
        "tgt_sizes": processed_data["tgt_sizes"],
        "image_bound": processed_data["image_bound"],
    }
    
    # Apply the same data_collator function as used in training
    from models.MiniCPM import data_collator
    batch_data = data_collator([sample_data], max_length=config.get('model_max_length', 768))
    
    # Debug: Print data shapes and config values that might affect consistency
    # print(f"Config - max_slice_nums: {config.get('max_slice_nums')}, model_max_length: {config.get('model_max_length')}")
    # print(f"Input IDs shape: {batch_data['input_ids'].shape}")
    # print(f"Pixel values length: {len(batch_data['pixel_values'][0]) if batch_data['pixel_values'] else 0}")
    
    # Move to GPU
    for key in ["input_ids", "position_ids", "labels", "attention_mask"]:
        if key in batch_data:
            batch_data[key] = batch_data[key].cuda()
    
    # Handle pixel_values - they should remain as a list but move tensors to GPU
    if "pixel_values" in batch_data:
        batch_data["pixel_values"] = [[pv.cuda() if torch.is_tensor(pv) else pv for pv in sample] for sample in batch_data["pixel_values"]]
    
    return (batch_data, [image_path]), torch.tensor([0.0])  # dummy label

def analyze_single_image(
    mllm_model: torch.nn.Module,
    mllm_tokenizer: AutoTokenizer,
    predictor_model: torch.nn.Module,
    config: Dict,
    image_path: str, 
    prompt: str,
    aspect: str
) -> Dict:
    """Analyze a single image and return results."""
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return {
            "image_path": image_path,
            "prompt": prompt,
            "aspect": aspect,
            "description": "Error: Could not open image",
            "score": 0.0,
            "rating": "bad"
        }
    
    # First: Generate description using MLLM with aspect-specific detailed prompt
    if aspect == "quality":
        analysis_prompt = f"Please closely examine this AI-generated image and provide a detailed analysis of its overall quality. The original prompt for the image was: '{prompt}'. Consider aspects like visual clarity, composition, color balance, and technical execution."
    elif aspect == "correspondence":
        analysis_prompt = f"Please closely examine this AI-generated image and provide a detailed analysis of its alignment with the original prompt. The original prompt for the image was: '{prompt}'. How well does the image match what was requested in the prompt?"
    elif aspect == "authenticity": 
        analysis_prompt = f"Please closely examine this AI-generated image and provide a detailed analysis of its authenticity. The original prompt for the image was: '{prompt}'. How closely does the image resemble real artworks or scenes? Highlight any parts of the image that appear non-real or artificial."
    else:
        raise ValueError(f"Invalid aspect: {aspect}")
    
    msgs = [{"role": "user", "content": f"{analysis_prompt}"}]
    description = mllm_model.chat(
        image=img,
        msgs=msgs,
        context=None,
        tokenizer=mllm_tokenizer,
        sampling=False
    )
    
    # Second: Generate score using predictor model
    try:
        # Prepare data for predictor with the correct conversation structure
        predictor_input = prepare_data_for_predictor(image_path, description, prompt, aspect, config, mllm_model)
        
        # Get prediction
        with torch.no_grad():
            # Set model to eval mode and ensure deterministic behavior
            predictor_model.eval()
            torch.manual_seed(42)  # Set seed for reproducibility
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
            
            outputs = predictor_model(predictor_input)
            score = outputs.item()
            # print(f"Raw score: {score}")
            # Clamp score to [0, 5] range
            score = max(0.0, min(5.0, score))
            
    except Exception as e:
        print(f"Error in score prediction for {image_path}: {e}")
        score = 0.0
    
    return {
        "image_path": image_path,
        "prompt": prompt,
        "aspect": aspect,
        "description": description,
        "score": score,
        "rating": score_to_rating(score)
    }

def process_single_image(args):
    """Process a single image file."""
    if not args.prompt:
        raise ValueError("Prompt must be provided when processing a single image")
    
    # Get the appropriate config path and MLLM checkpoint based on aspect
    config_path = get_config_path_for_aspect(args.aspect, args.config_path)
    mllm_checkpoint = get_mllm_checkpoint_for_aspect(args.aspect, args.mllm_checkpoint)
    print(f"Using config file: {config_path}")
    print(f"Using MLLM checkpoint: {mllm_checkpoint}")
    
    # Load config
    config = load_config(config_path)
    config = normalize_config_for_consistency(config, args.aspect)
    
    # Load models
    mllm_model, mllm_tokenizer = load_mllm_model(args.mllm_model_name, mllm_checkpoint)
    predictor_model = load_predictor_model(config, args.predictor_checkpoint, args.aspect, mllm_model)
    
    # Analyze image
    results = analyze_single_image(
        mllm_model, mllm_tokenizer, predictor_model, config,
        args.input, args.prompt, args.aspect
    )
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = str(input_path.parent / f"{input_path.stem}_{args.aspect}_results.json")
    
    # Save results to JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_path}")
    print("-" * 20)
    print(f"Description: {results['description']}")
    print(f"Score: {results['score']:.2f}, Rating: {results['rating']}")
    return results

def process_csv_file(args):
    """Process a CSV file containing multiple images."""
    # Get the appropriate config path and MLLM checkpoint based on aspect
    config_path = get_config_path_for_aspect(args.aspect, args.config_path)
    mllm_checkpoint = get_mllm_checkpoint_for_aspect(args.aspect, args.mllm_checkpoint)
    print(f"Using config file: {config_path}")
    print(f"Using MLLM checkpoint: {mllm_checkpoint}")
    
    # Load config
    config = load_config(config_path)
    config = normalize_config_for_consistency(config, args.aspect)
    
    # Load models
    mllm_model, mllm_tokenizer = load_mllm_model(args.mllm_model_name, mllm_checkpoint)
    predictor_model = load_predictor_model(config, args.predictor_checkpoint, args.aspect, mllm_model)
    
    # Read the CSV file
    image_data = []
    with open(args.input, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Read header
        
        # Determine column indices
        try:
            image_idx = header.index('image_path')
            prompt_idx = header.index('prompt')
        except ValueError:
            print(f"Error: CSV must contain 'image_path' and 'prompt' columns. Found: {header}")
            sys.exit(1)
            
        for row in reader:
            image_data.append({
                'image_path': row[image_idx],
                'prompt': row[prompt_idx]
            })
    
    # Process each image
    results = []
    for item in tqdm(image_data, desc="Processing images"):
        result = analyze_single_image(
            mllm_model, mllm_tokenizer, predictor_model, config,
            item['image_path'], item['prompt'], args.aspect
        )
        results.append(result)
    
    # Determine output paths
    if args.output:
        json_output_path = args.output
        csv_output_path = args.output.replace('.json', '.csv') if args.output.endswith('.json') else args.output + '.csv'
    else:
        input_path = Path(args.input)
        json_output_path = str(input_path.parent / f"{input_path.stem}_{args.aspect}_results.json")
        csv_output_path = str(input_path.parent / f"{input_path.stem}_{args.aspect}_results.csv")
    
    # Save results to JSON
    with open(json_output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save results to CSV
    with open(csv_output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'prompt', 'aspect', 'description', 'score', 'rating'])
        for result in results:
            writer.writerow([
                result['image_path'], 
                result['prompt'], 
                result['aspect'], 
                result['description'],
                result['score'],
                score_to_rating(result['score'])
            ])
    
    print(f"Results saved to {json_output_path} and {csv_output_path}")
    print(f"Average score: {np.mean([r['score'] for r in results]):.2f}")
    return results

def normalize_config_for_consistency(config: Dict, aspect: str) -> Dict:
    """Normalize config parameters to improve consistency across aspects."""
    # You can uncomment these lines to force consistent preprocessing
    # config['max_slice_nums'] = 4  # Use same slice nums for all aspects
    # config['model_max_length'] = 768  # Use same max length for all aspects
    
    print(f"Using config for {aspect}: max_slice_nums={config.get('max_slice_nums')}, model_max_length={config.get('model_max_length')}")
    return config

def get_mllm_checkpoint_for_aspect(aspect: str, custom_checkpoint: str = None) -> str:
    """Get the appropriate MLLM checkpoint based on the aspect."""
    if custom_checkpoint:
        return custom_checkpoint
    
    if aspect == "quality":
        return "strawhat/minicpm2.5-aigciqa-20k-ft"
    elif aspect == "correspondence":
        return "strawhat/minicpm2.5-agiqa-3k-corr-ft"
    elif aspect == "authenticity":
        return "strawhat/minicpm2.5-aigciqa2023-auth-ft"
    else:
        raise ValueError(f"Invalid aspect: {aspect}")

def get_config_path_for_aspect(aspect: str, custom_config_path: str = None) -> str:
    """Get the appropriate config file path based on the aspect."""
    if custom_config_path:
        return custom_config_path
    
    if aspect == "quality":
        return "cfg/minicpm-xlstm-aigciqa-30k.yaml"
    elif aspect == "correspondence":
        return "cfg/minicpm-xlstm-agiqa-3k.yaml"
    elif aspect == "authenticity":
        return "cfg/minicpm-xlstm-aigciqa2023.yaml"
    else:
        raise ValueError(f"Invalid aspect: {aspect}")

def main():
    args = parse_args()
    
    # Determine input type (single image or CSV file)
    if args.input.endswith('.csv'):
        process_csv_file(args)
    else:
        process_single_image(args)

if __name__ == "__main__":
    main()
