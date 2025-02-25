import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

WITHOUT_ID = False
# w/o ft, w/ ID
INPUT_TEST_RESULT_FILE = "./data/agiqa-3k/directly_test_cls_minicpm_quality_multiround_nonfinetuned.jsonl"

# w/ self ft, w/ ID
INPUT_TEST_RESULT_FILE = "./data/agiqa-3k/minicpm_agiqa-3k-quality_lora_07_27_2024_lr2e-6_3400_res.jsonl"

# w/ ft, w/ ID
INPUT_TEST_RESULT_FILE = "./data/agiqa-3k/gemini_agiqa-3k-quality_lora_08_12_2024_lr2e-6_3600_res.jsonl"

# # w/ ft, w/o ID
# INPUT_TEST_RESULT_FILE = "./data/agiqa-3k/directly_test_cls_minicpm_quality_res.jsonl"
# WITHOUT_ID = True

def rough_accuracy(y_true, y_pred):
    # if the difference between the true and predicted result is less than or equal 1, then it is correct
    correct = 0
    total = 0
    for true_result, pred_result in zip(y_true, y_pred):
        if abs(true_result - pred_result) <= 1:
            correct += 1
        total += 1

    return correct / total

def get_result_from_str(result: str):
    if 'bad' in result:
        return 0
    elif 'poor' in result:
        return 1
    elif 'fair' in result:
        return 2
    elif 'good' in result:
        return 3
    elif 'excellent' in result:
        return 4
    else:
        raise ValueError(f"Unknown result: {result}")

def get_predicted_and_groundtruth(entries: list):
    y_true = []
    y_pred = []

    for entry in tqdm(entries):
        if WITHOUT_ID:
            true_result = get_result_from_str(entry['conversations'][1]['content'])
            pred_result = get_result_from_str(entry['response'])
        else:
            true_result = get_result_from_str(entry['conversations'][3]['content'])
            pred_result = get_result_from_str(entry['response_1'])

        y_true.append(true_result)
        y_pred.append(pred_result)

    return y_true, y_pred

def calculate_metrics(y_true, y_pred):
    rough_accuracy_score = rough_accuracy(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return rough_accuracy_score, accuracy, precision, recall, f1

def main():
    with open(INPUT_TEST_RESULT_FILE, 'r') as f:
        entries = json.load(f)

    y_true, y_pred = get_predicted_and_groundtruth(entries)
    rough_accuracy_score, accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred)

    print(f"Rough accuracy: {rough_accuracy_score}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

if __name__ == "__main__":
    main()