import torch
import os

def find_save_dir(parent_dir, model_name):
    counter = 0
    save_dir = f'../{parent_dir}/{model_name}_{counter}'
    while os.path.exists(save_dir):
        counter += 1
        save_dir = f'../{parent_dir}/{model_name}_{counter}'
    os.mkdir(save_dir)
    print(f'save_dir is {save_dir}')
    return save_dir

def save_model_with_result(model, path, ner_f1, args):
    model_path = os.path.join(path, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f'save model at {model_path}')
    with open(os.path.join(path, 'valid_result.txt'), 'w') as f:
        f.write(f'ner:{ner_f1}\n')


def save_test_result(path, ner_f1, args):
    with open(os.path.join(path, 'test_result.txt'), 'w') as f:
        f.write(f'ner:{ner_f1}\n')


