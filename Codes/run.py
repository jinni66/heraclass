import argparse
import json
import os
from tqdm import tqdm
import torch
import time

from models.traffic_model import TrafficModel, TrafficEncoder, TrafficDecoder
from preprocess.data_loader import traffic_data_loader
from preprocess.traffic_tokenizer import TrafficTokenizer


# load model from cache
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'


def _acquire_device(use_gpu, use_multi_gpu, devices, gpu):
    if use_gpu and use_multi_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu) if not use_multi_gpu else devices
        device = torch.device('cuda:{}'.format(gpu))
        print('Use GPU: {}'.format(device))
    elif use_gpu:
        device = torch.device('cuda:{}'.format(gpu))
        print('Use GPU: {}'.format(device))
    else:
        device = torch.device('cpu')
        print('Use CPU')
    return device


def main():
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='Data file path or data dir path.')
    parser.add_argument('--config_path', default='./config', help='Config dir path.')
    parser.add_argument('--use_gpu', default=True, help='Use gpu')
    parser.add_argument('--use_multi_gpu', default=False, help='Use multiple gpu')
    parser.add_argument('--devices', default='1,2,3,4', help='GPU devices')
    parser.add_argument('--gpu', default='0', help='GPU ')
    parser.add_argument('--checkpoint', default='', help='checkpoint path')
    parser.add_argument('--res_file', default='res')
    parser.add_argument('--traffic_llm_task', default='EVD')

    args = parser.parse_args()

    with open(os.path.join(args.config_path, 'model_config.json')) as f_cfg:
        model_config = json.load(f_cfg)

    with open(os.path.join(args.config_path, 'dataset_config.json')) as f_cfg:
        dataset_config = json.load(f_cfg)

    device = _acquire_device(args.use_gpu, args.use_multi_gpu, args.devices, args.gpu)
    model_config['traffic_llm_config']['task'] = args.traffic_llm_task

    begin_time = time.time()

    model = TrafficModel(**model_config, device=device)
    if args.checkpoint:
        print(f'load model from {args.checkpoint}')
        model.load_state_dict(torch.load(args.checkpoint))
    data_loader = traffic_data_loader(args.data_path, **dataset_config)
    tokenizer = TrafficTokenizer(device=device, **model_config)

    steps = len(data_loader)
    correct = 0
    total = 0
    model.eval()

    with open(f'{args.res_file}.txt', 'w') as file:
        pass

    f = open(f'{args.res_file}.txt', 'a')
    for step, items in tqdm(enumerate(data_loader), total=steps, desc=f""):
        label = items.pop('labels')
        if len(set(label)) > 1:
            continue

        inputs = tokenizer(**items)
        output = model.generate(**inputs)
        generated_texts = tokenizer.decoder_tokenizer.batch_decode(output.tolist(), skip_special_tokens=True)
        generated_text = generated_texts[0].split(' ')[0]
        label = label[0]
        total += 1
        if generated_text == label:
            correct += 1

        print(f'generated_texts: {generated_text}, label: {label}')
        json.dump({'generated_texts': generated_texts, 'generated_text': generated_text, 'label': label}, f)
        f.write('\n')
    f.close()
    end_time = time.time()
    print(f'total: {total}, correct: {correct}, acc: {correct/total}')
    print(f'Test cost: {end_time - begin_time}')

    # test
    #_, first_item = next(enumerate(data_loader))
    #del first_item['labels']
    #inputs = tokenizer(**first_item)
    # print(inputs)
    # output = model(**inputs)
    # output = model.generate()
    # print(output)
    #model.eval()
    #output = model.generate(**inputs)
    #generated_texts = tokenizer.decoder_tokenizer.batch_decode(output.tolist(), skip_special_tokens=True)
    #print(generated_texts)


if __name__ == '__main__':
    main()
