from transformers import XLMRobertaTokenizer
import json
import os
from tqdm import tqdm

if __name__ == "__main__":
    train_file = "/root/paddlejob/workspace/env_run/liuqingyi01/data/eval_data/wmt18/tr-en/en-tr_tgt-en.json"
    output_file = "./en-tr_tgt-en_xlmr-ids.json"
    model_name_or_path = "/root/paddlejob/workspace/env_run/liuqingyi01/data/model/xlm_roberta_base/"
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name_or_path)
    if not os.path.exists(output_file):
        with open(output_file, "w", encoding='utf-8') as w:
            with open(train_file, "r", encoding='utf-8') as f:
                for line in tqdm(f.readlines()):
                    line = json.loads(line)
                    sample_id = line['sample_id']
                    src_text = line['src']
                    tgt_text = line['tgt']
                    input_text = src_text + ' ' + tgt_text
                    tokenized_ids = tokenizer.encode(input_text)
                    # print(input_text)
                    # print(tokenized_text)
                    # output_str = json.dumps({'sample_id': sample_id, 'input_ids': tokenized_ids}, ensure_ascii=False)
                    w.writelines(json.dumps({'id': sample_id, 'input_ids': tokenized_ids}, ensure_ascii=False) + '\n')
    else:
        print(f"File: {output_file} exists")
    
    test_file = '/root/paddlejob/workspace/env_run/liuqingyi01/data/eval_data/wmt18/tr-en/dev/newsdev2016-tren-tren.json'
    test_output_file = './newsdev2016-tren-tren_xlmr-ids.json'
    if not os.path.exists(test_output_file):
        with open(test_output_file, "w", encoding='utf-8') as w:
            with open(test_file, "r", encoding='utf-8') as f:
                for line in tqdm(f.readlines()):
                    line = json.loads(line)
                    sample_id = line['id']
                    src_text = line['src']
                    tgt_text = line['tgt']
                    input_text = src_text + ' ' + tgt_text
                    tokenized_ids = tokenizer.encode(input_text)
                    # print(input_text)
                    # print(tokenized_text)
                    # output_str = json.dumps({'id': sample_id, 'input_ids': tokenized_ids}, ensure_ascii=False)
                    w.writelines(json.dumps({'id': sample_id, 'input_ids': tokenized_ids}, ensure_ascii=False) + '\n')
    else:
        print(f"File: {test_output_file} exists")
