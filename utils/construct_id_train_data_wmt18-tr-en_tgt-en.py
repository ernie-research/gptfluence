'''
This script is used to construct the sample id data for wmt18-tr-en_tgt-en.
'''
from torch.utils.data import Dataset
import codecs
import xml.etree.cElementTree as ElementTree
import os
import json

def _load_data(path):
        """Loads data from a TMX file."""
        dataset = []
        generator = _parse_tmx(path)
        for item in generator:
            line_id, texts = item
            # 将line_id映射到为0, 1, 2, ...的sample_id
            line_id = line_id // 5 - 1
            src_text = texts["tr"]
            tgt_text = texts["en"]

            dataset.append({
                'sample_id': line_id,
                'src': src_text,
                'tgt': tgt_text,
            })
        return dataset

# https://huggingface.co/datasets/wmt18/blob/main/wmt_utils.py
def _parse_tmx(path):
    """Generates examples from TMX file."""

    def _get_tuv_lang(tuv):
        for k, v in tuv.items():
            if k.endswith("}lang"):
                return v
        raise AssertionError("Language not found in `tuv` attributes.")

    def _get_tuv_seg(tuv):
        segs = tuv.findall("seg")
        assert len(segs) == 1, "Invalid number of segments: %d" % len(segs)
        return segs[0].text

    with open(path, "rb") as f:
        # Workaround due to: https://github.com/tensorflow/tensorflow/issues/33563
        utf_f = codecs.getreader("utf-8")(f)
        for line_id, (_, elem) in enumerate(ElementTree.iterparse(utf_f)):
            if elem.tag == "tu":
                yield line_id, {_get_tuv_lang(tuv): _get_tuv_seg(tuv) for tuv in elem.iterfind("tuv")}
                elem.clear()

if __name__ == "__main__":
    wmt18_tr_en_train_data_path = "/root/paddlejob/workspace/env_run/liuqingyi01/data/eval_data/wmt18/tr-en/en-tr.tmx"
    output_file = "/root/paddlejob/workspace/env_run/liuqingyi01/data/eval_data/wmt18/tr-en/en-tr_tgt-en.json"
    dataset = _load_data(wmt18_tr_en_train_data_path)
    print(len(dataset))
    print(dataset[0])
    if os.path.exists(output_file):
        print(f"output file {output_file} exists")
        exit(0)
    
    with open(output_file, "w") as f:
        for data in dataset:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    print("finished")