# DotT5

**Pretrained T5 Seq2Seq Transformer Model for Braille to Korean Transfer Task**

- Build BPE Tokenizer based on Braille Unicode Characters & Korean Unicode Characters
- Trained T5 Seq2Seq Transformer Model for Braille to Korean Transfer Task

### Models

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# get tokenizer, config and pretrained weights
tokenizer = AutoTokenizer.from_pretrained("snoop2head/KoBrailleT5-small-v1")
model = AutoModelForSeq2SeqLM.from_pretrained("snoop2head/KoBrailleT5-small-v1")

# translate braille to korean
braille_pattern = "⠍⠗⠠⠪⠋⠕⠀⠘⠪⠐⠗⠒⠊⠕⠐⠀⠘⠮⠐⠍⠨⠟⠀⠚⠣⠕⠚⠕⠂</s>"
# translate braille to korean
input_ids = tokenizer(braille_pattern, return_tensors="pt").input_ids
outputs = model.generate(input_ids)
korean_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(korean_output) # ['위 스 키 브 랜 디 를 루 진 하 이 힐']

```

### Repository Structure

```
DotT5
ㄴ src
  ㄴ config.yaml - train / inference configs
  ㄴ build_tokenizer.py - BPE tokenizer training
  ㄴ utils.py - utils for training & inferencing
  ㄴ main.py - script for training
  ㄴ inference.py - script for inferencing
```

### Installation

```
pip install -r requirements.txt  # install
```

### References

```
@misc{raffel2020exploring,
    title   = {Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
    author  = {Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu},
    year    = {2020},
    eprint  = {1910.10683},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
