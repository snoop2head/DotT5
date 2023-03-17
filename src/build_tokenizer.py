import os
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

if __name__ == "__main__":
    dset = load_dataset(
        "PoolC/AIHUB-parallel-ko-braille-short64", use_auth_token=True
    )  # or replace with 128 for more data

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

    # BPE tokenizer for braille-korean individual letters
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # convert str iterator to list
    unique_strings = set()
    for i in dset["train"]:
        unique_strings.add(i["braille"])
        unique_strings.add(i["ko"])
    unique_strings = list(unique_strings)

    # train tokenizer
    trainer = BpeTrainer(special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(unique_strings, trainer)

    # iterate over vocab_map and print all tokens with length > 1
    vocab_map = tokenizer.get_vocab()
    for key, value in vocab_map.items():
        if len(key) > 1:
            print(key)

    # save and repack with the class PreTrainedTokenizerFast
    tokenizer.save("./korean-braille-BPE.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="./korean-braille-BPE.json")

    # check tokenizer output
    korean_output = tokenizer("세상은 돌아가는 연필깎이")
    braille_output = tokenizer("⠠⠝⠇⠶⠵⠀⠊⠥⠂⠣⠫⠉⠵⠀⠡⠙⠕⠂⠠⠫⠁⠁⠕")
    print(tokenizer.convert_ids_to_tokens(korean_output.input_ids))
    print(tokenizer.convert_ids_to_tokens(braille_output.input_ids))

    tokenizer.add_special_tokens(
        {
            "unk_token": "[UNK]",
            "bos_token": "[CLS]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "eos_token": "[SEP]",
            "pad_token": "[PAD]",
            "mask_token": "[MASK]",
        }
    )

    tokenizer.push_to_hub("snoop2head/KoBrailleT5-small-v1")
