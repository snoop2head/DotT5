from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, AutoModelForSeq2SeqLM

DEVICE = torch.device("cuda:0")
tokenizer = PreTrainedTokenizerFast.from_pretrained("snoop2head/KoBrailleT5-small-v1")


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(
            DEVICE
        )
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == tokenizer.eos_token_id:
            break
    return ys


def inference(transformer, src_input_id):
    # find the index where first pad_token_id appears in src_input_id
    num_tokens = src_input_id.shape[0]
    pad_token_id = tokenizer.pad_token_id

    # get index of the last True
    src_mask = (src_input_id != pad_token_id).squeeze()
    num_tokens_without_pad = src_mask.sum().item() - 1

    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    transformer.eval()
    tgt_tokens = greedy_decode(
        transformer,
        src_input_id,
        src_mask,
        max_len=num_tokens_without_pad,
        start_symbol=tokenizer.cls_token_id,
    ).flatten()

    result = " ".join(tokenizer.convert_ids_to_tokens(tgt_tokens))
    result = result.replace(" ##", "")
    result = result.replace("[UNK]", "")
    result = result.replace("[CLS]", "")
    return result


if __name__ == "__main__":
    list_inferenced = []
    slice_index = 1

    dset = load_dataset(
        "PoolC/AIHUB-parallel-ko-braille-short64", use_auth_token=True
    )  # or replace with 128 for more data

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        "snoop2head/KoBrailleT5-small-v1"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained("snoop2head/KoBrailleT5-small-v1")

    for index_num, item in enumerate(tqdm(dset["test"])):
        src_input_id = torch.tensor(item["input_ids"]).view(-1, 1)
        result = inference(model, src_input_id)
        list_inferenced.append(result)
        if index_num == slice_index:
            break
