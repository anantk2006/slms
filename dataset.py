import os
from typing import Optional, List
import torch
from torch.utils.data import Dataset

class LongTextDataset(Dataset):
    """
    Convert a plain text file into a corpus of long fixed-length token windows.
    - If a HuggingFace-style tokenizer is provided it will be used; otherwise
      raw UTF-8 bytes are used as token ids.
    - Produces windows of size block_size with a configurable stride (overlap).
    - Short final window(s) are padded with pad_token_id unless drop_last=True.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: Optional[object] = None,
        block_size: int = 2048,
        stride: Optional[int] = None,
        encoding: str = "utf-8",
        pad_token_id: int = 0,
        eos_token_id: Optional[int] = None,
        add_eos: bool = False,
        drop_last: bool = False,
    ):
        assert os.path.isfile(file_path), f"File not found: {file_path}"
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.stride = int(stride) if stride is not None else self.block_size
        self.encoding = encoding
        self.drop_last = drop_last

        # prefer tokenizer pad/eos ids if available
        if tokenizer is not None:
            pad_token = getattr(tokenizer, "pad_token_id", None)
            eos_token = getattr(tokenizer, "eos_token_id", None)
            if pad_token is not None:
                pad_token_id = pad_token
            if eos_token_id is None and eos_token is not None:
                eos_token_id = eos_token

        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.add_eos = add_eos

        # load and encode the entire file into a 1-D list of token ids
        with open(file_path, "r", encoding=self.encoding) as f:
            text = f.read()

        if self.tokenizer is None:
            # simple byte-level encoding fallback
            token_ids: List[int] = list(text.encode("utf-8"))
            if self.add_eos and self.eos_token_id is not None:
                token_ids.append(self.eos_token_id)
        else:
            # support both tokenizer.encode(text) and tokenizer(text)["input_ids"]
            if hasattr(tokenizer, "encode") and callable(getattr(tokenizer, "encode")):
                token_ids = tokenizer.encode(text, add_special_tokens=False)
            else:
                enc = tokenizer(text, add_special_tokens=False)
                token_ids = enc["input_ids"]
            if self.add_eos and self.eos_token_id is not None:
                token_ids = list(token_ids) + [self.eos_token_id]

        self.data = torch.tensor(token_ids, dtype=torch.long)
        self._compute_windows()

    def _compute_windows(self):
        n = self.data.size(0)
        if n < self.block_size:
            if self.drop_last:
                self.starts = []
            else:
                self.starts = [0]
            return

        step = max(1, self.stride)
        max_start = n - self.block_size
        self.starts = list(range(0, max_start + 1, step))
        # optionally include a final window that ends at the last token (right-aligned)
        last_start = n - self.block_size
        if last_start > 0 and (self.starts[-1] != last_start):
            # include final right-aligned window for full coverage
            self.starts.append(last_start)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx: int):
        if idx < 0:
            idx = len(self) + idx
        start = self.starts[idx]
        end = start + self.block_size
        slice_ = self.data[start:end]
        if slice_.size(0) < self.block_size:
            # pad on the right
            pad_len = self.block_size - slice_.size(0)
            pad = torch.full((pad_len,), fill_value=self.pad_token_id, dtype=torch.long)
            slice_ = torch.cat([slice_, pad], dim=0)
        # For language modeling it's common to return input_ids and labels==input_ids
        return {"input_ids": slice_.clone(), "labels": slice_.clone()}


# quick usage example (not executed on import):
# ds = LongTextDataset("my_corpus.txt", tokenizer=my_tokenizer, block_size=4096, stride=1024)
# loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True)