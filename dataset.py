# openwebmath from hugging faces (500mb)
import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"  # increase from default 10s

from datasets import load_dataset
ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
with open("openwebmath.txt", "w") as f:
      for i, example in enumerate(ds):
          f.write(example["text"] + "\n\n") 
          if i >= 70000:
              break
