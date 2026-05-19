import base64
import io

import torch
from PIL import Image
from torch import Tensor


class ImagePreprocessor:
    def __init__(self, *, size: int = 32) -> None:
        self._size = size

    def from_data_url(self, data_url: str) -> Tensor:
        _, _, payload = data_url.partition(",")
        image = Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
        image = image.resize((self._size, self._size))
        pixels = torch.tensor(list(image.getdata()), dtype=torch.uint8)
        return pixels.view(self._size, self._size, 3).permute(2, 0, 1).float() / 255.0
