from .remote_te_node import QwenImage21Remote, TextEncodeQwenImage21Remote
from .remote_te_edit import TextEncodeQwenImageEditRemote
from .remote_h3 import MiniMaxH3ImageToVideoRemote

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "TextEncodeQwenImage21Remote": TextEncodeQwenImage21Remote,
    "TextEncodeQwenImageEditRemote": TextEncodeQwenImageEditRemote,
    "QwenImage21Remote": QwenImage21Remote,
    "MiniMaxH3ImageToVideoRemote": MiniMaxH3ImageToVideoRemote,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeQwenImage21Remote": "Qwen Image 2.1 Text Encode (Remote)",
    "TextEncodeQwenImageEditRemote": "Qwen Image 2.1 Edit Encode (Remote)",
    "QwenImage21Remote": "Qwen Image 2.1 Positive / Negative / Latent (Remote)",
    "MiniMaxH3ImageToVideoRemote": "MiniMax H3 Image to Video (Remote TE)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
