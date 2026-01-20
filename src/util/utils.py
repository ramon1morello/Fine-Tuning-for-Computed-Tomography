from PIL import Image
import torch
import numpy as np

# Funções utilitárias locais
def img2tensor(img, bgr2rgb=True, float32=True):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    if bgr2rgb:
        img = img[:, :, [2, 1, 0]]
    img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
    if float32:
        img = img.float()
    return img

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    img_np = tensor.numpy().transpose(1, 2, 0)
    if rgb2bgr:
        img_np = img_np[:, :, [2, 1, 0]]
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    return img_np.astype(out_type)

def pil_to_tensor(image):
    """
    Converte uma imagem PIL para um tensor PyTorch de 4 dimensões (1, C, H, W),
    independente de ser grayscale ('L') ou RGB.
    Normaliza para [0, 1].
    """

    # Converte para array NumPy e normaliza
    arr = np.array(image).astype(np.float32) / 255.0  # [H, W] ou [H, W, 3]

    if arr.ndim == 2:
        # Caso 1: imagem em tons de cinza → adiciona canal
        t = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]
        t = t.unsqueeze(0)                      # [1, 1, H, W]
    else:
        # Caso 2: imagem RGB → permuta para [C, H, W]
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    return t

def pad_to_multiple(img, multiple=16):
    """Redimensiona a imagem para ser múltipla do tamanho da janela"""
    h, w = img.shape[:2]
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    
    if h != new_h or w != new_w:
        # Redimensionar usando PIL para manter qualidade
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)
        img = np.array(img_pil).astype(np.float32) / 255.0
    
    return img, (h, w)