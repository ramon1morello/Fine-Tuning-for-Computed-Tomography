import os
from glob import glob
from PIL import Image
import logging
import namelist as namelist
import util.utils as utils
from util.hat_arch import HAT
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from basicsr.archs.rrdbnet_arch import RRDBNet
from skimage.metrics import peak_signal_noise_ratio



#--------------------Dataset mínimo (PyTorch exige para DataLoader)-----------------------------
class PairedImageDataset(Dataset):
    """
    Dataset que recebe duas pastas: observation e ground truth.
    Apenas guarda os caminhos e carrega a imagem quando for pedida (__getitem__).
    Não carrega tudo na RAM -> economiza memória.
    """

    # Lista e ordena os arquivos das duas pastas.
    def __init__(self, obs_dir, gt_dir, transform=None, id=None, limit=None):
        """
        obs_dir: Pasta com imagens de observação (input)
        gt_dir: Pasta com imagens de ground truth (target)
        transform: Transformações a serem aplicadas (ToTensor)
        id: Identificação do dataset (treino ou validação) - Serve apenas para logs
        limit: Limita o número de pares carregados (Para testes rápidos)
        """

        self.obs_files = sorted(glob(os.path.join(obs_dir, "*.bmp")))
        self.gt_files = sorted(glob(os.path.join(gt_dir, "*.bmp")))
        self.transform = transform

        if limit is not None:
            self.obs_files = self.obs_files[:limit]
            self.gt_files = self.gt_files[:limit]

        if len(self.obs_files) != len(self.gt_files):
            raise AssertionError("Número de imagens diferente!")
        
        logging.info(f"{len(self.obs_files)} pares encontrados para {id}.")

    # Retorna o número de pares de imagens no dataset.  (É chamada pelo DataLoader)
    def __len__(self):          
        return len(self.obs_files)

    # Carrega uma imagem de observação e sua correspondente de ground truth. (É chamada pelo DataLoader)
    def __getitem__(self, idx):         
        input_image = Image.open(self.obs_files[idx]).convert("RGB")
        ground_truth_image = Image.open(self.gt_files[idx]).convert("RGB")
        if self.transform:
            input_image = self.transform(input_image)
            ground_truth_image = self.transform(ground_truth_image)
        return input_image, ground_truth_image
#-----------------------------------------------------------------------------------------------    

#---------------------------------Funções auxiliares--------------------------------------------

def psnr(output_image, ground_truth_image):
    # Converte tensores para numpy arrays e garante que estão no formato correto
    output_np = output_image.detach().cpu().numpy()
    gt_np = ground_truth_image.detach().cpu().numpy()

    # Ajusta o data_range conforme o intervalo dos dados [0,1]
    return peak_signal_noise_ratio(output_np, gt_np, data_range=1.0) 


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, use_amp=False):
    """
    Usa Mixed Precision para o HAT (muito grande para método convencional)
    """

    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for input_image, ground_truth_image in dataloader:
        
        # Move os tensores para o dispositivo correto
        input_image, ground_truth_image = input_image.to(device), ground_truth_image.to(device)
        
        optimizer.zero_grad()
       
        predicted_image = model(input_image)

        # Reduz a imagem gerada para o tamanho do ground truth
        if predicted_image.shape[2:] != ground_truth_image.shape[2:]:
            predicted_image = F.interpolate(predicted_image, size=ground_truth_image.shape[2:], mode='bilinear', align_corners=False)
        
        loss = loss_fn(predicted_image, ground_truth_image)
        loss.backward()
        optimizer.step()
    
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def _to_uint8_L_batch_pil(t):
    """t: [B,C,H,W] em [0,1] -> lista de imagens uint8 [H,W] usando PIL.convert('L')"""
    t = t.detach().float().cpu().clamp(0, 1).numpy()  # [B,C,H,W]
    out = []
    for i in range(t.shape[0]):
        x = t[i]  # [C,H,W]
        if x.shape[0] == 1:
            # 1 canal -> [H,W]
            u8 = (x[0] * 255.0).round().astype(np.uint8)
        else:
            # 3 canais -> [H,W,3] para PIL e depois convert('L')
            xhwc = (x.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
            u8 = np.array(Image.fromarray(xhwc).convert('L'))
        out.append(u8)
    return out

# Para desativar o cálculo de gradientes durante a validação (economiza memória e tempo)
@torch.no_grad()  

def validate(model, dataloader, device):
    model.eval()
    psnrs = []
    for input_image, ground_truth_image in dataloader:
        
        # Move os tensores para o dispositivo correto (CPU ou GPU)
        input_image, ground_truth_image = input_image.to(device), ground_truth_image.to(device)
        
        # Aplica o modelo ao tensor de entrada (super-resolução no domínio dos tensores)
        predicted_image = model(input_image)
        
        # Correção das dimensões da saída
        if predicted_image.shape[2:] != ground_truth_image.shape[2:]:
            predicted_image = F.interpolate(predicted_image, size=ground_truth_image.shape[2:], mode='bilinear', align_corners=False)
        
        # clamp explícito para evitar estouro numérico antes de converter
        predicted_image = predicted_image.clamp(0, 1)
        ground_truth_image = ground_truth_image.clamp(0, 1)

        pred_list = _to_uint8_L_batch_pil(predicted_image)
        gt_list   = _to_uint8_L_batch_pil(ground_truth_image)

        # Calcula o PSNR para cada imagem no batch
        for p, g in zip(pred_list, gt_list):
            psnrs.append(peak_signal_noise_ratio(g, p, data_range=255))

        # psnrs.append(psnr(predicted_image, ground_truth_image).item())

    # return sum(psnrs)/len(psnrs)
    return sum(psnrs)/len(psnrs) if psnrs else 0.0

#-----------------------------------------------------------------------------------------------

#----------------------------------- Preparação do dataset -------------------------------------
def get_dataloaders(limit_train=None, limit_val=None, batch_size=16, env_size="original"):
    """
    Retorna train_loader, val_loader.
    Especificação do HAT: patch_size(N) deve ser múltiplo de window_size(16).
    Também aplicável ao Real-ESRGAN, mas não necessário.
    """


    transform = transforms.ToTensor()
    root_path = namelist.root_path
    dataset_path = os.path.join(root_path, 'Datasets')
    
    # Define os diretórios conforme o tamanho do ambiente
    if env_size == "original":
        obs_train = os.path.join(dataset_path, 'train/observation_train/OBS_train_images_bmp')
        gt_train  = os.path.join(dataset_path, 'train/ground_truth_train/GT_train_images_bmp')
        obs_val   = os.path.join(dataset_path, 'validation/observation_validation/OBS_validation_images_bmp')
        gt_val    = os.path.join(dataset_path, 'validation/ground_truth_validation/GT_validation_images_bmp')
   
    elif env_size == "reduced":
        obs_train = os.path.join(dataset_path, 'train/observation_train/OBS_train_images_bmp_resized_240x240')
        gt_train  = os.path.join(dataset_path, 'train/ground_truth_train/GT_train_images_bmp_resized_240x240')
        obs_val   = os.path.join(dataset_path, 'validation/observation_validation/OBS_validation_images_bmp_resized_240x240')
        gt_val    = os.path.join(dataset_path, 'validation/ground_truth_validation/GT_validation_images_bmp_resized_240x240')

    # Criação dos datasets
    train_ds = PairedImageDataset(obs_train, gt_train, transform=transform, id='treino', limit=limit_train)
    val_ds   = PairedImageDataset(obs_val, gt_val, transform=transform, id='validacao', limit=limit_val)
    
    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader

#-----------------------------------------------------------------------------------------------



#-----------------------------------Funções principais--------------------------------------------
def fine_tune_esrgan(num_epoch, env_size="original"):
    try:
        logging.info(f"... Fazendo fine-tuning de {num_epoch} época(s) do Real-ESRGAN")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Usando dispositivo: {device}")

        batch_size_value  = 8
        train_loader, val_loader = get_dataloaders(limit_train=35820, limit_val=3522, batch_size=batch_size_value, env_size=env_size)
        logging.info(f"... Usando batch_size = {batch_size_value}")
        
        # Modelo Real-ESRGAN
        logging.info("... Definindo o modelo Real-ESRGAN")
        model = RRDBNet(num_in_ch=3, num_out_ch=3,
                        num_feat=64, num_block=23, num_grow_ch=32, scale=2).to(device)

        # Carregar pesos pré-treinados
        logging.info("... Carregando pesos pré-treinados")
        pretrained_model = f"Models/{namelist.esrgan_model_name}.pth"
        if os.path.exists(pretrained_model):
            ckpt = torch.load(pretrained_model, map_location="cpu")
            state = ckpt.get("params_ema", ckpt)
            model.load_state_dict(state, strict=False)
            logging.info(f"Carregado pré-treinado de {pretrained_model}")

        # Fine-tune apenas últimas camadas
        logging.info("... Configurando fine-tuning (apenas últimas camadas)")

        # Congela tudo
        for p in model.parameters():
            p.requires_grad = False

        # Libera apenas as camadas finais
        for name, module in reversed(list(model.named_modules())):  
            if any(k in name.lower() for k in ["conv_last", "conv_hr", "conv_up2"]):
                logging.info(f"Liberando camada: {name}") 
                for p in module.parameters():
                    p.requires_grad = True


        # Otimizador (ADAM)
        logging.info("... Definindo otimizador e função de perda")
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
        # Função de perda L1 (MSE)
        loss_fn = nn.L1Loss()

        # Loop de treino (só algumas épocas para teste)
        logging.info("... Iniciando o treino")
        
        for epoch in range(num_epoch):
            # Treina por uma época, recebendo a perda média
            loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device,  use_amp=False)
            # Valida o modelo, recebendo o PSNR médio
            psnr_val = validate(model, val_loader, device)

            logging.info(f"[ESRGAN] Epoch {epoch+1}: loss={loss:.4f}, val_psnr={psnr_val:.2f}")

            # Salva todos modelos gerados
            os.makedirs("../Models/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"../Models/checkpoints/FT_ESRGAN_PSNR_{psnr_val:.2f}.pth")


    except Exception as e:
        logging.error(f"Erro durante o fine-tuning do ESRGAN: {e}")

#-----------------------------------------------------------------------------------------------

#----------------------------------- Fine-tuning HAT -------------------------------------------
def fine_tune_hat(num_epoch, env_size="reduced"):
    try:
        logging.info(f"... Fazendo fine-tuning de {num_epoch} época(s) do HAT")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Usando dispositivo: {device}")

        batch_size_value  = 8
        train_loader, val_loader = get_dataloaders(limit_train=35820, limit_val=3522, batch_size=batch_size_value, env_size=env_size)

        logging.info(f"... Usando batch_size = {batch_size_value}")
        # Modelo HAT
        logging.info("... Definindo o modelo HAT")
        model = HAT(
            upscale=2,
            in_chans=3,
            img_size=64,
            window_size=16,
            compress_ratio=3,
            squeeze_factor=30,
            conv_scale=0.01,
            overlap_ratio=0.5,
            # depths=[6,6,6,6,6,6],
            depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            embed_dim=180,
            # num_heads=[6,6,6,6,6,6],
            num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="pixelshuffle",
            resi_connection="1conv"
        ).to(device)

        with torch.no_grad():
            model.mean.zero_()     # garante compatibilidade com TC (sem viés de RGB natural)
            logging.info("HAT: mean zerado para dataset em tons de cinza (TC)")

        # Carregar pesos pré-treinados
        logging.info("... Carregando pesos pré-treinados")
        pretrained_model = f"Models/{namelist.hat_model_name}.pth"
        if os.path.exists(pretrained_model):
            ckpt = torch.load(pretrained_model, map_location="cpu")
            state = ckpt.get("params_ema", ckpt)
            model.load_state_dict(state, strict=False)
            logging.info(f"Carregado pré-treinado de {pretrained_model}")


        # Fine-tune apenas últimas camadas
        logging.info("... Configurando fine-tuning (apenas últimas camadas)")

        # Congela tudo
        for p in model.parameters():
            p.requires_grad = False
        
        # Liberar somente as camadas conv2d
        # Libera apenas as camadas finais
        for name, module in reversed(list(model.named_modules())):
        # Última conv2d
            if name == "conv_last":  
                logging.info(f"Liberando camada: {name}")
                for p in module.parameters():
                    p.requires_grad = True

            # Penúltima e antepenúltima conv2d
            if name in ["upsample", "conv_before_upsample"]:
                    for submodule in module.children():
                        if isinstance(submodule, nn.Conv2d):  # Verifica se é Conv2d
                            logging.info(f"  Liberando subcamada {submodule} da camada {name}")
                            for p in submodule.parameters():
                                p.requires_grad = True

        # Otimizador (ADAM)
        logging.info("... Definindo otimizador e função de perda")
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        logging.info("... otimizador com Learning Rate de 1e-4")
        
        # Função de perda L1 (MSE)
        loss_fn = nn.L1Loss()

        # Loop de treino (só algumas épocas para teste)
        logging.info("... Iniciando o treino")
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parâmetros treináveis: {trainable:,}")

        for epoch in range(num_epoch):
            loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device,  use_amp=False)
            psnr_val = validate(model, val_loader, device)
            logging.info(f"[HAT] Epoch {epoch+1}: loss={loss:.4f}, val_psnr={psnr_val:.2f}")
            
            # Salva todos modelos gerados
            os.makedirs("Models/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"../Models/checkpoints/FT_HAT_PSNR_{psnr_val:.2f}.pth")
        
    except Exception as e:
        logging.error(f"Erro durante o fine-tuning do HAT: {e}")

#-----------------------------------------------------------------------------------------------
def _fine_tune_esrgan(num_epoch, env_size="original"):
    if env_size == "original":
        logging.info("... Treinando o modelo Real-ESRGAN com imagens na resolução original")
    else:
        logging.info("... Treinando o modelo Real-ESRGAN com imagens na resolução reduzida")
    
    fine_tune_esrgan(num_epoch, env_size)


def _fine_tune_hat(num_epoch, env_size="reduced"):
    logging.info("... Treinando o modelo HAT com imagens na resolução reduzida")
    fine_tune_hat(num_epoch, env_size)


if __name__ == "__main__":
    num_epoch = 3
    _fine_tune_esrgan(num_epoch)
    _fine_tune_hat(num_epoch)