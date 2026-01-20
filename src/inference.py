import os
import glob
import torch
import numpy as np
import logging
from PIL import Image
import namelist as name

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

from util.hat_arch import HAT
import util.utils as utils

def apply_realesrgan(env_size="original"):
    """
    Aplica o modelo RealESRGAN nas imagens de entrada.
    Lê as imagens da pasta input_observation_test_images_bmp e salva os resultados
    na pasta inference_output com o prefixo IF_ (inference). 
    """
    # Definindo caminhos
    logging.info("Iniciando aplicação do modelo RealESRGAN...")
    root_path = name.root_path
    dataset_path = os.path.join(root_path, 'Datasets')
    
    if env_size == "original":
        # Utiliza o dataset em sua resolução original (362x362)       
        source_path = os.path.join(dataset_path, f'{name.Img_group}/observation_{name.Img_group}/OBS_{name.Img_group}_images_bmp')     
        logging.info("... Direcionando inferência para o dataset em sua resolução original")
        output_path = os.path.join(dataset_path, 'inference_output/ESRGAN_362px')
    else:
        # Utiliza o dataset com resolução reduzida (240x240)
        source_path = os.path.join(dataset_path, f'{name.Img_group}/observation_{name.Img_group}/OBS_{name.Img_group}_images_bmp_resized_240x240')                                                  
        logging.info("... Direcionando inferência para o dataset em resolução reduzida (240x240)")
        output_path = os.path.join(dataset_path, 'inference_output/ESRGAN_240px')


    # Criando pasta de saída se não existir
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        logging.info(f"Criada pasta de saída: {output_path}")

    # Listando todas as imagens BMP de entrada
    source_images = glob.glob(os.path.join(source_path, '*.bmp'))
    logging.info(f"Encontradas {len(source_images)} imagens para processamento")

    # Selecionando dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Usando dispositivo: {device}")

    # Carregando o modelo
    esrgan_model_name = name.esrgan_model_name
    logging.info(f"...Carregando pesos do modelo: {esrgan_model_name}\n")
    model_dir = os.path.join(root_path, "Models")
    model_path = os.path.join(model_dir, esrgan_model_name)
    model_path = os.path.normpath(model_path)
    state_dict = torch.load(model_path, map_location=device)['params_ema']

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Inicializando o upscaler
    # Classe RealESRGANer: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
    # Referencia de upsampler: https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py

    upsampler = RealESRGANer(
        scale=2,
        model_path=model_path,
        model=model,
        tile=0,
        pre_pad=0,
        half=False
    )
    

    # Processando cada imagem
    for idx, source_image_path in enumerate(source_images, 1):
        try:
            # Gerando nome do arquivo de saída
            base_name = os.path.basename(source_image_path)
            output_name = f"IF_{base_name}"
            output_image_path = os.path.join(output_path, output_name)

            logging.info(f"Iniciando upscaling da imagem {idx}/{len(source_images)}: {base_name}")

            # Carregando e processando a imagem
            img = Image.open(source_image_path)
            original_size = img.size  # (altura, largura)
            img = np.array(img)
            
            # Aplicando o modelo
            output, _ = upsampler.enhance(img, outscale=4)
            
            # Convertendo e salvando o resultado
            output = Image.fromarray(output)
            
            # Redimensionando para o tamanho original, se necessário
            if output.size != (original_size[1], original_size[0]):  # PIL usa (largura, altura)
                output = output.resize((original_size[1], original_size[0]), Image.LANCZOS)
            
            # Salvando a imagem de saída
            output.save(output_image_path)
            

        except Exception as e:
            logging.error(f"Erro ao processar imagem {base_name}: {str(e)}")

logging.info("Processamento concluído!")

def apply_hat(env_size="original"):
    """
    Aplica o modelo HAT nas imagens de entrada.
    Lê as imagens da pasta input_observation_test_images_bmp e salva os resultados
    na pasta inference_output com o prefixo IF_.
    """
    # Definindo caminhos
    logging.info("Iniciando aplicação do modelo HAT...")
    root_path = name.root_path
    dataset_path = os.path.join(root_path, 'Datasets')

    if env_size == "original":
        # Utiliza o dataset em sua resolução original (362x362)  
        source_path = os.path.join(dataset_path, f'{name.Img_group}/observation_{name.Img_group}/OBS_{name.Img_group}_images_bmp')
        logging.info("... Direcionando inferência para o dataset em sua resolução original")
        output_path = os.path.join(dataset_path, 'inference_output/HAT_362px')   
    else:
        # Utiliza o dataset com resolução reduzida (240x240)
        source_path = os.path.join(dataset_path, f'{name.Img_group}/observation_{name.Img_group}/OBS_{name.Img_group}_images_bmp_resized_240x240')
        logging.info("... Direcionando inferência para o dataset em resolução reduzida (240x240)")
        output_path = os.path.join(dataset_path, 'inference_output/HAT_240px')

    weights_path = os.path.join(root_path, 'Models', name.hat_model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Usando dispositivo: {device}")

    # Criando pasta de saída se não existir
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        logging.info(f"Criada pasta de saída: {output_path}")

    # Examente igual do treinamento
    model = HAT(
        upscale=2,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv"
    ).to(device)

    # Carregando os pesos do modelo
    logging.info(f"Carregando pesos do modelo: {weights_path}")
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    # Filtrar apenas as chaves que existem no modelo
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['params_ema'].items() if k in model_dict}
    
    # Carregar apenas as chaves compatíveis
    model.load_state_dict(checkpoint['params_ema'], strict=False)
    logging.info(f"Carregadas {len(pretrained_dict)} de {len(checkpoint['params_ema'])} chaves")
    
    model.eval()

    # Listando todas as imagens BMP de entrada
    source_images = glob.glob(os.path.join(source_path, '*.bmp'))
    logging.info(f"Encontradas {len(source_images)} imagens para processamento")

    # Processando cada imagem
    for idx, source_image_path in enumerate(source_images, 1):
        try:
            # Gerando nome do arquivo de saída
            base_name = os.path.basename(source_image_path)
            output_name = f"IF_{base_name}"
            output_image_path = os.path.join(output_path, output_name)
            logging.info(f"Iniciando upscaling da imagem {idx}/{len(source_images)}: {base_name}")

            # Carregando e processando a imagem
            img = Image.open(source_image_path).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0

            
            if env_size == "original":
                # Redimensionar para ser múltipla do window_size (16)
                img_np, original_size = utils.pad_to_multiple(img_np, multiple=16)
            else:
                H, W = img_np.shape[:2]
                original_size = (H, W)
            
            # Convertendo para tensor
            img_tensor = utils.img2tensor(img_np, bgr2rgb=False, float32=True).unsqueeze(0).to(device)

            # Aplicando o modelo
            with torch.no_grad():
                # remove viés de imagens naturais
                model.mean.zero_()  
                output = model(img_tensor)

            # Convertendo o resultado de volta para imagem
            output_img = utils.tensor2img(output.squeeze(), rgb2bgr=True)

            # Redimensionar de volta para o tamanho original
            if output_img.shape[:2] != original_size:
                output_pil = Image.fromarray(output_img)
                output_pil = output_pil.resize((original_size[1], original_size[0]), Image.LANCZOS)
                output_img = np.array(output_pil)

            # Salvando o resultado
            Image.fromarray(output_img).save(output_image_path)

        except Exception as e:
            logging.error(f"Erro ao processar imagem {base_name}: {str(e)}")
            continue

    logging.info("Processamento HAT concluído!")

def inference(models: list, env_size: str) -> None:
    if "esrgan" in models:
        apply_realesrgan(env_size=env_size)
    elif "hat" in models:
        apply_hat(env_size=env_size)
    else:
        logging.error("Modelo desconhecido. Por favor, escolha entre 'esrgan' ou 'hat'.")
        raise ValueError("Modelo desconhecido. Por favor, escolha entre 'esrgan' ou 'hat'.")