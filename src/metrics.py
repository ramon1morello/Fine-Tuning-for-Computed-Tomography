import numpy as np
from PIL import Image
import skimage.metrics
import time
import logging
import os
import glob
import pyiqa
import namelist as name
import util.utils as utils

def calculate_metrics_for_pair(source_path, output_path):
    """
    Calcula PSNR e SSIM entre duas imagens.

    Entrada:
        source_path (str): Caminho da imagem original (ground truth).
        output_path (str): Caminho da imagem reconstruída (inference).

    Funcionamento:
        img1: Imagem Ground Truth
        img2: Imagem reconstruida (inferência)

    Saída:
        dict: Uma tabela com os valores de PSNR e SSIM e seus tempos de cálculo.
    """
    try:
        logging.info(f"Processando par - Ground Truth: {os.path.basename(source_path)}, inference: {os.path.basename(output_path)}")
        
        # Abri as imagens usando PIL
        img1_pil = Image.open(source_path)
        img2_pil = Image.open(output_path)

        # Converte para escala de cinza obrigatoriamente
        img1_pil = img1_pil.convert('L')
        img2_pil = img2_pil.convert('L')
        
        # Redimensiona a segunda imagem se necessário
        if img1_pil.size != img2_pil.size:   
            logging.info(f"Redimensionando imagem de {img2_pil.size} para {img1_pil.size}")
            img2_pil = img2_pil.resize(img1_pil.size)

        # Converte as imagens em arrays NumPy
        image1_array = np.array(img1_pil)
        image2_array = np.array(img2_pil)

        metrics_data = {}

        # PSNR (Peak Signal-to-Noise Ratio)
        start_time = time.time()
        psnr = skimage.metrics.peak_signal_noise_ratio(image1_array, image2_array, data_range=255)
        psnr_time = time.time() - start_time
        metrics_data['PSNR'] = (psnr, psnr_time)
        logging.info(f"PSNR calculado: {psnr:.4f} (tempo: {psnr_time:.4f}s)")

        # SSIM (Structural Similarity Index)
        start_time = time.time()
        ssim = skimage.metrics.structural_similarity(image1_array, image2_array, data_range=255)
        ssim_time = time.time() - start_time
        metrics_data['SSIM'] = (ssim, ssim_time)
        logging.info(f"SSIM calculado: {ssim:.4f} (tempo: {ssim_time:.4f}s)")

        # PI (Perceptual Index)
        # Inicializa os modelos NIQE e NRQM usando pyiqa para avaliações de qualidade de imagem sem referência
        niqe_model = pyiqa.create_metric('niqe')
        nrqm_model = pyiqa.create_metric('nrqm')

        # NIQE (Natural Image Quality Evaluator)
        niqe_score = float(niqe_model(utils.pil_to_tensor(img2_pil.convert('L'))).cpu().item())

        # NRQM (No-Reference Quality Metric)
        nrqm_score = float(nrqm_model(utils.pil_to_tensor(img2_pil.convert('RGB'))).cpu().item())


        def calculate_pi(niqe, nrqm):
            """
            Calculates the Perceptual Index (PI) metric based on NIQE and NRQM scores.

            PI = (10 - NRQM) / 2 + NIQE / 2

            Args:
                niqe (float): NIQE score of the image.
                nrqm (float): NRQM score of the image.

            Returns:
                float: Perceptual Index score.
            """
            return (10 - nrqm) / 2 + niqe / 2
        
        # Perceptual Index (PI)
        start_time = time.time()
        pi_score = calculate_pi(niqe_score, nrqm_score)
        pi_time = time.time() - start_time
        metrics_data['PI'] = (pi_score, pi_time)
        logging.info(f"PI calculado: {pi_score:.4f} (tempo: {pi_time:.4f}s)")

        return metrics_data

    except FileNotFoundError:
        logging.error(f"Erro: Uma das imagens não foi encontrada - GT: {source_path}, TF: {output_path}")
        return None
    except Exception as e:
        logging.error(f"Erro ao processar o par de imagens - GT: {source_path}, TF: {output_path}. Erro: {e}")
        return None

def calculate_all_metrics(models: list, env_size: str):
    """Calcula as métricas para todos os pares de imagens"""
    
    for model in models:
        logging.info(f"Iniciando cálculo de métricas para o modelo: {model}")

        # Definindo caminhos base
        root_path = name.root_path
        dataset_path = os.path.join(root_path, 'Datasets')
        if env_size == "original":
            # Resolução Original = 362x362
            source_dir = os.path.join(dataset_path, f'{name.Img_group}/ground_truth_{name.Img_group}/GT_{name.Img_group}_images_bmp')
            output_dir = os.path.join(dataset_path, f'inference_output/{model.upper()}_362px')
            metrics_output_path = os.path.join(root_path, 'Others', 'Metrics', f'metrics_{model}_362px_{time.strftime("%d_%m_%Y")}.csv')
        else:
            # Resolução Reduzida = 240x240
            source_dir = os.path.join(dataset_path, f'{name.Img_group}/ground_truth_{name.Img_group}/GT_{name.Img_group}_images_bmp_resized_240x240')
            output_dir = os.path.join(dataset_path, f'inference_output/{model.upper()}_240px')
            metrics_output_path = os.path.join(root_path, 'Others', 'Metrics', f'metrics_{model}_240px_{time.strftime("%d_%m_%Y")}.csv')

        # Obtendo lista de imagens ground truth
        ground_truth_images = glob.glob(os.path.join(source_dir, f'ground_truth_{name.Img_group}_*.bmp'))

        logging.info(f"Encontradas {len(ground_truth_images)} imagens ground truth para processamento")
        
        # Lista para armazenar todas as métricas
        all_metrics = []
        
        # Para cada imagem ground truth, encontra sua correspondente TF
        for gt_path in ground_truth_images:
            # Extrai o número da imagem do nome do arquivo
            base_name = os.path.basename(gt_path)
            img_number = base_name.split('_')[-1].split('.')[0]  # Pega o número antes do .bmp
            
            # Constrói o nome do arquivo TF correspondente
            tf_filename = f"IF_observation_{name.Img_group}_{img_number}.bmp"
            tf_path = os.path.join(output_dir, tf_filename)
            
            # Calcula as métricas para o par
            metrics = calculate_metrics_for_pair(gt_path, tf_path)
            
            if metrics:
                all_metrics.append({
                    'pair_number': img_number,
                    'metrics': metrics
                })
        
        # Calcula e salva as médias
        if all_metrics:
            total_psnr = sum(m['metrics']['PSNR'][0] for m in all_metrics)
            total_ssim = sum(m['metrics']['SSIM'][0] for m in all_metrics)
            total_pi = sum(m['metrics']['PI'][0] for m in all_metrics)
            avg_psnr = total_psnr / len(all_metrics)
            avg_ssim = total_ssim / len(all_metrics)
            avg_pi = total_pi / len(all_metrics)
            
            # Salva resultados em CSV
            with open(metrics_output_path, 'w') as f:
                # Cabeçalho
                f.write("PAR;PSNR;SSIM;PI\n")
                
                # Dados individuais
                for metric in all_metrics:
                    pair_num = metric['pair_number'].zfill(4)  # Garante 4 dígitos com zeros à esquerda
                    psnr_val = f"{metric['metrics']['PSNR'][0]:.4f}".replace('.', ',')
                    ssim_val = f"{metric['metrics']['SSIM'][0]:.4f}".replace('.', ',')
                    pi_val = f"{metric['metrics']['PI'][0]:.4f}".replace('.', ',')
                    f.write(f"{pair_num};{psnr_val};{ssim_val};{pi_val}\n")
                # Médias            
            logging.info(f"Processamento concluído. PSNR médio: {avg_psnr:.4f}, SSIM médio: {avg_ssim:.4f}, PI médio: {avg_pi:.4f}")
        else:
            logging.warning("Nenhum par de imagens foi processado com sucesso.")

