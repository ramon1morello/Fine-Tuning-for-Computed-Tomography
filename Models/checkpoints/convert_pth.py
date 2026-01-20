"""
Caso o modelo gerado pelo Fine-Tuning dê erro no momento da inferencia, pode ser necessário reempacotar o estado do modelo
"""


import torch

# Modelo gerado pelo fine-tuning
old_path = "nome_do_arquivo_treinado.pth"

# Compatabilidade forçada
new_path = "nome_do_arquivo_treinado_compat.pth"

# Reempacotar → Fazer dicionário com chave 'params_ema'
state_dict = torch.load(old_path, map_location="cpu")
torch.save({'params_ema': state_dict}, new_path)
