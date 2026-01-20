# Aprimoramento de Modelos de Super-Resolução para Imagens de Tomografia Computadorizada utilizando Fine-Tuning  

<p align="center">
  <a href="README_EN.md">
    <img src="https://img.shields.io/badge/README-English-🇬🇧" alt="English">
  </a>
  <a href="https://ele.ufes.br/pt-br/projetos-de-graduacao-202401-e-202402">
    <img src="https://img.shields.io/badge/-Projeto%20de%20Graduação-lightyellow" alt="Projeto de Graduação">
  </a>
</p>


## Índice
1. [Introdução](#introdução)  
2. [Conjunto de Dados](#conjunto-de-dados)  
3. [Etapas do Processo](#etapas-do-processo)  
4. [Escolhas Metodológicas](#escolhas-metodológicas)  
5. [Métricas de Avaliação](#métricas-de-avaliação)  
6. [Estrutura do Projeto](#estrutura-do-projeto)  
7. [Instalação e Execução](#instalação-e-execução)  


---

## Introdução

Este repositório contém o código desenvolvido no âmbito de um trabalho acadêmico cujo foco é o aprimoramento de imagens de tomografia computadorizada (TC) de baixa dosagem por meio de técnicas de super-resolução baseadas em aprendizado profundo. A motivação central está associada à possibilidade de melhorar a qualidade visual e estrutural das imagens médicas sem aumentar a exposição do paciente à radiação ionizante, utilizando exclusivamente pós-processamento computacional.

O projeto baseia-se na adaptação de modelos de super-resolução pré-treinados em imagens naturais para o domínio específico da tomografia computadorizada, utilizando a técnica de fine-tuning. O objetivo principal é avaliar o impacto desse ajuste fino no desempenho dos modelos, comparando a inferência direta com a inferência após o fine-tuning, e analisar se a especialização ao domínio médico contribui para ganho de qualidade perceptual e maior fidelidade estrutural nas imagens reconstruídas. Toda a implementação foi realizada com ferramentas gratuitas e de código aberto, visando reprodutibilidade e acessibilidade.

---

## Conjunto de Dados

O conjunto de dados utilizado neste projeto é o LoDoPaB-CT (Low-Dose Parallel Beam – Computed Tomography), comumente empregado para estudos de reconstrução e aprimoramento de imagens de tomografia computadorizada de baixa dosagem. Esse dataset é composto por imagens simuladas de TC do tórax humano, permitindo a formação de pares correspondentes de baixa resolução e alta resolução.

No contexto deste projeto, o dataset é organizado em subconjuntos de treino, validação e teste, possibilitando tanto o processo de fine-tuning supervisionado quanto a avaliação quantitativa dos resultados. As imagens originais são fornecidas em formato HDF5 e passam por uma etapa de processamento, na qual são convertidas para o formato BMP, tornando-se compatíveis com os modelos de super-resolução empregados.

---

## Etapas do Processo

O fluxo geral do projeto está organizado nas seguintes etapas:

1. **Preparação dos dados**  
   - Organização dos conjuntos de treino, validação e teste  
   - Conversão das imagens do formato HDF5 para BMP  

2. **Inferência direta**  
   - Aplicação dos modelos pré-treinados sem ajuste adicional  
   - Geração das imagens super-resolvidas de referência (baseline)  

3. **Fine-tuning dos modelos**  
   - Ajuste supervisionado dos pesos finais das redes  
   - Especialização dos modelos ao domínio de TC de baixa dosagem  

4. **Inferência pós fine-tuning**  
   - Geração das imagens reconstruídas com os modelos ajustados  

5. **Avaliação quantitativa**  
   - Cálculo das métricas de qualidade  
   - Comparação entre inferência direta e pós fine-tuning  

---

## Escolhas Metodológicas

As principais decisões adotadas no desenvolvimento do projeto foram:

- Uso de modelos pré-treinados, reduzindo custo computacional e tempo de treinamento  
- Aplicação de fine-tuning raso, com ajuste apenas das camadas finais das redes  
- Utilização do dataset LoDoPaB-CT
- Separação explícita entre treino, validação e teste para garantir avaliação consistente  

Essas escolhas buscam equilibrar desempenho, reprodutibilidade e viabilidade computacional.

---

## Métricas de Avaliação

A avaliação do desempenho dos modelos de super-resolução é realizada por meio das seguintes métricas:

**PSNR (Peak Signal-to-Noise Ratio)**
- Avalia a relação sinal-ruído entre a imagem reconstruída e a imagem de referência. Valores mais altos indicam melhor qualidade de reconstrução.

**SSIM (Structural Similarity Index Measure)**
- Mede a similaridade estrutural entre a imagem super-resolvida e a imagem de referência, considerando luminância, contraste e estrutura.

**PI (Perceptual Index)**
- Métrica perceptual que combina informações de qualidade visual para avaliar a naturalidade das imagens reconstruídas. Valores menores indicam melhor qualidade perceptual.

Essas métricas permitem analisar de forma complementar a fidelidade estrutural e a qualidade visual das imagens reconstruídas.

---

## Estrutura do Projeto

A organização dos arquivos e diretórios do repositório segue a estrutura abaixo.  
**Algumas pastas do projeto contêm um arquivo denominado `instruction.md`**, o qual descreve o propósito daquela pasta, os arquivos que ela armazena e o que acontece com seu conteúdo ao longo da execução do pipeline.

```text
Fine-Tuning-for-Computed-Tomography/
├─ Datasets/
│  ├─ test/         - Dataset com as imagens de teste
│  ├─ train/        - Dataset com as imagens de treino
│  ├─ validation/   - Dataset com as imagens de validação
├─ Models/
│  ├─ checkpoints/  - Modelos gerados pelo treinamento
├─ Others/
│  ├─ Logs/         - Registro de log da execução
│  └─ Metrics/      - Resultados das métricas calculadas
├─ src/
│  ├─ util/
│  │  ├─ hat_arch.py            - Dependências para execução do HAT
│  │  ├─ util_basicsr.py        - Ajustes na biblioteca BasicSR
│  │  └─ utils.py               - Funções auxiliares
│  ├─ fine_tune.py              - Treinamento (fine-tuning) dos modelos
│  ├─ inference.py              - Inferência dos modelos
│  ├─ main.py                   - Script principal de orquestração
│  ├─ metrics.py                - Cálculo das métricas
│  ├─ namelist.py               - Configurações da execução
│  └─ pre_processing.py         - Conversão HDF5 para BMP
├─ environment.yml              - Ambiente Conda com dependências
└─ README.md
```

---
## Instalação e Execução

As instruções de instalação e execução seguem o fluxo abaixo:

### 1. Instale o Anaconda
- https://www.anaconda.com/download/

### 2. Crie o ambiente virtual
```python
conda create -y -n .venv_FT python=3.10
```

### 3. Ative o ambiente virtual
```python
conda activate .venv_FT
```

### 4. Instale as bibliotecas partir do environment.yml
```python
conda env update -n .venv_FT -f environment.yml
```
### 5. Adicione o Dataset na pasta Datasets
- Siga as instruções descritas nos arquivos `instruction.md` dentro da pasta Datasets.

### 6. Edite o arquivo `namelist.py`
- Presente na pasta `src`, informe a informações solicitadas

### 7. Execute o código
```python
python src/main.py
```

### 8. Acompanhe o andamento através dos registros de log
- Disponível na pasta: `Others/logs`

---
