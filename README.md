![Machine Learning Engineering](docs/img/mlops_banner.png)

# Ecommerce Recsys MLOps

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Sistema de recomendação para e-commerce ponta a ponta, construído com PyTorch, MLflow, DVC e Docker. FIAP MLE Tech Challenge — Fase 2.

## 👥 Integrantes
| Nome | RM | Contato |
|--|--|--|
| Gabriel de Paula Vicente | RM373848 | [Github](https://github.com/gabrielpvicente) - [Linkedin](https://www.linkedin.com/in/gabriel-de-paula-vicente-796198102/)|
| Gustavo Dell Anhol Oliveira | RM372138 | [Github](https://github.com/gudaoliveira) - [Linkedin](https://www.linkedin.com/in/gustavodell/)|
| Kevin Pagrion Bela | RM371774 | [Github](https://github.com/kevinpabe) - [Linkedin](https://www.linkedin.com/in/kevinpb/)|
| Patrick Kwan | RM373172 | [Github](https://github.com/ptkwan) - [Linkedin](https://www.linkedin.com/in/patrick-kwan-617296220/)|
| Vitor Akira Ucha Ito | RM371483 | [Github](https://github.com/VitorAkira-me) - [Linkedin](https://www.linkedin.com/in/vitor-akira/)|

## 📋 Requisitos
- Python 3.13+
- `uv` instalado para gerenciamento de dependências
    - Aprenda como instalar o `uv` [aqui](https://docs.astral.sh/uv/getting-started/installation/)
- `Makefile` para comandos de conveniência (opcional, mas recomendado)
    - No Windows, você pode usar o [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) para acessar o `Makefile`.

## ⚙️ Setup
Realize o clone do repositório com

```bash
git clone https://github.com/fiap-postech-ml-engineering/ecommerce-recsys-mlops

cd ecommerce-recsys-mlops
```
Crie o ambiente virtual
```bash
uv venv

#############################################
# Para acessar o ambiente virtual
Windows 		-> .venv\Scripts\activate
Linux / macOS 	-> source .venv/bin/activate
```

Instale as dependências com:
```bash
# (para dependências de produção)
uv sync

# (para dependências de desenvolvimento)
uv sync --extra dev
uv run pre-commit install
uv run pre-commit run --all-files
```

## 📁 Organização do projeto

```
├── LICENSE            <- Licença open-source do projeto
├── Makefile           <- Comandos utilitários como `make test` ou `make lint`
├── README.md          <- README principal para desenvolvedores do projeto
├── data
│   ├── external       <- Dados de fontes externas/terceiros
│   ├── interim        <- Dados intermediários após transformações parciais
│   ├── processed      <- Dados finais prontos para modelagem
│   └── raw            <- Dados brutos originais, imutáveis
│
├── docs               <- Documentação do projeto (mkdocs)
│
├── models             <- Modelos treinados, serializados, predições e sumários
│
├── notebooks          <- Jupyter notebooks. Convenção de nomenclatura: número (ordenação),
│                         iniciais do autor e descrição curta separada por `-`, ex.:
│                         `1.0-gdl-exploracao-inicial-dos-dados`.
│
├── pyproject.toml     <- Configuração do projeto: metadados do pacote e ferramentas
│                         (ruff, pytest, coverage, taskipy)
│
├── references         <- Dicionários de dados, manuais e materiais de referência
│
├── reports            <- Análises geradas em HTML, PDF, LaTeX, etc.
│   └── figures        <- Gráficos e figuras gerados para os relatórios
│
├── src                <- Código-fonte principal do projeto
│   │
│   ├── __init__.py             <- Torna src um módulo Python
│   │
│   ├── config.py               <- Variáveis de configuração e constantes globais
│   │
│   ├── dataset.py              <- Scripts para download e geração de dados
│   │
│   ├── features.py             <- Criação de features para modelagem
│   │
│   ├── modeling
│   │   ├── __init__.py
│   │   ├── predict.py          <- Inferência com modelos treinados
│   │   └── train.py            <- Treinamento de modelos
│   │
│   └── plots.py                <- Geração de visualizações
│
└── tests              <- Testes automatizados (pytest)
```

## 📚 Documentações

lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec a diam lectus. Sed sit amet ipsum mauris. Maecenas congue ligula ac quam viverra nec consectetur ante hendrerit. Donec et mollis dolor. Praesent et diam eget libero egestas mattis sit amet vitae augue. Nam tincidunt congue enim, ut porta lorem lacinia consectetur.

## 🔄 Fluxograma do projeto

lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec a diam lectus. Sed sit amet ipsum mauris. Maecenas congue ligula ac quam viverra nec consectetur ante hendrerit. Donec et mollis dolor. Praesent et diam eget libero egestas mattis sit amet vitae augue. Nam tincidunt congue enim, ut porta lorem lacinia consectetur.

## 🚀 Execução do projeto

lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec a diam lectus. Sed sit amet ipsum mauris. Maecenas congue ligula ac quam viverra nec consectetur ante hendrerit. Donec et mollis dolor. Praesent et diam eget libero egestas mattis sit amet vitae augue. Nam tincidunt congue enim, ut porta lorem lacinia consectetur.

## ✅ Testes e validação

lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec a diam lectus. Sed sit amet ipsum mauris. Maecenas congue ligula ac quam viverra nec consectetur ante hendrerit. Donec et mollis dolor. Praesent et diam eget libero egestas mattis sit amet vitae augue. Nam tincidunt congue enim, ut porta lorem lacinia consectetur.

## 🔧 Pontos de melhoria

lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec a diam lectus. Sed sit amet ipsum mauris. Maecenas congue ligula ac quam viverra nec consectetur ante hendrerit. Donec et mollis dolor. Praesent et diam eget libero egestas mattis sit amet vitae augue. Nam tincidunt congue enim, ut porta lorem lacinia consectetur.
