# 1. Tech Challenge — Visão Geral

O Tech Challenge é o projeto da fase que **engloba os conhecimentos obtidos em todas as disciplinas**. A princípio é desenvolvido em grupo e tem prazo de entrega obrigatório.

É uma atividade **obrigatória e avaliada**, que vale **90% da nota de todas as disciplinas da fase**.

## 1.1 Modalidade

- **Tipo →** Atividade em grupo
- **Caráter →** Obrigatória
- **Avaliação →** Avaliada (90% da nota de todas as disciplinas da fase)

## 1.2 Entregas

- **Obrigatória →** Repositório GitHub + vídeo de 5 minutos pelo método STAR
- **Opcional →** Deploy em ambiente de produção em nuvem (AWS, Azure ou GCP)

---

# 2. O Problema

Uma empresa de **e-commerce** precisa de um **sistema de recomendação de produtos** baseado no comportamento de navegação dos usuários.

O modelo central é uma **rede neural** (MLP ou embedding-based) treinada com `PyTorch`, com pipeline completo containerizado em `Docker`, dados versionados com `DVC`, experimentos rastreados no `MLflow` e código seguindo padrões profissionais de clean code.

---

# 3. Requisitos Obrigatórios

## 3.1 Repositório GitHub

- **Estrutura clean code →** módulos curtos, nomes descritivos, SOLID, type hints
- **Gerenciamento de dependências →** `pyproject.toml` com `Poetry`/`uv`, dependências prod/dev separadas, lock file commitado
- **Arquivos de configuração →** `.dockerignore`, `.gitignore`, `.env.example` configurados
- **Versionamento →** histórico de commits semântico

## 3.2 Vídeo (5 minutos — método STAR)

### 3.2.1 Situation

- **Conteúdo →** problema de negócio e contexto do dataset

### 3.2.2 Task

- **Conteúdo →** objetivos técnicos e restrições

### 3.2.3 Action

- **Conteúdo →** decisões de arquitetura, modelo, versionamento e containerização

### 3.2.4 Result

- **Conteúdo →** resultados obtidos, trade-offs e lições aprendidas

## 3.3 Bibliotecas Requeridas

- **`PyTorch` →** rede neural para o modelo de recomendação
- **`Scikit-Learn` →** pré-processamento e baselines
- **`MLflow` →** tracking de experimentos e Model Registry
- **`DVC` →** versionamento de dados e pipeline reprodutível

## 3.4 Boas Práticas Obrigatórias

- **Clean code →** funções ≤ 20 linhas, naming conventions, type hints
- **Design patterns aplicados →** Factory, Strategy ou Template Method
- **Dockerfile →** multi-stage com imagem otimizada
- **Pipeline DVC →** ≥ 3 stages
- **Reprodutibilidade →** seeds fixados, lock file, `.env`

---

# 4. Etapas de Desenvolvimento (4 Etapas)

## 4.1 Etapa 1 — Clean Code e Estrutura (Disciplina 01)

### 4.1.1 Foco

- **Objetivo →** projeto limpo com padrões de engenharia desde o início

### 4.1.2 Tarefas

| **Tarefa** | **Referência** |
| --- | --- |
| Definir estrutura de projeto com `src/`, `tests/`, `data/`, `models/`, `configs/` | Clean Code, Aula 01 |
| Aplicar naming conventions e SOLID desde a primeira linha | Clean Code, Aulas 01–02 |
| Implementar ≥ 1 design pattern (Factory para criar modelos, Strategy para preprocessors) | Clean Code, Aula 03 |
| Type hints em todas as funções públicas + docstrings Google style | Clean Code, Aula 03 |
| Configurar `ruff` sem erros + pre-commit hooks | Clean Code, Aula 03 |

### 4.1.3 Entregável

- **Resultado →** repositório base com estrutura limpa e linting passando

## 4.2 Etapa 2 — Ambiente e Dependências (Disciplina 02)

### 4.2.1 Foco

- **Objetivo →** reprodutibilidade garantida com gerenciamento moderno de dependências

### 4.2.2 Tarefas

| **Tarefa** | **Referência** |
| --- | --- |
| Configurar `pyproject.toml` com Poetry → deps de prod (`pytorch`, `sklearn`, `mlflow`) e dev (`pytest`, `ruff`) | Dependências, Aula 02 |
| Gerar e commitar lock file | Dependências, Aula 03 |
| Externalizar configurações para `.env`  • `Pydantic Settings` | Dependências, Aula 03 |
| Script de validação de ambiente (`scripts/validate_env.py`) | Dependências, Aula 01 |
| Verificar instalação limpa em ambiente novo | Dependências, Aula 02 |

### 4.2.3 Entregável

- **Resultado →** projeto instalável do zero com `poetry install`

## 4.3 Etapa 3 — Containerização e Versionamento (Disciplinas 03 e 04)

### 4.3.1 Foco

- **Objetivo →** `Docker` + `DVC` + `MLflow` integrados em pipeline reprodutível

### 4.3.2 Tarefas

| **Tarefa** | **Referência** |
| --- | --- |
| Dockerfile multi-stage → builder (deps) + runtime (app) | Docker, Aulas 01–03 |
| `docker-compose.yml` com serviço de treino + MLflow server | Docker, Aula 04 |
| `dvc init`, versionar dataset, configurar remote (local ou S3) | DVC+MLflow, Aulas 01–03 |
| Pipeline DVC (`dvc.yaml`) → preprocess → feature_eng → train → evaluate | DVC+MLflow, Aula 04 |
| MLflow tracking → logar params, métricas, artefatos de cada run | DVC+MLflow, Aula 04 |

### 4.3.3 Entregável

- **Resultado →** pipeline reprodutível via `dvc repro` + Docker funcional

## 4.4 Etapa 4 — Rede Neural, Registry e Entrega (Disciplina 04 + consolidação)

### 4.4.1 Foco

- **Objetivo →** modelo neural treinado, registrado e documentado

### 4.4.2 Tarefas

| **Tarefa** | **Referência** |
| --- | --- |
| Treinar MLP/embedding model com `PyTorch` para recomendação | — (PyTorch) |
| Comparar com baselines (`Scikit-Learn`) usando ≥ 4 métricas | — (Scikit-Learn) |
| Registrar modelo no MLflow Model Registry → Staging → Production | DVC+MLflow, Aula 05 |
| Escrever Model Card com performance, limitações e vieses | — (boas práticas) |
| Finalizar README com instruções completas | — |
| Gravar vídeo STAR de 5 minutos | — |
| (Opcional) Deploy em nuvem via Docker | — |

### 4.4.3 Entregável

- **Resultado →** repositório final + modelo no Registry + vídeo STAR

---

# 5. Critérios de Avaliação

| **Critério** | **Peso** | **Descrição** |
| --- | --- | --- |
| Clean code e estrutura | 15% | SOLID, naming, type hints, design patterns, linting |
| Reprodutibilidade | 15% | Poetry, lock file, `.env`, instalação limpa |
| Docker | 15% | Multi-stage, imagem otimizada, compose funcional |
| DVC + Pipeline | 15% | Dataset versionado, pipeline ≥ 3 stages, `dvc repro` funcional |
| Rede neural (PyTorch) | 15% | MLP funcional, early stopping, comparação com baselines |
| MLflow + Registry | 10% | ≥ 3 runs rastreados, modelo promovido a Production |
| Vídeo STAR | 10% | Clareza, cobertura dos 4 elementos, ≤ 5 min |
| Bônus → deploy em nuvem | 5% | Container acessível via URL pública |

---

# 6. Dataset Sugerido

Dataset de **interações de e-commerce**, como por exemplo:

- **Instacart Market Basket →** dados de pedidos reais de supermercado online
- **RetailRocket →** eventos de navegação e transações em e-commerce
- **MovieLens →** alternativa clássica para sistemas de recomendação

## 6.1 Alternativa

- **Critério mínimo →** qualquer dataset com **≥ 10.000 interações user-item**

---

# 7. Passo a Passo Resumido

- **[Etapa 1] →** estrutura clean code + design patterns + linting
- **[Etapa 2] →** `Poetry` + lock file + `.env` + validação de ambiente
- **[Etapa 3] →** Docker multi-stage + DVC pipeline + MLflow tracking
- **[Etapa 4] →** MLP `PyTorch` + Model Registry + Model Card + vídeo STAR
