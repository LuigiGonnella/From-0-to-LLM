# From-0-to-LLM

A comprehensive learning path from traditional Machine Learning to Large Language Models. This repository documents my journey through practical implementations, experiments, and deep dives into modern ML/DL techniques, with a strong focus on Natural Language Processing and transformer architectures.

All materials are hands-on: Jupyter notebooks with reproducible code, preprocessing pipelines, training scripts, and detailed experiments on real-world datasets.

## 🎯 Mission

Build a solid foundation in ML/DL by progressing from classical algorithms to state-of-the-art language models. This repository serves as both a learning log and a reference for:

- **Classical ML**: Regression, clustering, classification, gradient boosting, and complete ML pipelines
- **Deep Learning Fundamentals**: PyTorch basics, CNN architectures, transfer learning
- **Computer Vision**: Image classification with CIFAR-10, Flowers-102, Caltech-101
- **NLP & Transformers**: Attention mechanisms, T5, BERT fine-tuning, text classification
- **Best Practices**: Hyperparameter optimization, model evaluation, artifact management

## 📁 Repository Structure

### `ML/` - Machine Learning Foundations

Classical machine learning implementations with scikit-learn:

- **`1) Regression/`** - Linear/polynomial regression, regularization techniques
- **`2) Clustering/`** - K-means, hierarchical clustering, DBSCAN
- **`3) Classification/`** - Logistic regression, decision trees, SVM, ensemble methods
- **`4) GradientBoosting/`** - XGBoost, LightGBM, gradient boosting techniques
- **`5) Preprocessing/`** - Data cleaning, imputation, encoding, scaling, feature engineering
- **`6) Time Series/`** - ARIMA, seasonal decomposition, forecasting
- **`7) RecapPipeline/`** - End-to-end ML pipelines with preprocessing + model training
- **`RecapML/`** - Summary notebooks and best practices

### `DL/` - Deep Learning & NLP

PyTorch-based deep learning experiments, from computer vision to language models:

#### **`0) Intro Pytorch/`**
Introduction to PyTorch: tensors, autograd, building neural networks, training loops

#### **`1) Training CIFAR10/`**
Image classification on CIFAR-10 dataset:
- Training CNNs from scratch
- Transfer learning with pre-trained models (ResNet, VGG)
- Performance comparison and evaluation

#### **`2) Training flowers/`**
Fine-grained classification on Oxford Flowers-102:
- Working with 102 flower species
- Data augmentation techniques
- Transfer learning strategies

#### **`3) training_CNN/`**
Deep dive into CNN architectures:
- **AlexNet**: Historic breakthrough architecture
- **ResNet**: Residual connections and transfer learning
- **Caltech-101**: Training from scratch vs transfer learning

#### **`4) T5&AttentionMechanism/`**
Introduction to transformers and attention:
- **T5 (Text-to-Text Transfer Transformer)**: Understanding the unified text-to-text framework
- **Attention Mechanism**: Self-attention, multi-head attention, visualization
- Sequence-to-sequence tasks

#### **`5) Hyperparameter optimization focus/`**
Systematic hyperparameter tuning:
- Grid search, random search, Bayesian optimization
- Learning rate scheduling
- Regularization techniques (dropout, weight decay)

#### **`6) BERT/`**
BERT (Bidirectional Encoder Representations from Transformers):
- **Architecture**: Understanding BERT's bidirectional context
- **Tokenization**: WordPiece tokenization, special tokens ([CLS], [SEP], [MASK])
- **Pre-training tasks**: Masked Language Modeling (MLM), Next Sentence Prediction (NSP)
- **Fine-tuning**: Text classification, sequence labeling
- **HuggingFace Transformers**: Using `BertModel`, `BertForSequenceClassification`, `BertForMaskedLM`
- **Training details**: Optimizer selection, learning rate scheduling, checkpointing

#### **`7) GPT-2/`**
GPT-2 (Generative Pre-trained Transformer 2):
- **Architecture**: Decoder-only transformer for autoregressive text generation
- **Causal language modeling**: Predicting next tokens in sequence
- **Fine-tuning**: Domain-specific text generation, dialogue systems
- **Generation strategies**: Greedy decoding, beam search, top-k/top-p sampling
- **HuggingFace integration**: `GPT2LMHeadModel`, `GPT2Tokenizer`

#### **`8) LLAMA&MISTRAL/`**
Modern open-source large language models:
- **LLAMA**: Meta's efficient LLM architecture (LLAMA 2, LLAMA 3)
- **Mistral**: High-performance 7B parameter model with sliding window attention
- **Model loading**: Handling large models with `device_map='auto'`, quantization
- **Instruction tuning**: Using instruct variants for chat and task completion
- **Inference optimization**: Memory management, batch processing

#### **`9) CLM TASK/`**
Causal Language Modeling tasks:
- **Medical text generation**: Fine-tuning GPT-2 on medical datasets
- **Domain adaptation**: Customizing LLMs for specific domains
- **Evaluation metrics**: Perplexity, BLEU scores, human evaluation
- **Training strategies**: Learning rate scheduling, gradient accumulation

#### **`10) Quantization&LoRA/`**
Model optimization and efficient fine-tuning:
- **Quantization**: 8-bit and 4-bit quantization with `bitsandbytes`
- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning
- **QLoRA**: Combining quantization with LoRA for memory-efficient training
- **PEFT library**: Practical implementation with HuggingFace PEFT
- **Memory optimization**: Reducing VRAM usage while maintaining performance

#### **`11) RAG_&_Prompt-Engineering/`**
Retrieval-Augmented Generation and advanced prompting:
- **RAG architecture**: Combining retrieval with generation
- **Document embeddings**: BERT-based embeddings for semantic search
- **Retrieval strategies**: Cosine similarity, top-k document selection
- **Prompt engineering**: System prompts, few-shot learning, chain-of-thought
- **Context management**: Handling long documents, truncation strategies

#### **`12) RAG_&_User_Stories_Generation/`**
Applying RAG to software engineering:
- **User story generation**: Creating structured requirements from natural language
- **Document retrieval**: Finding relevant examples and templates
- **Structured output**: JSON formatting, consistent schema generation
- **Use case**: Automated software requirement documentation

#### **`13) Advanced_Architectures_for_Vision&Multimedia/`**
Cutting-edge vision and multimodal models:
- **Vision Transformers (ViT)**: Applying transformers to computer vision
- **CLIP**: Contrastive language-image pre-training
- **Multimodal architectures**: Combining vision and language understanding
- **Advanced techniques**: Attention visualization, feature extraction

#### **`14) Keyword Spotting/`**
Audio processing with deep learning:
- **Speech recognition**: Wake word detection, command recognition
- **Audio preprocessing**: MFCC features, spectrograms
- **CNN/RNN architectures**: Models for audio classification
- **Real-time inference**: Optimizing for low-latency applications

#### **`15) ROC-ROUTER Design/`**
Model evaluation and routing strategies:
- **ROC curve analysis**: TPR/FPR trade-offs, AUC computation
- **Model routing**: Selecting appropriate models based on input characteristics
- **Performance metrics**: Precision, recall, F1-score, confusion matrices
- **Decision thresholds**: Optimizing classification cutoffs

#### **`16) Agentic-Code-Generation/`**
LLM-powered code generation and evaluation:
- **Code generation**: Using CodeLlama, Qwen, GPT for Python code synthesis
- **Automated testing**: Generating and running pytest test cases
- **Code quality metrics**: Cyclomatic complexity, maintainability index, LOC
- **Multi-model comparison**: Evaluating GPT vs CodeLlama vs Qwen
- **Functional correctness**: Test-driven validation of generated code
- **Radon analysis**: Static code analysis for quality assessment

#### **`17) GANs/`**
Generative Adversarial Networks:
- **GAN architectures**: Generator and discriminator networks
- **Training dynamics**: Adversarial training, mode collapse, stability
- **Applications**: Image generation, style transfer, data augmentation

### `ML vs DL/`
Comparative analysis between ML and DL approaches:
- When to use classical ML vs deep learning
- Performance/complexity trade-offs
- Practical decision-making guidelines


## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+ (with CUDA support recommended for deep learning)
- JupyterLab or Jupyter Notebook
- HuggingFace Transformers library

### Installation

1. Clone the repository:
```bash
git clone https://github.com/LuigiGonnella/From-0-to-LLM.git
cd From-0-to-LLM
```

2. Create a virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source venv/bin/activate      # Linux/Mac
```

3. Install dependencies:
```powershell
# Core deep learning
pip install torch torchvision torchaudio

# NLP and LLMs
pip install transformers datasets
pip install peft bitsandbytes accelerate
pip install sentencepiece

# Development tools
pip install jupyter jupyterlab
pip install pytest ipytest radon

# Classical ML
pip install scikit-learn pandas numpy matplotlib seaborn
pip install xgboost lightgbm

# Audio processing
pip install librosa
```

### Running Notebooks

1. Start Jupyter:
```powershell
jupyter lab
# or
jupyter notebook
```

2. Navigate to the desired notebook and execute cells sequentially
3. For reproducibility, always run setup cells first (imports, seeds, configurations)

### ⚠️ Important Notes

- **Large Model Files**: Model checkpoints (`.pt`, `.pth`, `.safetensors`) and training results are **not tracked in Git** (see `.gitignore`)
- **Datasets**: Some datasets are downloaded automatically by the notebooks; others may need manual download
- **GPU Memory**: Deep learning experiments (especially transformer fine-tuning) require significant GPU memory
- **Checkpointing**: Training checkpoints are saved locally but excluded from version control

## 🔑 Key Learnings

### Machine Learning
- Feature engineering and preprocessing pipelines
- Model selection and hyperparameter tuning
- Cross-validation strategies
- Handling imbalanced datasets
- Ensemble methods and stacking

### Deep Learning
- PyTorch fundamentals: tensors, autograd, custom layers
- CNN architectures and design patterns
- Transfer learning: when and how to use pre-trained models
- Data augmentation techniques for computer vision
- Training stability: batch normalization, dropout, learning rate scheduling

### NLP & Transformers
- **Tokenization**: Subword tokenization (WordPiece, BPE), special tokens
- **Attention Mechanisms**: Self-attention, multi-head attention, positional encodings
- **BERT Architecture**: Bidirectional context, [CLS] token for classification
- **GPT-2 & Causal LM**: Autoregressive generation, top-k/top-p sampling
- **LLAMA & Mistral**: Modern open-source LLMs, instruction tuning
- **Fine-tuning**: Task-specific heads, LoRA, QLoRA, learning rate strategies
- **RAG Systems**: Document retrieval, embedding-based search, context injection
- **Quantization**: 4-bit/8-bit quantization for memory efficiency
- **Code Generation**: LLM-based code synthesis, automated testing, quality metrics
- **HuggingFace Ecosystem**: `transformers`, `datasets`, `Trainer` API, `PEFT`, `bitsandbytes`

## 📊 Datasets Used

- **CIFAR-10**: 60k 32x32 color images in 10 classes
- **Oxford Flowers-102**: 102 flower categories with 8k images
- **Caltech-101**: 101 object categories for image classification
- **Text datasets**: Various for NLP tasks (classification, MLM, etc.)

## 🛠️ Technologies

- **Languages**: Python
- **DL Framework**: PyTorch
- **ML Libraries**: scikit-learn, XGBoost, LightGBM
- **NLP/LLMs**: HuggingFace Transformers, tokenizers, PEFT, bitsandbytes
- **Code Quality**: pytest, ipytest, Radon (complexity analysis)
- **Audio/Vision**: librosa, torchvision, timm
- **Visualization**: Matplotlib, Seaborn
- **Data**: Pandas, NumPy
- **Notebook**: Jupyter Lab/Notebook

## 📈 Progress Tracking

- ✅ Classical ML fundamentals (regression, clustering, classification)
- ✅ PyTorch basics and CNN training
- ✅ Transfer learning for computer vision
- ✅ Attention mechanisms and T5 architecture
- ✅ BERT fine-tuning and text classification
- ✅ GPT-2 and causal language modeling
- ✅ LLAMA and Mistral model deployment
- ✅ Quantization and LoRA for efficient fine-tuning
- ✅ RAG (Retrieval-Augmented Generation) systems
- ✅ Agentic code generation with LLMs
- ✅ Advanced vision architectures
- ✅ Audio processing and keyword spotting
- ✅ GANs for generative tasks

## 📝 Best Practices

1. **Reproducibility**: Set random seeds (`torch.manual_seed()`, `np.random.seed()`)
2. **Data Splits**: Never peek at test data; use validation for hyperparameter tuning
3. **Version Control**: Large model files are gitignored; use separate storage (DVC, Git LFS)
4. **Variable Naming**: Use descriptive names (`X_train_scaled` not `X2`)
5. **Documentation**: Add markdown cells explaining methodology and results

## 🤝 Contributing

This is a personal learning repository, but suggestions and discussions are welcome! Feel free to open issues for questions or recommendations.

## 📧 Contact

**Author**: Luigi Gonnella  
**Repository**: [From-0-to-LLM](https://github.com/LuigiGonnella/From-0-to-LLM)

---

*Last updated: December 2025*
