# Awesome Multi-Vector Retrieval

<!-- ascii-art:start -->
```text
    _                                           __  ____     ______
   / \__      _____  ___  ___  _ __ ___   ___  |  \/  \ \   / /  _ \
  / _ \ \ /\ / / _ \/ __|/ _ \| '_ ` _ \ / _ \ | |\/| |\ \ / /| |_) |
 / ___ \ V  V /  __/\__ \ (_) | | | | | |  __/ | |  | | \ V / |  _ <
/_/   \_\_/\_/ \___||___/\___/|_| |_| |_|\___| |_|  |_|  \_/  |_| \_\
```
<!-- ascii-art:end -->

_Generated with [pyfiglet](https://github.com/pwaller/pyfiglet)._

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/license-MIT-f59e0b)](LICENSE)
[![Topic](https://img.shields.io/badge/topic-Multi--Vector_Retrieval-e11d48)](#what-counts-as-multi-vector-retrieval)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-10b981)](#how-to-contribute)

A curated, taxonomy-driven reading list for **multi-vector retrieval (MVR)**, **late-interaction retrieval**, and **vector-set search** across text, multilingual, visual-document, image, video, and omni-modal settings.

## Recommended reading path

New to MVR? Follow the field from modeling fundamentals to scalable execution, then finish with multimodal extensions and critical analysis:

1. **Foundations —** start with ColBERT, ColBERTv2, and XTR in [Representation learning and interaction](#1-representation-learning-and-interaction) to understand token-level late interaction, MaxSim, supervision, and token retrieval.
2. **Representation capacity —** continue with Poly-encoders, multi-view retrieval, and ConstBERT in [Fixed-budget, multi-view, and aspect representations](#2-fixed-budget-multi-view-and-aspect-representations) to see how a controlled vector budget can preserve multiple meanings.
3. **Efficiency —** read token pruning, pooling, quantization, and merging work in [Compression, pooling, and pruning](#3-compression-pooling-and-pruning) to understand the storage–quality trade-off.
4. **Scale —** move to PLAID, DESSERT, MUVERA, WARP, and graph/GPU engines in [Indexing, approximate search, and retrieval systems](#4-indexing-approximate-search-and-retrieval-systems) to learn how MVR runs over large collections.
5. **Beyond text —** study FILIP, FLMR, ColPali, Video-ColBERT, and OmniRet in [Multimodal and visual retrieval](#5-multimodal-and-visual-retrieval) to follow late interaction from image patches to video, audio, and visual documents.
6. **Understand and verify —** finish with white-box analyses, theoretical capacity, robustness, and replication studies in [Analysis, theory, and reproducibility](#6-analysis-theory-and-reproducibility).

## Contents

- [Awesome Multi-Vector Retrieval](#awesome-multi-vector-retrieval)
  - [Recommended reading path](#recommended-reading-path)
  - [Contents](#contents)
  - [Scope and selection policy](#scope-and-selection-policy)
    - [What counts as multi-vector retrieval?](#what-counts-as-multi-vector-retrieval)
  - [Taxonomy](#taxonomy)
  - [Papers](#papers)
    - [1. Representation learning and interaction](#1-representation-learning-and-interaction)
    - [2. Fixed-budget, multi-view, and aspect representations](#2-fixed-budget-multi-view-and-aspect-representations)
    - [3. Compression, pooling, and pruning](#3-compression-pooling-and-pruning)
    - [4. Indexing, approximate search, and retrieval systems](#4-indexing-approximate-search-and-retrieval-systems)
    - [5. Multimodal and visual retrieval](#5-multimodal-and-visual-retrieval)
    - [6. Analysis, theory, and reproducibility](#6-analysis-theory-and-reproducibility)
  - [How to contribute](#how-to-contribute)

## Scope and selection policy

### What counts as multi-vector retrieval?

This list uses a representation-and-scoring definition. A query and/or a candidate is represented by a **set or short sequence of vectors**, the two sides can be encoded independently, and relevance is computed from fine-grained vector interactions. The canonical late-interaction score is **MaxSim / Chamfer similarity**:

$$
s(Q,D)=\sum_{i=1}^{|Q|}\max_{1\le j\le |D|}\mathrm{sim}(q_i,d_j),
$$

but the scope also includes learned alignment, weighted MaxSim, set-to-set search, fixed-cardinality latent vectors, multi-view/aspect representations, and hybrid single-/multi-vector pipelines.

The emphasis is the modern neural MVR lineage (roughly 2020 onward) and general vector-set or multi-reference search work that directly advances these workloads. Classical local-feature image matching and multi-interest recommendation are neighboring literatures rather than the primary scope of this list.

Included:

- token-, phrase-, span-, segment-, entity-, aspect-, patch-, frame-, or modality-level representations;
- model, objective, and scoring innovations whose main contribution is multi-vector retrieval;
- compression, pruning, quantization, indexing, ANN, execution-engine, and hardware work designed for vector sets or late interaction;
- multilingual, domain-specific, multimodal, visual-document, image, and video retrieval;
- theory, interpretability, empirical analysis, and reproducibility studies centered on MVR.

Not included by default:

- ordinary single-vector bi-encoders or dense retrievers;
- pure cross-encoder rerankers that cannot precompute the candidate representation;
- vector databases that merely expose several named vector fields without a retrieval algorithm, cost model, or systems contribution specific to multi-vector workloads;
- downstream RAG papers that only use an existing ColBERT-style retriever without studying or improving MVR itself.

## Taxonomy

The field is easier to navigate by separating **what is represented**, **how representations interact**, and **how search is executed**. A modality-only taxonomy would mix model and systems contributions, while a ColBERT-only taxonomy would miss fixed-budget multi-view and general vector-set work.

| Branch | Central question | Typical design space |
|---|---|---|
| Representation learning and interaction | How should vector sets be learned and scored? | token vectors, MaxSim, sparse alignment, lexical routing, token weighting |
| Fixed-budget, multi-view, and aspect representations | Can a small number of vectors preserve several meanings or regions? | learned views, aspect vectors, entity views, latent/vector budgets, multi-layer views |
| Compression, pooling, and pruning | How can storage and computation shrink without losing fine-grained evidence? | residual/PQ compression, token pruning, clustering, pooling, merging |
| Indexing, approximate search, and systems | How can vector-set similarity be searched at collection scale? | centroid indexes, inverted lists, graphs, FDEs, multi-reference ANN, query planning, SSD/GPU execution, kernels |
| Multimodal and visual retrieval | How should text interact with patches, frames, audio, or other modalities? | visual documents, image-text, video, omni-modal tokens, hybrid pipelines |
| Analysis, theory, and reproducibility | Why does late interaction work, fail, or reproduce? | interpretability, expressivity, length bias, robustness, controlled reproduction |

## Papers

### 1. Representation learning and interaction

Core architectures, objectives, routing mechanisms, multilingual/domain adaptations, and alternatives to uniform Sum-of-Max scoring.

| Paper | Venue | Authors | Institute | Keywords |
|---|---|---|---|---|
| [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://doi.org/10.1145/3397271.3401075) | SIGIR20 | Omar Khattab | Stanford University | late interaction; token embeddings; MaxSim; first-stage retrieval |
| [Sparse, Dense, and Attentional Representations for Text Retrieval](https://aclanthology.org/2021.tacl-1.20/) | TACL21 | Yi Luan | Google Research | ME-BERT; representation capacity; sparse/dense/attentional comparison |
| [COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List](https://aclanthology.org/2021.naacl-main.241/) | NAACL21 | Luyu Gao | Carnegie Mellon University | contextualized inverted lists; exact lexical routing; token interaction |
| [Baleen: Robust Multi-Hop Reasoning at Scale via Condensed Retrieval](https://proceedings.neurips.cc/paper/2021/hash/e8b1cbd05f6e6a358a81dee52493dd06-Abstract.html) | NeurIPS21 | Omar Khattab | Stanford University | multi-hop retrieval; condensed context; focused late interaction; weak supervision |
| [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://aclanthology.org/2022.naacl-main.272/) | NAACL22 | Keshav Santhanam | Stanford University | residual compression; denoised supervision; out-of-domain retrieval; LoTTE |
| [Transfer Learning Approaches for Building Cross-Language Dense Retrieval Models](https://doi.org/10.1007/978-3-030-99736-6_26) | ECIR22 | Suraj Nair | University of Maryland; HLTCOE, Johns Hopkins University | ColBERT-X; cross-language IR; XLM-R; translate-train |
| [Multi-Vector Retrieval as Sparse Alignment](https://arxiv.org/abs/2211.01267) | arXiv22 | Yujie Qian, Vincent Y. Zhao | Google Research | AligneR; sparse token alignment; unary salience; interpretable matching |
| [ColBERT-PRF: Semantic Pseudo-Relevance Feedback for Dense Passage and Document Retrieval](https://doi.org/10.1145/3572405) | TWEB23 | Xiao Wang | University of Glasgow | pseudo-relevance feedback; query expansion; feedback embeddings; clustering |
| [COILcr: Efficient Semantic Matching in Contextualized Exact Match Retrieval](https://doi.org/10.1007/978-3-031-28244-7_19) | ECIR23 | Zhen Fan | Carnegie Mellon University | canonical representations; contextual exact match; factorized semantics; efficiency |
| [CITADEL: Conditional Token Interaction via Dynamic Lexical Routing for Efficient and Effective Multi-Vector Retrieval](https://aclanthology.org/2023.acl-long.663/) | ACL23 | Minghan Li | University of Waterloo | dynamic lexical routing; conditional interaction; learned keys; efficiency |
| [LI-RAGE: Late Interaction Retrieval Augmented Generation with Explicit Signals for Open-Domain Table Question Answering](https://aclanthology.org/2023.acl-short.133/) | ACL23 | Weizhe Lin | Amazon Alexa AI; University of Cambridge | table retrieval; TableQA; joint retriever-reader training; explicit relevance |
| [SLIM: Sparsified Late Interaction for Multi-Vector Retrieval with Inverted Indexes](https://doi.org/10.1145/3539618.3591977) | SIGIR23 | Minghan Li | University of Waterloo | sparse token vectors; inverted index; Lucene; two-stage retrieval |
| [Rethinking the Role of Token Retrieval in Multi-Vector Retrieval](https://proceedings.neurips.cc/paper_files/paper/2023/hash/31d997278ee9069d6721bc194174bb4c-Abstract-Conference.html) | NeurIPS23 | Jinhyuk Lee | Google DeepMind | XTR; token retrieval; missing-similarity imputation; scoring objective |
| [M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation](https://aclanthology.org/2024.findings-acl.137/) | Findings-ACL24 | Jianlv Chen, Defu Lian, Zheng Liu | University of Science and Technology of China; Beijing Academy of Artificial Intelligence | BGE-M3; multilingual; dense/sparse/multi-vector; self-distillation |
| [Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever](https://aclanthology.org/2024.mrl-1.11/) | MRL@EMNLP24 | Rohan Jha | University of Texas at Austin | multilingual retrieval; long context; Matryoshka dimensions; training framework |
| [ColBERT-XM: A Modular Multi-Vector Representation Model for Zero-Shot Multilingual Information Retrieval](https://aclanthology.org/2025.coling-main.295/) | COLING25 | Antoine Louis | Maastricht University | modular multilingual retrieval; language adapters; zero-shot CLIR |
| [XTR meets ColBERTv2: Adding ColBERTv2 Optimizations to XTR](https://aclanthology.org/2025.coling-industry.30/) | COLING-Industry25 | Riyaz Ahmed Bhat | IBM Research India | XTR; ColBERTv2; residual compression; single-stage retrieval |
| [SCV: Light and Effective Multi-Vector Retrieval with Sequence Compressive Vectors](https://aclanthology.org/2025.coling-industry.63/) | COLING-Industry25 | Cheoneum Park | Hanbat National University | span-compressive vectors; coarse-to-fine search; sequence compression |
| [JaColBERTv2.5: Optimising Multi-Vector Retrievers to Create State-of-the-Art Japanese Retrievers with Constrained Resources](https://doi.org/10.5715/jnlp.32.176) | J. Nat. Lang. Process.25 | Benjamin Clavié | Answer.AI | Japanese retrieval; low-resource training; distillation; checkpoint merging |
| [Simple Projection Variants Improve ColBERT Performance](https://arxiv.org/abs/2510.12327) | arXiv25 | Benjamin Clavié | Mixedbread AI; National Institute of Informatics | projection heads; gradient flow; FFN/GLU blocks; residual connections |
| [Fantastic (small) Retrievers and How to Train Them: mxbai-edge-colbert-v0 Tech Report](https://arxiv.org/abs/2510.14880) | arXiv25 | Rikiya Takehi | Mixedbread AI; Waseda University | compact ColBERT; training ablations; long-context retrieval; edge deployment |
| [TRIAL: Token Relations and Importance Aware Late-interaction for Accurate Text Retrieval](https://aclanthology.org/2025.emnlp-main.854/) | EMNLP25 | Hyukkyu Kang, Wook-Shin Han | POSTECH | token relations; token importance; semantic units; weighted scoring |
| [Incorporating Token Importance in Multi-Vector Retrieval](https://ojs.aaai.org/index.php/AAAI/article/view/40566) | AAAI26 | Archish S | Microsoft Research | weighted Chamfer; IDF; few-shot weighting; frozen representations |
| [ColBERT-Zero: To Pre-train Or Not To Pre-train ColBERT Models](https://arxiv.org/abs/2602.16609) | LIR@ECIR26 | Antoine Chaffin, Luca Arnaboldi | LightOn; EPFL | multi-vector pretraining; contrastive learning; knowledge distillation; prompt alignment |
| [ColBERT-Att: Late-Interaction Meets Attention for Enhanced Retrieval](https://arxiv.org/abs/2603.25248) | arXiv26 | Raj Nath Patel | Huawei Research Center | attention weighting; token importance; late interaction; MaxSim |
| [ProtoCol: Late Interaction Retrieval for Protein Homolog Search](https://openreview.net/forum?id=n5HZ01EpNw) | GenBio@ICML26 | Gabrielle Cohn, Rohan Gumaste, Minh Hoang, Vihan Lakshman | Massachusetts Institute of Technology; Princeton University | protein homolog search; residue embeddings; MaxSim; specialized retrieval |
| [NumColBERT: Non-Intrusive Numeracy Injection for Late-Interaction Retrieval Models](https://arxiv.org/abs/2605.10109) | SIGIR26 | Haruki Fujimaki | University of Tsukuba | numerical gating; numerical contrastive learning; MaxSim compatibility |
| [PLAID-PRF: Pseudo-Relevance Feedback with Centroid-like Tokens in PLAID](https://eprints.gla.ac.uk/383294/) | SIGIR26 | Xiao Wang | University of International Business and Economics | pseudo-relevance feedback; centroid tokens; query expansion; PLAID |

### 2. Fixed-budget, multi-view, and aspect representations

Methods that use a small or structured collection of vectors to capture distinct meanings, regions, entities, layers, or views.

| Paper | Venue | Authors | Institute | Keywords |
|---|---|---|---|---|
| [Poly-encoders: Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring](https://openreview.net/forum?id=SkxgnnNFvH) | ICLR20 | Samuel Humeau | Facebook AI Research | learned context codes; fixed vector budget; attention pooling; response retrieval |
| [MuVER: Improving First-Stage Entity Retrieval with Multi-View Entity Representations](https://aclanthology.org/2021.emnlp-main.205/) | EMNLP21 | Xinyin Ma, Yong Jiang, Weiming Lu | Zhejiang University; DAMO Academy, Alibaba Group | entity retrieval; multi-view descriptions; first-stage retrieval; ANN |
| [Multi-View Document Representation Learning for Open-Domain Dense Retrieval](https://aclanthology.org/2022.acl-long.414/) | ACL22 | Shunyu Zhang | Beihang University | viewer tokens; global-local loss; max-over-views; open-domain QA |
| [Multi-Aspect Dense Retrieval](https://doi.org/10.1145/3534678.3539137) | KDD22 | Weize Kong | Google | aspect embeddings; e-commerce; controllable capacity; interpretability |
| [Dense Retrieval with Entity Views](https://doi.org/10.1145/3511808.3557285) | CIKM22 | Hai Dang Tran | Max Planck Institute for Informatics | entity views; entity-centric documents; multiple representations |
| [Learning Diverse Document Representations with Deep Query Interactions](https://arxiv.org/abs/2208.04232) | arXiv22 | Zehan Li | Beihang University | pseudo-queries; diverse views; query-informed representation; diversity |
| [Efficient Constant-Space Multi-Vector Retrieval](https://doi.org/10.1007/978-3-031-88714-7_22) | ECIR25 | Sean MacAvaney | University of Glasgow | ConstBERT; fixed vector budget; learned latent vectors; OS paging |
| [Investigating Multi-layer Representations for Dense Passage Retrieval](https://aclanthology.org/2025.findings-emnlp.1333/) | Findings-EMNLP25 | Zhongbin Xie | University of Oxford | encoder layers; multi-layer views; pooling; fixed-cardinality representations |

### 3. Compression, pooling, and pruning

Representation-side methods for reducing vector count, dimension, precision, or index footprint.

| Paper | Venue | Authors | Institute | Keywords |
|---|---|---|---|---|
| [Learned Token Pruning in Contextualized Late Interaction over BERT (ColBERT)](https://doi.org/10.1145/3477495.3531835) | SIGIR22 | Carlos Lassance | Naver Labs Europe | learned pruning; attention-based selection; index footprint |
| [Introducing Neural Bag of Whole-Words with ColBERTer: Contextualized Late Interactions using Enhanced Reduction](https://doi.org/10.1145/3511808.3557367) | CIKM22 | Sebastian Hofstätter | TU Wien | whole-word reduction; learned gating; hybrid retrieval; small dimensions |
| [Joint Optimization of Multi-vector Representation with Product Quantization](https://doi.org/10.1007/978-3-031-17120-8_49) | NLPCC22 | Yan Fang, Yiqun Liu | Tsinghua University | JMPQ; product quantization; joint optimization; index compression |
| [Static Pruning for Multi-Representation Dense Retrieval](https://doi.org/10.1145/3573128.3604896) | DocEng23 | Antonio Acquavia | University of Pisa | static pruning; low-IDF tokens; index-size reduction; query-independent |
| [Reducing the Footprint of Multi-Vector Retrieval with Minimal Performance Impact via Token Pooling](https://arxiv.org/abs/2409.14683) | arXiv24 | Benjamin Clavié | Answer.AI | hierarchical clustering; token pooling; drop-in compression; ColBERT |
| [Token Pruning Optimization for Efficient Multi-Vector Dense Retrieval](https://www.amazon.science/publications/token-pruning-optimization-for-efficient-multi-vector-dense-retrieval) | ECIR25 | Shanxiu He | University of California, Santa Barbara | adaptive pruning; Gumbel-Softmax; memory budget; zero-shot retrieval |
| [Towards Lossless Token Pruning in Late-Interaction Retrieval Models](https://doi.org/10.1145/3726302.3730100) | SIGIR25 | Yuxuan Zong | Sorbonne Université; CNRS; ISIR | dominance; pruning regularization; score preservation; token sparsity |
| [CRISP: Clustering Multi-Vector Representations for Denoising and Pruning](https://arxiv.org/abs/2505.11471) | arXiv25 | João Veneroso | Google | learned clusterability; end-to-end clustering; denoising; pruning |
| [Towards Storage-Efficient Visual Document Retrieval: An Empirical Study on Reducing Patch-Level Embeddings](https://aclanthology.org/2025.findings-acl.1003/) | Findings-ACL25 | Yubo Ma, Yuhang Zang | Nanyang Technological University; Shanghai AI Laboratory | Light-ColPali; patch merging; patch pruning; visual documents |
| [ReinPool: Reinforcement Learning Pooling Multi-Vector Embeddings for Retrieval System](https://arxiv.org/abs/2601.07125) | arXiv26 | Sungguk Cha | LG Uplus | reinforcement learning; vector filtering; learned pooling; visual documents |
| [Multi-Vector Index Compression in Any Modality](https://arxiv.org/abs/2602.21202) | SIGIR26 | Hanxiang Qin | Johns Hopkins University | attention-guided clustering; constant vector budget; omni-modal compression |
| [Comparing Token Pruning Approaches for Multi-Vector Retrieval](https://sigir2026.org/en-AU/pages/program/accepted-papers) | SIGIR26 | Ferdinand Schlatt | Friedrich Schiller University Jena | reproducibility; weighted pruning; IDF pruning; efficiency/effectiveness |
| [A Voronoi Cell Formulation for Principled Token Pruning in Late-Interaction Retrieval Models](https://arxiv.org/abs/2603.09933) | SIGIR26 | Yash Kankanampati | Université Sorbonne Paris Nord; CNRS; LIPN | Voronoi cells; geometric influence; principled pruning; interpretability |
| [ColBERTSaR: Sparsified ColBERT Index via Product Quantization](https://arxiv.org/abs/2606.05568) | SIGIR26 | Eugene Yang | Johns Hopkins University | residual-free PQ; inverted index; sparse MaxSim approximation; multilingual retrieval |
| [Sculpting the Vector Space: Towards Efficient Multi-Vector Visual Document Retrieval via Prune-then-Merge Framework](https://aclanthology.org/2026.findings-acl.1247/) | Findings-ACL26 | Yibo Yan, Xuming Hu | Hong Kong University of Science and Technology (Guangzhou); Alibaba Cloud Computing; Hong Kong University of Science and Technology | adaptive patch pruning; hierarchical merging; VDR; high compression |

### 4. Indexing, approximate search, and retrieval systems

Collection-scale candidate generation, set-search algorithms, indexes, execution engines, libraries, and hardware-aware kernels.

| Paper | Venue | Authors | Institute | Keywords |
|---|---|---|---|---|
| [PLAID: An Efficient Engine for Late Interaction Retrieval](https://doi.org/10.1145/3511808.3557325) | CIKM22 | Keshav Santhanam | Stanford University | centroid interaction; residual compression; staged pruning; engine |
| [DESSERT: An Efficient Algorithm for Vector Set Search with Vector Set Queries](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d6cc45de2e2dea14b96c1eba88fd8ef7-Abstract-Conference.html) | NeurIPS23 | Joshua Engels | ThirdAI | vector-set search; retrieval tables; randomized algorithm; guarantees |
| [Efficient Multi-vector Dense Retrieval with Bit Vectors](https://doi.org/10.1007/978-3-031-56060-6_1) | ECIR24 | Franco Maria Nardini, Cosimo Rulli | ISTI-CNR | EMVB; bit-vector filtering; SIMD; product quantization |
| [MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encoding](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b71cfefae46909178603b5bc6c11d3ae-Abstract-Conference.html) | NeurIPS24 | Laxman Dhulipala | Google Research; University of Maryland | fixed-dimensional encoding; MVR-to-MIPS reduction; approximation guarantees |
| [$\alpha$-Reachable Graphs for Multi-vector Nearest Neighbor Search](https://openreview.net/forum?id=v8jSxLHEE9) | VecDB@ICML25 | Siddharth Gollapudi, Ravishankar Krishnaswamy, Ben Landrum, Nikhil Rao, Kirankumar Shiragur, Sandeep Silwal, Harsh Wardhan | Microsoft Research | DiskANN; graph index; Chamfer distance; quasimetrics; approximation guarantees |
| [Storage Access Optimization for Efficient GPU-Centric Information Retrieval](https://link.springer.com/article/10.1007/s11227-025-07118-9) | J. Supercomputing25 | Susav Shrestha | Texas A&M University | ESPN; ESPN-LIVE; SSD offload; GPU Direct Storage; prefetching |
| [WARP: An Efficient Engine for Multi-Vector Retrieval](https://doi.org/10.1145/3726302.3729904) | SIGIR25 | Jan Luca Scheerer | ETH Zurich | XTR; dynamic similarity imputation; implicit decompression; C++ kernels |
| [IGP: Efficient Multi-Vector Retrieval via Proximity Graph Index](https://doi.org/10.1145/3726302.3730004) | SIGIR25 | Zheng Bian, Bo Tang | Southern University of Science and Technology; Hong Kong Polytechnic University | proximity graph; incremental greedy probe; candidate generation |
| [PyLate: Flexible Training and Retrieval for Late Interaction Models](https://doi.org/10.1145/3746252.3761608) | CIKM25 | Antoine Chaffin | LightOn | research library; Sentence Transformers; training; indexing; evaluation |
| [GEM: A Native Graph-based Index for Multi-Vector Retrieval](https://doi.org/10.1145/3802065) | SIGMOD26 | Yao Tian | Hong Kong University of Science and Technology | set-level graph; semantic shortcuts; beam search; quantized distance |
| [VecFlow-Chamfer: A GPU-based Data Management System for High-Performance Multi-Vector Search on Superchips](https://doi.org/10.1145/3786706) | SIGMOD26 | Chenghao Mo | SSAIL Lab, University of Illinois Urbana-Champaign | GPU data management; Chamfer search; MaxIVF; tiered memory |
| [LEMUR: Learned Multi-Vector Retrieval](https://arxiv.org/abs/2601.21853) | arXiv26 | Elias Jääsaari | University of Helsinki | learned reduction; latent single-vector search; ANNS; cross-modal evaluation |
| [Multivector Reranking in the Era of Strong First-Stage Retrievers](https://doi.org/10.1007/978-3-032-21324-2_4) | ECIR26 | Silvio Martinico | ISTI-CNR; University of Pisa | learned-sparse gathering; two-stage retrieval; reranking; compression |
| [Unified and Efficient Approach for Multi-Vector Similarity Search](https://arxiv.org/abs/2604.02815) | arXiv26 | Binhan Yang | Beihang University | MV-HNSW; native hierarchical graph; cardinality robustness; set search |
| [Breaking the Single-Reference-Vector Barrier in Approximate Nearest Neighbor Search](https://doi.org/10.1145/3774904.3792208) | WWW26 | Jiadong Xie, Yingfan Liu | Chinese University of Hong Kong; Xidian University | multi-reference ANN; all/any-kANN; graph index; set-to-point search |
| [MINT: Multi-Vector Search Index Tuning](https://arxiv.org/abs/2504.20018) | ICDE26 | Jiongli Zhu | University of California, San Diego | aligned multi-feature vectors; index tuning; query planning; HNSW/DiskANN |
| [No More K-means: Single-Stage Sparse Coding for Efficient Multi-Vector Retrieval](https://arxiv.org/abs/2605.30120) | ICML26 | Lixuan Guo, Chenyu You | Stony Brook University; Xidian University | SSR; sparse autoencoder; inverted index; clustering-free retrieval |
| [Efficient Multivector Retrieval with Token-Aware Clustering and Hierarchical Indexing](https://arxiv.org/abs/2604.28142) | SIGIR26 | Silvio Martinico | University of Pisa; ISTI-CNR | TACHIOM; token-aware clustering; hierarchical index; product quantization |
| [Scalable Multi-vector Retrieval for Policy Enforcement on E-commerce Marketplaces](https://sigir2026.org/en-AU/pages/program/accepted-papers) | SIGIR-Industry26 | Akshit Sarpal | Walmart Global Tech | policy enforcement; e-commerce; industrial scale; multi-vector retrieval |
| [GIGP+: A CPU-GPU Co-Processing Engine for Multi-Vector Retrieval](https://sigir2026.org/en-AU/pages/program/accepted-papers) | SIGIR26 | Zheng Bian, Bo Tang | Southern University of Science and Technology; Hong Kong Polytechnic University | CPU-GPU co-processing; proximity graph; heterogeneous scheduling |
| [TileMaxSim: IO-Aware GPU MaxSim Scoring with Dimension Tiling and Fused Product Quantization](https://arxiv.org/abs/2606.26439) | arXiv26 | Ashutosh Sharma | MIT-IBM Watson AI Lab | Triton; IO-aware tiling; fused PQ; GPU MaxSim kernel |

### 5. Multimodal and visual retrieval

Fine-grained interactions between text tokens and image patches, document pages, video frames, audio, or multiple modalities. Papers whose main novelty is generic compression or systems work are placed in Sections 3–4.

| Paper | Venue | Authors | Institute | Keywords |
|---|---|---|---|---|
| [FILIP: Fine-grained Interactive Language-Image Pre-Training](https://openreview.net/forum?id=cpDhcsEDC2) | ICLR22 | Lewei Yao, Hang Xu, Xiaodan Liang | Huawei Noah's Ark Lab; Hong Kong University of Science and Technology; Sun Yat-sen University | image patches; text tokens; cross-modal late interaction; pretraining |
| [Fine-grained Late-interaction Multi-modal Retrieval for Retrieval Augmented Visual Question Answering](https://proceedings.neurips.cc/paper_files/paper/2023/hash/47393e8594c82ce8fd83adc672cf9872-Abstract-Conference.html) | NeurIPS23 | Weizhe Lin | University of Cambridge | FLMR; KB-VQA; cross-modal tokens; knowledge retrieval |
| [MUST: An Effective and Scalable Framework for Multimodal Search of Target Modality](https://doi.org/10.1109/ICDE60146.2024.00361) | ICDE24 | Mengzhao Wang | Zhejiang University | multi-reference query; multimodal fusion; weighted similarity; proximity graph |
| [PreFLMR: Scaling Up Fine-Grained Late-Interaction Multi-modal Retrievers](https://aclanthology.org/2024.acl-long.289/) | ACL24 | Weizhe Lin | University of Cambridge | multimodal pretraining; M2KR; scaling; knowledge retrieval |
| [ColPali: Efficient Document Retrieval with Vision Language Models](https://openreview.net/forum?id=ogjBpZ8uSi) | ICLR25 | Manuel Faysse | Illuin Technology; CentraleSupélec, Université Paris-Saclay | OCR-free retrieval; visual documents; patch embeddings; ViDoRe |
| [Video-ColBERT: Contextualized Late Interaction for Text-to-Video Retrieval](https://openaccess.thecvf.com/content/CVPR2025/html/Reddy_Video-ColBERT_Contextualized_Late_Interaction_for_Text-to-Video_Retrieval_CVPR_2025_paper.html) | CVPR25 | Arun Reddy | Johns Hopkins University Applied Physics Laboratory | text-to-video; spatial-temporal tokens; MeanMaxSim; visual expansion |
| [CLaMR: Contextualized Late-Interaction for Multimodal Content Retrieval](https://arxiv.org/abs/2506.06144) | arXiv25 | David Wan | University of North Carolina at Chapel Hill | multimodal video; dynamic modality selection; frames; speech; OCR; metadata |
| [Llama Nemoretriever Colembed: Top-Performing Text-Image Retrieval Model](https://arxiv.org/abs/2507.05513) | arXiv25 | Mengyao Xu | NVIDIA | VLM retriever; bidirectional attention; visual documents; ColBERT scoring |
| [ColMate: Contrastive Late Interaction and Masked Text for Multimodal Document Retrieval](https://aclanthology.org/2025.emnlp-industry.145/) | EMNLP-Industry25 | Ahmed Masry | ServiceNow Research; York University | OCR pretraining; masked contrastive learning; visual documents; late-interaction scoring |
| [LILaC: Late Interacting in Layered Component Graph for Open-domain Multimodal Multihop Retrieval](https://aclanthology.org/2025.emnlp-main.1037/) | EMNLP25 | Joohyung Yun, Wook-Shin Han | POSTECH | layered component graph; multimodal multihop; subgraph retrieval; late interaction |
| [MetaEmbed: Scaling Multimodal Retrieval at Test-Time with Flexible Late Interaction](https://openreview.net/forum?id=yKDqg9HwZX) | ICLR26 | Zilin Xiao | Meta Superintelligence Labs; Rice University | meta tokens; Matryoshka multi-vector training; test-time scaling; MMEB |
| [Nemotron ColEmbed V2: Top-Performing Late Interaction Embedding Models for Visual Document Retrieval](https://openreview.net/forum?id=CkxZgxb87Q) | LIR@ECIR26 | Gabriel de Souza P. Moreira | NVIDIA | visual documents; VLM model family; hard negatives; model merging |
| [Efficient and High-Fidelity Omni Modality Retrieval](https://openaccess.thecvf.com/content/CVPR2026/html/Huynh_Efficient_and_High-Fidelity_Omni_Modality_Retrieval_CVPR_2026_paper.html) | CVPR26 | Chuong Huynh | University of Maryland, College Park | OmniRet; text/image/audio; fixed-size resampling; Wasserstein pooling |
| [Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy](https://aclanthology.org/2026.findings-acl.54/) | Findings-ACL26 | Juyeon Kim, Taeuk Kim, Kijung Shin | KAIST; Hanyang University | HEAVEN; hybrid two-stage retrieval; VIMDOC; query-token filtering |
| [Visual RAG at Scale: Tile-Level Spatial Pooling for Efficient Multi-Vector Document Retrieval](https://arxiv.org/abs/2602.12510) | SIGIR-Demo26 | Ara Yeroyan | Independent Researcher | visual RAG toolkit; spatial pooling; multi-stage search; MaxSim reranking |
| [Argus-Retriever: Vision-LLM Late-Interaction Retrieval with Region-Aware Query-Conditioned MoE for Visual Document Retrieval](https://arxiv.org/abs/2606.04300) | arXiv26 | Abdelrahman Abdallah | University of Innsbruck | query-conditioned documents; region-aware MoE; visual documents; agentic retrieval |

### 6. Analysis, theory, and reproducibility

Mechanistic and theoretical accounts of late interaction, plus controlled comparisons and reproduction studies.

| Paper | Venue | Authors | Institute | Keywords |
|---|---|---|---|---|
| [A White Box Analysis of ColBERT](https://doi.org/10.1007/978-3-030-72240-1_23) | ECIR21 | Thibault Formal | LIP6, Sorbonne Université; Naver Labs Europe | interpretability; exact matching; token contributions; white-box analysis |
| [On Single and Multiple Representations in Dense Passage Retrieval](https://ceur-ws.org/Vol-2947/paper5.pdf) | IIR-Workshop21 | Craig Macdonald | University of Glasgow | single vs. multi-vector; effectiveness/latency; query difficulty |
| [Reproducibility, Replicability, and Insights into Dense Multi-Representation Retrieval Models: from ColBERT to Col*](https://doi.org/10.1145/3539618.3591916) | SIGIR23 | Xiao Wang | University of Glasgow | reproducibility; ColBERT; ColBERT-PRF; Col*; implementation insights |
| [Beneath the \[MASK\]: An Analysis of Structural Query Tokens in ColBERT](https://doi.org/10.1007/978-3-031-56063-7_35) | ECIR24 | Ben Giacalone | Rochester Institute of Technology | structural tokens; query augmentation; interpretability; token order |
| [An Analysis on Matching Mechanisms and Token Pruning for Late-interaction Models](https://doi.org/10.1145/3639818) | TOIS24 | Qi Liu | Renmin University of China | co-occurrence; Sum-of-Max; document/query pruning; matching analysis |
| [Generative Retrieval as Multi-Vector Dense Retrieval](https://doi.org/10.1145/3626772.3657697) | SIGIR24 | Shiguang Wu | Shandong University | generative retrieval; alignment matrix; unified scoring; model equivalence |
| [A Reproducibility Study of PLAID](https://doi.org/10.1145/3626772.3657856) | SIGIR24 | Sean MacAvaney | University of Glasgow | PLAID parameters; Pareto frontier; BM25 reranking; cluster analysis |
| [Reproducibility, Replicability, and Insights into Visual Document Retrieval with Late Interaction](https://doi.org/10.1145/3726302.3730285) | SIGIR25 | Jingfen Qiao | University of Amsterdam | ColPali reproduction; query-patch matching; VLM backbones; scalability |
| [LIR: The First Workshop on Late Interaction and Multi Vector Retrieval @ ECIR 2026](https://doi.org/10.1007/978-3-032-21324-2_11) | ECIR26 | Benjamin Clavié | Mixedbread AI; National Institute of Informatics | research agenda; community; late interaction; multi-vector retrieval |
| [Diagnosable ColBERT: Debugging Late-Interaction Retrieval Models Using a Learned Latent Space as Reference](https://aclanthology.org/2026.bionlp-1.58/) | BioNLP@ACL26 | François Remy | Parallia Healthcare | biomedical retrieval; diagnostic latent space; concept grounding; debugging |
| [Working Notes on Late Interaction Dynamics: Analyzing Targeted Behaviors of Late Interaction Models](https://arxiv.org/abs/2603.26259) | LIR@ECIR26 | Antoine Edy | Illuin Technology | length bias; similarity distribution; causal vs. bidirectional encoders |
| [A Brief Comparison of Training-Free Multi-Vector Sequence Compression Methods](https://arxiv.org/abs/2603.22434) | LIR@ECIR26 | Rohan Jha | Johns Hopkins University | controlled comparison; token merging; token pruning; sequence compression |
| [Reproduction Beyond Benchmarks: ConstBERT and ColBERT-v2 Across Backends and Query Distributions](https://arxiv.org/abs/2604.09982) | SIGIR26 | Utshab Kumar Ghosh | Missouri University of Science and Technology | reproducibility; backend sensitivity; narrative queries; architectural robustness |
| [A Replicability Study of XTR](https://doi.org/10.1145/3805713.3820423) | ICTIR26 | Rohan Jha | Johns Hopkins University | XTR replication; score imputation; IVF engines; controlled comparison |
| [Spike Hijacking in Late-Interaction Retrieval](https://arxiv.org/abs/2604.05253) | arXiv26 | Karthik Suresh | Adobe | gradient concentration; hard MaxSim; document length; robustness |
| [Quantifying and Expanding the Theoretical Capacity of Late-Interaction Retrieval Models](https://arxiv.org/abs/2607.05803) | arXiv26 | Julian Killingback | Center for Intelligent Information Retrieval, University of Massachusetts Amherst | expressivity; Signed MaxSim; sparse inner products; logical functions |

## How to contribute

Pull requests are welcome. For a new entry, please:

1. Confirm that the paper satisfies the scope above and contributes to MVR itself, rather than merely using an off-the-shelf retriever.
2. Link the final proceedings/publisher page when one exists; use arXiv only when a final version cannot be verified.
3. Add exactly one row under the paper's **primary contribution** and use keywords for cross-cutting themes.
4. Follow the five-column schema: **Paper**, **Venue**, **Authors**, **Institute**, **Keywords**.
5. In **Authors**, start with the first author's full published name and append any explicitly designated corresponding authors with commas, omitting duplicate names. In **Institute**, list those authors' paper affiliations and remove duplicates.
6. Keep the venue label compact (`SIGIR25`, `Findings-ACL26`, `LIR@ECIR26`, or `arXiv26`) and preserve chronological ordering.

Corrections to venue status, author order, corresponding-author designation, affiliations, and final-version links are especially valuable for fast-moving 2026 work.
