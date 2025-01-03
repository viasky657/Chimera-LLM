Steps to Train Sonar Model List (TO-DO List)
1. Add Images (https://storage.googleapis.com/openimages/web/index.html and https://paperswithcode.com/dataset/stylegallery and Videos (https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid and https://huggingface.co/datasets/OpenGVLab/InternVid-Full and https://zhenglinpan.github.io/sakuga_dataset_webpage/) to abstract Conceptual learning Model with Binary encoder (Meta) instead of autotransformer (Sonar Encoder and Decoder). (Added code needed for training but need to train)
2. Add Gated Autoregressive encoder for the Sonar training to reduce gradiant learning knowledge saving errors (https://arxiv.org/abs/2404.16014).
3. Add Spause Synatic Smoothing with sparse encoders that eliminates all 0 values and keeps all the benefits of keeping rare tokens to better accomadate for rare situations: https://arxiv.org/abs/2410.11462
5. Add Smell Dataset training to SONAR (https://github.com/innovationcore/smell-datasets)
6. Add Touch (tactile feel) Dataset to SONAR (https://huggingface.co/datasets/mlfu7/Touch-Vision-Language-Dataset) and this touch dataset: (https://drive.google.com/drive/folders/1QOvbkIZtpJpz4Ry_Zg3ouXX-zLJNqu9m)

Steps to Train LCM (Large Concept Model) (Meta): https://ai.meta.com/research/publications/large-concept-models-language-modeling-in-a-sentence-representation-space/, LBM (Large Brain Model), B-Star (Model that can self-train and encourage exploration), Memory Layers (Meta) (Optmimized Self-Learning and Long Term Memory Storage), Binary Encoder (Meta), Flaming Filter (Meta):
1. Add Brain LLM Arch to the LCM Long Term Memory (Transformer Arch) replacement. (Started but not complete) (https://www.biorxiv.org/content/10.1101/2023.09.12.557460v2)
2. Add B-Star Model Arch to the Transformer Long Term Memory Part: https://github.com/hkust-nlp/B-STaR and https://arxiv.org/pdf/2412.17256. 
3. Add Memory Layers to the Transformer Long Term Memory Part: https://ai.meta.com/research/publications/memory-layers-at-scale/. 
4. Add Flaming Filter (if possible) at this step: https://ai.meta.com/research/publications/flame-factuality-aware-alignment-for-large-language-models/. 
5. Add Binary encoder instead of autotokenizer: https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/.
6. Add New Sonar model with additional training to the pipeline and replace the old sonar model. 
7. Train the LLM on MRI and EEG datasets: https://exhibits.stanford.edu/data/browse/openfmri-datasets, https://openneuro.org/, https://paperswithcode.com/dataset/raider, https://sites.google.com/site/depressiondatabase/, https://pmc.ncbi.nlm.nih.gov/articles/PMC10615764/
9. Add training for model Introspection for enhanced accuracy and preformance: https://arxiv.org/abs/2410.13787.
10. Add architecture for self-evolving LLM by saving the tokens to the memory layer (meta) instead of the one suggested by the self-evolving paper: https://writer.com/engineering/self-evolving-models/.
11. Add EEG Dataset (Taste, Hearing, etc.) (https://github.com/meagmohit/EEG-Datasets)

Improvements
1. Add complex thinking training datasets:
  - Marco-01: https://github.com/AIDC-AI/Marco-o1/blob/main/data/CoT_demo.json
  - Openai Ultra: https://huggingface.co/datasets/O1-OPEN/OpenO1-SFT-Ultra
  - Llava-COT: https://github.com/PKU-YuanGroup/LLaVA-CoT
2. EEG and Free Will Study(?)
3. Qwen Audio Datasets: https://github.com/QwenLM/Qwen-Audio
4. Omniparser Datasets: https://github.com/microsoft/OmniParser
5. Fine-Tuning Math Datasets: https://huggingface.co/datasets/HuggingFaceTB/finemath, https://huggingface.co/deepseek-ai/DeepSeek-Prover-V1,
6. Programming Datasets: https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1, https://github.com/src-d/datasets
7. The Pile (General Knowledge Large Dataset): https://pile.eleuther.ai/

