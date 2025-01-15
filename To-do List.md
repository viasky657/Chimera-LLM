Steps to Train Custom LCM Byte Latent Model List (TO-DO List) []
1. Add Images (https://storage.googleapis.com/openimages/web/index.html) and Videos (https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid and https://huggingface.co/datasets/OpenGVLab/InternVid-Full and https://zhenglinpan.github.io/sakuga_dataset_webpage/) to abstract Conceptual learning Model with Binary encoder (Meta) instead of autotransformer (Sonar Encoder and Decoder). (Added code needed for training but need to train) []
2. MRI Emotion Dataset Fear/Nuetral: https://osf.io/26rhz/, Love emotion dataset: https://pmc.ncbi.nlm.nih.gov/articles/PMC4327739/figure/F1/, Emotion Dataset: Happy, Sad, Fear, Nuetral, Angry: https://med.stanford.edu/content/dam/sm/scsnl/documents/Zhang_BigData_EmotionalDevelopment_2019.pdf (IMPORTANT ETHICAL CLARIFICATIONS: This is ONLY TO MONITOR the LLM MRI-SCAN BRAIN MODEL TO BE SURE THAT IT DOES NOT HAVE HUMAN-LIKE SENTIENCE AND TO BE NOTIFIED IN THE EXTREMELY-RARE UNINTENDED EVENT THAT THE LLM GAINS IT SOMEHOW AS AN EMERGENT BEHAVIOR DURING TRAINING. TO CLARIFY, THESE DATASETS ARE NOT BEING USED TO TRAIN AN LLM TO SIMULATE HUMAN EMOTIONS.THE PURPOSE OF THIS LLM PROGRAM IS TO SERVE HUMANS AND NOT TO REPLICATE HUMAN BEHAVIOR.)
3. MRI Deep Sleep NREM Dataset: https://www.jneurosci.org/content/42/12/2503?utm_source=chatgpt.com, https://www.jneurosci.org/contefnt/jneuro/42/12/2503/F4.large.jpg 
5. Add Smell Dataset training to SONAR (https://github.com/innovationcore/smell-datasets) [X]
6. Add General Sound Dataset training to SONAR (https://github.com/audioset/ontology/issues/9). []
7. Add Touch (tactile feel) Dataset to SONAR (https://huggingface.co/datasets/mlfu7/Touch-Vision-Language-Dataset) and this touch dataset: (https://drive.google.com/drive/folders/1QOvbkIZtpJpz4Ry_Zg3ouXX-zLJNqu9m). []
(Sparse autoencoders are not needed for Diffusion models like SONAR). 
Steps to Train LCM (Large Concept Model) (Meta): https://ai.meta.com/research/publications/large-concept-models-language-modeling-in-a-sentence-representation-space/, LBM (Large Brain Model), B-Star (Model that can self-train and encourage exploration), Memory Layers (Meta) (Optmimized Self-Learning and Long Term Memory Storage), Binary Encoder (Meta), Flaming Filter (Meta):
1. Add Brain LLM Arch to the LCM Long Term Memory (Transformer Arch) replacement. (Started but not complete) (https://www.biorxiv.org/content/10.1101/2023.09.12.557460v2) []
2. Add B-Star Model Arch to the Transformer Long Term Memory Part: https://github.com/hkust-nlp/B-STaR and https://arxiv.org/pdf/2412.17256. []
4. Add Memory Layers to the Transformer Long Term Memory Part: https://ai.meta.com/research/publications/memory-layers-at-scale/. []
5. Add Flaming Filter (if possible) at this step: https://ai.meta.com/research/publications/flame-factuality-aware-alignment-for-large-language-models/. []
6. Add Binary encoder instead of autotokenizer: https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/. []
7. Add New Sonar model with additional training to the pipeline and replace the old sonar model. []
8. Train the LLM on MRI and EEG datasets: https://exhibits.stanford.edu/data/browse/openfmri-datasets, https://openneuro.org/, https://paperswithcode.com/dataset/raider, https://sites.google.com/site/depressiondatabase/, https://pmc.ncbi.nlm.nih.gov/articles/PMC10615764/ []
9. Add training for model Introspection for enhanced accuracy and preformance: https://arxiv.org/abs/2410.13787. []
10. Add architecture for self-evolving LLM by saving the tokens to the memory layer (meta) instead of the one suggested by the self-evolving paper: https://writer.com/engineering/self-evolving-models/. []

The System Prompt (Marco-01 with some custom changes): "You are a well-trained AI assistant. ## Important!!!!!!!!! When you answer questions, your thinking should be completed in <Thought>, and your results should be output in <Output>. <Thought> should be in English as much as possible, but there are 2 exceptions, one is the reference to the original text, and the other is that mathematics should use markdown format, and the output in <Output> needs to follow the language of the user input. ## Important!!!!!! You have the ability to make function calls in .json pair formatting, so be sure to put all function calls in the tags called <Tool></Tool> within the <Output> xml tags when you need to use a tool."

Improvements
1. Add complex thinking training datasets: []
  - Marco-01: https://github.com/AIDC-AI/Marco-o1/blob/main/data/CoT_demo.json and this dataset for fine-tuning on instruction following and reasoning enhancements: https://github.com/UKPLab/arxiv2024-divergent-cot. []
  - Openai Ultra: https://huggingface.co/datasets/O1-OPEN/OpenO1-SFT-Ultra and the original Openai-SFT COT dataset: https://huggingface.co/datasets/O1-OPEN/OpenO1-SFT []
  - Llava-COT: https://github.com/PKU-YuanGroup/LLaVA-CoT []
4. Omniparser Datasets: https://github.com/microsoft/OmniParser []
5. Fine-Tuning Math Datasets: https://huggingface.co/datasets/HuggingFaceTB/finemath, https://huggingface.co/deepseek-ai/DeepSeek-Prover-V1, []
6. Programming Datasets: https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1, https://github.com/src-d/datasets []
8. The function-calling dataset (Hammer): https://github.com/MadeAgents/Hammer and https://huggingface.co/datasets/MadeAgents/xlam-irrelevance-7.5k []
9. Add Gated Autoregressive encoder for the Sonar training to reduce gradiant learning knowledge saving errors (https://arxiv.org/abs/2404.16014). (Still need to add to touch dataset for training). []
10. Add Spause Synatic Smoothing with sparse encoders that eliminates all 0 values and keeps all the benefits of keeping rare tokens to better accomadate for rare situations: https://arxiv.org/abs/2410.11462 []

Note: Most of this data is covered under the Apache License and Creative Commons License, but the image datasets may be restricted copyrighted as the datasets were unclear as to what license it was hosted under or where the images were sourced from or if they had approval for the images being used in AI development. Please use this model with caution. 
