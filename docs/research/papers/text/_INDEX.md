# Indice: papers/text

Indice generato; non modificare a mano.

## [FlashAttention](2205.14135-flashattention.md)

Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length.

- righe 8–9: **Testo estratto** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 10–1531: **FlashAttention : Fast and Memory-Efficient Exact Attention with IO-Awareness** — Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length.
        - righe 12–23: **Abstract** — Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length.
- righe 24–91: **1 Introduction** — Transformer models [ 82 ] have emerged as the most widely used architecture in applications such as natural language processing and image classification.
- righe 92–174: **2 Background** — We provide some background on the performance characteristics of common deep
  - righe 98–149: **2.1 Hardware Performance** — We focus here on GPUs.
  - righe 150–174: **2.2 Standard Attention Implementation** — Given input sequences where is the sequence length and
- righe 175–376: **3 FlashAttention : Algorithm, Analysis, and Extensions** — We show how to compute exact attention with fewer HBM reads/writes and without storing large intermediate matrices for the backward pass.
  - righe 187–250: **3.1 An Efficient Attention Algorithm With Tiling and Recomputation** — Given the inputs in HBM, we aim to compute the attention output and write it to HBM.
        - righe 246–250: **Theorem 1 .** — Algorithm 1 returns with FLOPs and
  - righe 251–327: **3.2 Analysis: IO Complexity of FlashAttention** — We analyze the IO complexity of FlashAttention , showing
        - righe 259–285: **Theorem 2 .** — Let be the sequence length, be the head dimension, and be size of
        - righe 286–327: **Proposition 3 .** — Let be the sequence length, be the head dimension, and be size of
  - righe 328–376: **3.3 Extension: Block-Sparse FlashAttention** — We extend FlashAttention to approximate attention:
        - righe 354–376: **Proposition 4 .** — Let be the sequence length, be the head dimension, and be size of
- righe 377–542: **4 Experiments** — We evaluate the impact of using FlashAttention to train Transformer models.
  - righe 406–467: **4.1 Faster Models with FlashAttention** — FlashAttention yields the fastest single-node BERT training speed that we know of.
      - righe 408–423: **BERT.** — FlashAttention yields the fastest single-node BERT training speed that we know of.
      - righe 424–438: **GPT-2.** — FlashAttention yields faster training times for GPT-2 [ 67 ] on the large OpenWebtext dataset [ 32 ] than the widely used HuggingFace [ 87 ] and Megatron-LM [ 77 ] implementations.
      - righe 439–467: **Long-range Arena.** — We compare vanilla Transformer (with either standard implementation or FlashAttention )
  - righe 468–519: **4.2 Better Models with Longer Sequences** — The runtime and memory-efficiency of FlashAttention allow us to increase the context length of
      - righe 470–482: **Language Modeling with Long Context.** — The runtime and memory-efficiency of FlashAttention allow us to increase the context length of
      - righe 483–497: **Long Document Classification.** — Training Transformers with longer sequences with FlashAttention improves performance on the MIMIC-III [ 47 ] and ECtHR [ 6 , 7 ] datasets.
      - righe 498–519: **Path-X and Path-256.** — The Path-X and Path-256 benchmarks are challenging tasks from the long-range arena benchmark designed to test long context.
  - righe 520–542: **4.3 Benchmarking Attention** — Figure 3 : Left: runtime of forward pass + backward pass.
      - righe 528–535: **Runtime.** — Figure 3 (left) reports the runtime in milliseconds of the forward + backward pass of FlashAttention and block-sparse FlashAttention compared to the baselines in exact, approximate, and sparse attention (exact numbers in…
      - righe 536–542: **Memory Footprint.** — Figure 3 (right) shows the memory footprint of FlashAttention and block-sparse FlashAttention compared to various exact, approximate, and sparse attention baselines.
- righe 543–574: **5 Limitations and Future Directions** — We discuss limitations of our approach and future directions.
    - righe 565–574: **Acknowledgments** — Our implementation uses Apex’s FMHA code ( https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/fmha ) as a starting point.
- righe 575–764: **References** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 765–847: **Appendix A Related Work** — IO-Aware Runtime Optimization.
- righe 848–1077: **Appendix B Algorithm Details** — We first derive the forward and backward passes of attention and show that
  - righe 859–895: **B.1 Memory-efficient forward pass** — The main challenge in making attention memory-efficient is the softmax that
  - righe 896–964: **B.2 Memory-efficient backward pass** — We derive the backward pass of attention and show that it can also be computed
  - righe 965–983: **B.3 FlashAttention : Forward Pass** — We describe the full details of FlashAttention forward pass.
  - righe 984–1034: **B.4 FlashAttention : Backward Pass** — We describe the full details of FlashAttention backward pass.
        - righe 1025–1034: **Theorem 5 .** — Let be the sequence length, be the head dimension, and be size of
  - righe 1035–1077: **B.5 Comparison with Rabe and Staats 2021** — We describe here some similarities and differences between our FlashAttention algorithm and the algorithm of Rabe and Staats 2021 .
- righe 1078–1237: **Appendix C Proofs** — We first count the number of FLOPs and extra memory required.
        - righe 1080–1137: **Proof of Theorem   1 .** — We first count the number of FLOPs and extra memory required.
        - righe 1138–1187: **Proof of Theorem   2 .** — We first analyze the IO complexity of standard attention implementation.
        - righe 1188–1200: **Proof of Proposition   3 .** — For contradiction, suppose that there exists an algorithm that computes
        - righe 1201–1237: **Proof of Theorem   5 .** — The IO complexity of the attention backward is very similar to the IO
- righe 1238–1301: **Appendix D Extension Details** — We describe the full block-sparse FlashAttention algorithm
  - righe 1240–1262: **D.1 Block-sparse FlashAttention** — We describe the full block-sparse FlashAttention algorithm
        - righe 1250–1262: **Proof of Proposition   4 .** — The proof is very similar to the proof of Theorem 2 .
  - righe 1263–1301: **D.2 Potential Extensions** — We discuss here a few potential extensions of the IO-aware approach to speed up
- righe 1302–1531: **Appendix E Full Experimental Results** — We train BERT-large following the training procedure and hyperparameters of the
  - righe 1304–1324: **E.1 BERT** — We train BERT-large following the training procedure and hyperparameters of the
  - righe 1325–1363: **E.2 GPT-2** — We use the standard implementations of
      - righe 1360–1363: **Long Document Classification.** — For MIMIC-III and ECtHR, we follow the hyperparameters of Dai et al.
  - righe 1364–1391: **E.3 LRA details** — We follow the hyperparameters from the Long-range arena
      - righe 1385–1391: **Path-X** — For Path-X and Path-256, we follow the hyperparameters from the PathFinder-32 experiments from the long-range arena paper [ 80 ] .
  - righe 1392–1430: **E.4 Comparison with Apex FMHA** — We compare our method/implementation with Apex FMHA
  - righe 1431–1467: **E.5 Speedup On Different Hardware and Configurations** — Speedup varies between different types of GPU types and generations depending on HBM bandwidth and SRAM size.
      - righe 1438–1444: **A100** — Figure 5 shows speedup on an A100 GPU with batch size 8, head dimension 64, and 12 attention heads, across different sequence lengths.
      - righe 1445–1453: **A100, Head Dimension 128** — Speedup also changes when we increase the head dimension.
      - righe 1454–1461: **RTX 3090** — Figure 7 shows speedup on an RTX 3090 GPU.
      - righe 1462–1467: **T4** — Figure 8 shows speedup on a T4 GPU.
  - righe 1468–1531: **E.6 Full Benchmarking Results** — We report the full benchmarking results and experimental details on A100.
      - righe 1472–1478: **Baselines** — We compare against reference implementations for exact attention from PyTorch/HuggingFace and Megatron, approximate attention, and sparse attention.
      - righe 1479–1500: **Setup** — We measure runtime and memory usage of the attention computation with 8 heads of dimension 64, and batch size 16 on a machine with one A100 GPU with 40 GB of GPU HBM.
      - righe 1501–1531: **Results** — Table 8 summarizes all the experimental configurations and contains pointers to the results tables.

## [GPTQ](2210.17323-gptq.md)

for Generative Pre-trained Transformers

- righe 8–9: **Testo estratto** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 10–438: **GPTQ: Accurate Post-Training Quantization** — for Generative Pre-trained Transformers
        - righe 14–20: **Abstract** — Generative Pre-trained Transformer models, known as GPT or OPT, set themselves apart through breakthrough performance across complex language modelling tasks, but also by their extremely high computational and storage co…
- righe 21–48: **1 Introduction** — Pre-trained generative models from the Transformer (Vaswani et al., 2017 ) family, commonly known as GPT or OPT (Radford et al., 2019 ; Brown et al., 2020 ; Zhang et al., 2022 ) , have shown breakthrough performance for …
      - righe 33–48: **Contribution.** — In this paper, we present a new post-training quantization method, called GPTQ, 1 1 1 This merges the name of the OPT model family with the abbreviation for post-training quantization (PTQ).
- righe 49–73: **2 Related Work** — Quantization methods fall broadly into two categories: quantization during training, and post-training methods.
      - righe 54–63: **Post-training Quantization.** — Most post-training methods have focused on vision models.
      - righe 64–73: **Large-model Quantization.** — With the recent open-source releases of language models like BLOOM (Laurençon et al., 2022 ) or OPT-175B (Zhang et al., 2022 ) , researchers have started to develop affordable methods for compressing such giant networks …
- righe 74–101: **3 Background** — At a high level, our method follows the structure of state-of-the-art post-training quantization methods (Nagel et al., 2020 ; Wang et al., 2020 ; Hubara et al., 2021 ; Frantar et al., 2022 ) , by performing quantization…
      - righe 76–84: **Layer-Wise Quantization.** — At a high level, our method follows the structure of state-of-the-art post-training quantization methods (Nagel et al., 2020 ; Wang et al., 2020 ; Hubara et al., 2021 ; Frantar et al., 2022 ) , by performing quantization…
      - righe 85–101: **Optimal Brain Quantization.** — Our approach builds on the recently-proposed Optimal Brain Quanization (OBQ) method (Frantar et al., 2022 ) for solving the layer-wise quantization problem defined above,
- righe 102–142: **4 The GPTQ Algorithm** — As explained in the previous section, OBQ quantizes weights in greedy order, i.e.
      - righe 104–116: **Step 1: Arbitrary Order Insight.** — As explained in the previous section, OBQ quantizes weights in greedy order, i.e.
      - righe 117–124: **Step 2: Lazy Batch-Updates.** — First, a direct implementation of the scheme described previously will not be fast in practice, because the algorithm has a relatively low compute-to-memory-access ratio.
      - righe 125–136: **Step 3: Cholesky Reformulation.** — The final technical issue we have to address is given by numerical inaccuracies, which can become a major problem at the scale of existing models, especially when combined with the block updates discussed in the previous…
      - righe 137–142: **The Full Algorithm.** — Finally, we present the full pseudocode for GPTQ in Algorithm 1 , including the optimizations discussed above.
- righe 143–256: **5 Experimental Validation** — We begin our experiments by validating the accuracy of GPTQ relative to other accurate-but-expensive quantizers, on smaller models, for which these methods provide reasonable runtimes.
      - righe 145–155: **Overview.** — We begin our experiments by validating the accuracy of GPTQ relative to other accurate-but-expensive quantizers, on smaller models, for which these methods provide reasonable runtimes.
      - righe 156–164: **Setup.** — We implemented GPTQ in PyTorch (Paszke et al., 2019 ) and worked with the HuggingFace integrations of the BLOOM (Laurençon et al., 2022 ) and OPT (Zhang et al., 2022 ) model families.
      - righe 165–168: **Baselines.** — Our primary baseline, denoted by RTN, consists of rounding all weights to the nearest quantized value on exactly the same asymmetric per-row grid that is also used for GPTQ, meaning that it corresponds precisely to the s…
      - righe 169–183: **Quantizing Small Models.** — As a first ablation study, we compare GPTQ’s performance relative to state-of-the-art post-training quantization (PTQ) methods, on ResNet18 and ResNet50, which are standard PTQ benchmarks, in the same setup as (Frantar e…
      - righe 184–187: **Runtime.** — Next we measure the full model quantization time (on a single NVIDIA A100 GPU) via GPTQ; the results are shown in Table 2 .
      - righe 188–203: **Language Generation.** — We begin our large-scale study by compressing the entire OPT and BLOOM model families to 3- and 4-bit.
      - righe 204–214: **175 Billion Parameter Models.** — We now examine BLOOM-176B and OPT-175B, the largest dense openly-available models.
      - righe 215–236: **Practical Speedups.** — Finally, we study practical applications.
      - righe 237–242: **Zero-Shot Tasks.** — While our focus is on language generation, we also evaluate the performance of quantized models on some popular zero-shot tasks, namely LAMBADA (Paperno et al., 2016 ) , ARC (Easy and Challenge) (Boratko et al., 2018 ) a…
      - righe 243–250: **Additional Tricks.** — While our experiments so far have focused exclusively on vanilla row-wise quantization, we want to emphasize that GPTQ is compatible with essentially any choice of quantization grid .
      - righe 251–256: **Extreme Quantization.** — Lastly, grouping also makes it possible to achieve reasonable performance for extreme quantization, to around 2-bits per component on average.
- righe 257–266: **6 Summary and Limitations** — We have presented GPTQ, an approximate second-order method for quantizing truly large language models.
- righe 267–271: **Acknowledgments** — Elias Frantar and Dan Alistarh gratefully acknowledge funding from the European Research Council (ERC) under the European Union’s Horizon 2020 programme (grant agreement No.
- righe 272–281: **7 Ethics Statement** — Our work introduces a general method for compressing large language models (LLMs) via quantization, with little-to-no accuracy loss in terms of standard accuracy metrics such as perplexity.
- righe 282–305: **8 Reproducibility Statement** — In the Supplementary Materials, we provide code to reproduce all experiments in this paper.
- righe 306–377: **References** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 378–438: **Appendix A Appendix** — We now provide an additional comparison between GPTQ and OBQ on BERT-base/SQuAD Rajpurkar et al.
  - righe 380–385: **A.1 Additional Comparison with OBQ** — We now provide an additional comparison between GPTQ and OBQ on BERT-base/SQuAD Rajpurkar et al.
  - righe 386–403: **A.2 Experiment Details** — This section provides additional details about our experiment setup, in particular regarding the model evaluation and the setup of our timing experiments.
    - righe 390–395: **A.2.1 Evaluation** — For language generation experiments, we calculate the perplexity, in standard fashion like Radford et al.
    - righe 396–403: **A.2.2 Timing Experiment Setup** — Our timing experiments are performed following the standard HuggingFace/accelerate 4 4 4 https://huggingface.co/docs/accelerate/index setup also used by the recent work LLM.int8() (Dettmers et al., 2022 ) .
  - righe 404–415: **A.3 Additional Language Generation Results** — Tables 9 , 10 , 11 and 12 show additional results for language generation tasks.
  - righe 416–438: **A.4 Additional ZeroShot Results** — This section contains additional results for zero-shot tasks.

## [SmoothQuant](2211.10438-smoothquant.md)

Post-Training Quantization for Large Language Models

- righe 8–9: **Testo estratto** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 10–408: **SmoothQuant: Accurate and Efficient** — Post-Training Quantization for Large Language Models
        - righe 14–26: **Abstract** — Large language models (LLMs) show excellent performance but are compute- and memory-intensive.
- righe 27–54: **1 Introduction** — Large-scale language models (LLMs) show excellent performance on various tasks (Brown et al., 2020a ; Zhang et al., 2022 ) .
- righe 55–76: **2 Preliminaries** — Quantization maps a high-precision value into discrete levels.
- righe 77–101: **3 Review of Quantization Difficulty** — LLMs are notoriously difficult to quantize due to the outliers in the activations (Dettmers et al., 2022 ; Wei et al., 2022 ; Bondarenko et al., 2021 ) .
- righe 102–132: **4 SmoothQuant** — Instead of per-channel activation quantization (which is infeasible), we propose to “smooth” the input activation by dividing it by a per-channel smoothing factor .
    - righe 108–127: **Migrate the quantization difficulty from activations to weights.** — We aim to choose a per-channel smoothing factor such that is easy to quantize.
    - righe 128–132: **Applying SmoothQuant to Transformer blocks.** — Linear layers take up most of the parameters and computation of LLM models.
- righe 133–257: **5 Experiments** — Table 2: Quantization setting of the baselines and SmoothQuant.
  - righe 135–163: **5.1 Setups** — Table 2: Quantization setting of the baselines and SmoothQuant.
    - righe 137–146: **Baselines.** — Table 2: Quantization setting of the baselines and SmoothQuant.
    - righe 147–153: **Models and datasets.** — We choose three families of LLMs to evaluate SmoothQuant: OPT (Zhang et al., 2022 ) , BLOOM (Scao et al., 2022 ) , and GLM-130B (Zeng et al., 2022 ) .
    - righe 154–159: **Activation smoothing.** — The migration strength is a general sweet spot for all the OPT and BLOOM models, and for GLM-130B since its activations are more difficult to quantize (Zeng et al., 2022 ) .
    - righe 160–163: **Implementation.** — We implement SmoothQuant with two backends: (1) PyTorch Huggingface ¶ ¶ ¶ https://github.com/huggingface/transformers for the proof of concept, and (2) FasterTransformer ∥ ∥ ∥ https://github.com/NVIDIA/FasterTransformer …
  - righe 164–204: **5.2 Accurate Quantization** — SmoothQuant can handle the quantization of very large LLMs, whose activations are more difficult to quantize.
    - righe 166–171: **Results of OPT-175B.** — SmoothQuant can handle the quantization of very large LLMs, whose activations are more difficult to quantize.
    - righe 172–181: **Results of different LLMs.** — SmoothQuant can be applied to various LLM designs.
    - righe 182–186: **Results on LLMs of different sizes.** — SmoothQuant works not only for the very large LLMs beyond 100B parameters, but it also works consistently for smaller LLMs.
    - righe 187–192: **Results on Instruction-Tuned LLM** — Table 5: SmoothQuant’s performance on the OPT-IML model.
    - righe 193–198: **Results on LLaMA models.** — Table 6: SmoothQuant can enable lossless W8A8 quantization for LLaMA models (Touvron et al., 2023a ) .
    - righe 199–204: **Results on Llama-2, Falcon, Mistral, and Mixtral models.** — Table 7: SmoothQuant can enable lossless W8A8 quantization for Llama-2 (Touvron et al., 2023b ) , Falcon (Almazrouei et al., 2023 ) , Mistral (Jiang et al., 2023 ) , and Mixtral (Jiang et al., 2024 ) models.
  - righe 205–237: **5.3 Speedup and Memory Saving** — In this section, we show the measured speedup and memory saving of SmoothQuant-O3 integrated into PyTorch and FasterTransformer.
    - righe 209–221: **Context-stage: PyTorch Implementation.** — We measure the end-to-end latency of generating all hidden states for a batch of 4 sentences in one pass, i.e.
    - righe 222–227: **Context-stage: FasterTransformer Implementation.** — As shown in Figure 9 (top), compared to FasterTransformer’s FP16 implementation of OPT, SmoothQuant-O3 can further reduce the execution latency of OPT-13B and OPT-30B by up to 1.56 when using a single GPU.
    - righe 228–237: **Decoding-stage.** — In Table 8 , we show SmoothQuant can significantly accelerate the autoregressive decoding stage of LLMs.
  - righe 238–244: **5.4 Scaling Up: 530B Model Within a Single Node** — We can further scale up SmoothQuant beyond 500B-level models, enabling efficient and accurate W8A8 quantization of MT-NLG 530B (Smith et al., 2022 ) .
  - righe 245–257: **5.5 Ablation Study** — Table 11 shows the inference latency of different quantization schemes based on our PyTorch implementation.
    - righe 247–251: **Quantization schemes.** — Table 11 shows the inference latency of different quantization schemes based on our PyTorch implementation.
    - righe 252–257: **Migration strength.** — We need to find a suitable migration strength (see Equation 4 ) to balance the quantization difficulty of weights and activations.
- righe 258–276: **6 Related Work** — Pre-trained language models have achieved remarkable performance on various benchmarks by scaling up .
    - righe 260–265: **Large language models (LLMs).** — Pre-trained language models have achieved remarkable performance on various benchmarks by scaling up .
    - righe 266–269: **Model quantization.** — Quantization is an effective method for reducing the model size and accelerating inference.
    - righe 270–276: **Quantization of LLMs.** — GPTQ (Frantar et al., 2022 ) applies quantization only to weights but not activations (please find a short discussion in Appendix A ).
- righe 277–282: **7 Conclusion** — We propose SmoothQuant, an accurate and efficient post-training quantization method to enable lossless 8-bit weight and activation quantization for LLMs up to 530B parameters.
- righe 283–291: **Acknowledgements** — We thank MIT-IBM Watson AI Lab,
- righe 292–387: **References** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 388–408: **Appendix A Discussion on Weight-Only Quantization** — In this work, we study W8A8 quantization so that we can utilize INT8 GEMM kernels to increase the throughput and accelerate inference.

## [Fast Inference from Transformers via Speculative Decoding](2211.17192-speculative-decoding.md)

Inference from large autoregressive models like Transformers is slow - decoding tokens takes serial runs of the model.

- righe 8–9: **Testo estratto** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 10–405: **Fast Inference from Transformers via Speculative Decoding** — Inference from large autoregressive models like Transformers is slow - decoding tokens takes serial runs of the model.
        - righe 12–23: **Abstract** — Inference from large autoregressive models like Transformers is slow - decoding tokens takes serial runs of the model.
- righe 24–54: **1 Introduction** — Large autoregressive models, notably large Transformers (Vaswani et al., 2017 ) , are much more capable than smaller models, as is evidenced countless times in recent years e.g., in the text or image domains, like GPT-3 …
- righe 55–73: **2 Speculative Decoding** — Let be the target model, inference from which we’re trying to accelerate, and the distribution we get from the model for a prefix .
  - righe 57–61: **2.1 Overview** — Let be the target model, inference from which we’re trying to accelerate, and the distribution we get from the model for a prefix .
  - righe 62–65: **2.2 Standardized Sampling** — First, note that while there are many methods and parameters of sampling, like argmax, top-k, nucleus, and setting a temperature, and popular implementations usually treat them differently at the logits level, they can a…
  - righe 66–73: **2.3 Speculative Sampling** — To sample , we instead sample , keeping it if , and in case we reject the sample with probability and sample again from an adjusted distribution instead.
- righe 74–214: **3 Analysis** — Let’s analyze the reduction factor in the number of serial calls to the target model, or equivalently, the expected number of tokens produced by a single run of Algorithm 1 .
  - righe 76–87: **3.1 Number of Generated Tokens** — Let’s analyze the reduction factor in the number of serial calls to the target model, or equivalently, the expected number of tokens produced by a single run of Algorithm 1 .
        - righe 80–87: **Definition 3.1 .** — The acceptance rate , given a prefix , is the probability of accepting by speculative sampling, as per Section 2.3 2 2 2 As before, we’ll omit the subscript whenever the prefix is clear from the context.
  - righe 88–117: **3.2 Calculating** — We’ll now derive a simple formula for calculating given a prefix and the two models and .
        - righe 92–95: **Definition 3.2 .** — where .
        - righe 96–97: **Lemma 3.3 .** — Sezione strutturale; consultare il contenuto locale indicato.
        - righe 98–103: **Proof.** — ∎
        - righe 104–105: **Corollary 3.4 .** — Sezione strutturale; consultare il contenuto locale indicato.
        - righe 106–107: **Theorem 3.5 .** — Sezione strutturale; consultare il contenuto locale indicato.
        - righe 108–113: **Proof.** — ∎
        - righe 114–117: **Corollary 3.6 .** — See Footnote for empirically observed values in our experiments.
  - righe 118–151: **3.3 Walltime Improvement** — We’ve shown that with the i.i.d.
        - righe 124–129: **Definition 3.7 .** — Let , the cost coefficient , be the ratio between the time for a single run of and the time for a single run of .
        - righe 130–133: **Theorem 3.8 .** — The expected improvement factor in total walltime by Algorithm 1 is .
        - righe 134–142: **Proof.** — Denote the cost of running a single step of by .
        - righe 143–146: **Corollary 3.9 .** — If , there exists for which we’ll get an improvement, and the improvement factor will be at least .
        - righe 147–151: **Proof.** — If we get an improvement for , we’d also get an improvement for any , so for our method to yield an improvement, we can evaluate Theorem 3.8 for , yielding .
  - righe 152–178: **3.4 Number of Arithmetic Operations** — Algorithm 1 does runs of in parallel, so the number of concurrent arithmetic operations grows by a factor of .
        - righe 159–162: **Definition 3.10 .** — Let be the ratio of arithmetic operations per token of the approximation model to that of the target model .
        - righe 163–166: **Theorem 3.11 .** — The expected factor of increase in the number of total operations of Algorithm 1 is .
        - righe 167–178: **Proof.** — Denote by the number of arithmetic operations done by a standard decoding baseline per token, i.e.
  - righe 179–196: **3.5 Choosing** — Given and and assuming enough compute resources (see Section 3.4 ), the optimal is the one maximizing the walltime improvement equation ( Theorem 3.8 ): .
  - righe 197–214: **3.6 Approximation Models** — Speculative sampling, and therefore speculative decoding, guarantee an identical output distribution for any choice of approximation model without restriction (see Section A.1 ).
- righe 215–249: **4 Experiments** — We implement our algorithm and compare it to the implementation in the T5X codebase for accelerating T5-XXL.
  - righe 217–230: **4.1 Empirical Walltime Improvement** — We implement our algorithm and compare it to the implementation in the T5X codebase for accelerating T5-XXL.
    - righe 221–224: **Setup** — We test a standard encoder-decoder T5 version 1.1 model (Raffel et al., 2020 ) on two tasks from the T5 paper: (1) English to German translation fine tuned on WMT EnDe, and (2) Text summarization fine tuned on CCN/DM.
    - righe 225–230: **Results** — Table 2 shows the empirical results from our method.
  - righe 231–249: **4.2 Empirical Values** — While we only implemented our method for T5, we measured values for various tasks, sampling methods, target models , and approximation models .
    - righe 235–238: **GPT-like (97M params)** — We test a decoder-only Transformer model on unconditional language generation, trained on lm1b (Chelba et al., 2013 ) .
    - righe 239–249: **LaMDA (137B params)** — We tested a decoder only LaMDA model on a dialog task (Thoppilan et al., 2022 ) .
- righe 250–265: **5 Related work** — The efficiency of inference from large models was studied extensively (Dehghani et al., 2021 ) .
- righe 266–280: **6 Discussion** — We presented speculative sampling which enables efficient stochastic speculative execution - i.e.
- righe 281–285: **Acknowledgments** — We would like to extend a special thank you to YaGuang Li for help with everything LaMDA related and for calculating the LaMDA figures in the paper, and to Blake Hechtman for great insights and help with XLA.
- righe 286–343: **References** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 344–405: **Appendix A Appendix** — We will now show that for any distributions and , the tokens sampled via speculative sampling from and are distributed identically to those sampled from alone.
  - righe 346–361: **A.1 Correctness of Speculative Sampling** — We will now show that for any distributions and , the tokens sampled via speculative sampling from and are distributed identically to those sampled from alone.
  - righe 362–379: **A.2 Speculative Sampling vs. Rejection Sampling** — Rejection sampling is the following iterative sampling procedure that looks superficially similar to ours:
  - righe 380–388: **A.3 Theoretical Predictions vs. Empirical Runtimes** — Table 4 compares the expected runtime improvements based on Theorem 3.8 to the empirically measured runtimes from Table 2 .
  - righe 389–392: **A.4 Application to Beam Search** — Our method can be applied, with some performance penalty, to beam search sampling.
  - righe 393–405: **A.5 Lenience** — A strong property of Algorithm 1 is that the output distribution is guaranteed to remain unchanged.

## [Accelerating Large Language Model Decoding with Speculative Sampling](2302.01318-speculative-sampling.md)

redacted

- righe 8–15: **Testo estratto** — redacted
- righe 16–366: **Accelerating Large Language Model Decoding with Speculative Sampling** — We present speculative sampling, an algorithm for accelerating transformer decoding by enabling the generation of multiple tokens from each transformer call.
        - righe 18–21: **Abstract** — We present speculative sampling, an algorithm for accelerating transformer decoding by enabling the generation of multiple tokens from each transformer call.
- righe 22–54: **1 Introduction** — Scaling transformer models to 500B+ parameters has led to large performance improvements on many natural language, computer vision and reinforcement learning tasks (Brown et al., 2020 ; Rae et al., 2021 ; Hoffmann et al.…
- righe 55–83: **2 Related Work** — There has been a substantial body of work focused on improving sampling latency of large transformers and other auto-regressive models.
- righe 84–97: **3 Auto-regressive Sampling** — Whilst transformers can be trained efficiently and in parallel on TPUs and GPUs, samples are typically drawn auto-regressively (See algorithm 1 ).
- righe 98–161: **4 Speculative Sampling** — For speculative sampling (See algorithm 2 ), we first make the observation that computing the logits of a short continuation of tokens in parallel has a very similar latency to that of sampling a single token.
  - righe 100–121: **4.1 Conditional Scoring** — For speculative sampling (See algorithm 2 ), we first make the observation that computing the logits of a short continuation of tokens in parallel has a very similar latency to that of sampling a single token.
  - righe 122–161: **4.2 Modified Rejection Sampling** — We require a method to recover the distribution of the target model from samples from the draft model, and logits of said tokens from both models.
- righe 162–184: **5 Choice of Draft Models** — Since the acceptance criterion guarantees the distribution of the target model in our samples, we are free to choose the method for drafting a continuation as long as it exposes logits, and there is a high enough accepta…
- righe 185–247: **6 Results** — We train a 4 billion parameter draft model optimised for sampling latency on 16 TPU v4s – the same hardware that is typically used to serve Chinchilla for research purposes.
  - righe 202–228: **6.1 Evaluation on XSum and HumanEval** — Table 1: Chinchilla performance and speed on XSum and HumanEval with naive and speculative sampling at batch size 1 and .
  - righe 229–234: **6.2 Acceptance rate changes per domain** — It is apparent that the acceptance rate is dependent on the application and the decoding method.
  - righe 235–247: **6.3 Trade off between longer drafts and more frequent scoring** — We visualise the trade-off of increasing , the number of tokens sampled by the draft model in Figure 1 .
- righe 248–259: **7 Conclusion** — In this work, we demonstrate a new algorithm and workflow for accelerating the decoding of language models.
- righe 260–305: **References** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 306–366: **Supplementary Materials** — Initial proposal: Charlie Chen, John Jumper and Geoffrey Irving
  - righe 308–337: **Author Contributions** — Initial proposal: Charlie Chen, John Jumper and Geoffrey Irving
  - righe 338–343: **Acknowledgements** — We’d like to thank Oriol Vinyals and Koray Kavukcuoglu for your kind advice and leadership.
  - righe 344–347: **Hyperparams** — Table 2: Hyperparameters for the draft model
  - righe 348–366: **Proofs** — Given discrete distributions , and a single draft sample , let be the final resulting sample.
        - righe 350–366: **Theorem 1 (Modified Rejection Sampling recovers the target distribution) .** — Given discrete distributions , and a single draft sample , let be the final resulting sample.

## [FlexGen](2303.06865-flexgen.md)

with a Single GPU

- righe 8–9: **Testo estratto** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 10–888: **FlexGen: High-Throughput Generative Inference of Large Language Models** — with a Single GPU
        - righe 14–27: **Abstract** — The high computational and memory requirements of large language model (LLM) inference make it feasible only with multiple high-end accelerators.
- righe 28–100: **1 Introduction** — In recent years, large language models (LLMs) have demonstrated strong performance across a wide range of tasks (Brown et al., 2020 ; Bommasani et al., 2021 ; Zhang et al., 2022 ; Chowdhery et al., 2022 ) .
- righe 101–116: **2 Related Work** — Given the recent advances of LLMs, LLM inference has become an important workload, encouraging active research from both the system side and the algorithm side.
- righe 117–146: **3 Background: LLM Inference** — In this section, we describe the LLM inference workflow and its memory footprint.
- righe 147–266: **4 Offloading Strategy** — In this section, we do not relax any computation of LLM inference and illustrate how to formalize the offloading procedure under the GPU, CPU, and disk memory hierarchy.
  - righe 152–178: **4.1 Problem Formulation** — Consider a machine with three devices: a GPU, a CPU, and a disk.
  - righe 179–217: **4.2 Search Space** — Given the formulation above, we construct a search space for possible valid strategies in FlexGen.
        - righe 195–217: **Theorem 4.1 .** — The I/O complexity of the zig-zag
  - righe 218–256: **4.3 Cost Model and Policy Search** — The schedule and placement in Section 4.2 constructs a search space with several parameters.
  - righe 257–266: **4.4 Extension to Multiple GPUs** — We discuss how to extend the offloading strategy in FlexGen if there are multiple GPUs.
- righe 267–295: **5 Approximate Methods** — The previous section focuses on the exact computation.
- righe 296–397: **6 Evaluation** — Table 1: Hardware Specs
  - righe 327–371: **6.1 Offloading** — Maximum throughput benchmark.
  - righe 372–379: **6.2 Approximations** — We use two tasks to show that our approximation methods exhibit negligible accuracy loss: next-word prediction on Lambada (Paperno et al., 2016 ) and language modeling on WikiText (Merity et al., 2016 ) .
  - righe 380–397: **6.3 Offloading vs. Collaborative Inference** — We compare FlexGen and Petals under different network conditions by setting a private Petals cluster on GCP with 4 nodes having one T4 GPU per node.
- righe 398–401: **7 Conclusion** — We introduce FlexGen, a high-throughput generation engine for LLM inference, which focuses on latency-insensitive batch-processing tasks for resource-constrained scenarios.
- righe 402–405: **Acknowledgements** — We would like to thank Clark Barrett and Joseph E.
- righe 406–497: **References** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 498–888: **Appendix A Appendix** — We use notations in Table 6 in this appendix.
  - righe 500–521: **A.1 Notations** — We use notations in Table 6 in this appendix.
  - righe 522–691: **A.2 Compute Schedule Optimality** — This subsection discusses the graph traversal problem described in Section 4.1 and only considers the case that the model cannot fit in a single GPU.
    - righe 539–601: **A.2.1 Zig-zag block schedule and Diagonal block schedule** — Zig-zag block schedule.
      - righe 561–601: **Diagonal block schedule** — Figure 5: diagonal block schedule
    - righe 602–691: **A.2.2 Proof of Theorem   4.1** — Note that in any case when we move from computing a square to another square, we need to offload and load the corresponding KV cache.
        - righe 610–621: **Definition A.1 .** — We define the working state at any time when the GPU is computing a square as follows.
        - righe 622–626: **Lemma A.2 .** — If there is a list of moves that start from state , and back to state at the end,
        - righe 627–644: **Proof.** — Suppose the start state .
        - righe 645–648: **Theorem A.3 .** — The diagonal block schedule is I/O-optimal asymptotically.
        - righe 649–691: **Proof.** — Notice that since the memory capacity is finite, the length of the state is finite, thus the number of the possible state is finite.
  - righe 692–782: **A.3 Cost Model** — In this section, we present the full cost model.
      - righe 741–744: **Objective** — Then the following constraints describe the calculation of total latency:
      - righe 745–782: **Peak Memory Constraints** — GPU peak memory constraints during prefill:
  - righe 783–888: **A.4 Tables and Additional Experimental Results** — Execution Breakdown

## [AWQ](2306.00978-awq.md)

Content selection saved.

- righe 8–42: **Testo estratto** — Content selection saved.
      - righe 10–42: **Report GitHub Issue** — Content selection saved.
        - righe 35–42: **Abstract** — Large language models (LLMs) have transformed numerous AI applications.
- righe 43–64: **1 Introduction** — Deploying large language models (LLMs) directly on edge devices is crucial.
- righe 65–78: **2 Related Work** — Quantization reduces the bit-precision of deep learning models Han et al.
    - righe 67–70: **Model quantization methods.** — Quantization reduces the bit-precision of deep learning models Han et al.
    - righe 71–74: **Quantization of LLMs.** — People study two settings for LLM quantization: (1) W8A8 quantization, where both activation and weights are quantized to INT8 Dettmers et al.
    - righe 75–78: **System support for low-bit quantized LLMs.** — Low-bit quantized LLMs have been a popular setting to reduce inference costs.
- righe 79–134: **3 AWQ: Activation-aware Weight Quantization** — Quantization maps a floating-point number into lower-bit integers.
  - righe 83–95: **3.1 Improving LLM Quantization by Preserving 1% Salient Weights** — Table 1: Keeping a small fraction of weights (0.1%-1%) in FP16 significantly improves the performance of the quantized models over round-to-nearest (RTN).
  - righe 96–134: **3.2 Protecting Salient Weights by Activation-aware Scaling** — We propose an alternative method to reduce the quantization error of the salient weight by per-channel scaling , which does not suffer from the hardware inefficiency issue.
- righe 135–172: **4 TinyChat: Mapping AWQ onto Edge Platforms** — AWQ can substantially reduce the size of LLMs.
  - righe 139–156: **4.1 Why AWQ Helps Accelerate On-Device LLMs** — Figure 4: SIMD-aware weight packing for ARM NEON with 128-bit SIMD units.
    - righe 145–148: **Context vs generation latency.** — As in Figure 3 (a), it takes 310 ms to generate 20 tokens, while summarizing a prompt with 200 tokens only takes 10 ms.
    - righe 149–152: **Generation stage is memory-bound.** — To accelerate the generation phase, we conduct a roofline analysis in Figure 3 (b).
    - righe 153–156: **Weight access dominates memory traffic.** — We therefore further break down the memory access for weight and activation in Figure 3 (c).
  - righe 157–172: **4.2 Deploy AWQ with TinyChat** — To this end, we demonstrated that 4-bit weight quantization could lead to a 4 theoretical peak performance.
    - righe 161–164: **On-the-fly weight dequantization.** — For quantized layers, as the hardware does not provide multiplication instructions between INT4 and FP16, we need to dequantize the integers to FP16 before performing matrix computation.
    - righe 165–168: **SIMD-aware weight packing.** — On-the-fly weight dequantization reduces intermediate DRAM access, but remains expensive.
    - righe 169–172: **Kernel fusion.** — We also extensively apply kernel fusion to optimize on-device LLM inference.
- righe 173–285: **5 Experiments** — Table 4: AWQ improves over round-to-nearest quantization (RTN) for different model sizes and different bit-precisions.
  - righe 179–196: **5.1 Settings** — We focus on weight-only grouped quantization in this work.
    - righe 181–184: **Quantization.** — We focus on weight-only grouped quantization in this work.
    - righe 185–188: **Models.** — We benchmarked our method on LLaMA Touvron et al.
    - righe 189–192: **Evaluations.** — Following previous literature Dettmers et al.
    - righe 193–196: **Baselines.** — Our primary baseline is vanilla round-to-nearest quantization (RTN).
  - righe 197–258: **5.2 Evaluation** — Figure 5: Comparing INT3-g128 quantized Vicuna models with FP16 counterparts under GPT-4 evaluation protocol Chiang et al.
    - righe 206–210: **Results on LLaMA models.** — We focus on LLaMA models (LLaMA Touvron et al.
    - righe 211–218: **Results on Mistral / Mixtral models.** — We also evaluated AWQ on the Mistral and Mixtral models, which are among the most popular open-source LLMs and Mixture-of-Experts (MoE) models, respectively Jiang et al.
    - righe 219–227: **Quantization of instruction-tuned models.** — Instruction tuning can significantly improve the models’ performance and usability
    - righe 228–244: **Quantization of multi-modal language models.** — Large multi-modal models (LMMs) or visual language models (VLMs) are LLMs augmented with vision inputs Alayrac et al.
    - righe 245–248: **Visual reasoning results.** — We further provide some qualitative visual reasoning examples of the LLaVA-13B Liu et al.
    - righe 249–253: **Results on programming and math tasks** — To further evaluate the performance of AWQ on tasks involving complex generations, we also tested AWQ on MBPP Austin et al.
    - righe 254–258: **Extreme low-bit quantization.** — We further quantize LLM to INT2 to accommodate limited device memory (Table 9 ).
  - righe 259–269: **5.3 Data Efficiency and Generalization** — Our method requires a smaller calibration set since we do not rely on regression/backpropagation; we only measure the average activation scale from the calibration set, which is data-efficient.
    - righe 261–265: **Better data-efficiency for the calibration set.** — Our method requires a smaller calibration set since we do not rely on regression/backpropagation; we only measure the average activation scale from the calibration set, which is data-efficient.
    - righe 266–269: **Robust to the calibration set distributions.** — Our method is less sensitive to the calibration set distribution since we only measure the average activation scale from the calibration set, which is more generalizable across different dataset distributions.
  - righe 270–285: **5.4 Speedup Evaluation** — Table 10: TinyChat also enables seamless deployment of VILA Lin et al.
    - righe 274–277: **Settings.** — In Figure 9 , we demonstrate the system acceleration results from TinyChat.
    - righe 278–281: **Results.** — As in Figure 9 (a), TinyChat brings 2.7-3.9 speedup to three families of LLMs (Llama-2, MPT and Falcon) on 4090 compared with the Huggingface FP16 implementation.
    - righe 282–285: **Comparisons against other systems.** — We compare TinyChat against existing edge LLM inference systems AutoGPTQ, llama.cpp and exllama in Figure 10 .
- righe 286–291: **6 Conclusion** — In this work, we propose Activation-aware Weight Quantization (AWQ), a simple yet effective method for low-bit weight-only LLM compression.
- righe 292–295: **Acknowledgements** — We thank MIT AI Hardware Program, National Science Foundation (CNS-2112562), MIT-IBM Watson AI Lab, Amazon and MIT Science Hub, Microsoft Turing Academic Program, and Samsung for supporting this research.
- righe 296–439: **References** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 440–455: **Instructions for reporting errors** — We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile

## [FlashAttention-2](2307.08691-flashattention-2.md)

Faster Attention with Better Parallelism and Work Partitioning

- righe 8–9: **Testo estratto** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 10–663: **FlashAttention-2 :** — Faster Attention with Better Parallelism and Work Partitioning
        - righe 14–41: **Abstract** — Scaling Transformers to longer sequence lengths has been a major problem in the
- righe 42–116: **1 Introduction** — Scaling up the context length of Transformers [ 18 ] is a
- righe 117–247: **2 Background** — We provide some background on the performance characteristics and execution
  - righe 124–150: **2.1 Hardware characteristics** — GPU performance characteristics.
  - righe 151–184: **2.2 Standard Attention Implementation** — Given input sequences where is the sequence length and
  - righe 185–247: **2.3 FlashAttention** — To speed up attention on hardware accelerators such as GPU,
    - righe 191–229: **2.3.1 Forward pass** — FlashAttention applies the classical technique of tiling to reduce memory IOs, by
    - righe 230–247: **2.3.2 Backward pass** — In the backward pass, by re-computing the values of the attention matrices
- righe 248–459: **3 FlashAttention-2 : Algorithm, Parallelism, and Work Partitioning** — We describe the FlashAttention-2 algorithm, which includes several tweaks to FlashAttention to reduce the number of non-matmul FLOPs.
  - righe 257–348: **3.1 Algorithm** — We tweak the algorithm from FlashAttention to reduce the number of non-matmul
    - righe 270–328: **3.1.1 Forward pass** — We revisit the online softmax trick as shown in Section 2.3 and make
      - righe 299–319: **Causal masking.** — One common use case of attention is in auto-regressive language modeling, where
      - righe 320–328: **Correctness, runtime, and memory requirement.** — As with FlashAttention , Algorithm 1 returns the correct output
    - righe 329–348: **3.1.2 Backward pass** — The backward pass of FlashAttention-2 is almost the same as that of FlashAttention .
      - righe 338–348: **Multi-query attention and grouped-query attention.** — Multi-query attention (MQA) [ 15 ] and grouped-query attention
  - righe 349–400: **3.2 Parallelism** — The first version of FlashAttention parallelizes over batch size and number of
      - righe 365–382: **Forward pass.** — We see that the outer loop (over sequence length) is embarrassingly parallel,
      - righe 383–400: **Backward pass.** — Notice that the only shared computation between different column blocks is in
  - righe 401–459: **3.3 Work Partitioning Between Warps** — As Section 3.2 describe how we schedule thread blocks, even within
      - righe 409–433: **Forward pass.** — For each block, FlashAttention splits and across 4 warps while keeping
      - righe 434–443: **Backward pass.** — Similarly for the backward pass, we choose to partition the warps to avoid the
      - righe 444–459: **Tuning block sizes** — Increasing block sizes generally reduces shared memory loads/stores, but
- righe 460–579: **4 Empirical Validation** — We evaluate the impact of using FlashAttention-2 to train Transformer models.
  - righe 484–555: **4.1 Benchmarking Attention** — We measure the runtime of different attention methods on an A100 80GB SXM4 GPU
  - righe 556–579: **4.2 End-to-end Performance** — We measure the training throughput of GPT-style models with either 1.3B or 2.7B
- righe 580–622: **5 Discussion and Future Directions** — FlashAttention-2 is 2 faster than FlashAttention , which means that we can train models
    - righe 601–622: **Acknowledgments** — We thank Phil Tillet and Daniel Haziza, who have implemented versions of
- righe 623–663: **References** — Sezione strutturale; consultare il contenuto locale indicato.

## [Efficient Memory Management for Large Language Model Serving with PagedAttention](2309.06180-pagedattention-vllm.md)

High throughput serving of large language models (LLMs) requires batching sufficiently many requests at a time.

- righe 8–9: **Testo estratto** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 10–682: **Efficient Memory Management for Large Language Model Serving with PagedAttention** — High throughput serving of large language models (LLMs) requires batching sufficiently many requests at a time.
        - righe 12–22: **Abstract.** — High throughput serving of large language models (LLMs) requires batching sufficiently many requests at a time.
- righe 23–96: **1. Introduction** — The emergence of large language models ( LLMs ) like GPT (OpenAI, 2023b ; Brown et al .
- righe 97–146: **2. Background** — In this section, we describe the generation and serving procedures of typical LLMs and the iteration-level scheduling used in LLM serving.
  - righe 101–110: **2.1. Transformer-Based Large Language Models** — The task of language modeling is to model the probability of a list of tokens Since language has a natural sequential ordering, it is common to factorize the joint probability over the whole sequence as the product of co…
  - righe 111–129: **2.2. LLM Service & Autoregressive Generation** — Once trained, LLMs are often deployed as a conditional generation service (e.g., completion API (OpenAI, 2020 ) or chatbot (OpenAI, 2022 ; Google, 2023 ) ).
  - righe 130–146: **2.3. Batching Techniques for LLMs** — The compute utilization in serving LLMs can be improved by batching multiple requests.
- righe 147–171: **3. Memory Challenges in LLM Serving** — Although fine-grained batching reduces the waste of computing and enables requests to be batched in a more flexible way,
  - righe 163–171: **3.1. Memory Management in Existing Systems** — Since most operators in current deep learning frameworks (Paszke et al .
- righe 172–284: **4. Method** — Figure 4 .
  - righe 180–191: **4.1. PagedAttention** — To address the memory challenges in § 3 , we introduce PagedAttention , an attention algorithm inspired by the classic idea of paging (Kilburn et al .
  - righe 192–199: **4.2. KV Cache Manager** — The key idea behind vLLM’s memory manager is analogous to the virtual memory (Kilburn et al .
  - righe 200–215: **4.3. Decoding with PagedAttention and vLLM** — Figure 6 .
  - righe 216–256: **4.4. Application to Other Decoding Scenarios** — § 4.3 shows how PagedAttention and vLLM handle basic decoding algorithms, such as greedy decoding and sampling, that take one user prompt as input and generate a single output sequence.
  - righe 257–273: **4.5. Scheduling and Preemption** — When the request traffic surpasses the system’s capacity, vLLM must prioritize a subset of requests.
  - righe 274–284: **4.6. Distributed Execution** — Many LLMs have parameter sizes exceeding the capacity of a single GPU (Brown et al .
- righe 285–334: **5. Implementation** — Table 1.
  - righe 296–315: **5.1. Kernel-level Optimization** — Since PagedAttention introduces memory access patterns that are not efficiently supported by existing systems, we develop several GPU kernels for optimizing it.
  - righe 316–334: **5.2. Supporting Various Decoding Algorithms** — vLLM implements various decoding algorithms using three key methods: fork , append , and free .
- righe 335–450: **6. Evaluation** — In this section, we evaluate the performance of vLLM under a variety of workloads.
  - righe 339–387: **6.1. Experimental Setup** — Model and server configurations.
  - righe 388–411: **6.2. Basic Sampling** — We evaluate the performance of vLLM with basic sampling (one sample per request) on three models and two datasets.
  - righe 412–423: **6.3. Parallel Sampling and Beam Search** — We evaluate the effectiveness of memory sharing in PagedAttention with two popular sampling methods: parallel sampling and beam search.
  - righe 424–436: **6.4. Shared prefix** — We explore the effectiveness of vLLM for the case a prefix is shared among different input prompts, as illustrated in Fig.
  - righe 437–450: **6.5. Chatbot** — A chatbot (OpenAI, 2022 ; Google, 2023 ; Chiang et al .
- righe 451–496: **7. Ablation Studies** — In this section, we study various aspects of vLLM and evaluate the design choices we make with ablation experiments.
  - righe 455–468: **7.1. Kernel Microbenchmark** — The dynamic block mapping in PagedAttention affects the performance of the GPU operations involving the stored KV cache, i.e., block read/writes and attention.
  - righe 469–486: **7.2. Impact of Block Size** — The choice of block size can have a substantial impact on the performance of vLLM.
  - righe 487–496: **7.3. Comparing Recomputation and Swapping** — vLLM supports both recomputation and swapping as its recovery mechanisms.
- righe 497–512: **8. Discussion** — Applying the virtual memory and paging technique to other GPU workloads.
- righe 513–540: **9. Related Work** — General model serving systems.
- righe 541–546: **10. Conclusion** — This paper proposes PagedAttention, a new attention algorithm that allows attention keys and values to be stored in non-contiguous paged memory, and presents vLLM, a high-throughput LLM serving system with efficient memo…
- righe 547–551: **Acknowledgement** — We would like to thank Xiaoxuan Liu, Zhifeng Chen, Yanping Huang, anonymous SOSP reviewers, and our shepherd, Lidong Zhou, for their insightful feedback.
- righe 552–682: **References** — Sezione strutturale; consultare il contenuto locale indicato.

## [Atom](2310.19102-atom.md)

marginparsep has been altered.

- righe 8–33: **Testo estratto** — marginparsep has been altered.
        - righe 28–33: **Abstract** — The growing demand for Large Language Models (LLMs) in applications such as content generation, intelligent chatbots, and sentiment analysis poses considerable challenges for LLM service providers.
- righe 34–73: **1 Introduction** — Large Language Models (LLMs) are increasingly being integrated into our work routines and daily lives, where we use them for summarization, code completion, and decision-making.
- righe 74–89: **2 Background** — Quantization techniques use discrete low-bit values to approximate high-precision floating points.
- righe 90–106: **3 Performance analysis of low-bit LLM serving** — In this section, we first analyze the performance bottleneck of LLM inference in serving scenarios and then establish the importance of low-bit weight-activation quantization.
- righe 107–152: **4 Design** — Low-bit precision enables efficient utilization of the underlying hardware, leading to increased throughput.
  - righe 113–122: **4.1 Mixed-precision quantization** — Prior works observed that a key challenge of LLM quantization is the outlier phenomena in activations Dettmers et al.
  - righe 123–132: **4.2 Fine-grained group quantization** — Even if Atom quantizes outliers and normal values separately, the latter is still challenging to perform accurately due to the limited representation capability of 4-bit precision (Section 5.4 ).
  - righe 133–142: **4.3 Dynamic quantization process** — Although fine-grained quantization can better preserve the local variations inside each channel of activations, this advantage would diminish if we statically calculated the quantization parameters based on calibration d…
  - righe 143–148: **4.4 KV-cache quantization** — As described in § 3 , the self-attention layer in the decode stage is highly memory-bound.
  - righe 149–152: **4.5 Implementation of quantization workflow** — To demonstrate the feasibility of our design choices, we implement Atom on Llama models Touvron et al.
- righe 153–223: **5 Evaluation** — We conduct a comprehensive evaluation of Atom’s accuracy and efficiency.
  - righe 157–169: **5.1 Quantization setup** — Atom uses symmetric quantization on weights and activations while using asymmetric quantization on the KV-cache.
  - righe 170–183: **5.2 Accuracy evaluation** — Benchmarks.
  - righe 184–206: **5.3 Efficiency evaluation** — To demonstrate the efficiency of Atom, we conduct experiments profiling both per-kernel and end-to-end performance.
    - righe 188–195: **5.3.1 Kernel evaluation** — Matrix multiplication.
    - righe 196–206: **5.3.2 End-to-end evaluation** — Serving setup.
  - righe 207–223: **5.4 Ablation study of quantization techniques** — In this subsection, we comprehensively evaluate the effectiveness of quantization techniques used in Atom, in terms of both accuracy and efficiency, to better illustrate our design choices and the trade-off between accur…
    - righe 211–217: **5.4.1 Ablation study to evaluate accuracy** — We examine the accuracy gain or loss of different quantization techniques used in Atom.
    - righe 218–223: **5.4.2 Ablation study to evaluate efficiency** — We then showcase the GEMM kernel throughput with different fused quantization techniques 3 3 3 Kernel performance is profiled by NVBench NVIDIA ( 2024b ) with the Llama-7b config and a batch size of on RTX 4090.
- righe 224–236: **6 Discussion** — With innovations of model architectures like Mixture of Experts (MoE) Jiang et al.
- righe 237–247: **7 Related Work** — LLM serving.
- righe 248–251: **8 Conclusion** — We presented Atom, a low-bit quantization method that leverages the underlying hardware efficiently to achieve both high accuracy and high throughput for LLM serving.
- righe 252–255: **Acknowledgments** — We thank Jiaming Tang and Yixin Dong for their discussion and insightful feedback.
- righe 256–380: **References** — Sezione strutturale; consultare il contenuto locale indicato.

## [FlashDecoding++](2311.01282-flashdecoding-plus-plus.md)

As the Large Language Model (LLM) becomes increasingly important in various domains, the performance of LLM inference is crucial to massive LLM applications.

- righe 8–9: **Testo estratto** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 10–283: **FlashDecoding++: Faster Large Language Model Inference on GPUs** — As the Large Language Model (LLM) becomes increasingly important in various domains, the performance of LLM inference is crucial to massive LLM applications.
        - righe 12–18: **Abstract** — As the Large Language Model (LLM) becomes increasingly important in various domains, the performance of LLM inference is crucial to massive LLM applications.
- righe 19–57: **1 Introduction** — Figure 1: Overview of comparison between FlashDecoding++ and state-of-the-art designs.
- righe 58–88: **2 Background** — The task of LLM inference is to generate tokens from the input sequence, which can be used to complete a sentence or answer a question.
  - righe 60–66: **2.1 LLM Inference Dataflow Overview** — The task of LLM inference is to generate tokens from the input sequence, which can be used to complete a sentence or answer a question.
  - righe 67–81: **2.2 Operations in LLM Inference** — The main operations in LLM inference are depicted as operation ① to ⑥ in Figure 2 , including the linear projection (① and ⑤), the attention (②, ③, and ④), and the feedforward network (⑥).
  - righe 82–88: **2.3 Attention Optimization** — The softmax operation shown in Figure 4 (a) requires all global data to be calculated and stored before it can proceed.
- righe 89–112: **3 Asynchronized Softmax with** — Unified Maximum Value
- righe 113–132: **4 Flat GEMM Optimization with Double Buffering** — Motivation.
- righe 133–150: **5 Heuristic Dataflow with Hardware Resource Adaption** — Motivation.
- righe 151–196: **6 Evaluation** — We evaluate the performance of FlashDecoding++ on different GPUs with various Large Language Models.
  - righe 153–184: **6.1 Experiments Setup** — We evaluate the performance of FlashDecoding++ on different GPUs with various Large Language Models.
    - righe 157–162: **6.1.1 Hardware Platforms** — We evaluate the performance of FlashDecoding++ and other LLM engines on both NVIDIA and AMD platforms to make a comprehensive comparison.
    - righe 163–168: **6.1.2 LLM Engine Baselines** — We implement our FlashDecoding++ using the Pytorch-based front-end with the C++ and CUDA backend for NVIDIA GPUs while ROCm for AMD GPUs.
    - righe 169–184: **6.1.3 Models** — We evaluate the performance of FlashDecoding++ with other LLM inference engines on three typical Large Language Models: Llama2, OPT, and ChatGLM2.
  - righe 185–196: **6.2 Comparison with State-of-the-art** — We compare FlashDecoding++ with state-of-the-art LLM inference engines in Figure 10 and Figure 11 on NVIDIA GPUs, Figure 12 and Figure 13 for AMD GPUs.
- righe 197–200: **7 Related Works** — Large language model inference acceleration has gained significant attention in recent research, with several notable approaches and techniques emerging in the field.
- righe 201–206: **8 Conclusion** — We propose FlashDecoding++ , a fast Large Language Model inference engine in this paper.
- righe 207–283: **References** — Sezione strutturale; consultare il contenuto locale indicato.

## [SparQ Attention](2312.04985-sparq-attention.md)

The computational difficulties of large language model (LLM) inference remain a significant obstacle to their widespread deployment.

- righe 8–9: **Testo estratto** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 10–650: **SparQ Attention: Bandwidth-Efficient LLM Inference** — The computational difficulties of large language model (LLM) inference remain a significant obstacle to their widespread deployment.
        - righe 12–17: **Abstract** — The computational difficulties of large language model (LLM) inference remain a significant obstacle to their widespread deployment.
- righe 18–32: **1 Introduction** — Transformer models trained on large corpora of text have recently shown remarkable performance on complex natural language processing tasks (Achiam et al., 2023 ; Touvron et al., 2023 ) .
- righe 33–54: **2 Background** — In this section we provide a straightforward framework to understand the computational efficiency of sequence generation using transformer models (similar to the modelling introduced by Kaplan et al.
      - righe 39–46: **Arithmetic intensity** — Consider a compute unit capable of scalar arithmetic operations per second that is connected to a memory via an interface which can transfer scalar elements per second.
      - righe 47–54: **Time in attention** — Sequence generation with transformers is dominated by two types of computation.
- righe 55–119: **3 Approximating Attention** — (a)
      - righe 87–94: **Attention scores sparsity** — First, consider the attention scores in Equation 2 :
      - righe 95–102: **Mean value reallocation** — Table 1: Excess correlation ratio (Roche et al., 1998 ) along axes of (excess: subtract , so uniform random data ).
      - righe 103–111: **Query sparsity** — In order to improve the lower bound on memory transfers, we further consider efficiently approximating the mask by calculating approximate attention scores without using the full matrix .
      - righe 112–119: **Mean value reallocation with query sparsity** — As a final consideration, we look at combining the mean value reallocation improvement of Equation 6 with the approach in Equation 8 .
- righe 120–150: **4 SparQ Attention** — Algorithm 1 SparQ Attention
      - righe 146–150: **Grouped query attention** — For models using GQA, groups of queries access the same KV head.
- righe 151–210: **5 Experiments** — Table 2: Results for the largest model of each family tested are presented below.
  - righe 155–170: **5.1 Setup** — We evaluate our method on five widely-used open-source language model variants: Llama (Touvron et al., 2023 ) , Llama (Meta AI, 2024 ) , Mistral (Jiang et al., 2023 ) , Gemma (Mesnard et al., 2024 ) and Pythia (Biderman …
      - righe 157–160: **Models** — We evaluate our method on five widely-used open-source language model variants: Llama (Touvron et al., 2023 ) , Llama (Meta AI, 2024 ) , Mistral (Jiang et al., 2023 ) , Gemma (Mesnard et al., 2024 ) and Pythia (Biderman …
      - righe 161–166: **Tasks** — In order to evaluate our method on a spectrum of relevant NLP tasks that present a particular challenge to sparse attention techniques, our evaluation setup consists of various tasks requiring information retrieval and r…
      - righe 167–170: **Baselines** — We consider the cache eviction technique H 2 O (Zhang et al., 2023 ) , top- sparse attention in the form of FlexGen (Sheng et al., 2023 ) , and LM-Infinite, a local windowing scheme with initial-tokens included proposed …
  - righe 171–176: **5.2 Results** — Our experiments span eight distinct models: Llama with and billion parameters, Llama with billion parameters, Mistral with billion parameters, Gemma with billion parameters, and three Pythia models with , and billion par…
  - righe 177–188: **5.3 Sequence Length Scaling** — Figure 6: SQuAD performance vs input sequence length.
  - righe 189–210: **5.4 Ablations** — The first step in SparQ Attention involves reading components of the key cache to approximately determine which keys yield the highest attention scores.
      - righe 191–194: **Key cache compression** — The first step in SparQ Attention involves reading components of the key cache to approximately determine which keys yield the highest attention scores.
      - righe 195–198: **Approximate softmax temperature** — To empirically support our statistical analysis of agreement shown in Figure 4f , we evaluate a number of different viable temperature settings, including the square root of the head dimension ( ), the square root of the…
      - righe 199–210: **Hyperparameter selection** — The reduction of data transfer attained by SparQ Attention is controlled by its two hyperparameters, and .
- righe 211–246: **6 Benchmarking** — The results above use a theoretical cost model of total memory transfers (the number of scalar elements transferred to and from memory per token), allowing us to evaluate SparQ Attention independently of the specific har…
  - righe 217–230: **6.1 Microbenchmarks** — We tested multiple implementations of baseline and SparQ Attention on IPU using the Poplar C++ interface and GPU using PyTorch (Paszke et al., 2019 ) .
  - righe 231–246: **6.2 End-to-End Performance** — Figure 9: End-to-end CPU speedup results for SparQ Attention compared to the dense baseline with Llama B.
      - righe 237–242: **CPU benchmarking** — We evaluated CPU benchmarking performance on AMD EPYC systems with up to GB memory.
      - righe 243–246: **GPU benchmarking** — Our end-to-end GPU implementation is evaluated on a single H PCIe with GB memory.
- righe 247–260: **7 Related Work** — Efficient attention methods have been a very active area of research (Tay et al., 2020b ) .
- righe 261–266: **8 Conclusion** — In this work we have presented SparQ Attention, a novel technique for unlocking faster inference for pre-trained LLMs.
- righe 267–270: **Impact Statement** — This paper presents work whose goal is to advance the field of Machine Learning.
- righe 271–276: **Acknowledgements** — We would like to thank Oscar Key for implementing SparQ Attention on GPU and benchmarking its end-to-end performance.
- righe 277–384: **References** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 385–396: **Appendix A Detailed Results** — Figures A 1 , A 2 and A 3 report the compression/performance trade-off curves for all models and tasks that were evaluated.
- righe 397–400: **Appendix B Code** — [fontsize= ]python3algorithms/code_snippet.py
- righe 401–426: **Appendix C Arithmetic Intensity** — Consider a full transformer layer, with parameters, batch size , elements in the attention KV cache per batch element and grouped-query heads per key-value head.
      - righe 419–426: **Hardware** — Properties of selected machine learning hardware.
- righe 427–446: **Appendix D Measuring Time Spent in Attention** — (a)
- righe 447–470: **Appendix E Attention Sparsity Analysis** — (a)
- righe 471–495: **Appendix F Benchmarking Detail** — Benchmarking code is made available from:
      - righe 478–481: **IPU measurements** — We tested custom fully-fused Poplar implementations of both dense attention and SparQ Attention, compiled using Poplar SDK 3.3.0+1403.
      - righe 482–485: **GPU measurements** — All experiments use PyTorch 2.1.2+cu121 on Ubuntu AWS instances.
      - righe 486–489: **Additional results** — In addition to the headline results shared in Section 6 and Figure 8 , we give an aggregate picture of the trends in Figure F 1 .
      - righe 490–495: **Storing twice** — One limitation of a theoretical model of data transfer is that it does not account for the granularity of memory access.
- righe 496–650: **Appendix G Methodology** — We provide a comprehensive description of our experimental setup for reference in Table G 1 .
      - righe 500–503: **Baselines** — We use our own implementation of H 2 O ( Zhang et al.
      - righe 504–511: **Compression ratio** — We define the compression ratio as the ratio of attention data transfers required for the sparse technique and the dense data transfers.
  - righe 512–518: **G.1 Examples** — We illustrate the task setup with a single example per task, showing the prompt formatting and a cherry-picked example.
    - righe 516–518: **G.1.1 Question Answering (SQuAD -shot)** — Sezione strutturale; consultare il contenuto locale indicato.
  - righe 519–533: **PROMPT (5725e1c4271a42140099d2d9)** — Title: University of Chicago.
  - righe 534–543: **OUTPUT** — DENSE: 1978
    - righe 541–543: **G.1.2 Question Answering (TriviaQA 0-shot)** — Sezione strutturale; consultare il contenuto locale indicato.
  - righe 544–559: **PROMPT (dpql_5685)** — Apritifs and digestifs ( and) are drinks, typical...
  - righe 560–571: **OUTPUT** — DENSE: Dubonnet
    - righe 569–571: **G.1.3 Summarisation (CNN/DailyMail)** — Sezione strutturale; consultare il contenuto locale indicato.
  - righe 572–575: **PROMPT (a62bbf503be06e8b1f8baa4f3cd537310d5aa3bc)** — Article: Prince William arrived in China tonight for one of the most high-profil...
  - righe 576–585: **OUTPUT** — DENSE: Prince William arrived in China tonight for one of the most high-profile ...
    - righe 583–585: **G.1.4 Repetition (Shakespeare)** — Sezione strutturale; consultare il contenuto locale indicato.
  - righe 586–609: **PROMPT (210496)** — you mistake me much;
  - righe 610–619: **OUTPUT** — DENSE: can; for my good uncle Gloucester
    - righe 617–619: **G.1.5 Language Modelling (WikiText-103)** — Sezione strutturale; consultare il contenuto locale indicato.
  - righe 620–644: **QUERY (2)** — = Mellor hill fort =
  - righe 645–650: **BPC** — DENSE: 0.669

## [DistServe](2401.09670-distserve.md)

Large Language Model Serving

- righe 8–9: **Testo estratto** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 10–614: **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized** — Large Language Model Serving
        - righe 14–22: **Abstract** — DistServe improves the performance of large language models (LLMs) serving by disaggregating the prefill and decoding computation.
- righe 23–70: **1 Introduction** — Large language models (LLMs), such as GPT-4 [ 37 ] , Bard [ 2 ] , and LLaMA [ 51 ] , represent a groundbreaking shift in generative AI.
- righe 71–132: **2 Background and Motivation** — An LLM service follows a client-server architecture: the client submits a sequence of text as a request to the server; the server hosts the LLM on GPUs, runs inference over the request, and responds (or streams) the gene…
  - righe 76–86: **2.1 LLM Inference** — Modern LLMs [ 37 , 51 ] predict the next token given an input sequence.
  - righe 87–105: **2.2 LLM Serving Optimization** — In real-time online serving, multiple requests come and must be served within SLOs.
  - righe 106–132: **2.3 Problems and Opportunities** — Colocating and batching the prefill and decoding computation to maximize the overall system throughput, as in existing systems, is cost-effective for service providers.
- righe 133–206: **3 Tradeoff Analysis** — Disaggregation uncouples the two phases and allows a distinct analysis of the characteristics of each phase, providing valuable insights into the algorithm design.
  - righe 139–177: **3.1 Analysis for Prefill Instance** — (a) Prefill phase (b) Decoding phase
  - righe 178–195: **3.2 Analysis for Decoding Instance** — Unlike the prefill instance, a decoding instance follows a distinct computational pattern: it receives the KV caches and the first output token from the prefill instance and generates subsequent tokens one at a time.
  - righe 196–206: **3.3 Practical Problems** — We have developed foundational principles for selecting batching and parallelisms for each phase.
- righe 207–267: **4 Method** — We built DistServe to solve the above challenges.
  - righe 214–236: **4.1 Placement for High Node-Affinity Cluster** — Algorithm 1 High Node-Affinity Placement Algorithm
  - righe 237–247: **4.2 Placement for Low Node-Affinity Cluster** — A straightforward solution is to always colocate prefill and decoding instances on the same node, utilizing the NVLINK, which is commonly available inside a GPU node.
  - righe 248–267: **4.3 Online scheduling** — The runtime architecture of DistServe is shown in Figure 6 .
- righe 268–273: **5 Implementation** — DistServe is an end-to-end distributed serving system for LLMs with a placement algorithm module, a RESTful API frontend, an orchestration layer, and a parallel execution engine.
- righe 274–366: **6 Evaluation** — In this section, we evaluate DistServe under different sizes of LLMs ranging from 13B to 175B and various application datasets including chatbot, code-completion, and summarization.
  - righe 288–321: **6.1 Experiments Setup** — Cluster testbed.
  - righe 322–339: **6.2 End-to-end Experiments** — In this Section, we compare the end-to-end performance of DistServe against the baselines on real application datasets.
  - righe 340–349: **6.3 Latency Breakdown** — To understand DistServe’s performance in detail, we make a latency breakdown of the requests in DistServe.
  - righe 350–360: **6.4 Ablation Studies** — We study the effectiveness of the two key innovations in DistServe: disaggregation and the placement searching algorithm.
  - righe 361–366: **6.5 Algorithm Running Time** — Figure 12 shows the running time for Alg.
- righe 367–376: **7 Discussion** — In this paper, we focus on the goodput-optimized setting and propose DistServe under the large-scale LLM serving scenario.
- righe 377–387: **8 Related Work** — Inference serving.
- righe 388–401: **9 Conclusion** — We present DistServe, a new LLM serving architecture that disaggregates the prefill and decoding computation.
- righe 402–523: **References** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 524–605: **Appendix A Latency Model for LLM Inference** — To accurately simulate the goodput of different placement strategies, we use an analytical model to predict the execution time of the prefill and decoding phases in LLM inference.
  - righe 530–573: **A.1 Symbol Definition** — Here are symbols related to the architecture of the model:
  - righe 574–585: **A.2 Prefill Phase Latency Modeling** — Since the attention operation uses specially optimized kernels, we first discuss the other four matrix multiplications in the prefill phase:
  - righe 586–605: **A.3 Decoding Phase Latency Modeling** — Similarly, we first focus on the following GEMMs in the decoding phase:
- righe 606–611: **Appendix B DistServe Placements in End-to-end Experiments** — Table 3 shows the tensor parallelism (TP) and pipeline parallelism (PP) configurations for prefill and decoding instances chosen by DistServe in the end-to-end experiments § 6.2 .
- righe 612–614: **Appendix C End-to-end Results under 99% SLO attainment** — Figure 13 and Figure 14 show the end-to-end performance between DistServe and baselines with the same setup in § 6.2 except that the SLO attainment goal is changed to 99%.

## [Medusa](2401.10774-medusa.md)

Large Language Models (LLMs) employ auto-regressive decoding that requires sequential computation, with each step reliant on the previous one’s output.

- righe 8–9: **Testo estratto** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 10–571: **Medusa : Simple LLM Inference Acceleration Framework with Multiple Decoding Heads** — Large Language Models (LLMs) employ auto-regressive decoding that requires sequential computation, with each step reliant on the previous one’s output.
        - righe 12–22: **Abstract** — Large Language Models (LLMs) employ auto-regressive decoding that requires sequential computation, with each step reliant on the previous one’s output.
- righe 23–47: **1 Introduction** — The recent advancements in Large Language Models (LLMs) have demonstrated that the quality of language generation significantly improves with an increase in model size, reaching billions of parameters (Brown et al., 2020…
- righe 48–156: **2 Methodology** — Medusa follows the same framework as speculative decoding, where each decoding step primarily consists of three substeps: (1) generating candidates, (2) processing candidates, and (3) accepting candidates.
  - righe 54–79: **2.1 Key Components** — In speculative decoding, subsequent tokens are predicted by an auxiliary draft model.
    - righe 56–68: **2.1.1 Medusa Heads** — In speculative decoding, subsequent tokens are predicted by an auxiliary draft model.
    - righe 69–79: **2.1.2 Tree Attention** — Through Medusa heads, we obtain probability predictions for the subsequent tokens.
  - righe 80–113: **2.2 Training Strategies** — At the most basic level, we can train Medusa heads by freezing the backbone model and fine-tuning Medusa heads.
    - righe 86–91: **2.2.1 Medusa -1: Frozen Backbone** — To train Medusa heads with a frozen backbone model, we can use the cross-entropy loss between the prediction of Medusa heads and the ground truth.
    - righe 92–109: **2.2.2 Medusa -2: Joint Training** — To further improve the accuracy of Medusa heads, we can train Medusa heads together with the backbone model.
    - righe 110–113: **2.2.3 How to Select the Number of Heads** — Empirically, we found that five heads are sufficient at most.
  - righe 114–156: **2.3 Extensions** — In speculative decoding papers (Leviathan et al., 2022 ; Chen et al., 2023 ) , authors employ rejection sampling to yield diverse outputs that align with the distribution of the original model.
    - righe 116–130: **2.3.1 Typical Acceptance** — In speculative decoding papers (Leviathan et al., 2022 ; Chen et al., 2023 ) , authors employ rejection sampling to yield diverse outputs that align with the distribution of the original model.
    - righe 131–142: **2.3.2 Self-Distillation** — In Section 2.2 , we assume the existence of a training dataset that matches the target model’s output distribution.
    - righe 143–156: **2.3.3 Searching for the Optimized Tree Construction** — In Section 2.1.2 , we present the simplest way to construct the tree structure by taking the Cartesian product.
- righe 157–231: **3 Experiments** — In this section, we present experiments to demonstrate the effectiveness of Medusa under different settings.
  - righe 163–178: **3.1 Case Study: Medusa -1 v.s. Medusa -2 on Vicuna 7B and 13B** — Experimental Setup.
  - righe 179–202: **3.2 Case Study: Training with Self-Distillation on Vicuna-33B and Zephyr-7B** — Experimental Setup.
  - righe 203–231: **3.3 Ablation Study** — The study of tree attention is conducted on the writing and roleplay categories from the MT-Bench dataset using Medusa -2 Vicuna-7B.
    - righe 205–213: **3.3.1 Configuration of Tree Attention** — The study of tree attention is conducted on the writing and roleplay categories from the MT-Bench dataset using Medusa -2 Vicuna-7B.
    - righe 214–223: **3.3.2 Thresholds of Typical Acceptance** — The thresholds of typical acceptance are studied on the writing and roleplay categories from the MT-Bench dataset (Zheng et al., 2023 ) using Medusa -2 Vicuna 7B.
    - righe 224–231: **3.3.3 Effectiveness of Two-stage Fine-tuning** — Table 2 shows the performance differences between various fine-tuning strategies for the Vicuna-7B model.
- righe 232–238: **4 Discussion** — In conclusion, Medusa enhances LLM inference speed by 2.3-2.8 times by equipping models with additional predictive decoding heads, allowing for generating multiple tokens simultaneously and bypassing the sequential decod…
- righe 239–264: **Acknowledgements** — We extend our heartfelt gratitude to several individuals whose contributions were invaluable to this project:
- righe 265–296: **Impact Statement** — The introduction of Medusa , an innovative method to improve the inference speed of Large Language Models (LLMs), presents a range of broader implications for society, technology, and ethics.
  - righe 269–282: **Societal and Technological Implications** — Accessibility and Democratization of AI : By significantly enhancing the efficiency of LLMs, Medusa makes advanced AI technologies more accessible to a wider range of users and organizations.
  - righe 283–296: **Ethical Considerations** — Bias and Fairness : While Medusa aims to improve LLM efficiency, it inherits the ethical considerations of its backbone models, including issues related to bias and fairness.
- righe 297–402: **References** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 403–429: **Appendix A Related Work** — The inefficiency of Large Language Model (LLM) inference is primarily attributed to the memory-bandwidth-bound nature of the auto-regressive decoding process.
  - righe 405–421: **A.1 LLM Inference Acceleration** — The inefficiency of Large Language Model (LLM) inference is primarily attributed to the memory-bandwidth-bound nature of the auto-regressive decoding process.
      - righe 409–412: **Reducing KV Cache.** — Methods such as Multi-query attention (Shazeer, 2019 ) and Grouped-query attention (Ainslie et al., 2023 ) adopt a direct approach to diminish the KV cache.
      - righe 413–416: **Quantization.** — Quantization techniques are extensively used to shrink LLMs’ memory consumption.
      - righe 417–421: **Speculative Decoding.** — As an approach orthogonal to the aforementioned methods, speculative decoding (Leviathan et al., 2022 ; Chen et al., 2023 ) aims to execute several decoding steps in parallel, thus reducing the total number of steps requ…
  - righe 422–429: **A.2 Sampling Scheme** — The manner in which text is sampled from Large Language Models (LLMs) can significantly influence the quality of the generated output.
- righe 430–451: **Appendix B Experiment Settings** — We clarify three commonly used terms:
  - righe 432–439: **B.1 Common Terms** — We clarify three commonly used terms:
  - righe 440–443: **B.2 Shared Settings** — For all the experiments, we use the Axolotl (Axolotl, 2023 ) framework for training.
  - righe 444–447: **B.3 Medusa -1 v.s. Medusa -2 on Vicuna 7B and 13B** — We use a global batch size of and a peak learning rate of for the backbone and for Medusa heads and warmup for steps.
  - righe 448–451: **B.4 Training with Self-Distillation on Vicuna-33B and Zephyr-7B** — We use Medusa -2 for both models instead of using a two-stage training procedure.
- righe 452–457: **Appendix C Visualization of optimized tree attention** — Fig.
- righe 458–471: **Appendix D Results of Speculative Decoding** — In this study, speculative decoding was applied to Vicuna models (Chiang et al., 2023 ) with varying sizes, specifically 7B, 13B, and 33B.
- righe 472–477: **Appendix E Additional Results for All Models** — We show speedup on various models in Fig.
- righe 478–483: **Appendix F Additional Results on AlpacalEval Dataset** — We conduct further experiments on the AlpacaEval (Li et al., 2023 ) dataset.
- righe 484–571: **Appendix G Exploration and Modeling of Hardware Constraints and Medusa** — We explore the hardware constraints, specifically memory-bandwidth bound, and their impact on Medusa -style parallel decoding by incorporating a simplified Llama-series model.
  - righe 491–527: **G.1 Roofline Model of Operators** — We present an analysis of the roofline model for various operators in large language models (LLMs), specifically focusing on Llama-7B, Llama-13B, and Llama-33B (Touvron et al., 2023 ) .
  - righe 528–552: **G.2 FLOP/s vs. Operational Intensity Variations in Medusa** — We investigate how Medusa can change Operational Intensity and elevate the FLOP/s.
  - righe 553–571: **G.3 Predicting Medusa Performance** — We further employ a straightforward analytical model for the acceleration rate.

## [KVQuant](2401.18079-kvquant.md)

LLMs are seeing growing use for applications which require large context windows, and with these large context windows KV cache activations surface as the dominant contributor to memory consumption during inference.

- righe 8–9: **Testo estratto** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 10–730: **KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization** — LLMs are seeing growing use for applications which require large context windows, and with these large context windows KV cache activations surface as the dominant contributor to memory consumption during inference.
        - righe 12–24: **Abstract** — LLMs are seeing growing use for applications which require large context windows, and with these large context windows KV cache activations surface as the dominant contributor to memory consumption during inference.
- righe 25–76: **1 Introduction** — Large language models (LLMs) have revolutionized many natural language processing (NLP) tasks.
- righe 77–120: **2 Background** — When inferring a decoder-only LLM, inference proceeds in two distinct phases.
  - righe 79–92: **2.1 LLM Inference** — When inferring a decoder-only LLM, inference proceeds in two distinct phases.
  - righe 93–106: **2.2 LLM Quantization** — There have been many prior works on LLM quantization.
  - righe 107–120: **2.3 KV Cache Compression** — There have also been several prior works on compressing the KV cache.
- righe 121–223: **3 Method** — To inform our approach, we first performed a detailed analysis to understand the KV cache distributions.
  - righe 123–142: **3.1 Per-Channel Key Quantization** — To inform our approach, we first performed a detailed analysis to understand the KV cache distributions.
  - righe 143–156: **3.2 Pre-RoPE Key Quantization** — One issue when quantizing Keys is handling the rotary positional embedding (RoPE), which is applied to Keys and Queries in most public LLMs, including LLaMA and Llama-2 [ 35 ] .
  - righe 157–173: **3.3 nuqX: An X-Bit Per-Layer Sensitivity-Weighted Non-Uniform Datatype** — Uniform quantization is suboptimal for KV cache quantization since the Query and Key activations are non-uniform.
  - righe 174–188: **3.4 Per-Vector Dense-and-Sparse Quantization** — As shown in Figure 4 in Appendix F , for both Keys and Values, the majority of elements are contained within a small percentage of the dynamic range.
  - righe 189–199: **3.5 Attention Sink-Aware Quantization** — Prior work has demonstrated that after the first few layers in LLMs, the model tends to allocate a large attention score to the first token [ 42 ] .
  - righe 200–207: **3.6 Offline Calibration versus Online Computation** — A crucial challenge for activation quantization is that we either need to compute statistics on-the-fly (which is potentially expensive) or else we need to use offline calibration data (which potentially has negative acc…
  - righe 208–223: **3.7 Kernel Implementation** — In order to efficiently perform activation quantization on-the-fly, we leverage dedicated kernel implementations with our 4-bit quantization method for compressing vectors to reduced precision and extracting the sparse o…
- righe 224–315: **4 Results** — We used the LLaMA-7B/13B/30B/65B, Llama-2-7B/13B/70B, Llama-3-8B/70B, and Mistral-7B models to evaluate our methodology by measuring perplexity on both Wikitext-2 and C4 [ 36 , 37 , 1 , 16 , 27 , 31 ] .
  - righe 226–267: **4.1 Main Evaluation** — We used the LLaMA-7B/13B/30B/65B, Llama-2-7B/13B/70B, Llama-3-8B/70B, and Mistral-7B models to evaluate our methodology by measuring perplexity on both Wikitext-2 and C4 [ 36 , 37 , 1 , 16 , 27 , 31 ] .
  - righe 268–290: **4.2 Long Context Length Evaluation** — Perplexity Evaluation.
  - righe 291–297: **4.3 Joint Weight and KV Cache Quantization** — Table 5 provides results for our KV cache quantization method when the weights are also quantized using the methodology in SqueezeLLM [ 17 ] .
  - righe 298–315: **4.4 Performance Analysis and Memory Savings** — Table 6 shows kernel benchmarking results using a batch size of 1 for
- righe 316–328: **5 Conclusion** — As context lengths in LLMs increase, the KV cache activations surface as the dominant contributor to memory consumption.
- righe 329–347: **Acknowledgements** — The authors would like to acknowledge Nicholas Lee for helpful discussions and feedback.
- righe 348–439: **References** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 440–459: **Appendix A Memory Bottlenecks for Long Context Length Inference** — Table 7 shows the model size and KV cache memory requirements for different LLaMA models with different sequence lengths.
- righe 460–477: **Appendix B Additional Related Works** — LLMs have been known to have distinct outliers both in weights and activations [ 7 , 9 , 17 ] .
  - righe 462–469: **B.1 Outlier-Aware LLM Quantization** — LLMs have been known to have distinct outliers both in weights and activations [ 7 , 9 , 17 ] .
  - righe 470–477: **B.2 Non-uniform LLM Quantization** — Non-uniform quantization has also been applied in the context of LLMs.
- righe 478–485: **Appendix C RoPE Equation** — The rotation matrix for RoPE is provided in Equation 2 , where and are cosine and sine functions, , is the attention head dimension, and is the current position in the sequence:
- righe 486–499: **Appendix D Derivation for Sensitivity Analysis** — To compute the sensitivity analysis we largely follow the derivation in [ 29 ] , which was originally provided to compute
- righe 500–514: **Appendix E Derivation for Quantization Error** — In our work, before applying the sensitivity-weighted K-means to derive quantization signposts,
- righe 515–524: **Appendix F Key and Value Dynamic Range** — Figure 4 shows the portion of the elements contained within difference percentages of the dynamic range for both Keys and Values.
- righe 525–540: **Appendix G Per-Channel Key Quantization Ablations** — As shown in Table 9 , per-channel quantization for Keys and per-token quantization for Values outperforms the standard per-token quantization approach for both Keys and Values, yielding an improvement of 3.82 perplexity …
- righe 541–546: **Appendix H Pre-RoPE Key Quantization Ablations** — As shown in Table 10 , pre-RoPE Key quantization achieves higher accuracy than post-RoPE quantization, with an improvement of 0.82 perplexity for 3-bit quantization with the LLaMA-7B model.
- righe 547–561: **Appendix I Sensitivity-Weighted Non-Uniform Quantization Ablations** — Table 11 shows perplexity evaluation results across different LLaMA, Llama-2, and Mistral models on Wikitext-2 for different datatypes, including nuq3, nuq3 without using sensitivity-weighting, as well as nuq3 with sensi…
- righe 562–574: **Appendix J Per-Vector Dense-and-Sparse Quantization Ablations** — Table 12 shows the performance improvements we observe when isolating a small portion of outliers and storing them in a sparse format.
- righe 575–584: **Appendix K Attention Sink-Aware Quantization Ablations** — Table 13 provides perplexity with and without Attention Sink-Aware quantization across different LLaMA, Llama-2, and Llama-3 models.
- righe 585–619: **Appendix L Calibration Ablations** — Figure 5 outlines the accuracy and efficiency challenges which our work addresses in order to enable accurate and efficient KV cache quantization.
- righe 620–634: **Appendix M Additional Experimental Details** — For our empirical evaluation, we use 16 calibration samples of sequence length 2K from the Wikitext-2 training set (as well as the corresponding gradients) to derive the per-channel scaling factors and zero-points, and t…
- righe 635–644: **Appendix N Comparison Between Different Datatypes and Sparsity Levels** — Table 17 shows perplexity across LLaMA, Llama-2, Llama-3, and Mistral models on Wikitext-2 using different datatypes.
- righe 645–662: **Appendix O Full Perplexity Evaluation** — Tables 18 and 19 show perplexity evaluation across all LLaMA, Llama-2, Llama-3, and Mistral models on Wikitext-2 and C4, respectively.
- righe 663–674: **Appendix P Post-RoPE Per-Token Quantization Ablation** — Table 20 shows perplexity evaluation on Wikitext-2 for the LLaMA-7B model with uniform quantization, with Keys quantized pre-RoPE and post-RoPE.
- righe 675–683: **Appendix Q Experiments on Calibration Data Robustness** — To evaluate the robustness of our quantization method to the choice of calibration datasets,
- righe 684–722: **Appendix R Kernel Implementation Details** — We implemented 4-bit lookup table-based kernels for matrix-vector multiplication between the Key or Value activations (packed as a lookup table (LUT) plus indices into the LUT per-element) and a full-precision activation…
- righe 723–730: **Appendix S Limitations** — While our work enables accurate long-context length inference by reducing the memory requirements,

## [KIVI](2402.02750-kivi.md)

Efficiently serving large language models (LLMs) requires batching of many requests to reduce the cost per request.

- righe 8–9: **Testo estratto** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 10–467: **KIVI : A Tuning-Free Asymmetric 2bit Quantization for KV Cache** — Efficiently serving large language models (LLMs) requires batching of many requests to reduce the cost per request.
        - righe 12–17: **Abstract** — Efficiently serving large language models (LLMs) requires batching of many requests to reduce the cost per request.
- righe 18–83: **1 Introduction** — Large Language Models (LLMs) have demonstrated strong performance across a wide range of tasks (Brown et al., 2020 ; Taylor et al., 2022 ; Yuan et al., 2023 ; Chuang et al., 2024 ) .
- righe 84–115: **2 Background: Attention Inference-Time Workflow** — The LLM attention inference-time workflow involves two phases: i) the prefill phase, where the input prompt is used to generate KV cache for each transformer layer of LLMs; and ii) the decoding phase, where the model use…
      - righe 88–95: **Prefill Phase.** — Let be the input tensor, where is the batch size, is the length of the input prompt, and is the model hidden size.
      - righe 96–106: **Decoding Phase.** — Let be the current input token embedding.
      - righe 107–115: **Memory and Speed Analysis.** — The above process is repeated until a special token indicating the sentence’s conclusion is reached.
- righe 116–227: **3 Methodology** — In scenarios with long contexts or batched inferences, the memory and speed bottlenecks are storing and loading KV cache.
  - righe 124–153: **3.1 Preliminary Study of KV Cache Quantization** — As we analyzed in Section 2 , KV cache functions as a streaming data structure, where the new tensor arrives sequentially.
      - righe 139–153: **Setting.** — In Table 1 , we show the results of fake KV cache group-wise quantization with different configurations on the Llama-2-13B model for the CoQA and TruthfulQA tasks.
  - righe 154–189: **3.2 Why Key and Value Cache Should Quantize Along Different Dimensions?** — In Table 1 , we observe that quantizing key cache per-channel and value cache per-token to 2bit results in a very small accuracy drop.
      - righe 161–173: **Analysis of Key Cache.** — The above observation for key cache aligns with previous findings that certain fixed columns in activations exhibit larger outliers (Dettmers et al., 2022 ; Lin et al., 2023 ) .
      - righe 174–189: **Analysis of Value Cache.** — Unlike key cache, value cache does not show the channel-wise outlier pattern.
  - righe 190–227: **3.3 KIVI : Algorithm and System Support** — As we previously analyzed, key cache should be quantized per-channel and value cache should be quantized per-token.
      - righe 192–211: **Algorithm.** — As we previously analyzed, key cache should be quantized per-channel and value cache should be quantized per-token.
      - righe 212–220: **Analysis.** — In KIVI , the grouped key cache and value cache is quantized, while the residual key cache and value cache is kept in full precision.
      - righe 221–227: **System Support.** — We provide a hardware-friendly implementation for running KIVI on GPUs.
- righe 228–322: **4 Experiments** — We evaluate KIVI using three popular model families: Llama/Llama-2 (Touvron et al., 2023a , b ) , Falcon (Penedo et al., 2023 ) and Mistral (Jiang et al., 2023 ) .
  - righe 230–249: **4.1 Settings** — We evaluate KIVI using three popular model families: Llama/Llama-2 (Touvron et al., 2023a , b ) , Falcon (Penedo et al., 2023 ) and Mistral (Jiang et al., 2023 ) .
      - righe 232–238: **Models.** — We evaluate KIVI using three popular model families: Llama/Llama-2 (Touvron et al., 2023a , b ) , Falcon (Penedo et al., 2023 ) and Mistral (Jiang et al., 2023 ) .
      - righe 239–249: **Tasks.** — As we analyzed in Section 2 , the KV cache size grows larger with a longer context.
  - righe 250–322: **4.2 Accuracy and Efficiency Analysis** — We first utilize the fake quantization to demonstrate the effectiveness of our asymmetric quantization, namely, quantizing key cache per-channel and value cache per-token.
    - righe 252–264: **4.2.1 Comparison Between Different Quantization Configurations** — We first utilize the fake quantization to demonstrate the effectiveness of our asymmetric quantization, namely, quantizing key cache per-channel and value cache per-token.
    - righe 265–296: **4.2.2 Accuracy Comparison on Generation Tasks** — Table 3 : Performance comparison between 16bit, 4-bit per-token quantization, four fake 2bit KV cache quantization similar to those in Table 1 , KIVI-2 (2bit) / KIVI-4 (4bit) across various models.
      - righe 276–283: **LM-Eval Results.** — We benchmark KIVI in CoQA, TruthfulQA and GSM8K tasks using the LM-Eval framework.
      - righe 284–289: **LongBench Results.** — The performance of KIVI over various models in the LongBench dataset is summarised in Table 4 .
      - righe 290–296: **NIAH Results.** — From Figure 15 , we observe that KIVI can still maintain the retrieval ability of LLMs even with 2bit KV Cache.
    - righe 297–308: **4.2.3 Ablation** — In this section, we benchmark KIVI on GSM8K, one of the hardest generation tasks, to show the effect of hyperparameters group size and residual length on the model performance.
    - righe 309–322: **4.2.4 Efficiency Comparison** — To evaluate the wall-clock time efficiency of KIVI ,
- righe 323–340: **5 Related Work** — Many machine learning systems and benchmark works consider scaling up LLM inference process (Pope et al., 2023 ; Yuan et al., 2024 ) .
- righe 341–347: **6 Conclusion and Future Work** — In this paper, we systematically analyze KV cache element distribution in popular LLMs.
- righe 348–355: **Acknowledgments** — The authors thank the anonymous reviewers for their helpful comments.
- righe 356–359: **Impact Statement** — This paper presents work whose goal is to advance the field of Machine Learning.
- righe 360–425: **References** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 426–431: **Appendix A Detailed Implementations** — In this section, we present the algorithm for KIVI as discussed in Section 3.3 .
- righe 432–445: **Appendix B NIAH Setting** — We largely follows the passkey retrieval prompt template of Mohtashami and Jaggi [ 2023 ] but using 7-digit passkey and Paul Graham Essays 2 2 2 https://paulgraham.com/articles.html as the background filler, as set forth…
- righe 446–453: **Appendix C More Ablation Results** — In our efficiency evaluation, we observe that with a residual length of 32, KIVI achieves a significantly higher memory compression rate, which in turn leads to increased throughput.
- righe 454–467: **Appendix D More Experimental Results** — We present additional results using Llama3-8B, Mistral-7B-v0.2, and LongChat-7B-v1.5 in LongBench, which can be found in Table 8 , Table 9 and Table 10 , respectively.

## [QUICK](2402.10076-quick.md)

derivation

- righe 8–22: **Testo estratto** — derivation
- righe 23–199: **QUICK: Quantization-aware Interleaving and Conflict-free Kernel for efficient LLM inference** — We introduce QUICK, a group of novel optimized CUDA kernels for the efficient inference of quantized Large Language Models (LLMs).
        - righe 25–30: **Abstract** — We introduce QUICK, a group of novel optimized CUDA kernels for the efficient inference of quantized Large Language Models (LLMs).
- righe 31–38: **1 Introduction** — Enhancing the efficiency of Large Language Models (LLMs) has become increasingly crucial due to the escalating demand for deploying state-of-the-art models in real-world scenarios [ 2 , 8 , 9 , 15 , 16 ] .
- righe 39–76: **2 Preliminary** — Quantization involves the reduction of precision or range of a continuous variable to a discrete set of values.
  - righe 41–48: **2.1 Quantization and Dequantization** — Quantization involves the reduction of precision or range of a continuous variable to a discrete set of values.
  - righe 49–60: **2.2 GEMM kernel using Tensor Core** — A substantial portion of the computational workload associated with LLMs primarily comprises GEMMs.
  - righe 61–76: **2.3 Mixed precision GEMM kernel** — Figure 2 : Computation overview of original kernel and QUICK.
- righe 77–109: **3 Avoiding Bank Conflict** — In this section, we propose QUICK, a novel way to remove the shared memory write-back bank conflicts of mixed precision matrix multiplication.
  - righe 81–89: **3.1 Skipping Shared Memory Write-back During Mixed Precision GEMM** — As previously discussed, state-of-the-art mixed precision GEMM kernels rely on a specific sequence involving dequantization, shared memory write-back, ldmatrix , and mma .
  - righe 90–103: **3.2 Interleaving Data Pattern** — Figure 4 : ldmatrix instruction-aware weight interleaving to avoid shared memory conflicts.
  - righe 104–109: **3.3 Tile Size Optimization** — Optimizing the number of active warps per multiprocessor plays an important role in improving the performance of computation kernels.
- righe 110–150: **4 Experimental Results** — In this section, we evaluate the performance improvement provided by QUICK in comparison to both the baseline fp16 kernel and AutoAWQ-Kernel.
  - righe 117–129: **4.1 Matrix Multiplication Performance** — We initially evaluate the performance of QUICK with unit matrix multiplications, with the matrix multiplication dimensions set to .
  - righe 130–144: **4.2 End-to-end Throughput** — Figure 8 : End-to-end token generation throughput benchmarks of (a) Mistral-7B [ 8 ] on RTX 4090, (b) Vicuna-13B [ 3 ] on RTX A6000, (c) LLaMA-2-13B [ 15 ] on L40, and (d) LLaMA-33B [ 16 ] on A100.
  - righe 145–150: **4.3 vLLM Throughput** — In this section, we present the throughput benchmark results of our initial version of vLLM [ 10 ] integrated with QUICK (Table 1 ).
- righe 151–157: **5 Limitation and Future Work** — While the proposed QUICK technique has demonstrated enhanced throughput at larger batch sizes, such as 128, enabling the utilization of weight-only quantization for larger batch sizes, it still falls short of the efficie…
- righe 158–164: **6 Conclusion** — In this work, we introduce QUICK, a suite of optimized CUDA kernels designed for efficient execution of mixed precision GEMM operations.
- righe 165–199: **References** — Sezione strutturale; consultare il contenuto locale indicato.

## [LLM Inference Performance Engineering](2402.16363-llm-inference-roofline.md)

\ul

- righe 8–11: **Testo estratto** — \ul
- righe 12–1203: **LLM Inference Unveiled: Survey and Roofline Model Insights** — The field of efficient Large Language Model (LLM) inference is rapidly evolving, presenting a unique blend of opportunities and challenges.
        - righe 14–24: **Abstract** — The field of efficient Large Language Model (LLM) inference is rapidly evolving, presenting a unique blend of opportunities and challenges.
- righe 25–47: **1 Introduction** — Figure 1 : Workflow of our designed LLM-Viewer.
- righe 48–142: **2 Delve into LLM Inference and Deployment** — Figure 3 : Demonstration of the architecture of LLMs.
  - righe 50–87: **2.1 LLM Inference** — Figure 3 : Demonstration of the architecture of LLMs.
  - righe 88–127: **2.2 Roofline Model** — Assessing the efficiency at which LLMs deploy onto specific hardware involves a comprehensive consideration of both hardware and model characteristics.
  - righe 128–142: **2.3 LLM-Viewer** — There are multiple Transformer layers in LLMs, each containing various operations.
- righe 143–313: **3 Model Compression** — The formidable size and computational demands of Large Language Models (LLMs) present significant challenges for practical deployment, especially in resource-constrained environments.
  - righe 149–253: **3.1 Quantization** — In the realm of LLM compression, quantization has become a pivotal technique for mitigating the substantial storage and computational overhead associated with these models.
    - righe 156–193: **3.1.1 A Use Case of LLM-Viewer:** — Roofline Analysis for Quantization
    - righe 194–233: **3.1.2 Quantization for Compressing Pre-trained LLMs** — In Quantization-Aware Training (QAT) [ Courbariaux et al., , 2015 , Choi et al., , 2018 , Dong et al., , 2019 ] , the quantization process is seamlessly integrated into the training of Large Language Models (LLMs), enabl…
    - righe 234–249: **3.1.3 Quantization for Parameter Efficient Fine-Tuning (Q-PEFT)** — Parameter Efficient Fine-Tuning (PEFT) is an important topic for LLMs.
    - righe 250–253: **3.1.4 Discussion on LLM Quantiztaion** — Figure 10 presents a timeline of LLM quantization techniques, highlighting the evolution from Post-Training Quantization (PTQ) as the initial mainstream approach to the rising prominence of Quantization-Aware Training (Q…
  - righe 254–274: **3.2 Pruning** — Pruning [ LeCun et al., , 1989 , Liang et al., , 2021 ] , which concentrates on identifying and eliminating model parameters that are deemed unnecessary or redundant, is another popular technique for compressing LLMs.
    - righe 260–267: **3.2.1 Unstructured pruning** — Unstructured pruning selectively eliminates individual weights or neurons from a model, leading to a sparser, yet more irregularly structured network.
    - righe 268–274: **3.2.2 Structured pruning** — Structured pruning removes entire neurons or layers, resulting in a cleaner, more regular structure.
  - righe 275–304: **3.3 Knowledge Distillation** — Knowledge distillation [ Hinton et al., , 2015 , Gou et al., , 2021 ] is a technique that facilitates the transfer of capabilities from a larger model (referred to as the “teacher”) to a smaller model (referred to as the…
    - righe 281–289: **3.3.1 White-Box Knowledge Distillation** — In white-box distillation, the architecture and weights of the teacher model are fully accessible.
    - righe 290–304: **3.3.2 Black-Box Knowledge Distillation** — Contrary to white-box distillation, black-box distillation does not require access to the internal information of the teacher model.
  - righe 305–313: **3.4 Factorization** — The use of low-rank matrix decomposition [ Kishore Kumar and Schneider, , 2017 ] as a technique for compressing Deep Neural Networks (DNNs) represents a straightforward yet effective approach, garnering considerable atte…
- righe 314–434: **4 Algorithmic Methods for Fast Decoding** — Figure 12 : Illustration of Input-Dependent Dynamic Network Technique
  - righe 333–391: **4.1 Minimum Parameter Used Per Token Decoded** — Interestingly, Simoulin and Crabbé, [ 2021 ] has shown that although language models tend to have a huge number of parameters, not all parameters are needed to generate the accurate tokens.
    - righe 337–367: **4.1.1 Early Exiting** — Early exiting (or layer skipping) has been a well-explored idea in various network architectures, particularly for the encoder-only models [ Baier-Reinio and Sterck, , 2020 , Hou et al., , 2020 , Li et al., , 2021 , Liu …
    - righe 368–375: **4.1.2 Contextual Sparsity** — While early exiting aims to select parameters on the depth dimension, some techniques have also been proposed to exploit the dynamic sparsity on the width dimension.
    - righe 376–384: **4.1.3 Mixture-of-Expert Models** — Language Models, especially transformer architecture, exhibit strong power-law scaling ( Kaplan et al., [ 2020 ] , Hoffmann et al., [ 2022 ] ) of performance when the training dataset is scaled up.
    - righe 385–391: **4.1.4 Roofline Model Analysis for Dynamic Parameter Reducing** — The Minimum Parameter Used Per Token Decoded methods simultaneously decrease computational and memory access overhead.
  - righe 392–434: **4.2 Maximum Tokens Decoded Per LLM Forward Propagation** — Another angle to reduce the latency of LLM inference is to relax the LLM from the limitation of autoregressive decoding and have more than one token decoded per one LLM forward propagation.
    - righe 397–412: **4.2.1 Speculative Decoding** — Due to the demanding memory loading challenges and autoregressive properties, LLMs are inefficient in inference.
    - righe 413–434: **4.2.2 Parallel Decoding** — Alternatively, abundant works have been proposed to enable the large model to directly perform parallel decoding without the help of a small transformer model.
- righe 435–523: **5 Compiler/System Optimization** — After model compression and algorithm optimization for LLMs, the next step is to compile and deploy them on hardware devices.
  - righe 442–483: **5.1 Operator Fusion** — Figure 14 : Demonstration of operator fusion for a linear operator followed by a SiLU operator.
  - righe 484–507: **5.2 Memory Management and Workload Offloading** — When using an LLM to generate responses, the number of input and output tokens can change each time.
  - righe 508–523: **5.3 Parallel Serving** — Parallel serving handles multiple user requests to a server at the same time.
- righe 524–622: **6 Hardware Optimization** — Designing hardware to efficiently support inference for LLMs is a challenging task due to the varying arithmetic intensity 5 5 5 Arithmetic intensity refers to the ratio of arithmetic operations to memory access, which h…
  - righe 533–546: **6.1 Spatial Architecture** — The decoding process of LLM involves predicting words one at a time based on previously generated ones.
  - righe 547–572: **6.2 Processing in Memory** — The decoding phase of LLM inference experiences the so-called ”Memory Wall” problem, primarily due to its low arithmetic intensity.
  - righe 573–613: **6.3 New Data Format** — Neural networks typically employ high-precision floating-point numbers (16 or 32 bits) for training.
  - righe 614–622: **6.4 New Processing Element** — Except for the high demand for memory access, there has been a growing interest in developing specialized processing elements (PEs) to boost the computation.
- righe 623–684: **7 Discussion** — The discussion above significantly enhances the inference and training efficiency of LLMs in practical scenarios.
  - righe 625–638: **7.1 Reliability** — The discussion above significantly enhances the inference and training efficiency of LLMs in practical scenarios.
    - righe 631–638: **7.1.1 Hallucination** — The ability of an LLM to suppress hallucinations is critically affected by modifications to its parameters.
  - righe 639–642: **7.2 Safety Alignment** — Based on previous research findings Yuan et al., 2023c , moderate model compression, such as 8-bit quantization, does not significantly compromise the safety capabilities of models.
  - righe 643–646: **7.3 OOD Generalization** — Large language models, when deployed in real-world scenarios, are often influenced by decision shortcuts, leading to erroneous decisions within the long-tail subgroup distributions Geirhos et al., [ 2020 ] .
  - righe 647–660: **7.4 Efficient Large Multimodal Models** — Large Multimodal Models (LMMs), particularly Visual Language Models (VLMs), have emerged as a promising avenue for creating general-purpose assistants, showcasing significant enhancements in perception and reasoning capa…
    - righe 649–652: **7.4.1 Large Multimodal Models (LMMs)** — Large Multimodal Models (LMMs), particularly Visual Language Models (VLMs), have emerged as a promising avenue for creating general-purpose assistants, showcasing significant enhancements in perception and reasoning capa…
    - righe 653–660: **7.4.2 Efficient LMMs** — The need for cross-modality capabilities in resource-limited scenarios has become increasingly apparent.
  - righe 661–684: **7.5 Long Context Modeling** — When used for tasks like Chatbot or document summarization tools, Large Language Models’ long context language modeling and reasoning capabilities are challenged.
    - righe 665–671: **7.5.1 Alternative Attention Design** — Lying in the core of transformer architecture is the self-attention mechanism.
    - righe 672–678: **7.5.2 Recurrence and Retrieval** — Transformer-XL Dai et al., [ 2019 ] proposes to introduce the segment-level recurrence structure to the Language Model to boost the current language model in long-context capabilities.
    - righe 679–684: **7.5.3 Maneuvering Position Encodings** — During pretraining, the position encoding of the transformer hasn’t seen an input sequence length longer than a fixed limit.
- righe 685–692: **8 Conclusion** — In this work, we review on efficient large language model (LLM) inference.
- righe 693–1203: **References** — Sezione strutturale; consultare il contenuto locale indicato.

## [Sarathi-Serve](2403.02310-sarathi-serve.md)

Each LLM serving request goes through two phases.

- righe 8–9: **Testo estratto** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 10–525: **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve** — Each LLM serving request goes through two phases.
        - righe 12–19: **Abstract** — Each LLM serving request goes through two phases.
- righe 20–68: **1 Introduction** — Large language models (LLMs) [ 71 , 34 , 35 , 57 , 52 ] have shown impressive abilities in a wide variety of tasks spanning natural language processing, question answering, code generation, etc.
- righe 69–120: **2 Background** — In this section, we describe the typical LLM model architecture along with their auto-regressive inference process.
  - righe 73–80: **2.1 The Transformer Architecture** — Popular large language models, like, GPT-3 [ 18 ] , LLaMA [ 66 ] , Yi [ 24 ] etc.
  - righe 81–94: **2.2 LLM Inference Process** — Autoregressive decoding: LLM inference consists of two distinct phases – a prefill phase followed by a decode phase.
  - righe 95–102: **2.3 Multi-GPU LLM Inference** — With ever-increasing growth in model sizes, it becomes necessary to scale LLMs to multi-GPU or even multi-node deployments [ 59 , 22 ] .
  - righe 103–108: **2.4 Performance Metrics** — There are two primary latency metrics of interest for LLM serving: TTFT (time-to-first-token) and TBT (time-between-tokens).
  - righe 109–120: **2.5 Scheduling Policies for LLM Inference** — The scheduler is responsible for admission control and batching policy.
- righe 121–173: **3 Motivation** — Figure 3 : Throughput of the prefill and decode phases with different batch sizes for Mistral-7B running on a single A100 GPU.
  - righe 127–149: **3.1 Cost Analysis of Prefill and Decode** — As discussed in §2.2 , while the prefill phase processes all input tokens in parallel and effectively saturates GPU compute, the decode phase processes only a single token at a time and is very inefficient.
  - righe 150–163: **3.2 Throughput-Latency Trade-off** — Iteration-level batching improves system throughput but we show that it comes at the cost of high TBT latency due to a phenomenon we call generation stalls .
  - righe 164–173: **3.3 Pipeline Bubbles waste GPU Cycles** — Pipeline-parallelism (PP) is a popular strategy for cross-node deployment of large models, owing to its lower communication overheads compared to Tensor Parallelism (TP).
- righe 174–220: **4 Sarathi-Serve: Design and Implementation** — We now discuss the design and implementation of Sarathi-Serve — a system that provides high throughput with predictable tail latency via two key techniques – chunked-prefills and stall-free batching .
  - righe 184–191: **4.1 Chunked-prefills** — As we show in §3.1 , decode batches are heavily memory bound with low arithmetic intensity.
  - righe 192–199: **4.2 Stall-free batching** — The Sarathi-Serve scheduler is an iteration-level scheduler that leverages chunked-prefills and coalescing of prefills and decodes to improve throughput while minimizing latency.
  - righe 200–213: **4.3 Determining Token Budget** — The token budget is determined based on two competing factors — TBT SLO requirement and chunked-prefills overhead.
  - righe 214–220: **4.4 Implementation** — We implement Sarathi-Serve on top of the open-source implementation of vLLM [ 53 , 23 ] .
- righe 221–317: **5 Evaluation** — We evaluate Sarathi-Serve on a variety of popular models and GPU configurations (see Table 1 ) and two datasets (see Table 2 ).
  - righe 258–275: **5.1 Capacity Evaluation** — (a) Dataset: openchat_sharegpt4 .
  - righe 276–286: **5.2 Throughput-Latency Tradeoff** — To fully understand the throughput-latency tradeoff in LLM serving systems, we vary the P99 TBT latency SLO and observe the impact on system capacity for vLLM and Sarathi-Serve.
  - righe 287–301: **5.3 Making Pipeline Parallel Viable** — We now show that Sarathi-Serve makes it feasible to efficiently serve LLM inference across commodity networks with efficient pipeline parallelism.
  - righe 302–317: **5.4 Ablation Study** — In this subsection, we conduct an ablation study on different aspects on Sarathi-Serve.
    - righe 306–313: **5.4.1 Overhead of chunked-prefills** — Figure 14 shows how much overhead chunking adds in Yi-34B – on overall prefill runtime.
    - righe 314–317: **5.4.2 Impact of individual techniques** — Finally, Table 4 shows the TTFT and TBT latency with each component of Sarathi-Serve evaluated in isolation i.e., chunked-prefills -only, hybrid-batching -only (mixed batches with both prefill and decode requests) and wh…
- righe 318–329: **6 Related Work** — Model serving systems: Systems such as Clipper [ 37 ] , TensorFlow-Serving [ 56 ] , Clockwork [ 45 ] and BatchMaker [ 44 ] study various placement, caching and batching strategies for model serving.
- righe 330–335: **7 Conclusion** — Optimizing LLM inference for high throughput and low latency is desirable but challenging.
- righe 336–340: **8 Acknowledgement** — We would like to thank OSDI reviewers and our shepherd for their insightful feedback.
- righe 341–496: **References** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 497–525: **Appendix A Artifact Appendix** — Our open source artifact is available on GitHub .
  - righe 499–505: **Abstract** — Our open source artifact is available on GitHub .
  - righe 506–512: **Scope** — This artifact allows the readers to validate the claims made
  - righe 513–516: **Contents** — The repository is structured as follows, the primary source code for the system is contained in directory /sarathi .
  - righe 517–522: **Hosting** — You can obtain our artifacts from GitHub:
  - righe 523–525: **Requirements** — Sarathi-Serve has been tested with CUDA 12.1 on A100 and A40 GPUs.

## [Better and Faster Large Language Models via Multi-token Prediction](2404.19737-multi-token-prediction.md)

Large language models such as GPT and Llama are trained with a next-token prediction loss.

- righe 8–9: **Testo estratto** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 10–572: **Better & Faster Large Language Models via Multi-token Prediction** — Large language models such as GPT and Llama are trained with a next-token prediction loss.
        - righe 12–20: **Abstract** — Large language models such as GPT and Llama are trained with a next-token prediction loss.
- righe 21–58: **1 Introduction** — Humanity has condensed its most ingenious undertakings, surprising findings and beautiful productions into text.
    - righe 40–58: **Contributions** — While multi-token prediction has been studied in previous literature (Qi et al., 2020 ) , the present work offers the following contributions:
- righe 59–94: **2 Method** — Standard language modeling learns about a large text corpus by implementing a next-token prediction task.
    - righe 78–87: **Memory-efficient implementation** — One big challenge in training multi-token predictors is reducing their GPU memory utilization.
    - righe 88–94: **Inference** — During inference time, the most basic use of the proposed architecture is vanilla next-token autoregressive prediction using the next-token prediction head , while discarding all others.
- righe 95–169: **3 Experiments on real data** — Table 1 : Multi-token prediction improves performance and unlocks efficient byte level training.
  - righe 118–124: **3.1 Benefits scale with model size** — To study this phenomenon, we train models of six sizes in the range 300M to 13B parameters from scratch on at least 91B tokens of code.
  - righe 125–128: **3.2 Faster inference** — We implement greedy self-speculative decoding (Stern et al., 2018 ) with heterogeneous batch sizes using xFormers (Lefaudeux et al., 2022 ) and measure decoding speeds of our best 4-token prediction model with 7B paramet…
  - righe 129–134: **3.3 Learning global patterns with multi-byte prediction** — To show that the next-token prediction task latches to local patterns, we went to the extreme case of byte-level tokenization by training a 7B parameter byte-level transformer on 314B bytes, which is equivalent to around…
  - righe 135–138: **3.4 Searching for the optimal** — To better understand the effect of the number of predicted tokens, we did comprehensive ablations on models of scale 7B trained on 200B tokens of code.
  - righe 139–142: **3.5 Training for multiple epochs** — Multi-token training still maintains an edge on next-token prediction when trained on multiple epochs of the same data.
  - righe 143–149: **3.6 Finetuning multi-token predictors** — Pretrained models with multi-token prediction loss also outperform next-token models for use in finetunings.
  - righe 150–169: **3.7 Multi-token prediction on natural language** — To evaluate multi-token prediction training on natural language, we train models of size 7B parameters on 200B tokens of natural language with a 4-token, 2-token and next-token prediction loss, respectively.
- righe 170–193: **4 Ablations on synthetic data** — What drives the improvements in downstream performance of multi-token prediction models on all of the tasks we have considered?
  - righe 175–185: **4.1 Induction capability** — Figure 7 : Induction capability of -token prediction models.
  - righe 186–193: **4.2 Algorithmic reasoning** — Figure 8 : Accuracy on a polynomial arithmetic task with varying number of operations per expression.
- righe 194–226: **5 Why does it work? Some speculation** — Why does multi-token prediction afford superior performance on coding evaluation benchmarks, and on small algorithmic reasoning tasks?
  - righe 200–211: **5.1 Lookahead reinforces choice points** — Figure 9 : Multi-token prediction loss assigns higher implicit weights to consequential tokens.
  - righe 212–226: **5.2 Information-theoretic argument** — Language models are typically trained by teacher-forcing, where the model receives the ground truth for each future token during training.
- righe 227–245: **6 Related work** — Dong et al.
    - righe 229–232: **Language modeling losses** — Dong et al.
    - righe 233–237: **Multi-token prediction in language modelling** — Qi et al.
    - righe 238–241: **Self-speculative decoding** — Stern et al.
    - righe 242–245: **Multi-target prediction** — Multi-task learning is the paradigm of training neural networks jointly on several tasks to improve performance on the tasks of interest (Caruana, 1997 ) .
- righe 246–256: **7 Conclusion** — We have proposed multi-token prediction as an improvement over next-token prediction in training language models for generative or reasoning tasks.
- righe 257–261: **Impact statement** — The goal of this paper is to make language models more compute and data efficient.
- righe 262–265: **Environmental impact** — In aggregate, training all models reported in the paper required around 500K GPU hours of computation on hardware of type A100-80GB and H100.
- righe 266–269: **Acknowledgements** — We thank Jianyu Zhang, Léon Bottou, Emmanuel Dupoux, Pierre-Emmanuel Mazaré, Yann LeCun, Quentin Garrido, Megi Dervishi, Mathurin Videau and Timothée Darcet and other FAIR PhD students and CodeGen team members for helpfu…
- righe 270–385: **References** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 386–395: **Appendix A Additional results on self-speculative decoding** — Figure S10 : Decoding speeds and latencies with self-speculative decoding relative to standard autoregressive decoding.
- righe 396–419: **Appendix B Alternative architectures** — Table S4 : Alternative architectures improve on baseline but not as consistently.
    - righe 402–405: **Replicated unembeddings** — Replicating the unembedding matrix times is a simple method for implementing multi-token prediction architectures.
    - righe 406–409: **Linear heads** — Apart from using a single transformer layer for the heads , other architectures are conceivable.
    - righe 410–419: **Causal and anticausal variant** — Instead of making the prediction heads architecturally independent of each other, we can also allow them to rely on other heads’ (pre-unembedding) outputs.
- righe 420–423: **Appendix C Training speeds** — Table S5 : Training time relative to next-token prediction training.
- righe 424–427: **Appendix D Finetuning** — Table S6 : Finetuning LLama 2 with multi-token prediction does not significantly improve performance.
- righe 428–431: **Appendix E Additional results on model scaling behavior** — Table S7 : Scaling model size Full results of scaling model size with n=1,2 and 4.
- righe 432–435: **Appendix F Details on CodeContests finetuning** — We use the Python subset of the CodeContests [Li et al., 2022 ] train split with reward annotations (“correct” / “incorrect”) and condition on correct solutions at evaluation time.
- righe 436–441: **Appendix G Additional results on natural language benchmarks** — We evaluate the models from Section 3.7 on standard natural language processing benchmarks: ARC Challenge [Yadav et al., 2019 ] , COPA [Roemmele et al., 2011 ] , Hellaswag [Zellers et al., 2019 ] , Natural Questions [Kwi…
- righe 442–455: **Appendix H Additional results on abstractive text summarization** — Table S8 : Comprehensive evaluation on abstractive text summarization.
- righe 456–459: **Appendix I Additional results on mathematical reasoning in natural language** — Figure S13 : Performance on the mathematical reasoning benchmark GSM8K [Cobbe et al., 2021 ] .
- righe 460–464: **Appendix J Additional results on induction learning** — Figure S14 : Induction capability of -token prediction models trained on higher-quality data.
- righe 465–486: **Appendix K Additional results on algorithmic reasoning** — We investigate the following computation-sharing hypothesis for explaining the efficacy of multi-token prediction as training loss.
- righe 487–567: **Appendix L Additional intuitions on multi-token prediction** — In Section 5.2 , we argued that multi-token prediction reduces the distribution mismatch between teacher-forced training and autoregressive evaluation of language models.
  - righe 489–494: **L.1 Comparison to scheduled sampling** — In Section 5.2 , we argued that multi-token prediction reduces the distribution mismatch between teacher-forced training and autoregressive evaluation of language models.
  - righe 495–528: **L.2 Information-theoretic argument** — We give details on the information-theoretic terms appearing in the decomposition in Section 5.2 and derive a relative version that similarly allows to decompose multi-token prediction losses.
        - righe 503–506: **Definition L.1 .** — The conditional cross-entropy of conditioned on from to is defined as the expectation under of the cross-entropy between the distributions and conditioned on , in formulas:
        - righe 507–514: **Definition L.2 .** — The relative mutual information of and from relative to is defined by
        - righe 515–516: **Lemma L.3 .** — Sezione strutturale; consultare il contenuto locale indicato.
        - righe 517–528: **Proof.** — We calculate
  - righe 529–551: **L.3 Lookahead reinforces choice points** — Figure S17 : Example of a sequential prediction task with derailing.
  - righe 552–567: **L.4 Factorization orders** — Causal language modelling factorizes probabilities over text sequences classically as
- righe 568–572: **Appendix M Training hyperparameters** — Table S13 : Overview of all training hyperparameters used.

## [QServe](2405.04532-qserve.md)

\gappto

- righe 8–36: **Testo estratto** — \gappto
        - righe 32–36: **Abstract** — Quantization can accelerate large language model (LLM) inference.
- righe 37–53: **1 Introduction** — Figure 1: QServe achieves higher throughput when running Llama models on L40S compared with TensorRT-LLM on A100, effectively saves the dollar cost for LLM serving by 3 through system-algorithm codesign.
- righe 54–73: **2 Background** — Large Language Models (LLMs) are a family of causal transformer models with multiple identically-structured layers.
  - righe 56–63: **2.1 Large Language Models** — Large Language Models (LLMs) are a family of causal transformer models with multiple identically-structured layers.
  - righe 64–73: **2.2 Integer Quantization** — Integer quantization maps high-precision numbers to discrete levels.
- righe 74–105: **3 Motivation** — In this paper, we denote -bit weight, -bit activation and -bit KV cache quantization in LLMs as WxAyKVz , and use the abbreviated notation WxAy if y=z .
  - righe 80–91: **3.1 W4A8KV4 Has Superior Roofline Over W8A8 , W4A16** — Figure 2: Left : Both attention and GEMM are crucial for end-to-end LLM latency.
  - righe 92–105: **3.2 Why Not W4A4KV4 : Main Loop Overhead in GEMM** — Figure 4: Illustration of GPU GEMM : are parallel dimensions and the reduction dimension has a sequential main loop.
- righe 106–186: **4 QoQ Quantization** — To this end, we have discussed why W4A8KV4 is a superior quantization precision choice.
  - righe 110–141: **4.1 Progressive Group Quantization** — Figure 6: Progressive Group Quantization first employs per-channel INT8 quantization with protective range [-119, 119], followed by per-group INT4 quantization, so that the dequantized intermediate values remain within t…
      - righe 124–133: **Protective Quantization Range.** — Naively applying Equation 4 and 5 does not guarantee that the intermediate dequantized weights perfectly lie in the 8-bit integer representation range ( i.e .
      - righe 134–141: **Compared to previous two-level quantization.** — Progressive group quantization introduces two levels of scales and .
  - righe 142–157: **4.2 SmoothAttention** — Figure 7: SmoothAttention effectively smooths the outliers in Keys.
  - righe 158–186: **4.3 General LLM Quantization Optimizations** — One of the key challenges of low-bit LLM quantization is the activation outliers for every linear layers.
    - righe 166–171: **4.3.1 Block Input Module Rotation** — In transformer blocks, we define the components that take in the block inputs as input modules, such as the QKV Projection Layer and the FFN 1st Layer.
    - righe 172–175: **4.3.2 Block Output Module Smoothing** — Output modules refer to those layers that generate block outputs, such as the Output Projection Layer and FFN 2nd Layer.
    - righe 176–181: **4.3.3 Activation-Aware Channel Reordering** — Figure 10: Reorder weight input channels based on their salience in group quantization.
    - righe 182–186: **4.3.4 Weight Clipping** — Weight clipping is another popular quantization optimization technique.
- righe 187–263: **5 QServe Serving System** — To this end, we have presented the QoQ quantization algorithm, which aims to minimize accuracy loss incurred by W4A8KV4 quantization.
  - righe 191–201: **5.1 QServe System Runtime** — Figure 11: QServe’s precision mapping for an FP16 in, FP16 out LLM block.
  - righe 202–243: **5.2 W4A8 GEMM in QServe** — As discussed in Section 3 , the main loop overhead poses a significant obstacle in allowing quantized GEMMs to attain the theoretical performance gains projected by the roofline model (Figure 3 ).
    - righe 206–215: **5.2.1 Compute-Aware Weight Reorder** — Figure 12: QServe applies compute-aware weight reoder to minimize the pointer arithmetics in W4A8 GEMM main loop.
    - righe 216–231: **5.2.2 Fast Dequantization in Per-Channel W4A8 GEMM** — Figure 13: QServe exploits register-level parallelism to significantly reduce the number of required logical operations in UINT4 to UINT8 weight unpacking.
    - righe 232–239: **5.2.3 Fast Dequantization in Per-Group W4A8 GEMM** — Figure 14: Our progressive quantization algorithm ensures that all intermediate results in the subtraction after multiplication computation order will not overflow, thereby enabling register-level parallelism and reducin…
    - righe 240–243: **5.2.4 General Optimizations** — In our W4A8 kernel, we also employ general techniques for GEMM optimization.
  - righe 244–263: **5.3 KV4 Attention in QServe** — Table 1: A naive KV4 attention implementation is 1.7 faster on L40S than TRT-LLM- KV8 , but is 1.1-1.2 slower on A100 due to earlier CUDA core roofline turning point.
- righe 264–344: **6 Evaluation** — Algorithm.
  - righe 266–273: **6.1 Evaluation Setup** — Algorithm.
  - righe 274–323: **6.2 Accuracy Evaluation** — We evaluated QoQ on the Llama-1 Touvron et al.
      - righe 276–279: **Benchmarks.** — We evaluated QoQ on the Llama-1 Touvron et al.
      - righe 280–284: **Baselines.** — We compared QoQ to widely used post-training LLM quantization techiniques, SmoothQuant Xiao et al.
      - righe 285–288: **WikiText2 perplexity.** — Table 2 compares the Wikitext2 perplexity results between QoQ and other baselines.
      - righe 289–323: **Zero-shot Accuracy and Long-Context Accuracy.** — We report the zero-shot accuracy of five common sense tasks in Table 3 .
  - righe 324–329: **6.3 Efficiency Evaluation** — We assessed the efficiency of QServe on A100-80G-SXM4 and L40S-48G GPUs by comparing it against TensorRT-LLM (using FP16 , W8A8 , and W4A16 precisions), Atom ( W4A4 ), and QuaRot ( W4A4 ).
  - righe 330–344: **6.4 Analysis and Discussion.** — Ablation study on quantization techniques.
- righe 345–356: **7 Related Work** — Quantization of LLMs.
- righe 357–360: **8 Conclusion** — We introduce QServe, an algorithm and system co-design framework tailored to quantize large language models (LLMs) to W4A8KV4 precision, facilitating their efficient deployment on GPUs.
- righe 361–364: **Acknowledgements** — We thank MIT-IBM Watson AI Lab, MIT AI Hardware Program, MIT Amazon Science Hub, and NSF for supporting this research.
- righe 365–456: **References** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 457–597: **Appendix A Artifact Appendix** — This artifact contains necessary scripts and dependencies
  - righe 459–463: **A.1 Abstract** — This artifact contains necessary scripts and dependencies
  - righe 464–549: **A.2 Artifact check-list (meta-information)** — Program: Efficiency benchmarking code for QServe; efficiency benchmarking code for baseline systems such as TensorRT-LLM.
  - righe 550–563: **A.3 Description** — We will provide AE reviewers with a pre-built docker image containing QServe, TensorRT-LLM and all necessary dependencies.
    - righe 552–555: **A.3.1 How delivered** — We will provide AE reviewers with a pre-built docker image containing QServe, TensorRT-LLM and all necessary dependencies.
    - righe 556–559: **A.3.2 Hardware dependencies** — A host machine with x86_64 CPUs and at least one NVIDIA A100 GPU (recommended) or L40S GPU.
    - righe 560–563: **A.3.3 Software dependencies** — A GPU-compatible Docker runtime environment is required.
  - righe 564–567: **A.4 Installation** — We recommend that users utilize our pre-built Docker images to set up the environment and run all experiments within the GPU-supported Docker container.
  - righe 568–571: **A.5 Experiment workflow** — The generation throughputs of QServe and baseline system (i.e., TensorRT-LLM) can be measured with the following commands.
  - righe 572–577: **A.6 Evaluation and expected result** — Table 6: Generation throughput of QServe and baseline (TensorRT-LLM).
  - righe 578–582: **A.7 Experiment customization** — The users are encouraged to carry out experiments with different models and batch sizes by modifying the benchmarking scripts.
  - righe 583–597: **A.8 Methodology** — Submission, reviewing and badging methodology:

## [Marlin](2408.11743-marlin.md)

marginparsep has been altered.

- righe 8–37: **Testo estratto** — marginparsep has been altered.
        - righe 32–37: **Abstract** — As inference on Large Language Models (LLMs) emerges as an important workload in machine learning applications, weight quantization has become a standard technique for efficient GPU deployment.
- righe 38–74: **1 Introduction** — The capabilities of large language models (LLMs) Radford et al.
      - righe 50–74: **Contribution.** — In this work, we investigate software support for LLM inference acceleration via mixed-precision in the general batched case.
- righe 75–129: **2 Background** — We continue with an overview of GPU architecture, and the CUDA programming and execution model.
  - righe 79–109: **2.1 Graphics Processing Units** — NVIDIA GPUs comprise an array of Streaming Multiprocessor (SM) elements that share a DRAM memory, known as Global MEMory (GMEM) and an L2 cache.
    - righe 93–109: **2.1.1 Modern Tensor Core Units** — Figure 2: Illustration of asynchronous copy operation with and without L1 bypass (right) vs.
  - righe 110–129: **2.2 Mixed-Precision Inference on LLMs** — Mixed-precision LLM inference offers the potential to reduce a model’s large memory footprint, and correspondingly accelerate memory-bound workloads by statically compressing pretrained model weights while decompressing …
      - righe 114–129: **Weight Quantization** — .
- righe 130–257: **3 The MARLIN Kernel** — LLM weight quantization is motivated by the fact that modern GPUs have large FLOPs/Bytes ratios, meaning that they can execute floating point operations much faster than they can read from memory.
  - righe 132–140: **3.1 Motivation** — LLM weight quantization is motivated by the fact that modern GPUs have large FLOPs/Bytes ratios, meaning that they can execute floating point operations much faster than they can read from memory.
  - righe 141–156: **3.2 Ampere Matrix Multiplication** — We begin by describing the general concepts used to implement peak performing (uniform precision) matrix multiplication kernels on GPUs, in particular on Ampere class devices.
      - righe 145–148: **SM Level.** — As a first step, is partitioned into blocks , into blocks and into blocks .
      - righe 149–152: **Warp Level.** — Within the sub-problem considered by a single SM, another equivalent partitioning, this time with parameters , , and , is performed.
      - righe 153–156: **Tensor Core Level.** — Eventually, each warp will repeatedly multiply and matrices.
  - righe 157–178: **3.3 Mixed-Precision Challenges** — Adapting the above uniform precision matmul to the mixed-precision case while maintaining peak performance, in particular for medium where the operation is (close to) memory-bound, is challenging for the following reason…
  - righe 179–249: **3.4 Kernel Design** — In what follows, we assume that the matrix is in full FP16 precision, while the matrix has been (symmetrically) quantized to INT4, either with one FP16 scale for each of the columns or one scale per consecutive weights i…
      - righe 183–192: **Bound By Weight Loading.** — Executing our target matmul requires, in theory, touching exactly bits of memory (reading both operands and writing the results) while executing exactly multiply-accumulate operations, each counted as 2 FLOPs.
      - righe 193–198: **Maximizing Loading Bandwidth.** — In order to maximize practical loading bandwidth, we aim to utilize the widest loads possible; on current GPUs 16 bytes (128 bits) per thread.
      - righe 199–209: **Shared Memory Layouts.** — Overall, we always load asynchronously via Ampere’s cp.async instruction from global (or L2) to shared memory; this requires no temporary registers and also makes overlapping these loads with computation much easier.
      - righe 210–215: **Memory Load Pipelining.** — The key to simultaneously reaching close to maximum bandwidth and close to maximum compute is to fully overlap memory loading and Tensor Core math.
      - righe 216–221: **Warp Layout.** — The computation of on a single SM must further be subdivided across warps: if done in direct fashion, each warp would compute an tile of the output.
      - righe 222–231: **Dequantization and Tensor Cores.** — Doing naive type-casts from INT4 to FP16 is slow; instead, we follow a modified version of the binary manipulations of Kim et al.
      - righe 232–237: **Groups and Instruction Ordering.** — For per-output quantization, we can simply scale the final output once before the global write-out.
      - righe 238–249: **Striped Partitioning.** — With all the techniques described so far, we can reach near optimal compute and bandwidth performance, provided matrices are large and can be perfectly partitioned across all SMs over the axis.
  - righe 250–257: **3.5 GPTQ Modifications** — The quantization format used by MARLIN, designed for peak inference efficiency, is slightly different than the original GPTQ implementation Frantar et al.
- righe 258–301: **4 The Sparse-MARLIN Kernel** — To further improve FLOPS/Byte ratios, we can integrate a : sparsity scheme on top of the 4-bit quantized weight representation.
      - righe 275–289: **Quantized non-zero values.** — Figure 7 , left side, illustrates a 4-bit quantized matrix of size which has been pruned to : sparsity.
      - righe 290–301: **Metadata indices.** — In order to select the elements from the Right-Hand-Side (RHS) operand that will be necessary in the sparse computation, a metadata structure containing the indexes of non-zero elements in the original matrix is required…
- righe 302–379: **5 Experimental Results** — In our first set of experiments, we examine the efficiency of MARLIN relative to an ideal kernel, and compare its performance with other popular 4-bit inference kernels, notably the well-optimized PyTorch kernel Paszke e…
  - righe 304–339: **5.1 Kernel Benchmarks** — In our first set of experiments, we examine the efficiency of MARLIN relative to an ideal kernel, and compare its performance with other popular 4-bit inference kernels, notably the well-optimized PyTorch kernel Paszke e…
      - righe 322–332: **Roofline Analysis.** — To gain a deeper understanding of the computational efficiency of MARLIN, we perform a roofline analysis, which is a widely accepted methodology for evaluating performance potential.
      - righe 333–339: **Performance of Sparse-MARLIN.** — We now examine improvements due to 2:4 sparsity.
  - righe 340–379: **5.2 End-to-End Experiments** — Next, we validate our approach end-to-end (i.e., on full models) in a realistic LLM serving setting.
      - righe 344–349: **Accuracy.** — In Table 1 we briefly examine the accuracy difference between the baseline and sparse and sparse-quantized versions of Llama2.
      - righe 350–357: **Integration with vLLM.** — We first compare the end-to-end performance of MARLIN and Sparse-MARLIN with the default 16-bit kernel via a vLLM integration.
      - righe 358–368: **GPU and Model Types.** — Next, Table 2 shows MARLIN speedups under a variety of settings using several popular (quantized) models on different GPU types.
      - righe 369–379: **Client Counts.** — Finally, we perform a serving benchmark in a simulated server-client setting and measured the standard TPOT metric (Time Per Output Token, the average latency to generate an output token for each queried sequence) under …
- righe 380–388: **6 Related Work** — Due to space constraints, we focus on closely related work about providing efficient support for quantized LLM inference.
- righe 389–394: **7 Discussion and Future Work** — We have presented MARLIN, a general approach for implementing mixed-precision kernels for LLM generative inference, which achieves near-optimal efficiency by leveraging new GPU hardware instructions and parallelization t…
- righe 395–398: **Acknowledgments** — The authors would like to thank the Neural Magic team, in particular Michael Goin, Alexander Matveev, and Rob Shaw, for support during the writing of this paper, in particular with the vLLM integration.
- righe 399–465: **References** — Sezione strutturale; consultare il contenuto locale indicato.

## [Gated Delta Networks](2412.06464-gated-delta-networks.md)

Improving Mamba2 with Delta Rule

- righe 8–9: **Testo estratto** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 10–453: **Gated Delta Networks:** — Improving Mamba2 with Delta Rule
        - righe 14–19: **Abstract** — Linear Transformers have gained attention as efficient alternatives to standard Transformers, but their performance in retrieval and long-context tasks has been limited.
- righe 20–39: **1 Introduction** — The Transformer architecture has significantly advanced the capabilities of Large Language Models (LLMs), showcasing exceptional performance across a wide range of tasks due to its effective attention mechanism.
- righe 40–91: **2 Preliminary** — It is known that the linear transformer (Katharopoulos et al., 2020b ) can be formulated as the following linear recurrence when excluding normalization and query/key activations:
  - righe 42–59: **2.1 Linear Attention with Chunkwise Parallel Form** — It is known that the linear transformer (Katharopoulos et al., 2020b ) can be formulated as the following linear recurrence when excluding normalization and query/key activations:
    - righe 52–59: **Chunkwise parallel form.** — To summarize, the chunkwise parallel form splits inputs and outputs into several chunks of size , and computes outputs for each chunk based on the final state of the previous chunk and the query/key/value blocks of the c…
  - righe 60–79: **2.2 Mamba2: Linear attention with scalar-valued data-dependent decay** — Mamba2 (Dao & Gu, 2024a ) can be represented by the following linear recurrence (up to specific parameterization):
    - righe 72–79: **Chunkwise parallel form.** — Slightly abusing the notation, we define the local cumulative product of decays within the chunk as .
  - righe 80–91: **2.3 Delta Networks: Linear Attention with Delta Rule** — The delta update rule (Widrow et al., 1960 ; Schlag et al., 2021b ) dynamically erases the value ( ) associated with the current input key ( ) and writes a new value ( ), which is a linear combination of the current inpu…
    - righe 84–91: **Chunkwise parallel form.** — By partially expanding the recurrence, we have
- righe 92–150: **3 Gated Delta Networks** — The proposed gated delta rule is simple yet effective:
  - righe 94–111: **3.1 Formulation: Gated Delta Rule** — The proposed gated delta rule is simple yet effective:
  - righe 112–138: **3.2 Algorithm: Hardware-efficient Chunkwise training** — In this subsection, we describe an efficient chunkwise algorithm for gated delta rule.
    - righe 116–124: **Chunkwise parallel form.** — By partially expanding the recurrence, we have
    - righe 125–128: **Comparison to Eq. 1 - 2** — We can see that the key distinction lies in the replacement of the value block with the “pseudo”-value term .
    - righe 129–134: **UT transform.** — To maximize hardware efficiency, we apply the UT transform (Joffrain et al., 2006 ) to Eq.
    - righe 135–138: **Remark on speed.** — Similar to Mamba2, the gating term (colored in blue) only performs elementwise multiplication with (intermediate) variables without affecting matrix multiply structures, enabling tensor core GPU optimization.
  - righe 139–150: **3.3 Gated Delta Networks and Hybrid Models** — Figure 1: Visualization of the (hybrid) architecture and block design of Gated DeltaNet models.
    - righe 143–146: **Token mixer block.** — The basic Gated DeltaNet follows Llama’s macro architecture, stacking token mixer layers with SwiGLU MLP layers, but replaces self-attention with gated delta rule token mixing.
    - righe 147–150: **Hybrid models.** — Linear transformers have limitations in modeling local shifts and comparisons, and their fixed state size makes it hard for retrieval tasks (Arora et al., 2024a ) .
- righe 151–202: **4 Experiments** — Our experiments encompass a comprehensive comparison of recent state-of-the-art architectures, including pure Transformer models, RNN-based approaches, and hybrid architectures.
    - righe 153–160: **Setup** — Our experiments encompass a comprehensive comparison of recent state-of-the-art architectures, including pure Transformer models, RNN-based approaches, and hybrid architectures.
    - righe 161–166: **Common-sense reasoning** — In Table 2 , we present the language modeling perplexity and zero-shot accuracy on commonsense reasoning benchmarks for models with 400M and 1.3B parameters.
    - righe 167–173: **In-context retrieval on synthetic data** — Table 3 shows the results on Single Needle-In-A-Haystack (S-NIAH) benchmark suite from RULER (Hsieh et al., 2024 ) .
    - righe 174–183: **In-context retrieval on real-world data** — Table 4: Accuracy on recall-world retrieval tasks with input truncated to 2K tokens.
    - righe 184–187: **Length extrapolation on long sequences.** — As shown in Fig.
    - righe 188–195: **Long context understanding** — As demonstrated in Table 5 , we evaluated the models’ performance on LongBench (Bai et al., 2023 ) .
    - righe 196–202: **Throughput Comparison.** — The training throughput comparison across different models is presented in Fig.
- righe 203–219: **5 Related Work** — Large linear recurrent language models have attracted significant attention due to their training and inference efficiency.
    - righe 205–211: **Gated linear RNN.** — Large linear recurrent language models have attracted significant attention due to their training and inference efficiency.
    - righe 212–219: **Delta rule.** — The delta learning rule has been shown to offer superior memory capacity compared to the Hebbian learning rule (Gardner, 1988 ; Prados & Kak, 1989 ) .
- righe 220–223: **6 Conclusion** — In this work, we introduced Gated DeltaNet, which enables better key-value association learning compared to Mamba2 and more adaptive memory clearance than DeltaNet, leading to consistently better empirical results across…
- righe 224–227: **Acknowledgment** — We thank Yu Zhang for assistance with figure drawing, Simeng Sun and Zhixuan Lin for valuable discussions on the evaluation, and Eric Alcaide for insightful feedback on the online learning perspective of DeltaNet.
- righe 228–409: **References** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 410–437: **Appendix A Appendix** — To reduce notation clutter, we only consider the first chunk here.
  - righe 412–429: **A.1 Extended WY representation for gated delta rule** — To reduce notation clutter, we only consider the first chunk here.
        - righe 420–429: **Proof.** — ∎
  - righe 430–437: **A.2 Ablation Study** — Table S.1: Ablation study on the Gated DeltaNet block.
- righe 438–453: **Appendix B Experimental settings** — Following Gu & Dao ( 2023 ) , we evaluate our model on multiple commonsense reasoning benchmarks: PIQA (Bisk et al., 2020 ) , HellaSwag (Hella.; Zellers et al., 2019 ) , WinoGrande (Wino.; Sakaguchi et al., 2020 ) , ARC-…
  - righe 440–453: **B.1 Evaluation** — Following Gu & Dao ( 2023 ) , we evaluate our model on multiple commonsense reasoning benchmarks: PIQA (Bisk et al., 2020 ) , HellaSwag (Hella.; Zellers et al., 2019 ) , WinoGrande (Wino.; Sakaguchi et al., 2020 ) , ARC-…
    - righe 442–445: **Commonsense reasoning** — Following Gu & Dao ( 2023 ) , we evaluate our model on multiple commonsense reasoning benchmarks: PIQA (Bisk et al., 2020 ) , HellaSwag (Hella.; Zellers et al., 2019 ) , WinoGrande (Wino.; Sakaguchi et al., 2020 ) , ARC-…
    - righe 446–450: **In-context retrieval** — Our evaluation comprises both synthetic and real-world tasks.
    - righe 451–453: **Long context understanding** — We evaluate on 14 tasks from Longbench (Bai et al., 2023 ) , encompassing: narrative comprehension (Narrative QA (Kočiský et al., 2018 ) ), scientific understanding (QasperQA (Dasigi et al., 2021 ) ), multi-hop reasoning…

## [FlashInfer](2501.01005-flashinfer.md)

marginparsep has been altered.

- righe 8–36: **Testo estratto** — marginparsep has been altered.
        - righe 28–36: **Abstract** — Transformers, driven by attention mechanisms, form the foundation of large language models (LLMs).
- righe 37–77: **1 Introduction** — The Transformer architecture has become the primary backbone for large language models (LLMs), prominently featuring attention mechanism Vaswani et al.
- righe 78–105: **2 Background** — FlashAttention Dao et al.
  - righe 80–86: **2.1 FlashAttention** — FlashAttention Dao et al.
  - righe 87–97: **2.2 Attention Composition** — Block-Parallel Transformer (BPT) Liu & Abbeel ( 2023 ) demonstrates that attention outputs for the same query and different keys/values can be composed by preserving both the attention outputs and their scales.
  - righe 98–105: **2.3 Block/Vector Sparsity** — Block Compressed Sparse Row (BSR) is a hardware-efficient sparse format that groups non-zero elements into contiguous matrices of size , as opposed to the random scattering found in unstructured sparsity.
- righe 106–211: **3 Design** — In this section, we introduce the system design of FlashInfer.
  - righe 110–129: **3.1 KV-Cache Storage** — Recent advancements in KV-Cache storage, such as PageAttention Kwon et al.
    - righe 112–123: **3.1.1 Block-Sparse Matrix as Unified Format** — Recent advancements in KV-Cache storage, such as PageAttention Kwon et al.
    - righe 124–129: **3.1.2 Composable Formats for Memory Efficiency** — Inspired by SparseTIR Ye et al.
  - righe 130–181: **3.2 Compute Abstraction** — We developed CUDA/CUTLASS Thakkar et al.
    - righe 134–143: **3.2.1 Global to Shared Memory Data Movement** — The FlashInfer attention template supports any block size, requiring a specialized data loading approach since blocks may not align with tensor core shapes.
    - righe 144–159: **3.2.2 Microkernel with Different Tile Sizes** — To adapt to the varying operational intensities of LLM applications, FlashInfer implements the FA2 algorithm across multiple sizes.
    - righe 160–181: **3.2.3 JIT Compiler for Attention Variants** — For query tile size , we use CUDA Cores template since tensor core instruction (minimum rows) is , and use Tensor Cores for other query tile sizes.
  - righe 182–199: **3.3 Dynamism-Aware Runtime** — In this section we introduce the runtime design of FlashInfer, including the dynamic scheduling framework, and the composable formats for memory efficient attention.
    - righe 186–199: **3.3.1 Load-balanced Scheduling** — In FlashInfer, the load-balanced scheduling algorithm aims to minimize SM idle time by distributing the workload evenly across all SMs.
  - righe 200–211: **3.4 Programming Interface** — FlashInfer offers a programming interface designed for seamless integration with existing LLM serving frameworks such as vLLM Kwon et al.
- righe 212–249: **4 Evaluation** — In this section, we evaluate FlashInfer v0.2 on kernel-level and end-to-end performance showing how FlashInfer’s design address the challenges of LLM serving.
  - righe 216–223: **4.1 End-to-end LLM serving performance** — We evaluate FlashInfer with SGLang v0.3.4 Zheng et al.
  - righe 224–231: **4.2 Kernel Performance for Input Dynamism** — In this section we measure FlashInfer’s generated kernel performance against state-of-the-art open-source FlashAttention library under different sequence length distributions, we use the latest main branch 6 6 6 Commit: …
  - righe 232–239: **4.3 Customizability for Long-Context Inference** — In this section, we demonstrate how FlashInfer’s customized attention kernels significantly accelerate LLM inference.
  - righe 240–249: **4.4 Parallel-Generation Performance** — In this section, we illustrate how the composable formats of FlashInfer can enhance parallel decoding.
- righe 250–271: **5 Related Work** — Multi-Head Attention (MHA) (Vaswani et al., 2017 ) faces computational and IO challenges.
  - righe 252–259: **5.1 Attention Optimizations** — Multi-Head Attention (MHA) (Vaswani et al., 2017 ) faces computational and IO challenges.
  - righe 260–263: **5.2 Sparse Optimizations on GPUs** — FusedMM (Rahman et al., 2021 ) explores Sparse-dense Matrix Multiplication (SpMM) fusion, though it omits softmax computation, limiting direct applicability for accelerating attention.
  - righe 264–267: **5.3 Attention Compilers** — FlexAttention (He et al., 2024 ) provides a user-friendly interface for programming attention variants, compiling them into block-sparse flashattention implemented in Triton (Tillet et al., 2019 ) .
  - righe 268–271: **5.4 LLM Serving Systems** — Orca (Yu et al., 2022 ) introduces continuous batching for enhanced throughput.
- righe 272–279: **6 Discussions** — Currently, FlashInfer only supports the forward pass for attention computation.
- righe 280–283: **7 Conclusion and Future Work** — In this paper, we present FlashInfer, an versatile and efficient attention engine for LLM serving.
- righe 284–289: **8 acknowledgements** — We thank anonymous MLSys reviewers for providing constructive comments, LMSYS ORG, UW Syslab and SAMPL research group, CMU Catalyst group for their useful feedback and discussions, Yaxing Cai, Junru Shao, Lianmin Zheng, …
- righe 290–447: **References** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 448–456: **Appendix A Head Group Fusion for Grouped-Query Attention** — Grouped-Query Attention (GQA) Ainslie et al.
- righe 457–473: **Appendix B Overhead of Sparse Gathering** — In Section 3.2.1 , we detailed the design of FlashInfer’s sparse loading module, which transfers sparse rows from global memory into contiguous shared memory.
- righe 474–487: **Appendix C The Choice of Backend** — For NVIDIA GPUs, we build FlashInfer on top of CUDA/CUTLASS Thakkar et al.
- righe 488–516: **Appendix D Memory Management** — FlashInfer manages a page-locked (pinned) host buffer and a device workspace buffer to store scheduler metadata and split-k partial outputs.
  - righe 492–495: **D.1 CUDAGraph-Compatible Workspace Layout** — Once a kernel is captured by CUDA Graph, its arguments (pointers and scalars) become fixed, implying that each section of the device workspace buffer must maintain a consistent address for the entire captured graph’s lif…
  - righe 496–499: **D.2 Split-K Writethrough Optimizations** — In FlashInfer’s load-balancing scheduler (Section 3.3.1 ), KV-splitting is only applied to requests that have large KV lengths.
  - righe 500–516: **D.3 Workspace Buffer Size Estimation** — The workspace buffer size depends on two main factors:
      - righe 506–509: **Scheduler Metadata.** — The maximum size of each metadata section is derived from the largest possible number of concurrent requests and the maximum accumulated request length.
      - righe 510–516: **Partial Outputs.** — The size of partial outputs depends on both the problem dimensions (i.e., the number of heads and the head dimension) and the number of CTAs per kernel launch.
- righe 517–520: **Appendix E Overlap of Attention with Other Operations** — Nanoflow Zhu et al.
- righe 521–524: **Appendix F FP8–FP16 Mixed-Precision Attention** — Recent LLMs frequently adopt fp8 KV-Cache to reduce memory bandwidth and storage costs Micikevicius et al.
- righe 525–585: **Appendix G Additional Evaluation** — In this section, we present additional evaluation results to further validate the performance, scalability, and robustness of FlashInfer across diverse experimental conditions.
  - righe 529–544: **G.1 Comparison with FlexAttention** — We compare FlashInfer and FlexAttention He et al.
  - righe 545–553: **G.2 Evaluation of Shared-Prefix Attention Kernels** — We measure shared-prefix attention kernels with suffix length .
  - righe 554–563: **G.3 Ablation Study on Variable Sequence Length and load-balancing scheduler** — We conduct ablations on the effect of load-balancing scheduler (Section 3.3.1 ).
  - righe 564–570: **G.4 vLLM Integration Evaluation** — We compare the vLLM with FlashInfer backend and its default backend with a fixed request rate of , reporting throughput (tokens/s), inter-token latency (ITL, ms), and time-to-first-token (TTFT, ms) in Table 8 .
  - righe 571–585: **G.5 Fine-Grained Block-Sparsity Evaluation** — FlashInfer supports fine-grained block-sparse matrices, which is useful in many KV-Cache pruning algorithms.
