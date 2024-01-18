# ParameterNet: Parameters Are All You Need
Kai Han, Yunhe Wang, Jianyuan Guo, and Enhua Wu. [[arXiv link]](https://arxiv.org/pdf/2306.14525v2.pdf)

The large-scale visual pretraining has significantly improve the performance of large vision models. However, we observe the \emph{low FLOPs pitfall} that the existing low-FLOPs models cannot benefit from large-scale pretraining. In this paper, we introduce a novel design principle, termed ParameterNet, aimed at augmenting the number of parameters in large-scale visual pretraining models while minimizing the increase in FLOPs. We leverage dynamic convolutions to incorporate additional parameters into the networks with only a marginal rise in FLOPs. The ParameterNet approach allows low-FLOPs networks to take advantage of large-scale visual pretraining. Furthermore, we extend the ParameterNet concept to the language domain to enhance inference results while preserving inference speed. Experiments on the large-scale ImageNet-22K have shown the superiority of our ParameterNet scheme. For example, ParameterNet-600M can achieve higher accuracy on ImageNet than the widely-used Swin Transformer (81.6\% \emph{vs.} 80.9\%) and has much lower FLOPs (0.6G \emph{vs.} 4.5G). In the language domain, LLaMA-1B enhanced with ParameterNet achieves 2\% higher accuracy over vanilla LLaMA. 

![image](../fig/moe_layer.png) !

## Vision Model

The ParameterNet-600M code is at `./cv/`, just training it using [timm](https://github.com/huggingface/pytorch-image-models).

![image](../fig/parameternet_result.png)

## Large Language Model

The LLaMA-1B code is at `./nlp/`, just training it using standard Huggingface transformers framework such as [LLaMA](https://github.com/facebookresearch/llama) and [InternLM](https://github.com/InternLM/InternLM).

![image](https://parameternet.github.io/static/images/llama.PNG)

## Citation
```
@article{han2023parameternet,
  title={ParameterNet: Parameters Are All You Need for Large-scale Visual Pretraining of Mobile Networks},
  author={Han, Kai and Wang, Yunhe and Guo, Jianyuan and Wu, Enhua},
  journal={arXiv preprint arXiv:2306.14525},
  year={2023}
}
```

## Acknowledgement
This repo partially uses code from [transformers](https://github.com/huggingface/transformers/tree/main).
