---
title: 2024-04-01-Daily-Note
author: <rc> 
date: 2024-04-01
categories: [Daily]
tags: [AI,LLM]
share: true
---

# 04-01
Retrospect what have been done in the past one week, what can be learnt from the past ? 

# 04-02
## Create attack_range on azure 
create multiple windows clients and join the AD


https://uncoder.io/
It's a great platform for threat hunters to jump around different hunting platforms that have their-own query language. 
https://www.splunk.com/en_us/blog/security/approaching-linux-post-exploitation-with-splunk-attack-range.html

I just have a thought of presenting Splunk based attack simulation on upcoming TECH VT session. 

# 04-03
https://ubuntu.com/server/docs/nvidia-drivers-installation
Fix the Nvidia driver and CUDA on azure VM 
```
reid@llm-01:~$ cat /proc/driver/nvidia/version
NVRM version: NVIDIA UNIX x86_64 Kernel Module  535.154.05  Thu Dec 28 15:37:48 UTC 2023
GCC version:  gcc version 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04)

reid@llm-01:~$ sudo ubuntu-drivers list --gpgpu
nvidia-driver-535-server, (kernel modules provided by linux-modules-nvidia-535-server-azure)
nvidia-driver-525-open, (kernel modules provided by linux-modules-nvidia-525-open-azure)
nvidia-driver-525-server, (kernel modules provided by linux-modules-nvidia-525-server-azure)
nvidia-driver-535-server-open, (kernel modules provided by linux-modules-nvidia-535-server-open-azure)
nvidia-driver-545-open, (kernel modules provided by linux-modules-nvidia-545-open-azure)
nvidia-driver-470-server, (kernel modules provided by linux-modules-nvidia-470-server-azure)
nvidia-driver-535-open, (kernel modules provided by linux-modules-nvidia-535-open-azure)
nvidia-driver-470, (kernel modules provided by linux-modules-nvidia-470-azure)
nvidia-driver-550-server-open, (kernel modules provided by linux-modules-nvidia-550-server-open-azure)
nvidia-driver-550, (kernel modules provided by nvidia-dkms-550)
nvidia-driver-550-open, (kernel modules provided by nvidia-dkms-550-open)
nvidia-driver-550-server, (kernel modules provided by linux-modules-nvidia-550-server-azure)

```

**CUDA 12.2 is compatible with Nvidia GPU driver 535, it is vital to keep both Nvidia GPU driver and CUDA compatible**
upgrading Nvidia Driver to the latest one, and check out its compatible CUDA. 

NC64as_T4_v3
- The [NCv3-series](https://learn.microsoft.com/en-us/azure/virtual-machines/ncv3-series) and [NC T4_v3-series](https://learn.microsoft.com/en-us/azure/virtual-machines/nct4-v3-series) sizes are optimized for compute-intensive GPU-accelerated applications. Some examples are CUDA and OpenCL-based applications and simulations, AI, and Deep Learning. The NC T4 v3-series is focused on inference workloads featuring NVIDIA's Tesla T4 GPU and AMD EPYC2 Rome processor. The NCv3-series is focused on high-performance computing and AI workloads featuring NVIDIA’s Tesla V100 GPU.
    
- The [NC 100 v4-series](https://learn.microsoft.com/en-us/azure/virtual-machines/nc-a100-v4-series) sizes are focused on midrange AI training and batch inference workload. The NC A100 v4-series offers flexibility to select one, two, or four NVIDIA A100 80GB PCIe Tensor Core GPUs per VM to leverage the right-size GPU acceleration for your workload.
    
- The [ND A100 v4-series](https://learn.microsoft.com/en-us/azure/virtual-machines/nda100-v4-series) sizes are focused on scale-up and scale-out deep learning training and accelerated HPC applications. The ND A100 v4-series uses 8 NVIDIA A100 TensorCore GPUs, each available with a 200 Gigabit Mellanox InfiniBand HDR connection and 40 GB of GPU memory.



![|400](../Attachments/Pasted%20image%2020240403155946.png)


![[../Attachments/Pasted image 20240403173719.png|500]]



![[../Attachments/Pasted image 20240403211736.png|400]]

```
text-generation-launcher --huggingface-hub-cache /home/azureuser/LLM/llm_models/Jamba-v0.1 --model-id ai21labs/Jamba-v0.1

2024-04-04T06:49:36.458268Z ERROR text_generation_launcher: exllamav2_kernels not installed.

2024-04-04T06:49:36.475237Z  WARN text_generation_launcher: Could not import Flash Attention enabled models: No module named 'vllm'

2024-04-04T06:49:36.490824Z  WARN text_generation_launcher: Could not import Mamba: cannot import name 'FastRMSNorm' from 'text_generation_server.utils.layers' (/home/azureuser/LLM/text-generation-inference/server/text_generation_server/utils/layers.py)

2024-04-04T06:49:36.671668Z ERROR text_generation_launcher: Error when initializing model

```

https://github.com/turboderp/exllamav2
https://zhuanlan.zhihu.com/p/668126399
Now, TGI has successfully kicked off and Web API is ready for remote accessing. 
```
time curl 127.0.0.1:3000/generate_stream -X POST -d '{"inputs": "What is deep learning?","parameters":{"max_new_tokens":20}}' -H 'Content-Type: application/json'
```
Install text-generation-benchmark
```
 make install-benchmark
```
For downloaded model, to locate the tokenizer json file under the model folder
```
text-generation-benchmark -t /home/azureuser/LLM/llm_models/Llama-2-13b-chat-hf/models--meta-llama--Llama-2-13b-chat-hf/snapshots/29655417e51232f4f2b9b5d3e1418e5a9b04e80e
```
Install and run Jmeter
```
(base) ➜  bin export JAVA_HOME=/opt/homebrew/opt/openjdk
(base) ➜  bin ./jmeter.sh
```
How to simulate the concurrent requests of 200 users on Jmeter ? 

https://www.youtube.com/watch?v=TRjq7t2Ms5I&ab_channel=AIEngineer

GNN in lateral movement detection in the context of Windows environment
https://github.com/pl247/ai-toolkit
https://www.microsoft.com/en-us/security/blog/2021/04/08/gamifying-machine-learning-for-stronger-security-and-ai-models/
[attack_data/datasets/attack_techniques at master · splunk/attack_data · GitHub](https://github.com/splunk/attack_data/tree/master/datasets/attack_techniques)

security dataset
https://csr.lanl.gov/data/cyber1/

Privilege Escalation Tool
https://github.com/ohpe/juicy-potato


# 04-05

## Studying "Serving LLM" on deeplearning.ai
https://learn.deeplearning.ai/courses/efficiently-serving-llms/lesson/2/text-generation
## KV caching
![[../../Attachments/Pasted image 20240405204839.png|500]]
```python
def generate_token_with_past(inputs):
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    last_logits = logits[0, -1, :]
    next_token_id = last_logits.argmax()
    return next_token_id, outputs.past_key_values

generated_tokens = []
next_inputs = inputs
durations_cached_s = []
for _ in range(10):
    t0 = time.time()
    next_token_id, past_key_values = \
        generate_token_with_past(next_inputs)
    durations_cached_s += [time.time() - t0]
    
    next_inputs = {
        "input_ids": next_token_id.reshape((1, 1)),
        "attention_mask": torch.cat(
            [next_inputs["attention_mask"], torch.tensor([[1]])],
            dim=1),
        "past_key_values": past_key_values,
    }
    
    next_token = tokenizer.decode(next_token_id)
    generated_tokens.append(next_token)

print(f"{sum(durations_cached_s)} s")
print(generated_tokens)
```
The model object takes "past_key_values" as one of the input parameters, KV caching happens inside it, like a black-box.



==How does the output tensor looks like after processing with "tokenizer" ?== 


## Page Attention 

## Batching
![[../../Attachments/Pasted image 20240405210209.png|400]]![[../../Attachments/Pasted image 20240405210249.png|400]]


## Continuous Batching

![[../../Attachments/Pasted image 20240406092828.png|400]]![[../../Attachments/Pasted image 20240406092850.png|400]]


[[../../0x00-AI/LLM/Serving LLM/Lesson_3-Continuous_Batching|Lesson_3-Continuous_Batching]]

https://towardsdatascience.com/increase-llama-2s-latency-and-throughput-performance-by-up-to-4x-23034d781b8c

## Quantization
### Theory

![[../../Attachments/Pasted image 20240406213051.png|400]]![[../../Attachments/Pasted image 20240408112655.png|400]]

[[../../0x00-AI/LLM/Serving LLM/Lesson_4-Quantization|Lesson_4-Quantization]]
==What is quantization in plain English?== 
Quantization is used to reduce the memory footprint of LLM. 
**the size of Model = the number of parameters X the precision of data (date type)

Two types of Quantization:
1. Post Training Quantization(PTQ): quantize the weights of pre-trained model, might lead to potential performance degradation
2. Quantization Aware Training (QAT): incorporates weight conversion process during model training with expensive computational cost

https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html
This article mentions a method of measuring the output of LLM model, perplexity, in simple term, how many "guesses" LLM needs to make prediction correctly, the lower perplexity indicates that LLM is more reliable. 
This point leads to another critical avenue of making successful LLM, LLM measurement.
The blog describes THREE quantization approaches:
1. absolute maximum (absmax) quantization -- **Symmetric
2. Zero Point Quantization -- **Asymmetric
3. LLM.int8()
**Their goal is to map an original FP32 tensor into INT8 tensor.**

==Why AbsMax is symmetric and ZP is asymmetric ? Why ZP is useful to deal with ReLU like activation function ?== 
The quantized tensor is INT8 which can represent numerical value from -128 to 127, clamping the original tensor into that range. 
To answer the second question, ReLU activation function has a lot of zero point, while quantizing it especially zero-point into 8-bit range (-128,127), it is vital to keep the zero-point (during training phrase) intact at inference phrase, it necessitates one extra step compared with AbsMax, where the range should be centered around zero-point.
[[../../Reading/Zero-point quantization - How do we get those formulas- - by Luis Antonio Vasquez - Medium|Zero-point quantization - How do we get those formulas- - by Luis Antonio Vasquez - Medium]]


```python
def absmax_quantize(X):
    # Calculate scale
    scale = 127 / torch.max(torch.abs(X))
    # Quantize
    X_quant = (scale * X).round()
    # Dequantize
    X_dequant = X_quant / scale
    return X_quant.to(torch.int8), X_dequant

def zeropoint_quantize(X):
    # Calculate value range (denominator)
    x_range = torch.max(X) - torch.min(X)
    x_range = 1 if x_range == 0 else x_range
    # Calculate scale
    scale = 255 / x_range
    # Shift by zero-point
    zeropoint = (-scale * torch.min(X) - 128).round()
    # Scale and round the inputs
    X_quant = torch.clip((X * scale + zeropoint).round(), -128, 127)
    # Dequantize
    X_dequant = (X_quant - zeropoint) / scale
    return X_quant.to(torch.int8), X_dequant
```

[[../../Reading/LLM Quantization- Techniques, Advantages, and Models|LLM Quantization- Techniques, Advantages, and Models]]
![[../../Attachments/Pasted image 20240409154649.png|400]]
* NF4 is implemented in **bitsandbytes** library which closely work with HF transformers. It is primarily used by QLoRA to load models with 4-bits.
* GTPQ is the most innovative way of quantizing LLM, it can only quantize models into **4 INT data type**

> *The NormalFloat (NF) data type is an enhancement of the Quantile Quantization technique. It has shown better results than both 4-bit Integers and 4-bit Floats.*

>ExLLama is a standalone implementation of Llama for use with 4-bit GPTQ weights

### ZQ
[[../../Reading/Zero-point quantization - How do we get those formulas- - by Luis Antonio Vasquez - Medium|Zero-point quantization - How do we get those formulas- - by Luis Antonio Vasquez - Medium]]
Wow, this post thoroughly explains the motivation of where ZQ comes in, and the steps of deriving ZQ equation. 
1. the boundary of new range [-128,127], the Max and Min value of old range has to be projected to "-128" and "127" 
$$ \left \{ \begin{array}\{ 127 = Scale * X_{max} + Offset \\ -128= Scale * X_{min} + Offset \end{array} \right. $$
2. Subtract "Offset" from both equations, we can get:
$$255 = Scale * (X_{max} - X_{min})$$
$$ Scale = \frac {255}{X_{max} - X_{min}}$$
$$ Offset = -(Scale * X_{min}) - 128$$
**Time to answer why "zero point", to put it simple, zero_point = offset, this can guarantee the zero in old range can still be projected into zero in new range**
Like ReLu gates in Neural Network, "0" would turn off its corresponding weights, whereas other positive value would turn on its weights. In original range, checking 0 is trivial, but in new range, after quantizing, the offset value will be counted as "0".

[[../Reading/Tensor Quantization- The Untold Story - by Dhruv Matani - Towards Data Science|Tensor Quantization- The Untold Story - by Dhruv Matani - Towards Data Science]]
This article will try to answer below questions:
1. What do the terms scale and zero-point mean for quantization?
2. What are the different types of quantization schemes?
3. How to compute scale and zero-point for the different quantization schemes
4. Why is zero-point important for quantization?
5. How do normalization techniques benefit quantization
The last two are most significant, as they are somehow linking with important decisions that have to make in practical engineering. 

### GPTQ

### AWQ


### Bitsandbytes
**Dequanitization could be used to boost inference performance**


### LoRA
[[../Attachments/LoRA- Low-Rank Adaptation of Large Language Models .pdf|LoRA- Low-Rank Adaptation of Large Language Models ]]
[[../Attachments/Lesson_5-Low-Rank_Adaptation|Lesson_5-Low-Rank_Adaptation]]
First of all, what is rank in context of Matrix ? 
* The maximum number of linear independent rows or columns
* Square Matrix : The number of "unique" rows or columns, the number is always the same from row and column perspective
* Non-Square Matrix: The rank can NOT be greater than the smallest dimension of the Matrix
* Full rank, when rank equals to the smallest dimension of the Matrix
https://www.mathsisfun.com/algebra/matrix-rank.html

What Rank can do ? 


![[../Attachments/Pasted image 20240410144721.png|400]]


### QLoRA


### Further Reading
https://www.tensorops.ai/post/what-are-quantized-llms
https://huggingface.co/blog/hf-bitsandbytes-integration
https://towardsdatascience.com/tensor-quantization-the-untold-story-d798c30e7646
https://medium.com/@luis.vasquez.work.log/zero-point-quantization-how-do-we-get-those-formulas-4155b51a60d6


## LLM Benchmark 
https://huggingface.co/docs/transformers/en/benchmarks
HF benchmark is deprecated

https://github.com/run-ai/llmperf
LLM Perf appears to be a good choice

https://medium.com/@walotta2001/towards-realistic-benchmarking-of-llm-serving-endpoints-ba3cca28f246

https://huggingface.co/docs/transformers/perplexity

https://towardsdatascience.com/deploying-llms-locally-with-apples-mlx-framework-2b3862049a93
An interesting found,  MLX. Try it out for GPT2 

https://netron.app/
Open Source Tool to view LLM Model Object

https://www.microsoft.com/en-us/research/group/experimentation-platform-exp/articles/how-to-evaluate-llms-a-complete-metric-framework/
* Time to first token render
* Requests per second
* Tokens render per second 

### Reference
https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html
https://symbl.ai/developers/blog/a-guide-to-quantization-in-llms/


----

# 04-07
* Azure Spot VM 
The access to Azure portal is not stable, come back later. 
* Use Mojo to learn Matrix
## Mojo 
https://github.com/modularml/mojo/tree/main/examples/notebooks#readme
Mojo supports running code in Jupyter notebook the same as python, it sounds attractive. 
Follow the steps below:
1. create a file with prefix ".ipynb"
2. select mojo" as the kernel of Jupyter 
3. fire up
https://www.modular.com/blog/ais-compute-fragmentation-what-matrix-multiplication-teaches-us
==Does Mojo support matmul on GPU ? if so, delve into the specific code==

https://medium.com/@bennynottonson/optimizing-matrix-multiplication-for-ml-with-mojo-bfc428112360
https://towardsai.net/p/machine-learning/the-multilayer-perceptron-built-and-implemented-from-scratch

While the NN frameworks like Keras, Pytorch, Tensorflow offers convenient and powerful API, without solid understanding of the underlying techniques and algorithms could present some huge challenges to deliver production-ready solution, especially in the fields of tuning hyperparemeters, debugging etc., The rapid evolution of NN frameworks makes it hard to stay current. 
I have decided to go further deeper to the algorithms and techniques behind the curtain, instead of just dabbling all sorts of new tools. 

[Mojo for building MLP](https://medium.com/data-and-beyond/mojo-build-a-simple-mlp-neural-network-in-the-fastest-programming-language-5d13cdea6c9e)
![[../Excalidraw/Raw_MLP.excalidraw|1400]]


MLP can be described as a composition of multiple functions:   P(LP(H(LH(X))))  
* LH(X) is the linear transformation of input tensor X
* H( ) is the activation function, which in our case will use sigmod function
* LP ( ) is the linear transformation, using the output of hidden layer as the input
* P( ) is the last activation function



# 04-11
## Cybersecurity In Nvidia
https://resources.nvidia.com/en-us-morpheus-developer/cve-demo
# 04-14
Take action on the theory
* use one or two sentences to summarize each pages or blog you have just read
* Link with Anki to boost your memory
When you are working on something that is the pipeline of waiting non-control tasks to complete, you can switch your focus on similar or relevant content, but do not go further otherwise you might get distracted, what you can do is to quickly note down what have just off the top of your heads, then come back to look into it more deeper. 

