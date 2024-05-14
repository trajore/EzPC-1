
# LLMs (Layered Language Models)
LLMs, or Layered Language Models, are a class of natural language processing models that utilize foundational elements like Feed-Forward Networks (FFNs) and MultiHeadAttention layers. These components, supported by Sytorch, allow for the construction of sophisticated architectures like Transformers. By chaining together these building blocks in specific arrangements, one can design and define an LLM tailored to a particular task or design, such as sequence classification in the BERT model.

## Pre-Generated LLM Architectures defined in repo as of now:
- BERT
- BERTSequenceClassification
- GPT2
- GPT2SequenceClassification
- GPTNEO
- GPTNEONextWordLogits

Note: These are also the values for the key "llm_version" in the below config file, and the models are defined under `~/path/EzPC/sytorch/include/sytorch/LLMs`.

## Configuration to define an LLM:
```json
{
    "llm_version": "BERTSequenceClassification",
    "n_vocab" : 50257,
    "n_ctx" : 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer" : 12,
    "n_label" : 2,
    "window_size" : 256,
    "scale" : 12
}
```

In the context of LLMs, let's break down the parameters listed in the config.json file:

Config Parameters:
1. **llm_version**:
The specific version or variant of the LLM. For instance, "BERTSequenceClassification" refers to the BERT model specialized for sequence classification tasks.

2. **n_vocab**:
The size of the vocabulary. It represents the number of unique tokens the model can recognize. For example, a value of 50257 would mean the model recognizes 50,257 distinct tokens.

3. **n_ctx**:
The maximum sequence length or context size the model can handle. In this case, the model can take sequences up to a length of 1024 tokens.

4. **n_embd**:
Dimensionality of the embeddings. Each token is represented in the model as a vector of this size. Here, each token gets a 768-dimensional vector.

5. **n_head**:
Number of attention heads in the MultiHeadAttention mechanism. More heads allow the model to focus on different parts of the input simultaneously. A value of 12 means there are 12 parallel attention mechanisms.

6. **n_layer**:
Number of Transformer layers or blocks in the model. Each additional layer allows the model to capture more complex patterns and relationships. Here, the model has 12 layers.

7. **n_label**:
Number of labels or classes in a classification task. For a binary classification task, this would be 2.

8. **window_size**:
In models that operate on segments or chunks of data rather than the full sequence, this parameter specifies the size of each segment. Here, the model operates on chunks of 256 tokens at a time.

9. **scale**:
A scaling factor for converting floating-point weights to fixed-point weights. Specific to sytorch.

These parameters allow for the flexible configuration and customization of LLMs to meet the demands of a wide variety of NLP tasks and datasets. Adjusting these values can lead to performance improvements or adaptations to specific data constraints.


## Usage:

### Strip weights from existing model:

### Prepare input data:

#### 1. Generate Input metadata for Dealer:
Dealer need some public metadata for the input dataset denoting the n_seq for each datapoint.
To generate the metadata use the `generate_metadata.sh` script as below:
```bash
Usage: ./generate_metadata.sh --batch_size BATCH_SIZE --n_embd N_EMBD --path PATH

Arguments:
  --batch_size  Number of data points to consider.
  --n_embd      Number of embedding dimensions.
  --path        Path to the directory containing the datapoints as {1.dat, 2.dat,...}.
```
This saves a file `metadata.txt` which has entries for n_seq for all data points.

### Compilation and Inference:
Once you have selected a LLM from the list above:
1. Modify the config.json with values correspoding to the LLM. 
```bash
# run this while inside `/path/EzPC/sytorch/LLMs/`
./compile_llm <name_for_excutable>  
# this executable will be saved in dir `executables/`
```

```bash
# LLAMA
# Run the dealer- 
./executables/model 1 n_seq=<n_seq> [id=<id>] [nt=<nt>]

# Run the server- 
./executables/model 2 wt_file=<weights.dat> [id=<id>] [nt=<nt>]

# Run the client- 
./executables/model 3 ip=<server-ip> in_file=<input.dat> [id=<id>] [nt=<nt>]
```

```bash
# Cleartext
./executables/model 0 wt_file=<weights.dat> in_file=<input.dat> [ct_float=<0/1>]
```
