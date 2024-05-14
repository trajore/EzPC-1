#pragma once
#include <sytorch/LLMs/llm_base.h>

template <typename T>
class TransformerBlockBERT : public SytorchModule<T>
{
public:
    using SytorchModule<T>::add;

    MultiHeadAttentionBERT<T> *attn;
    FFN<T> *ffn;
    LayerNorm<T> *ln0;
    LayerNorm<T> *ln1;

    u64 n_heads, n_embd;

public:
    TransformerBlockBERT(u64 n_heads, u64 n_embd) : n_heads(n_heads), n_embd(n_embd)
    {
        attn = new MultiHeadAttentionBERT<T>(n_heads, n_embd);
        ffn = new FFN<T>(n_embd, 4 * n_embd);
        ln0 = new LayerNorm<T>(n_embd);
        ln1 = new LayerNorm<T>(n_embd);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &attn_out = attn->forward(input);
        auto &add0_out = add(attn_out, input);
        auto &ln0_out = ln0->forward(add0_out);

        auto &ffn_out = ffn->forward(ln0_out);
        auto &add1_out = add(ffn_out, ln0_out);
        auto &ln1_out = ln1->forward(add1_out);
        return ln1_out;
    }
};

template <typename T>
class BERT : public SytorchModule<T>
{
public:
    using SytorchModule<T>::tanh;
    using SytorchModule<T>::view;
    using SytorchModule<T>::add;
    using SytorchModule<T>::unsqueeze;
    std::vector<TransformerBlockBERT<T> *> blocks;
    LayerNorm<T> *ln_f;
    FC<T> *pool;
    u64 n_layer, n_heads, n_embd;

public:
    BERT(u64 n_layer, u64 n_heads, u64 n_embd) : n_layer(n_layer), n_heads(n_heads), n_embd(n_embd)
    {
        for (u64 i = 0; i < n_layer; ++i)
        {
            blocks.push_back(new TransformerBlockBERT<T>(n_heads, n_embd));
        }
        ln_f = new LayerNorm<T>(n_embd);
        pool = new FC<T>(n_embd, n_embd, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &y = ln_f->forward(input);
        Tensor<T> *x = &y;

        for (u64 i = 0; i < n_layer; ++i)
        {
            auto &block = blocks[i];
            auto &x_out = block->forward(*x);
            x = &x_out;
        }

        auto &x0 = view(*x, 0);
        auto &x0_unsqueeze = unsqueeze(x0);
        auto &pool_out = pool->forward(x0_unsqueeze);
        auto &tanh_out = tanh(pool_out);
        // return view(tanh_out, 0);
        return tanh_out;
    }
};

template <typename T>
class BERTSequenceClassification : public SytorchModule<T>
{
public:
    using SytorchModule<T>::view;
    BERT<T> *gpt2;
    FC<T> *fc;
    u64 n_layer, n_heads, n_embd, n_labels;

public:
    BERTSequenceClassification(u64 n_layer, u64 n_heads, u64 n_embd, u64 n_labels) : n_layer(n_layer), n_heads(n_heads), n_embd(n_embd), n_labels(n_labels)
    {
        gpt2 = new BERT<T>(n_layer, n_heads, n_embd);
        fc = new FC<T>(n_embd, n_labels, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &fc_in = gpt2->forward(input);
        auto &fc_out = fc->forward(fc_in);
        return view(fc_out, 0);
    }
};
