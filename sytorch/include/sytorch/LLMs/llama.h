#pragma once
#include <sytorch/LLMs/llm_base.h>

template <typename T>
class TransformerBlockLLAMA : public SytorchModule<T>
{
    using SytorchModule<T>::add;

    MultiHeadAttentionLLAMA<T> *attn;
    FFN_LLAMA<T> *ffn;
    RMSNorm<T> *ln0;
    RMSNorm<T> *ln1;

    u64 n_heads, n_embd, intermediate_size;

public:
    TransformerBlockLLAMA(u64 n_heads, u64 n_embd, u64 intermediate_size) : n_heads(n_heads), n_embd(n_embd), intermediate_size(intermediate_size)
    {
        attn = new MultiHeadAttentionLLAMA<T>(n_heads, n_embd);
        ffn = new FFN_LLAMA<T>(n_embd, intermediate_size);
        ln0 = new RMSNorm<T>(n_embd, false);
        ln1 = new RMSNorm<T>(n_embd, false);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &ln0_out = ln0->forward(input);
        auto &attn_out = attn->forward(ln0_out);
        auto &attn_out_add = add(attn_out, input);
        auto &ln1_out = ln1->forward(attn_out_add);
        auto &ffn_out = ffn->forward(ln1_out);
        auto &ffn_out_add = add(ffn_out, attn_out_add);
        return ffn_out_add;
    }
};

template <typename T>
class LLAMA_MODEL : public SytorchModule<T>
{
    std::vector<TransformerBlockLLAMA<T> *> blocks;
    RMSNorm<T> *ln_f;
    u64 n_layer, n_heads, n_embd, intermediate_size;

public:
    LLAMA_MODEL(u64 n_layer, u64 n_heads, u64 n_embd, u64 intermediate_size) : n_layer(n_layer), n_heads(n_heads), n_embd(n_embd), intermediate_size(intermediate_size)
    {
        for (u64 i = 0; i < n_layer; ++i)
        {
            blocks.push_back(new TransformerBlockLLAMA<T>(n_heads, n_embd, intermediate_size));
        }
        ln_f = new RMSNorm<T>(n_embd);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        Tensor<T> *x = &input;

        // for(u64 i = 0; i < n_layer - 1; ++i)
        // {
        //     auto &block = blocks[i];
        //     auto &x_out = block->forward(*x);
        //     x = &x_out;
        // }

        // auto &block = blocks[n_layer - 1];
        // return block->forward(*x);

        for (u64 i = 0; i < n_layer; ++i)
        {
            auto &block = blocks[i];
            auto &x_out = block->forward(*x);
            x = &x_out;
        }
        return ln_f->forward(*x);
    }
};

template <typename T>
class LlamaNextWordLogits : public SytorchModule<T>
{
    using SytorchModule<T>::view;
    using SytorchModule<T>::transpose;
    LLAMA_MODEL<T> *llama_model;
    FC<T> *fc;
    u64 n_layer, n_heads, n_embd, n_vocab, intermediate_size;

public:
    LlamaNextWordLogits(u64 n_layer, u64 n_heads, u64 n_embd, u64 n_vocab, u64 intermediate_size) : n_layer(n_layer), n_heads(n_heads), n_embd(n_embd), n_vocab(n_vocab), intermediate_size(intermediate_size)
    {
        llama_model = new LLAMA_MODEL<T>(n_layer, n_heads, n_embd, intermediate_size);
        fc = new FC<T>(n_embd, n_vocab, false);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &fc_in = llama_model->forward(input);
        // printshape(fc_in.shape);
        auto &fc_out = fc->forward(fc_in);
        return view(fc_out, -1);
    }
};
