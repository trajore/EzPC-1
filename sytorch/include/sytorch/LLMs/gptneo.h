#pragma once
#include <sytorch/LLMs/llm_base.h>

template <typename T>
class TransformerBlockGPTNEO : public SytorchModule<T>
{
    using SytorchModule<T>::add;

    MultiHeadAttentionNEO<T> *attn;
    FFN<T> *ffn;
    LayerNorm<T> *ln0;
    LayerNorm<T> *ln1;
    
    u64 n_heads, n_embd;
    u64 attention_type; 
    u64 window_size;
public:

    TransformerBlockGPTNEO(u64 n_heads, u64 n_embd, u64 attention_type, u64 window_size): n_heads(n_heads), n_embd(n_embd)
    {
        attn = new MultiHeadAttentionNEO<T>(n_heads, n_embd, attention_type, window_size);
        ffn = new FFN<T>(n_embd, 4*n_embd);
        ln0 = new LayerNorm<T>(n_embd);
        ln1 = new LayerNorm<T>(n_embd);
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
class GPTNEO : public SytorchModule<T>
{
    std::vector<TransformerBlockGPTNEO<T> *> blocks;
    LayerNorm<T> *ln_f;
    u64 n_layer, n_heads, n_embd;
    u64 window_size;

public:
    
    GPTNEO(u64 n_layer, u64 n_heads, u64 n_embd, u64 window_size): n_layer(n_layer), n_heads(n_heads), n_embd(n_embd)
    {
        for(u64 i = 0; i < n_layer; ++i)
        {
            blocks.push_back(new TransformerBlockGPTNEO<T>(n_heads, n_embd, i, window_size));
        }
        ln_f = new LayerNorm<T>(n_embd);
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
        
        for(u64 i = 0; i < n_layer; ++i)
        {
            auto &block = blocks[i];
            auto &x_out = block->forward(*x);
            x = &x_out;
        }
        return ln_f->forward(*x);
    }
};

template <typename T>
class GPTNEONextWordLogits : public SytorchModule<T>
{
    using SytorchModule<T>::view;
    using SytorchModule<T>::transpose;
    GPTNEO<T> *gptNEO;
    FC<T> *fc;
    u64 n_layer, n_heads, n_embd, n_vocab;
    u64 window_size;
public:
    
    GPTNEONextWordLogits(u64 n_layer, u64 n_heads, u64 n_embd, u64 n_vocab, u64 window_size): n_layer(n_layer), n_heads(n_heads), n_embd(n_embd), n_vocab(n_vocab)
    {
        gptNEO = new GPTNEO<T>(n_layer, n_heads, n_embd, window_size);
        fc = new FC<T>(n_embd, n_vocab, false);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &fc_in = gptNEO->forward(input);
        // printshape(fc_in.shape);
        auto &fc_out = fc->forward(fc_in);
        return view(fc_out, -1);
    }
};