#pragma once

#include <sytorch/backend/llama_extended.h>
#include <sytorch/backend/llama_transformer.h>
#include <sytorch/backend/piranha_cleartext.h>
#include <sytorch/backend/secureml_cleartext.h>
#include <sytorch/backend/float.h>
#include <sytorch/layers/layers.h>
#include <sytorch/module.h>
#include <llama/utils.h>
#include <llama/api.h>

template <typename T>
class FFN : public SytorchModule<T>
{
public:
    using SytorchModule<T>::gelu;

    u64 in;
    u64 hidden;

public:
    FC<T> *up;
    FC<T> *down;

    FFN(u64 in, u64 hidden) : in(in), hidden(hidden)
    {
        up = new FC<T>(in, hidden, true);
        down = new FC<T>(hidden, in, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        return down->forward(gelu(up->forward(input)));
    }
};

template <typename T>
class FFN_LLAMA : public SytorchModule<T>
{
    using SytorchModule<T>::silu;
    using SytorchModule<T>::mul;

    u64 in;
    u64 intermediate_size;

public:
    FC<T> *up1;
    FC<T> *up2;
    FC<T> *down;

    FFN_LLAMA(u64 in, u64 intermediate_size) : in(in), intermediate_size(intermediate_size)
    {
        up1 = new FC<T>(in, intermediate_size, false);
        up2 = new FC<T>(in, intermediate_size, false);
        down = new FC<T>(intermediate_size, in, false);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &a = up1->forward(input);
        auto &b = up2->forward(input);
        return down->forward(mul(silu(a), b));
    }
};

template <typename T>
class MultiHeadAttentionBERT : public SytorchModule<T>
{
public:
    using SytorchModule<T>::split;
    using SytorchModule<T>::view;
    using SytorchModule<T>::add;
    using SytorchModule<T>::transpose;
    using SytorchModule<T>::matmul;
    using SytorchModule<T>::scalarmul;
    using SytorchModule<T>::invsqrt;
    using SytorchModule<T>::softmax;
    using SytorchModule<T>::concat;
    using SytorchModule<T>::attention_mask;

public:
    FC<T> *c_attn;
    FC<T> *c_proj;

    u64 n_heads;
    u64 n_embd;

    MultiHeadAttentionBERT(u64 n_heads, u64 n_embd) : n_heads(n_heads), n_embd(n_embd)
    {
        always_assert(n_embd % n_heads == 0);
        c_attn = new FC<T>(n_embd, 3 * n_embd, true);
        c_proj = new FC<T>(n_embd, n_embd, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &x = c_attn->forward(input);
        auto &qkv_heads = split(x, 3);
        auto &q_heads = view(qkv_heads, 0);
        auto &k_heads = view(qkv_heads, 1);
        auto &v_heads = view(qkv_heads, 2);
        auto &qs = split(q_heads, n_heads);
        auto &ks = split(k_heads, n_heads);
        auto &vs = split(v_heads, n_heads);

        double divisor = 1 / sqrt(double(n_embd) / double(n_heads));

        std::vector<Tensor<T> *> qks_sm_vs;
        for (u64 i = 0; i < n_heads; ++i)
        {
            auto &q = view(qs, i);
            auto &k = view(ks, i);
            auto &v = view(vs, i);
            auto &kt = transpose(k);
            auto &qk = matmul(q, kt);
            auto &qks = scalarmul(qk, divisor);

            auto &qks_sm = softmax(qks);

            auto &qks_sm_v = matmul(qks_sm, v);
            qks_sm_vs.push_back(&qks_sm_v);
        }

        auto &qks_sm_vs_cat = concat(qks_sm_vs);
        auto &res = c_proj->forward(qks_sm_vs_cat);
        return res;
    }
};

template <typename T>
class MultiHeadAttentionGPT2 : public SytorchModule<T>
{
    using SytorchModule<T>::split;
    using SytorchModule<T>::view;
    using SytorchModule<T>::transpose;
    using SytorchModule<T>::matmul;
    using SytorchModule<T>::matmul_triangular;
    using SytorchModule<T>::scalarmul;
    using SytorchModule<T>::softmax_triangular;
    using SytorchModule<T>::concat;

public:
    FC<T> *c_attn;
    FC<T> *c_proj;

    u64 n_heads;
    u64 n_embd;

    MultiHeadAttentionGPT2(u64 n_heads, u64 n_embd) : n_heads(n_heads), n_embd(n_embd)
    {
        always_assert(n_embd % n_heads == 0);
        c_attn = new FC<T>(n_embd, 3 * n_embd, true);
        c_proj = new FC<T>(n_embd, n_embd, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &x = c_attn->forward(input);
        auto &qkv_heads = split(x, 3);
        auto &q_heads = view(qkv_heads, 0);
        auto &k_heads = view(qkv_heads, 1);
        auto &v_heads = view(qkv_heads, 2);
        auto &qs = split(q_heads, n_heads);
        auto &ks = split(k_heads, n_heads);
        auto &vs = split(v_heads, n_heads);

        double divisor = 1 / sqrt(double(n_embd) / double(n_heads));

        std::vector<Tensor<T> *> qks_sm_vs;
        for (u64 i = 0; i < n_heads; ++i)
        {
            auto &q = view(qs, i);
            auto &k = view(ks, i);
            auto &v = view(vs, i);
            auto &kt = transpose(k);
            auto &qk = matmul_triangular(q, kt);
            auto &qks = scalarmul(qk, divisor);

            auto &qks_sm = softmax_triangular(qks);

            auto &qks_sm_v = matmul(qks_sm, v);
            qks_sm_vs.push_back(&qks_sm_v);
        }

        auto &qks_sm_vs_cat = concat(qks_sm_vs);
        auto &res = c_proj->forward(qks_sm_vs_cat);
        return res;
    }
};

template <typename T>
class MultiHeadAttentionNEO : public SytorchModule<T>
{
    using SytorchModule<T>::split;
    using SytorchModule<T>::view;
    using SytorchModule<T>::add;
    using SytorchModule<T>::transpose;
    using SytorchModule<T>::matmul;
    using SytorchModule<T>::scalarmul;
    using SytorchModule<T>::invsqrt;
    using SytorchModule<T>::softmax;
    using SytorchModule<T>::concat;
    using SytorchModule<T>::attention_mask;
    // using SytorchModule<T>::local_attention_mask;
    ///////////////////////////
    using SytorchModule<T>::matmul_triangular;
    using SytorchModule<T>::softmax_triangular;

public:
    // FC<T> *c_attn;
    FC<T> *k_attn;
    FC<T> *v_attn;
    FC<T> *q_attn;
    FC<T> *c_proj;

    u64 n_heads;
    u64 n_embd;
    u64 attention_type;
    u64 window_size;

    MultiHeadAttentionNEO(u64 n_heads, u64 n_embd, u64 attention_type, u64 window_size): n_heads(n_heads), n_embd(n_embd)
    {
        always_assert(n_embd % n_heads == 0);
        // c_attn = new FC<T>(n_embd, 3*n_embd, true);
        k_attn = new FC<T>(n_embd, n_embd, false);
        v_attn = new FC<T>(n_embd, n_embd, false);
        q_attn = new FC<T>(n_embd, n_embd, false);
        c_proj = new FC<T>(n_embd, n_embd, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        // auto &x = c_attn->forward(input);
        // auto &qkv_heads = split(x, 3);
        // auto &q_heads = view(qkv_heads, 0);
        // auto &k_heads = view(qkv_heads, 1);
        // auto &v_heads = view(qkv_heads, 2);
        auto &k_heads = k_attn->forward(input);
        auto &v_heads = v_attn->forward(input);
        auto &q_heads = q_attn->forward(input);
        auto &qs = split(q_heads, n_heads);
        auto &ks = split(k_heads, n_heads);
        auto &vs = split(v_heads, n_heads);

        // double divisor = 1 / sqrt(double(n_embd) / double(n_heads));
        // double divisor = 1;

        std::vector<Tensor<T>*> qks_sm_vs;
        for(u64 i = 0; i < n_heads; ++i)
        {
            auto &q = view(qs, i);
            auto &k = view(ks, i);
            auto &v = view(vs, i);
            auto &kt = transpose(k);
            // auto &qks = matmul(q, kt);
            auto &qks = matmul_triangular(q, kt);
            // auto &qk = matmul(q, kt);
            // auto &qks = scalarmul(qk, divisor);

            /*
            Tensor<T> *x = &input;
            if(attention_type % 2 == 0)
            {   
                // printf("global\n");
                auto &qks_masked = attention_mask(qks, 10000.0);
                x = &qks_masked;
            }
            else 
            {
                auto &qks_masked = local_attention_mask(qks, 10000.0);
                x = &qks_masked;
            }
            auto &qks_sm = softmax(*x);
            auto &qks_sm_v = matmul(qks_sm, v);
            */

           Tensor<T> *x = &input;
            if(attention_type % 2 == 0)
            {   
                auto &qks_sm = softmax_triangular(qks);
                x = &qks_sm;
            }
            else 
            {
                // auto &qks_masked = local_attention_mask(qks, 10000.0);
                // auto &qks_sm = softmax_triangular(qks_masked);

                auto &qks_sm = softmax_triangular(qks);
                x = &qks_sm;
            }
            auto &qks_sm_v = matmul(*x, v);

            qks_sm_vs.push_back(&qks_sm_v);
        }

        auto &qks_sm_vs_cat = concat(qks_sm_vs);
        auto &res = c_proj->forward(qks_sm_vs_cat);
        return res;
    }
};

template <typename T>
class MultiHeadAttentionLLAMA : public SytorchModule<T>
{
    using SytorchModule<T>::split;
    using SytorchModule<T>::view;
    using SytorchModule<T>::transpose;
    using SytorchModule<T>::matmul;
    using SytorchModule<T>::matmul_triangular;
    using SytorchModule<T>::scalarmul;
    using SytorchModule<T>::softmax_triangular;
    using SytorchModule<T>::concat;

    using SytorchModule<T>::mul;
    using SytorchModule<T>::add;
    using SytorchModule<T>::silu;
    using SytorchModule<T>::sin;
    using SytorchModule<T>::cos;
    using SytorchModule<T>::rotate_half;
    using SytorchModule<T>::rotary_embedding_before_sin_cos;

public:
    // FC<T> *c_attn;
    FC<T> *q_attn;
    FC<T> *k_attn;
    FC<T> *v_attn;
    FC<T> *c_proj;

    u64 n_heads;
    u64 n_embd;

    MultiHeadAttentionLLAMA(u64 n_heads, u64 n_embd) : n_heads(n_heads), n_embd(n_embd)
    {
        always_assert(n_embd % n_heads == 0);
        // c_attn = new FC<T>(n_embd, 3*n_embd, false);
        q_attn = new FC<T>(n_embd, n_embd, false);
        k_attn = new FC<T>(n_embd, n_embd, false);
        v_attn = new FC<T>(n_embd, n_embd, false);
        c_proj = new FC<T>(n_embd, n_embd, false);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        // auto &x = c_attn->forward(input);
        // auto &qkv_heads = split(x, 3);
        // auto &q_heads = view(qkv_heads, 0);
        // auto &k_heads = view(qkv_heads, 1);
        // auto &v_heads = view(qkv_heads, 2);
        auto &q_heads = q_attn->forward(input);
        auto &k_heads = k_attn->forward(input);
        auto &v_heads = v_attn->forward(input);
        auto &qs = split(q_heads, n_heads);
        auto &ks = split(k_heads, n_heads);
        auto &vs = split(v_heads, n_heads);

        double divisor = 1 / sqrt(double(n_embd) / double(n_heads));

        std::vector<Tensor<T> *> qks_sm_vs;
        for (u64 i = 0; i < n_heads; ++i)
        {
            auto &q_before = view(qs, i);
            auto &k_before = view(ks, i);
            auto &v = view(vs, i);

            /////////////////////
            // Apply Rotary Embedding

            auto &rotary_embedding_intermediate = rotary_embedding_before_sin_cos(v);
            auto &sin_emb = sin(rotary_embedding_intermediate);
            auto &cos_emb = cos(rotary_embedding_intermediate);
            auto &q = add(mul(q_before, cos_emb), mul(rotate_half(q_before), sin_emb));
            auto &k = add(mul(k_before, cos_emb), mul(rotate_half(k_before), sin_emb));

            /*
                        if (done_rotary == false)
                        {
                            auto &rotary_embedding_intermediate = rotary_embedding_before_sin_cos(v);
                            auto &sin_emb2 = sin(rotary_embedding_intermediate);
                            auto &cos_emb2 = cos(rotary_embedding_intermediate);

                            sin_emb = &sin_emb2;
                            cos_emb = &cos_emb2;

                            // std::cout << "sin" << std::endl;
                            // print(sin_emb2, (u64)12);
                            // std::cout << std::endl;
                            // std::cout << "cos" << std::endl;
                            // print(cos_emb2, (u64)12);
                            // std::cout << std::endl;

                            // std::cout << "Isin" << std::endl;
                            // auto sini64 = toi64(*sin_emb, (u64)48);
                            // printfe(sini64, 100);
                            // std::cout << std::endl;
                            // std::cout << "Icos" << std::endl;
                            // auto cosi64 = toi64(*cos_emb, (u64)48);
                            // printfe(cosi64, 100);
                            // std::cout << std::endl;


                            done_rotary = true;
                        }
                        auto &q = add(mul(q_before, *cos_emb), mul(rotate_half(q_before), *sin_emb));
                        auto &k = add(mul(k_before, *cos_emb), mul(rotate_half(k_before), *sin_emb));
            */
            /////////////////////

            auto &kt = transpose(k);
            auto &qk = matmul_triangular(q, kt);
            auto &qks = scalarmul(qk, divisor);

            auto &qks_sm = softmax_triangular(qks);

            auto &qks_sm_v = matmul(qks_sm, v);
            qks_sm_vs.push_back(&qks_sm_v);
        }

        auto &qks_sm_vs_cat = concat(qks_sm_vs);
        auto &res = c_proj->forward(qks_sm_vs_cat);
        return res;
    }
};
