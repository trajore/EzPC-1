#include <sytorch/backend/llama_transformer.h>
#include <sytorch/layers/layers.h>
#include <sytorch/module.h>
#include <llama/utils.h>
#include <llama/api.h>

////////////////////////////////
// bool done_rotary = false;
// Tensor<i64> *sin_emb;
// Tensor<i64> *cos_emb;
////////////////////////////////


template <typename T>
void printfe(Tensor<T> &t, u64 n = 1)
{
    u64 nm = std::min(n, t.size());
    for (int i = 0; i < nm; ++i)
    std::cout << t.data[i] << " ";
    std::cout << std::endl;
    
}

Tensor<i64> toi64(Tensor<u64> &t, u64 bw)
{
    Tensor<i64> res(t.shape);
    
    for (int i = 0; i < t.size(); ++i)
        res.data[i] = (t.data[i] + (1LL << (bw - 1))) % (1LL << bw) - (1LL << (bw - 1));
    return res;
}

template <typename T>
class FFN : public SytorchModule<T>
{
    using SytorchModule<T>::silu;
    using SytorchModule<T>::mul;

    u64 in;
    u64 intermediate_size;
public:
    FC<T> *up1;
    FC<T> *up2;
    FC<T> *down;

    FFN(u64 in, u64 intermediate_size) : in(in), intermediate_size(intermediate_size) 
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
class MultiHeadAttention : public SytorchModule<T>
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

    MultiHeadAttention(u64 n_heads, u64 n_embd): n_heads(n_heads), n_embd(n_embd)
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

        std::vector<Tensor<T>*> qks_sm_vs;
        for(u64 i = 0; i < n_heads; ++i)
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

template <typename T>
class TransformerBlock : public SytorchModule<T>
{
    using SytorchModule<T>::add;

    MultiHeadAttention<T> *attn;
    FFN<T> *ffn;
    RMSNorm<T> *ln0;
    RMSNorm<T> *ln1;
    
    u64 n_heads, n_embd, intermediate_size;
public:

    TransformerBlock(u64 n_heads, u64 n_embd, u64 intermediate_size): n_heads(n_heads), n_embd(n_embd), intermediate_size(intermediate_size)
    {
        attn = new MultiHeadAttention<T>(n_heads, n_embd);
        ffn = new FFN<T>(n_embd, intermediate_size);
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
    std::vector<TransformerBlock<T> *> blocks;
    RMSNorm<T> *ln_f;
    u64 n_layer, n_heads, n_embd, intermediate_size;

public:
    
    LLAMA_MODEL(u64 n_layer, u64 n_heads, u64 n_embd, u64 intermediate_size): n_layer(n_layer), n_heads(n_heads), n_embd(n_embd), intermediate_size(intermediate_size)
    {
        for(u64 i = 0; i < n_layer; ++i)
        {
            blocks.push_back(new TransformerBlock<T>(n_heads, n_embd, intermediate_size));
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
        
        for(u64 i = 0; i < n_layer; ++i)
        {
            auto &block = blocks[i];
            auto &x_out = block->forward(*x);
            x = &x_out;
        }
        return ln_f->forward(*x);
    }
};

u64 get_n_seq(std::string filename, u64 n_embd)
{
    u64 n_elements = std::filesystem::file_size(filename);
    assert(n_elements % (4 * n_embd) == 0);
    return n_elements / (4 * n_embd);
}


template <typename T>
class LlamaNextWordLogits : public SytorchModule<T>
{
    using SytorchModule<T>::view;
    using SytorchModule<T>::transpose;
    LLAMA_MODEL<T> *llama_model;
    FC<T> *fc;
    u64 n_layer, n_heads, n_embd, n_vocab, intermediate_size;
public:
    
    LlamaNextWordLogits(u64 n_layer, u64 n_heads, u64 n_embd, u64 n_vocab, u64 intermediate_size): n_layer(n_layer), n_heads(n_heads), n_embd(n_embd), n_vocab(n_vocab), intermediate_size(intermediate_size)
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

/*
void ct_main()
{
    sytorch_init();

    const u64 n_embd = 4096;
    const u64 n_head = 32;
    const u64 scale = 12;
    const u64 n_seq = 128;
    const u64 intermediate_size = 11008;

    // TransformerBlock<i64> net(n_head, n_embd);
    TransformerBlock<float> net(n_head, n_embd, intermediate_size);
    net.init(scale);
    net.zero();

    // Tensor<i64> input({n_seq, n_embd});
    Tensor<float> input({n_seq, n_embd});
    input.fill(1LL << (scale-2));
    net.forward(input);
}
*/


int test_llamaNextWordLogits_ct(int __argc, char**__argv) {
    sytorch_init();

    const u64 n_vocab = 32000;
    const u64 n_ctx = 2048;
    const u64 n_embd = 4096;
    const u64 n_head = 32;
    const u64 n_layer = 32;
    const u64 intermediate_size = 11008;
    const u64 scale = 12;

    // LLAMA_MODEL<i64> llama_model(n_layer, n_head, n_embd, intermediate_size);
    // LlamaNextWordLogits<float> llama_model(n_layer, n_head, n_embd, n_vocab, intermediate_size);
    LlamaNextWordLogits<i64> llama_model(n_layer, n_head, n_embd, n_vocab, intermediate_size);
    llama_model.init(scale);
    // gpt2.setBackend(new BaselineClearText<i64>());
    // llama_model.load("../../../open_llama_7b.dat");
    llama_model.load("../../../meta_llama2_7b.dat");

    auto t1 = std::chrono::high_resolution_clock::now();

    // int arr[50] = {3577, 3189, 2165, 2582, 2556, 1808, 2374, 2064, 3081, 221, 1480, 
    // 3368, 241, 4235, 5018, 929, 3865, 4429, 3329, 4908, 2864, 1794, 3002, 3338, 3222, 
    // 4813, 745, 3444, 3859, 3102, 4830, 1247, 3459, 4328, 1753, 975, 4664, 4749, 1362, 
    // 1441, 3026, 442, 3194, 3808, 1196, 3017, 3134, 706, 1331, 326};
    // for (int j = 0; j < 13; ++j) {
    
    int start = atoi(__argv[1]);
    int end = atoi(__argv[2]);

    for (int i = start; i < end; ++i) {
        // std::string fname = std::string("../../../lambada-open-llama-7b/") + std::to_string(i) + ".dat";
        std::string fname = std::string("../../../lambada-meta-llama2-7b/") + std::to_string(i) + ".dat";
        // std::string fname = std::string("3.dat");
        u64 n_seq = get_n_seq(fname, n_embd);
        // std::cout << "n_seq = " << n_seq << std::endl;
        Tensor<i64> input({n_seq, n_embd});
        // Tensor<i64> input({n_seq, n_embd});
        input.load(fname, scale);
       

        // auto t1 = std::chrono::high_resolution_clock::now();
        auto &res = llama_model.forward(input);


        // printshape(res.shape);
        // auto t2 = std::chrono::high_resolution_clock::now();
        // auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        // std::cout << "Total time = " << compute_time / (1000.0)  << " ms" << std::endl;
        // print(gpt2.activation, scale);
        // print(res, scale);
        // res.print();
        // printfe(res, 100);

        i64 max = INT_MIN;
        int argmax = 0;
        for(int i=0; i < n_vocab; i++)
        {   
            if(res.data[i]>max)
            {
                max = res.data[i];
                argmax = i;
            }
        }
        std::cout << argmax << std::endl;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Total time = " << compute_time / (1000.0)  << " ms" << std::endl;

    return 0;
}


/*
template <typename T>
class SIN : public SytorchModule<T>
{
    using SytorchModule<T>::silu;
public:

    Tensor<T> &_forward(Tensor<T> &input)
    {
       return silu(input);
    }
};
*/

/*
template <typename T>
class NORM : public SytorchModule<T>
{
    RMSNorm<T> *ln0;
    // LayerNorm<T> *ln0;
    u64 n_embd;
public:
    NORM(u64 n_embd): n_embd(n_embd)
    {
        ln0 = new RMSNorm<T>(n_embd, false);
        // ln0 = new LayerNorm<T>(n_embd);
    }
    Tensor<T> &_forward(Tensor<T> &input)
    {
       return ln0->forward(input);
    }
};
*/


void lt_main(int __argc, char**__argv){
    
    sytorch_init();

    const u64 n_vocab = 32000;
    const u64 n_ctx = 2048;
    const u64 n_embd = 4096;
    const u64 n_head = 32;
    const u64 n_layer = 32;
    const u64 intermediate_size = 11008;

    // const u64 n_embd = 2048;

    int party = atoi(__argv[1]);
    std::string ip = "127.0.0.1";

    using LlamaVersion = LlamaTransformer<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));

    const u64 scale = 12;

    LlamaConfig::bitlength = 48;
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = false;
    LlamaConfig::stochasticRT = false;
    LlamaConfig::num_threads = 4;

    if(__argc > 2){
        ip = __argv[2];
    }
    llama->init(ip, true);

    // SIN<u64> net;
    // NORM<u64> net(n_embd);
    // LLAMA_MODEL<u64> net(n_layer, n_head, n_embd, intermediate_size);
    LlamaNextWordLogits<u64> net(n_layer, n_head, n_embd, n_vocab, intermediate_size);
    net.init(scale);
    net.setBackend(llama);
    net.optimize();
    if(party == DEALER){
        net.zero();
    }
    if(party == SERVER){
        // net.zero();
        net.load("../../../meta_llama2_7b.dat");
        // net.load("../../../transformers/gpt-neo-lambada/gpt-neo-1pt3B-weights.dat");
    }

    llama->initializeInferencePartyA(net.root);

    // std::string fname = std::string("../../../transformers/datasets/lambada_large/") + std::to_string(3) + ".dat";
    std::string fname = std::string("3.dat");
    u64 n_seq = get_n_seq(fname, n_embd);
    // u64 n_seq = 128;
    Tensor<u64> input({n_seq, n_embd});

    if(party == CLIENT){
        input.load(fname, scale);
        // input.load("3.dat", scale);
        // input.fill(1LL << (scale-2));
    }
    llama->initializeInferencePartyB(input);

    llama::start();
    net.forward(input);
    llama::end();

    auto &output = net.activation;
    llama->outputA(output);
    if (party == CLIENT) {
        // print(output, scale, LlamaConfig::bitlength);
        auto outputi64 = toi64(output, LlamaConfig::bitlength);
        printfe(outputi64, 100);
    }
    llama->finalize();

}






int main(int __argc, char**__argv)
{
    // ct_main();
    // lt_main(__argc, __argv);
    test_llamaNextWordLogits_ct(__argc, __argv);

    // const u64 scale = 12;
    // const u64 n_embd = 2048;
    // sytorch_init();
    // std::string fname = std::string("../../../transformers/datasets/lambada_large/") + std::to_string(3) + ".dat";
    // u64 n_seq = get_n_seq(fname, n_embd);
    // Tensor<i64> input({n_seq, n_embd});
    // input.load(fname, scale);
    // // SIN<i64> sin_test;
    // // sin_test.init(scale);
    // // auto &res = sin_test.forward(input);
    // NORM<i64> norm_test(n_embd);
    // norm_test.init(scale);
    // norm_test.load("../../../transformers/gpt-neo-lambada/gpt-neo-1pt3B-weights.dat");
    // auto &res = norm_test.forward(input);
    // // print(res, scale);
    // // res.print();

    // printfe(res, 100);
    // // printfe(input, 100);

    

    return 0;
}