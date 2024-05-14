#include <sytorch/LLMs/llm_library.h>

int ct_main(std::string weights_file, std::string input_file)
{
    sytorch_init();

    ModelFactory<i64> model_factory;
    auto net = model_factory.create(llm_version);

    net->init(scale);
    net->load(weights_file);

    u64 n_seq = get_n_seq(input_file, n_embd);
    Tensor<i64> input({n_seq, n_embd});
    input.printshape();
    input.load(input_file, scale);

    auto t1 = std::chrono::high_resolution_clock::now();

    net->forward(input);

    auto t2 = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Total time = " << compute_time / (1000.0) << " ms" << std::endl;

    print(net->activation, scale);

    return 0;
}

int float_main(std::string weights_file, std::string input_file)
{
    sytorch_init();

    ModelFactory<float> model_factory;
    auto net = model_factory.create(llm_version);

    net->init(0);
    net->load(weights_file);

    auto t1 = std::chrono::high_resolution_clock::now();

    u64 n_seq = get_n_seq(input_file, n_embd);
    std::cout << n_seq << std::endl;
    Tensor<float> input({n_seq, n_embd});
    input.load(input_file, scale);

    net->forward(input);
    net->activation.print();

    auto t2 = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Total time = " << compute_time / (1000.0) << " ms" << std::endl;

    return 0;
}

int lt_main(int party, std::string weights_file, std::string input_file,std::string key_path, std::string ip = "127.0.0.1", u64 n_seq = 10, std::string id = "0", int nt = 4, bool no_reveal = false)
{

    sytorch_init();

    using LlamaVersion = LlamaTransformer<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));

    LlamaConfig::bitlength = 50;
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = false;
    LlamaConfig::stochasticRT = false;
    LlamaConfig::num_threads = nt;

    llama->init(ip, id,key_path,false);

    ModelFactory<u64> model_factory;
    auto net = model_factory.create(llm_version);

    net->init(scale);
    net->setBackend(llama);

    if (party == CLIENT)
    {
        n_seq = get_n_seq(input_file, n_embd);
        llama->send_n_seq(n_seq);
    }
    else if (party == SERVER)
    {
        n_seq = llama->recv_n_seq();
    }
    Tensor<u64> input({n_seq, n_embd});

    if (party == SERVER)
    {
        net->load(weights_file);
    }
    else if (party == DEALER)
    {
        net->zero();
    }
    llama->initializeInferencePartyA(net->root);

    if (party == CLIENT)
    {
        input.load(input_file, scale);
    }
    llama->initializeInferencePartyB(input);

    std::cout << "n_seq = " << n_seq << std::endl;

    llama::start();
    net->forward(input);
    llama::end();

    auto &output = net->activation;

    if (!no_reveal)
    {
        llama->outputA(output);
        if (party == CLIENT)
        {
            print(output, scale, LlamaConfig::bitlength);
        }
    }
    else
    {
        llama->outputShares(output.data, output.size());
        if (party == CLIENT or party == SERVER)
        {
            print_shares("secret_shares.txt", output, scale, LlamaConfig::bitlength);
        }
    }
    llama->finalize();

    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <party> [ name=value ]... \n "
                  << "\twt_file\t Weights file : fed by server[default = ] \n"
                  << "\tin_file\t Input file : fed by client[default = ]\n"
                  <<" \tkey_path\t Path to key files : fed by all parties[default =./ ]\n"
                  << "\tip\t IP address of server : fed by client[default = 127.0.0.1]\n"
                  << "\tn_seq\t Number of sequences : fed by dealer[default = 10]\n"
                  << "\tid\t ID of party : fed by all parties<optional>[default = 0]\n"
                  << "\tnt\t Number of threads : fed by all parties<optional>[default = 4]\n "
                  << "\tno_reveal\t Reveal output : fed by all parties<optional>[default = 0]\n "
                  << "\tct_float\t Use float for cleartext computation : fed by all parties<optional>[default = 0] "
                  << std::endl;
        return 0;
    }

    int party = atoi(argv[1]);
    std::string weights_file = "";
    std::string input_file = "";
    std::string ip = "127.0.0.1";
    u64 n_seq = 10;
    std::string id = "0";
    std::string key_path = "./";
    int nt = 4;
    bool no_reveal = 0;
    bool ct_float = 0;

    ArgMapping amap;
    amap.arg("wt_file", weights_file, "Weights file: fed by server");
    amap.arg("in_file", input_file, "Input file: fed by client");
    amap.arg("key_path", key_path, "Path to key files: fed by all parties  [Default: ./]");
    amap.arg("ip", ip, "IP address of server: fed by client  [Default: 127.0.0.1]");
    amap.arg("n_seq", n_seq, "Number of sequences: fed by dealer");
    amap.arg("id", id, "ID of party: fed by all parties  [Default: 0]");
    amap.arg("nt", nt, "Number of threads: fed by all parties  [Default: 4]");
    amap.arg("no_reveal", no_reveal, "Reveal output: fed by all parties  [Default: 0]");
    amap.arg("ct_float", ct_float, "Use float for cleartext computation  [Default: 0]");

    if (party == DEALER)
    {
        amap.setRequired("n_seq", true);
        amap.setRequired("key_path", true);
    }
    else if (party == SERVER)
    {
        amap.setRequired("wt_file", true);
        amap.setRequired("key_path", true);
    }
    else if (party == CLIENT)
    {
        amap.setRequired("in_file", true);
        amap.setRequired("key_path", true);
        amap.setRequired("ip", true);
    }

    read_config("config.json");

    if (party == 0)
    {
        amap.setRequired("wt_file", true);
        amap.setRequired("in_file", true);

    }

    amap.parse(argc, argv);

    if (party == 0)
    {
        if (ct_float)
            float_main(weights_file, input_file);
        else
            ct_main(weights_file, input_file);
    }
    else
    {
        lt_main(party, weights_file, input_file,key_path, ip, n_seq, id, nt, no_reveal);
    }
}