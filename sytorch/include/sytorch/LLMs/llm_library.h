#pragma once

#include <sytorch/LLMs/ArgMapping.h>
#include <sytorch/LLMs/bert.h>
#include <sytorch/LLMs/gpt2.h>
#include <sytorch/LLMs/gptneo.h>
#include <sytorch/LLMs/llama.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <functional>
#include <filesystem>
#include <fstream>

template <typename T>
using llm = std::function<std::unique_ptr<SytorchModule<T>>()>;

using json = nlohmann::json;

u64 n_vocab = 50257;
u64 n_ctx = 1024;
u64 n_embd = 768;
u64 n_head = 12;
u64 n_layer = 12;
u64 n_label = 2;
u64 window_size = 256;
u64 intermediate_size = 11008;
u64 scale = 12;
std::string llm_version = "BERT";

template <typename T>
class ModelFactory
{
public:
    std::unordered_map<std::string, llm<T>> factories;

    ModelFactory()
    {
        factories["BERT"] = []() -> std::unique_ptr<SytorchModule<T>>
        {
            return std::make_unique<BERT<T>>(n_layer, n_head, n_embd);
        };

        factories["BERTSequenceClassification"] = []() -> std::unique_ptr<SytorchModule<T>>
        {
            return std::make_unique<BERTSequenceClassification<T>>(n_layer, n_head, n_embd, n_label);
        };

        factories["GPT2"] = []() -> std::unique_ptr<SytorchModule<T>>
        {
            return std::make_unique<GPT2<T>>(n_layer, n_head, n_embd);
        };

        factories["GPT2SequenceClassification"] = []() -> std::unique_ptr<SytorchModule<T>>
        {
            return std::make_unique<GPT2SequenceClassification<T>>(n_layer, n_head, n_embd, n_label);
        };

        factories["GPT2NextWordLogits"] = []() -> std::unique_ptr<SytorchModule<T>>
        {
            return std::make_unique<GPT2NextWordLogits<T>>(n_layer, n_head, n_embd, n_vocab);
        };

        factories["GPTNEO"] = []() -> std::unique_ptr<SytorchModule<T>>
        {
            return std::make_unique<GPTNEO<T>>(n_layer, n_head, n_embd, window_size);
        };

        factories["GPTNEONextWordLogits"] = []() -> std::unique_ptr<SytorchModule<T>>
        {
            return std::make_unique<GPTNEONextWordLogits<T>>(n_layer, n_head, n_embd, n_vocab, window_size);
        };

        factories["LLAMA7NextWordLogits"] = []() -> std::unique_ptr<SytorchModule<T>>
        {
            return std::make_unique<LlamaNextWordLogits<T>>(n_layer, n_head, n_embd, n_vocab, intermediate_size);
        };
    }

    std::unique_ptr<SytorchModule<T>> create(const std::string &name)
    {
        if (factories.find(name) != factories.end())
        {
            return factories[name]();
        }
        return nullptr; // Or throw an exception.
    }
};

void read_config(std::string filename)
{
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    json j;
    file >> j;

    if (j.is_object())
    {
        for (auto &[key, value] : j.items())
        {
            if (value.is_string())
            {
                if (key == "llm_version")
                {
                    llm_version = value.get<std::string>();
                }
            }
            else if (value.is_number_unsigned())
            {
                if (key == "n_vocab")
                {
                    n_vocab = value;
                }
                else if (key == "n_ctx")
                {
                    n_ctx = value;
                }
                else if (key == "n_embd")
                {
                    n_embd = value;
                }
                else if (key == "n_head")
                {
                    n_head = value;
                }
                else if (key == "n_layer")
                {
                    n_layer = value;
                }
                else if (key == "n_label")
                {
                    n_label = value;
                }
                else if (key == "scale")
                {
                    scale = value;
                }
                else if (key == "window_size")
                {
                    window_size = value;
                }
                else if (key == "intermediate_size")
                {
                    intermediate_size = value;
                }
            }
        }
    }

    file.close();
}

u64 get_n_seq(std::string filename, u64 n_embd)
{
    u64 n_elements = std::filesystem::file_size(filename);
    assert(n_elements % (4 * n_embd) == 0);
    std::cout << "n_elements: " << n_elements / (4 * n_embd) << std::endl;
    return n_elements / (4 * n_embd);
}
