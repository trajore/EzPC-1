#include <iostream>
#include <fstream>
#include <string>
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#define BUFFER_SIZE 1024

int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <server_address> <server_port> <ssl_certificate> <ca_certificate>" << std::endl;
        return 1;
    }

    const char *server_address = argv[1];
    const char *server_port = argv[2];
    const char *ssl_certificate = argv[3];
    const char *ca_certificate = argv[4];
    const char *file_path = "/home/trajore/eval_website/gpt-ezpc/sytorch/LLMs/dataset.tar.gz"; // Path to the file to send

    // Initialize SSL
    SSL_library_init();
    OpenSSL_add_all_algorithms();
    SSL_load_error_strings();
    ERR_load_BIO_strings();
    ERR_load_crypto_strings();

    // Create SSL context
    SSL_CTX *ctx = SSL_CTX_new(TLS_client_method());
    if (!ctx) {
        std::cerr << "Error creating SSL context" << std::endl;
        ERR_print_errors_fp(stderr);
        return 1;
    }

    // Load CA certificate
    if (!SSL_CTX_load_verify_locations(ctx, ca_certificate, nullptr)) {
        std::cerr << "Error loading CA certificate" << std::endl;
        ERR_print_errors_fp(stderr);
        return 1;
    }

    // Create SSL connection
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        std::cerr << "Error creating socket" << std::endl;
        return 1;
    }

    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(std::stoi(server_port));
    inet_pton(AF_INET, server_address, &server_addr.sin_addr);

    // Connect to server
    if (connect(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1) {
        std::cerr << "Error connecting to server" << std::endl;
        close(sockfd);
        return 1;
    }

    // Set up SSL connection
    SSL *ssl = SSL_new(ctx);
    if (!ssl) {
        std::cerr << "Error creating SSL structure" << std::endl;
        close(sockfd);
        SSL_CTX_free(ctx);
        return 1;
    }
    SSL_set_fd(ssl, sockfd);

    // Perform SSL handshake
    if (SSL_connect(ssl) <= 0) {
        std::cerr << "Error performing SSL handshake" << std::endl;
        ERR_print_errors_fp(stderr);
        close(sockfd);
        SSL_free(ssl);
        SSL_CTX_free(ctx);
        return 1;
    }

    // Open file to send
    std::ifstream input_file(file_path, std::ios::binary);
    if (!input_file) {
        std::cerr << "Error opening file" << std::endl;
        close(sockfd);
        SSL_free(ssl);
        SSL_CTX_free(ctx);
        return 1;
    }

    // Send file data
    char buffer[BUFFER_SIZE];
    int bytes_read;
    while ((bytes_read = input_file.readsome(buffer, BUFFER_SIZE)) > 0) {
        int bytes_sent = SSL_write(ssl, buffer, bytes_read);
        if (bytes_sent <= 0) {
            std::cerr << "Error sending data" << std::endl;
            close(sockfd);
            SSL_free(ssl);
            SSL_CTX_free(ctx);
            return 1;
        }
    }

    // Clean up
    input_file.close();
    SSL_shutdown(ssl);
    SSL_free(ssl);
    close(sockfd);
    SSL_CTX_free(ctx);

    std::cout << "File sent successfully" << std::endl;
    return 0;
}
