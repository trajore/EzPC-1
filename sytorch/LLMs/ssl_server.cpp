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

    // Initialize SSL
    SSL_library_init();
    OpenSSL_add_all_algorithms();
    SSL_load_error_strings();
    ERR_load_BIO_strings();
    ERR_load_crypto_strings();

    // Create SSL context
    SSL_CTX *ctx = SSL_CTX_new(TLS_server_method());
    if (!ctx) {
        std::cerr << "Error creating SSL context" << std::endl;
        ERR_print_errors_fp(stderr);
        return 1;
    }

    // Load server certificate
    if (SSL_CTX_use_certificate_file(ctx, ssl_certificate, SSL_FILETYPE_PEM) <= 0) {
        std::cerr << "Error loading server certificate" << std::endl;
        ERR_print_errors_fp(stderr);
        return 1;
    }

    // Load private key
    if (SSL_CTX_use_PrivateKey_file(ctx, ssl_certificate, SSL_FILETYPE_PEM) <= 0) {
        std::cerr << "Error loading private key" << std::endl;
        ERR_print_errors_fp(stderr);
        return 1;
    }

    // Load CA certificate
    if (!SSL_CTX_load_verify_locations(ctx, ca_certificate, nullptr)) {
        std::cerr << "Error loading CA certificate" << std::endl;
        ERR_print_errors_fp(stderr);
        return 1;
    }

    // Set up TCP socket
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        std::cerr << "Error creating socket" << std::endl;
        return 1;
    }

    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(std::stoi(server_port));
    inet_pton(AF_INET, server_address, &server_addr.sin_addr);

    // Bind socket
    if (bind(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1) {
        std::cerr << "Error binding socket" << std::endl;
        close(sockfd);
        return 1;
    }

    // Listen for connections
    if (listen(sockfd, 1) == -1) {
        std::cerr << "Error listening for connections" << std::endl;
        close(sockfd);
        return 1;
    }

    // Accept connection
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    int clientfd = accept(sockfd, (struct sockaddr *)&client_addr, &client_len);
    if (clientfd == -1) {
        std::cerr << "Error accepting connection" << std::endl;
        close(sockfd);
        return 1;
    }

    // Set up SSL connection
    SSL *ssl = SSL_new(ctx);
    if (!ssl) {
        std::cerr << "Error creating SSL structure" << std::endl;
        close(clientfd);
        close(sockfd);
        SSL_CTX_free(ctx);
        return 1;
    }
    SSL_set_fd(ssl, clientfd);

    // Perform SSL handshake
    if (SSL_accept(ssl) <= 0) {
        std::cerr << "Error performing SSL handshake" << std::endl;
        ERR_print_errors_fp(stderr);
        close(clientfd);
        close(sockfd);
        SSL_free(ssl);
        SSL_CTX_free(ctx);
        return 1;
    }

    // Receive file data
    char buffer[BUFFER_SIZE];
    int bytes_received;
    std::ofstream output_file("received_file.tar.gz", std::ios::binary);
    if (!output_file) {
        std::cerr << "Error creating output file" << std::endl;
        close(clientfd);
        close(sockfd);
        SSL_free(ssl);
        SSL_CTX_free(ctx);
        return 1;
    }

    while ((bytes_received = SSL_read(ssl, buffer, BUFFER_SIZE)) > 0) {
        output_file.write(buffer, bytes_received);
    }

    if (bytes_received < 0) {
        std::cerr << "Error receiving data" << std::endl;
    }

    // Clean up
    output_file.close();
    SSL_shutdown(ssl);
    SSL_free(ssl);
    close(clientfd);
    close(sockfd);
    SSL_CTX_free(ctx);

    return 0;
}
