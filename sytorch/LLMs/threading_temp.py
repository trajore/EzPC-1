import socket
import threading

def handle_client(client_socket):
    # Handle client communication
    try:
        while True:
            request_data = client_socket.recv(1024)
            if not request_data:
                break
            print(f"Received: {request_data}")
            # Process the data...
            client_socket.send(b"ACK!")
    finally:
        client_socket.close()

def server_listen(port):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', port))
    server.listen(5)  # Set the backlog queue to 5
    print(f"Listening on port {port}...")
    
    while True:
        client, addr = server.accept()
        print(f"Accepted connection from: {addr}")
        client_handler = threading.Thread(target=handle_client, args=(client,))
        client_handler.start()

# Start the server
server_listen(8000)