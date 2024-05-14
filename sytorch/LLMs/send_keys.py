"""
This script sends the keys to the server.
Usage:
python3 send_keys.py dataset {id} {dataset_ip_received} {dataset_port_received}
python3 send_keys.py model {id} {model_ip_received} {model_port_received}

"""

import socket
import ssl
import os
import argparse
from tqdm import tqdm
import glob


def send_file(folder_path, server_address, server_port, ca, type):
    # Check if the file exists
    # try:
    #     with open(file_path, "rb") as file:
    #         file_data = file.read()
    # except FileNotFoundError:
    #     print(f"File not found: {file_path}")
    #     return

    # Create a TCP/IP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Wrap the socket in TLS encryption
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.load_verify_locations(cafile=ca)
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = False

        with context.wrap_socket(
            s, server_side=False, server_hostname=server_address
        ) as ssl_socket:
            try:
                # Connect to the server
                ssl_socket.connect((server_address, server_port))
                print(f"Connected to {server_address}:{server_port}")
                # send the folder
                ssl_socket.sendall(folder_path.encode())
                # iterate over the folder and send the number of files
                dataset_keys = glob.glob(os.path.join(folder_path, "client*.dat"))
                dataset_keys = sorted([x.split("/")[-1] for x in dataset_keys])
                print(dataset_keys)
                model_keys = glob.glob(os.path.join(folder_path, "server*.dat"))
                # only keep the file name and sort them
                model_keys = sorted([x.split("/")[-1] for x in model_keys])
                print(model_keys)
                if type == "dataset":
                    num_files = len(dataset_keys)
                    folder = dataset_keys
                elif type == "model":
                    num_files = len(model_keys)
                    folder = model_keys
                else:
                    print('Invalid type. Use "dataset" or "model"')
                    exit(1)

                ssl_socket.sendall(str(num_files).encode())
                print(f"Sending {num_files} files")
                # iterate over the folder and send the files

                for file_name in tqdm(folder):
                    file_path = os.path.join(folder_path, file_name)
                    # Send the file name
                    ssl_socket.sendall(file_name.encode())

                    # Send the file size
                    file_size = os.path.getsize(file_path)
                    ssl_socket.sendall(str(file_size).encode())

                    # Send the file data
                    with open(file_path, "rb") as f:
                        while True:
                            file_data = f.read(1024)
                            if not file_data:
                                break
                            ssl_socket.sendall(file_data)
                    # Send end of file delimiter
                    # ssl_socket.sendall(b"EOF")
                    # Receive confirmation
                    confirmation = ssl_socket.recv(1024).decode()
                    print(f"Sent {file_name}: {confirmation}")

                print("Folder sent successfully")
            except ConnectionRefusedError:
                print(f"Connection refused: {server_address}:{server_port}")
            except ssl.SSLError as e:
                print(f"SSL Error: {e}")
            except Exception as e:
                print(f"Error: {e}")


# Usage example
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send keys")
    parser.add_argument(
        "type", type=str, help="Type of keys to send (dataset or model)"
    )
    parser.add_argument("id", type=str, help="Max number of iterations")
    parser.add_argument("server_address", type=str, help="Server address")
    parser.add_argument("server_port", type=int, help="Server port")
    args = parser.parse_args()
    print(
        f"Sending keys of type {args.type} with id {args.id} to {args.server_address}:{args.server_port}"
    )

    if args.type == "dataset":
        # send the files
        send_file(
            "ezpc_keys/",
            args.server_address,
            args.server_port,
            "/home/trajore/eval_website/ca.crt",
            args.type,
        )
    elif args.type == "model":
        # send the files
        send_file(
            "ezpc_keys/",
            args.server_address,
            args.server_port,
            "/home/trajore/eval_website/ca.crt",
            args.type,
        )
    else:
        print('Invalid type. Use "dataset" or "model"')
        exit(1)
