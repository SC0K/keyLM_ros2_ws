"""Connection defaults for SSH-forwarded Ollama servers."""

DEFAULT_SERVER = "tars"
DEFAULT_LOCAL_PORT = 11434
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.6:27b"
SERVER_PROFILES = {
    # SSH host, remote Ollama port, default SSH user.
    "tars": ("tars", 11434, "sitchen"),
    "case": ("case.inf.ethz.ch", 11434, "sitchen"),
    "tailscale": ("100.99.254.46", 11434, "sitongchen"),
}


def tunnel_destination(server, host=None, remote_port=None):
    profile_host, profile_port, _ = SERVER_PROFILES[server]
    return (host if host is not None else profile_host,
            remote_port if remote_port is not None else profile_port)


def ssh_tunnel_command(server=DEFAULT_SERVER, *, user=None,
                       local_port=DEFAULT_LOCAL_PORT, host=None, remote_port=None):
    if user is None:
        user = SERVER_PROFILES[server][2]
    host, remote_port = tunnel_destination(server, host, remote_port)
    if not (1 <= local_port <= 65535 and 1 <= remote_port <= 65535):
        raise ValueError("Tunnel ports must be between 1 and 65535")
    return [
        "ssh", *(["-p", "2222"] if server == "tailscale" else []),
        "-N", "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30", "-L",
        f"{local_port}:localhost:{remote_port}", f"{user}@{host}",
    ]
