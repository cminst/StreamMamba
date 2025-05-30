# InternVideo2 6B Embeddings Server

In order to start the server, first run:

```
$ huggingface-cli download OpenGVLab/InternVideo2-Stage2_6B-224p-f4 internvideo2-s2_6b-224p-f4.pt --local-dir .
[It will show the path to the file]
```

Then run:

```
$ export IV2_6B_CKPT=/path/to/internvideo2-s2_6b-224p-f4.pt
```

And then

```
$ python3 server.py
```

## Testing the server

The script starts a FastAPI service bound to `0.0.0.0:8008`. To query the
service from another machine, replace `<server-ip>` with the address of the
server and send requests to `http://<server-ip>:8008/infer`.

Example request:

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"window_tensor":[[[[0]]]]}' \
     http://<server-ip>:8008/infer
```

If the command fails or times out, verify that the server is running and that
firewall rules allow access to port `8008`.
