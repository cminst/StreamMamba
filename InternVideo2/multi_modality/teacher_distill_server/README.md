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
