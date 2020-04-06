## Image-Caption

> NTU-AI6127 NLP Final Project

### Reference

* [a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)

### TODO

- [x] Baseline.
- [x] [self-attentiion:Transformer](https://github.com/huggingface/transformers)
- [x]  Glove pre-trained word embedding
- [ ] [Multi-Eva metrics](https://github.com/tylin/coco-caption)
- [ ] [try BERT](https://github.com/ajamjoom/Image-Captions)

### Requirements

Recommend for `Conda` Env.

imread is deprecated in the newer version of SciPy. To use it:

```shell
pip uninstall scipy
pip install scipy==1.2.1
```

We're using `Pytorch 1.4.0` and `Python 3.7`.

### Data

```json
{"images": [{"sentids": [0, 1, 2, 3, 4], "imgid": 0, "sentences": [{"tokens": ["a", "black", "dog", "is", "running", "after", "a", "white", "dog", "in", "the", "snow"], "raw": "A black dog is running after a white dog in the snow .", "imgid": 0, "sentid": 0}, {"tokens": ["black", "dog", "chasing", "brown", "dog", "through", "snow"], "raw": "Black dog chasing brown dog through snow", "imgid": 0, "sentid": 1}, {"tokens": ["two", "dogs", "chase", "each", "other", "across", "the", "snowy", "ground"], "raw": "Two dogs chase each other across the snowy ground .", "imgid": 0, "sentid": 2}, {"tokens": ["two", "dogs", "play", "together", "in", "the", "snow"], "raw": "Two dogs play together in the snow .", "imgid": 0, "sentid": 3}, {"tokens": ["two", "dogs", "running", "through", "a", "low", "lying", "body", "of", "water"], "raw": "Two dogs running through a low lying body of water .", "imgid": 0, "sentid": 4}], "split": "train", "filename": "2513260012_03d33305cf.jpg"}, {"sentids": [5, 6, 7, 8, 9], ...}, {}, {}, ...
```

```json
# if COCO
{"filepath": "train2014", "sentids": [283074, 283110, 284385, 284799, 285885], "filename": "COCO_train2014_000000537772.jpg", "imgid": 116634, "split": "train", "sentences": [{"tokens": ["a", "white", "car", "has", "stopped", "in", "front", "of", "a", "white", "truck"], "raw": "A white car has stopped in front of a white truck", "imgid": 116634, "sentid": 283074}, {"tokens": ["unloaded", "flat", "bed", "truck", "and", "car", "stopped", "in", "parking", "lot"], "raw": "Unloaded flat bed truck and car stopped in parking lot.", "imgid": 116634, "sentid": 283110}, {"tokens": ["a", "truck", "faces", "a", "car", "in", "front", "of", "a", "house"], "raw": "A truck faces a car in front of a house.", "imgid": 116634, "sentid": 284385}, {"tokens": ["a", "flatbed", "semi", "facing", "a", "car", "in", "front", "of", "a", "house"], "raw": "A flatbed semi facing a car in front of a house.", "imgid": 116634, "sentid": 284799}, {"tokens": ["a", "tractor", "trailer", "and", "a", "white", "car", "facing", "each", "other"], "raw": "a tractor trailer and a white car facing each other", "imgid": 116634, "sentid": 285885}], "cocoid": 537772},
```

