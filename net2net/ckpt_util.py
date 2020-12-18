import os, hashlib
import requests
from tqdm import tqdm

URL_MAP = {
    "biggan_128": "https://heibox.uni-heidelberg.de/f/56ed256209fd40968864/?dl=1",
    "biggan_256": "https://heibox.uni-heidelberg.de/f/437b501944874bcc92a4/?dl=1",
    "dequant_vae": "https://heibox.uni-heidelberg.de/f/e7c8959b50a64f40826e/?dl=1",
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1",
    "coco_captioner": "https://heibox.uni-heidelberg.de/f/b03aae864a0f42f1a2c3/?dl=1",
    "coco_word_map": "https://heibox.uni-heidelberg.de/f/1518aa8461d94e0cb3eb/?dl=1"
}

CKPT_MAP = {
    "biggan_128": "biggan-128.pth",
    "biggan_256": "biggan-256.pth",
    "dequant_vae": "dequantvae-20000.ckpt",
    "vgg_lpips": "autoencoders/lpips/vgg.pth",
    "coco_captioner": "captioning_model_pt16.ckpt",
}

MD5_MAP = {
    "biggan_128": "a2148cf64807444113fac5eede060d28",
    "biggan_256": "e23db3caa34ac4c4ae922a75258dcb8d",
    "dequant_vae": "5c2a6fe765142cbdd9f10f15d65a68b6",
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a",
    "coco_captioner": "db185e0f6791e60d27c00de0f40c376c",
}


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path
