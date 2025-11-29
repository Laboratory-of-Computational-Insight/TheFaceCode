from utils.filesystem.fs import FS
from config import configuration

def init(
        fs_url,
        fs_key,
        fs_secret,
        fs_bucket,
):
    print(fs_url, fs_key, fs_secret, fs_bucket)
    fs = FS(fs_url, fs_key, fs_secret, fs_bucket)
init(
    configuration.wasabi.wasabi_url,
    configuration.wasabi.key,
    configuration.wasabi.secret,
    configuration.wasabi.bucket,
)