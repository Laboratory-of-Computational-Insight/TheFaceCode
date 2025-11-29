# @title File System infra

import glob
import os
from typing import Optional

import boto3
import botocore

from utils.singleton.singlton import Singleton
from io import IOBase, BytesIO
import os
import shutil


class FS(metaclass=Singleton):
    def __init__(self, url=None, key=None, secret=None, bucket=None, read_timeout=300, connect_timeout=300):
        # if url is None or key is None or secret is None or bucket is None:
        #     return

        self.s3 = boto3.resource(
            "s3",
            endpoint_url=url,
            aws_access_key_id=key,
            aws_secret_access_key=secret,
            config=botocore.client.Config(read_timeout=read_timeout, connect_timeout=connect_timeout),
        )
        self.boto_bucket = self.s3.Bucket(bucket)
        self.bucket = bucket

    def upload_data(self, data, remote):
        """
        Local mock of S3 upload.
        Saves to ./fs_mock instead of cloud.

        Accepts:
        - BytesIO (e.g. from torch.save into a buffer)
        - bytes / bytearray / memoryview
        """
        remote_path = remote.replace("\\", "/")
        local_path = os.path.join("./fs_mock", remote_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if isinstance(data, BytesIO):
            # We KNOW torch.save wrote the full checkpoint here
            data_bytes = data.getvalue()
        elif isinstance(data, (bytes, bytearray, memoryview)):
            data_bytes = bytes(data)
        else:
            raise TypeError(
                f"upload_data expected BytesIO or bytes-like object, got {type(data)}"
            )

        with open(local_path, "wb") as f:
            f.write(data_bytes)

    def get_data(self, remote, delimiter=""):
        # Normalize path separators
        remote_path = remote.replace("\\", "/")

        # 1) Try local fs_mock first
        local_path = os.path.join("./fs_mock", remote_path).replace("\\", "/")
        if os.path.exists(local_path):
            with open(local_path, "rb") as f:
                return f.read()  # bytes

        # 2) Fallback to S3
        if not delimiter:
            delimiter = remote_path

        body = None
        for obj in self.boto_bucket.objects.filter(Prefix=delimiter):
            key = obj.key
            if key == remote_path:
                body = obj.get()["Body"].read()  # bytes
                break

        return body  # bytes or None
    def get_file(self, remote, local=None, download=True, override=False):
        # Normalize remote path to use forward slashes (S3-style)
        remote_path = remote.replace("\\", "/")

        # Default local path if none provided
        if local is None:
            local = os.path.normpath(os.path.join("tmp", *os.path.split(remote_path)))

        local_folder = os.path.dirname(local)
        os.makedirs(local_folder, exist_ok=True)

        # 1) Try local mock FS first: ./fs_mock/<remote_path>
        mock_path = os.path.join("./fs_mock", remote_path)
        if os.path.exists(mock_path):
            # Copy from mock to requested local path if needed
            if override or not os.path.exists(local):
                os.makedirs(os.path.dirname(local), exist_ok=True)
                shutil.copy2(mock_path, local)
            return local

        # 2) If download is disabled and we already have a local file, just return it
        if download and not override:
            if os.path.exists(local):
                return local

        # 3) Fallback: download from S3
        self.boto_bucket.download_file(remote_path, local)
        return local