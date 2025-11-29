from typing import Optional

import pydantic
from pydantic import BaseModel

class Model(BaseModel):
    restore_path: Optional[str]
    save_path: str

class WasabiConf(BaseModel):
    bucket: str
    data_path: str
    model_path: str
    wasabi_url: str
    wasabi_base: str
    region: str
    key: str
    secret: str


class DataConf(BaseModel):
    batch_size: int
    train_limit: Optional[int]
    val_limit: Optional[int]
    test_limit: Optional[int]


class Train(BaseModel):
    train_data_path: str
    validation_data_path: str
    test_data_path: str
    save: Optional[str]
    log: Optional[str]
    device: str

    class Config:
        use_enum_values = True
        schema_extra = {
            "example": {
                "train_data_path": "example",
                "validation_data_path": "example",
                "test_data_path": "example",
                "save": "example",
                "log": "example",
                "device": "example",
            }
        }


class Conf(BaseModel):
    wasabi: WasabiConf
    data: DataConf
    model: Model

    class Config:
        use_enum_values = True
        schema_extra = {"example": {"train": Train.Config.schema_extra.get("example")}}
