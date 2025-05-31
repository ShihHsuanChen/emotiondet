import os

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class AppSettings(BaseSettings):
    model_config= SettingsConfigDict(
        env_file=('.env'),
        env_file_encoding='utf8',
    )

    model_dir: str
    """ path to the DebertaV2ForSequenceClassification model """
    device: str = 'cpu'
    """ device string for running the model in (e.g. 'cpu', 'cuda', 'cuda:0', ...)"""
    max_length: int = Field(72, ge=10)
    """ maximum token length """
    batch_size: int = Field(8, ge=1)
    """ maximim batch size for batch inference """

    @field_validator('model_dir', mode='after')
    def validate_model_dir(cls, v: str) -> str:    
        v = os.path.abspath(os.path.expanduser(v))
        if not os.path.exists(v):
            raise ValueError(f'{v} does not exist')
        if not os.path.isdir(v):
            raise ValueError('{v} is not a directory')
        for fname in ['config.json', 'model.safetensors']:
            if not os.path.isfile(os.path.join(v, fname)):
                raise ValueError(f'Cannot find "{fname}" in {v}')
        return v


    @field_validator('device', mode='after')
    def validate_device(cls, v: str) -> str:
        if v in ['cpu', 'cuda']:
            pass
        elif v.startswith('cuda:'):
            if not v[5:].isdigit():
                raise ValueError(f'Invalid device value: {v}')
        else:
            raise ValueError(f'Invalid device value: {v}')
        return v

