from dataclasses import dataclass

@dataclass
class Files:
    train_data: str
    dev_data: str

@dataclass
class LibriSpeechConfig:
    files: Files