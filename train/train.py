import torchaudio
import hydra
from config import LibriSpeechConfig
from squeezeformer.model import Squeezeformer

@hydra.main(version_base=None, config_path='../conf', config_name='config')
def main(cfg: LibriSpeechConfig):
    print(cfg)
    train_dataset = torchaudio.datasets.LIBRISPEECH(cfg.files.train_data)
    dev_dataset = torchaudio.datasets.LIBRISPEECH(cfg.files.dev_data)
    model = Squeezeformer(2)

if __name__ == "__main__":
    main()