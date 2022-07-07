# # Prepare Dev Dataset
#
# Load paths from `Librispeech` folder. From paths load actual labels. Zip labels with corresponding audio paths. Audio paths can be played using the `playsound` library.

from glob import glob

# Just for one subdirectory
PATH = '..\\LibriSpeech\\'
dirs = ['dev-clean', 'train-clean-100']

def return_file_paths(dir: str, file_ext: str):
    return glob(f'{PATH + dir}\\**\\*.{file_ext}', recursive=True)

def write_to_file(name: str, data: 'list[str]'):
    with open(f"..\\manifest\\{name}.txt", 'w') as file:
        [file.write(d + "\n") for d in data]

def create_manifest(dirs: 'list[str]'):
    for d in dirs:
        audio_paths = return_file_paths(d, 'flac')
        label_paths = return_file_paths(d, 'txt')

        # Intermediary files
        write_to_file(f'{d}-audios', audio_paths)
        # write_to_file(d, label_paths)

        # Read string data from file path
        lines = []

        for l in label_paths:
            with open(l) as file:
                for a in file.readlines():
                    lines.append(a)

        write_to_file(f'{d}-labels', lines)

        # Write as CSV
        with open(f'..\\manifest\\{d}.csv', 'w') as file:
            for a, l in zip(audio_paths, lines):
                file.write(f'{a},{l}')

create_manifest(dirs)