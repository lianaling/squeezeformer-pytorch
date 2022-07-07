import pandas as pd

PATH = '../manifest/'
dirs = map(lambda f: PATH + f, ['dev-clean.csv', 'train-clean-100.csv'])

def create_manifest(dirs: 'list[str] or str') -> None:
    for d in dirs:
        df = pd.read_csv(d)
        df.columns = ['audio', 'label']
        # Take only label data, remove file
        df['proc_label'] = [*map(lambda s: ' '.join(["_" + w for w in str.split(str(s).split(" ", maxsplit=1)[1].upper(), ' ')]), df['label'].tolist())]
        df = df.drop(labels='label', axis=1)
        df.to_csv(f'{d[:-4]}-proc.csv', sep='\t', index=False)

create_manifest(dirs)