import hashlib
import json
from pathlib import Path
from datetime import datetime

def sha256_file(filepath: Path) -> str:
    """Считает SHA-256 файла, читая его блоками по 64 КБ."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def build_manifest(data_dir: str, output_file: str = 'manifest.json'):
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f'Папка не найдена: {base.resolve()}')

    manifest = {
        'generated_at': datetime.now().isoformat(),
        'algorithm': 'SHA-256',
        'files': []
    }

    # Ищем все файлы (включая .gz) рекурсивно
    for path in sorted(base.rglob('*')):
        if path.is_file():
            print(f'Хеширую: {path.name} ...')
            manifest['files'].append({
                'path': str(path.relative_to(base)),   # путь относительно data_dir
                'size': path.stat().st_size,
                'sha256': sha256_file(path)
            })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f'\nГотово. Манифест сохранён в {output_file}')
    print(f'Всего файлов: {len(manifest["files"])}')

if __name__ == '__main__':
    # Укажите папку, где лежит скачанный EMNIST
    build_manifest('./data/EMNIST', 'hash_manifest.json')
