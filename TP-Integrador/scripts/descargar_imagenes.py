"""Descarga imágenes de prueba desde picsum.photos."""

import argparse
import os

import requests

DEFAULT_COUNT: int = 100
DEFAULT_DEST: str = 'public/images/in'
IMAGE_WIDTH: int = 800
IMAGE_HEIGHT: int = 600
BASE_URL: str = 'https://picsum.photos'


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos.

    Returns:
        Namespace con los atributos ``count``, ``dest``, ``width`` y ``height``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-n', '--count', type=int, default=DEFAULT_COUNT,
        help='cantidad de imágenes a descargar (default: %(default)s)')
    parser.add_argument(
        '-d', '--dest', type=str, default=DEFAULT_DEST,
        help='ruta de destino (default: %(default)s)')
    parser.add_argument(
        '-W', '--width', type=int, default=IMAGE_WIDTH,
        help='ancho de las imágenes en píxeles (default: %(default)s)')
    parser.add_argument(
        '-H', '--height', type=int, default=IMAGE_HEIGHT,
        help='alto de las imágenes en píxeles (default: %(default)s)')
    return parser.parse_args()


def download_image(index: int, dest: str, width: int, height: int) -> None:
    """Descarga una imagen y la guarda en ``dest``.

    Args:
        index: Índice de la imagen, usado como semilla aleatoria.
        dest: Carpeta de destino donde guardar el archivo.
        width: Ancho de la imagen en píxeles.
        height: Alto de la imagen en píxeles.
    """
    url = f'{BASE_URL}/{width}/{height}?random={index}'
    img = requests.get(url, timeout=30).content
    path = os.path.join(dest, f'imagen_{index + 1}.jpg')
    with open(path, 'wb') as file:
        file.write(img)


def main() -> None:
    """Descarga la cantidad de imágenes indicada en la ruta indicada."""
    args = parse_args()
    os.makedirs(args.dest, exist_ok=True)
    for index in range(args.count):
        download_image(index, args.dest, args.width, args.height)
    print(f'Listo. {args.count} imágenes en {args.dest}')


if __name__ == '__main__':
    main()
