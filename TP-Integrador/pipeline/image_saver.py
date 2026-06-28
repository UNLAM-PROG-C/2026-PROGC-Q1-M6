"""Consumidor dedicado para guardar imágenes procesadas en disco."""

from __future__ import annotations

import datetime
import logging
import os
import threading

import cv2

from core.queue_manager import ImageQueue

_LOGGER = logging.getLogger(__name__)


class ImageSaver(threading.Thread):
    """Hilo dedicado para escribir imágenes a disco sin bloquear."""

    def __init__(self, io_queue: ImageQueue, output_dir: str) -> None:
        """Inicializa el hilo guardián de imágenes.

        Args:
            io_queue: Cola con tuplas (nombre, operacion, imagen).
            output_dir: Carpeta destino donde guardar las imágenes.
        """
        super().__init__(name='image-saver')
        self._io_queue = io_queue
        self._output_dir = output_dir

    def run(self) -> None:
        """Consume imágenes de la cola y las guarda en disco."""
        _LOGGER.info('Hilo de I/O iniciado (Guardado de imágenes)')
        while True:
            item = self._io_queue.get()
            try:
                if item is None:
                    break
                name, operation, image = item
                self._save_image(name, operation, image)
            except Exception as e:
                _LOGGER.error('Error guardando imagen %s: %s', name, e)
            finally:
                self._io_queue.task_done()
        _LOGGER.info('Hilo de I/O finalizado')

    def _save_image(self, name: str, operation: str, image: "np.ndarray") -> None:
        """Construye el nombre final y guarda la imagen.

        El formato es: nombreOriginal_operacion_YYYYMMDD_HHMMSS.ext
        """
        base_name, ext = os.path.splitext(name)
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        new_name = f"{base_name}_{operation}_{now}{ext}"
        out_path = os.path.join(self._output_dir, new_name)
        
        success = cv2.imwrite(out_path, image)
        if not success:
            _LOGGER.error('Fallo cv2.imwrite al escribir: %s', out_path)
