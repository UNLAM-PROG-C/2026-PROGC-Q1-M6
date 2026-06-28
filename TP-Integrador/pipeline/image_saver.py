"""Consumidor dedicado para guardar imágenes procesadas en disco."""

from __future__ import annotations

import datetime
import json
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
        self._manifest: list[dict[str, str]] = []

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
        self._save_manifest()
        _LOGGER.info('Hilo de I/O finalizado')

    def _save_image(self, name: str, operation: str, image: "np.ndarray") -> None:
        """Construye el nombre final y guarda la imagen.

        El formato es: nombreOriginal_operacion_YYYYMMDD_HHMMSS.ext
        """
        base_name, ext = os.path.splitext(name)
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        new_name = f"{base_name}_{operation}_{now}{ext}"
        out_path = os.path.join(self._output_dir, new_name)
        
        import numpy as np
        try:
            is_success, im_buf_arr = cv2.imencode(ext, image)
            if is_success:
                im_buf_arr.tofile(out_path)
                success = True
            else:
                success = False
        except Exception:
            success = False

        if success:
            self._manifest.append({
                "original_filename": name,
                "transformed_filename": new_name,
                "operation": operation
            })
        else:
            _LOGGER.error('Fallo cv2.imencode/tofile al escribir: %s', out_path)

    def _save_manifest(self) -> None:
        """Guarda el archivo manifest.json con las imágenes procesadas."""
        if not self._manifest:
            return
        manifest_path = os.path.join(self._output_dir, 'manifest.json')
        try:
            # Si ya existe, podríamos querer añadir o sobrescribir. 
            # Por ahora sobrescribimos por cada ejecución del pipeline.
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump({"results": self._manifest}, f, indent=2)
            _LOGGER.info('Manifest JSON guardado en %s', manifest_path)
        except Exception as e:
            _LOGGER.error('Error guardando manifest.json: %s', e)
