"""Tests de ImageQueue: correctitud y acceso concurrente."""

from __future__ import annotations

import threading

from core.queue_manager import ImageQueue

QUEUE_SIZE: int = 8
MAXSIZE_ONE: int = 1
ITEM_A: str = 'a'
ITEM_B: str = 'b'
ITEM_C: str = 'c'
ITEMS: list[str] = [ITEM_A, ITEM_B, ITEM_C]
ITEM_COUNT: int = len(ITEMS)
THREAD_TIMEOUT_S: float = 2.0


def _enqueue_all(img_queue: ImageQueue, items: list[str]) -> None:
    """Encola todos los ítems de la lista."""
    for item in items:
        img_queue.put(item)


def _dequeue_all(
    img_queue: ImageQueue,
    count: int,
    received: list[str],
) -> None:
    """Extrae count elementos de la cola y los agrega a received."""
    for _ in range(count):
        received.append(img_queue.get())


def _do_blocked_put(
    img_queue: ImageQueue,
    event: threading.Event,
    item: str,
) -> None:
    """Señala el evento y pone item en la cola (bloquea si está llena)."""
    event.set()
    img_queue.put(item)


def _consume_with_task_done(img_queue: ImageQueue, count: int) -> None:
    """Extrae count ítems y llama task_done en cada uno."""
    for _ in range(count):
        img_queue.get()
        img_queue.task_done()


def test_put_get_single_thread() -> None:
    """put + get en el mismo hilo mantiene orden FIFO."""
    img_queue: ImageQueue = ImageQueue(maxsize=QUEUE_SIZE)
    for item in ITEMS:
        img_queue.put(item)
    results: list[str] = [img_queue.get() for _ in range(ITEM_COUNT)]
    assert results == ITEMS


def test_concurrent_put_get() -> None:
    """Productor y consumidor en hilos separados: todos los ítems llegan."""
    img_queue: ImageQueue = ImageQueue(maxsize=QUEUE_SIZE)
    received: list[str] = []
    prod = threading.Thread(target=_enqueue_all, args=(img_queue, ITEMS))
    cons = threading.Thread(
        target=_dequeue_all, args=(img_queue, ITEM_COUNT, received))
    prod.start()
    cons.start()
    prod.join(timeout=THREAD_TIMEOUT_S)
    cons.join(timeout=THREAD_TIMEOUT_S)
    assert sorted(received) == sorted(ITEMS)


def test_maxsize_blocks_producer() -> None:
    """maxsize=1: el productor bloquea hasta que el consumidor extrae."""
    img_queue: ImageQueue = ImageQueue(maxsize=MAXSIZE_ONE)
    img_queue.put(ITEM_A)
    event = threading.Event()
    thread = threading.Thread(
        target=_do_blocked_put, args=(img_queue, event, ITEM_B))
    thread.start()
    event.wait(timeout=THREAD_TIMEOUT_S)
    assert thread.is_alive()
    img_queue.get()
    thread.join(timeout=THREAD_TIMEOUT_S)
    assert not thread.is_alive()


def test_empty_after_get() -> None:
    """empty() retorna True luego de extraer todos los ítems."""
    img_queue: ImageQueue = ImageQueue(maxsize=QUEUE_SIZE)
    for item in ITEMS:
        img_queue.put(item)
    for _ in range(ITEM_COUNT):
        img_queue.get()
    assert img_queue.empty()


def test_task_done_join() -> None:
    """join() retorna luego de task_done() por cada ítem encolado."""
    img_queue: ImageQueue = ImageQueue(maxsize=QUEUE_SIZE)
    for item in ITEMS:
        img_queue.put(item)
    consumer = threading.Thread(
        target=_consume_with_task_done, args=(img_queue, ITEM_COUNT))
    consumer.start()
    img_queue.join()
    consumer.join(timeout=THREAD_TIMEOUT_S)
    assert not consumer.is_alive()
