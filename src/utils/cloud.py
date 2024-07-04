from functools import partial
import multiprocessing
import threading
import queue
import pandas as pd

registers = {}


def init_queue(queue_name):
    if queue_name not in registers:
        registers[queue_name] = {"queue": multiprocessing.Queue(), "callbacks": []}
        start_subscriber(queue_name)


def subscribe(queue_name):
    def decorator(func):
        init_queue(queue_name)

        def wrapper(instance_method):
            def callback(message):
                instance_method(message)

            registers[queue_name]["callbacks"].append(callback)
            return instance_method  # 确保返回实例方法本身

    return decorator


def publish(queue_name, message):
    if queue_name in registers:
        registers[queue_name]["queue"].put(message)
        print(f"publised to Queue {registers[queue_name]['queue']}")


def subscriber_thread(queue_name):
    if queue_name in registers:
        while True:
            try:
                message = registers[queue_name]["queue"].get(timeout=5)
                print(f"CALLBACK =====>")
                print(f"{message}")
                print(f"CALLBACK END =====>")
                print(f'{queue_name}: {registers[queue_name]["callbacks"]=}:')
                for callback in registers[queue_name]["callbacks"]:
                    print(f"callback: {callback}({message})")
                    callback(message)
            except queue.Empty:
                continue


def start_subscriber(queue_name):
    subscriber = threading.Thread(target=subscriber_thread, args=(queue_name,))
    subscriber.daemon = True
    subscriber.start()
