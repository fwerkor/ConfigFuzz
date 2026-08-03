import os
import argparse
import unittest

import torch.distributed as dist


class TestTCPStore1p(unittest.TestCase):
    def test_api_completeness(self):
        self.test_set_get()
        self.test_add()
        self.test_multi_store()

    def test_set_get(self):
        server_store = dist.TCPStore("127.0.0.1", 22345, None, True)
        client_store = dist.TCPStore("127.0.0.1", 22345, None, False)

        server_store.set("key1", "value1")
        assert server_store.get("key1") == b"value1"
        assert client_store.get("key1") == b"value1"

        client_store.set("key1", "value2")
        assert server_store.get("key1") == b"value2"
        assert client_store.get("key1") == b"value2"

        client_store.set("key2", "value2")
        assert server_store.get("key2") == b"value2"
        assert client_store.get("key2") == b"value2"

        assert not server_store.delete_key("__xx__")
        assert not client_store.delete_key("__xx__")

        assert server_store.delete_key("key1")
        assert client_store.delete_key("key2")

    def test_add(self):
        server_store = dist.TCPStore("127.0.0.1", 22345, None, True)
        client_store = dist.TCPStore("127.0.0.1", 22345, None, False)
        assert server_store.add('key', 2) == 2
        assert client_store.add('key', 3) == 5
        assert server_store.delete_key('key')

    def test_multi_store(self):
        server1 = dist.TCPStore("127.0.0.1", 22345, None, True)
        client1 = dist.TCPStore("127.0.0.1", 22345, None, False)
        server2 = dist.TCPStore("127.0.0.1", 22346, None, True)
        client2 = dist.TCPStore("127.0.0.1", 22346, None, False)

        server1.set("key1", "value1")
        server2.set("key1", "value2")
        assert server1.get("key1") == b"value1"
        assert client1.get("key1") == b"value1"
        assert server2.get("key1") == b"value2"
        assert client2.get("key1") == b"value2"

        client1.set("key2", "value1")
        client2.set("key2", "value2")
        assert server1.get("key2") == b"value1"
        assert client1.get("key2") == b"value1"
        assert server2.get("key2") == b"value2"
        assert client2.get("key2") == b"value2"

        assert server1.add('key3', 2) == 2
        assert server2.add('key3', 3) == 3
        assert client1.add('key3', 3) == 5
        assert client2.add('key3', 3) == 6

        assert server1.delete_key("key1")
        assert server2.delete_key("key1")
        assert server1.delete_key("key2")
        assert server2.delete_key("key2")
        assert client1.delete_key("key3")
        assert client2.delete_key("key3")

class TestTCPStore2p(unittest.TestCase):
    def test_api_completeness(self):
        self.test_multiprocessing(self.do_test_set_get)
        self.test_multiprocessing(self.do_test_add)

    def test_multiprocessing(self, fn):
        world_size = 2

        processes = []
        import multiprocessing
        for rank in range(world_size):
            p = multiprocessing.Process(target=fn, args=(rank, world_size))
            processes.append(p)
            p.start()

        exitcode = 0
        for p in processes:
            p.join()
            if p.exitcode != 0:
                exitcode = p.exitcode

        if exitcode != 0:
            exit(exitcode)

    def do_test_set_get(self, rank, world_size):
        if rank == 0:
            store = dist.TCPStore("127.0.0.1", 22345, None, True)
            assert not store.delete_key("__xx__")
            store.set("key1", "value1")
            assert store.get("key1") == b"value1"
            assert store.get("key2") == b"value2"
            store.set("key3", "value3")
            assert store.delete_key("key1")
            assert store.get("finish") == b""
            assert store.delete_key("finish")
        else:
            store = dist.TCPStore("127.0.0.1", 22345, None, False)
            assert not store.delete_key("__xx__")
            assert store.get("key1") == b"value1"
            store.set("key2", "value2")
            assert store.get("key2") == b"value2"
            assert store.get("key3") == b"value3"
            assert store.delete_key("key2")
            store.set("finish", "")

    def do_test_add(self, rank, world_size):
        if rank == 0:
            store = dist.TCPStore("127.0.0.1", 22345, None, True)
            assert store.add("key", 2) == 2
            assert store.get("finish") == b""
            assert store.delete_key("key")
            assert store.delete_key("finish")
        else:
            store = dist.TCPStore("127.0.0.1", 22345, None, False)
            assert store.get("key") == b"2"
            assert store.add("key", 3) == 5
            store.set("finish", "")


if __name__ == "__main__":
    print(f"PYTHONPATH is: {os.getenv('PYTHONPATH')}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'performance', 'precision', 'outlier'],
                        help="test mode")
    args, _ = parser.parse_known_args()

    if args.test_mode == 'completeness':
        TestTCPStore1p().test_api_completeness()
        TestTCPStore2p().test_api_completeness()
