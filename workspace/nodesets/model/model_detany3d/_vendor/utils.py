import posix_ipc
import mmap
import numpy as np
import atexit
import time
import pickle
import struct

class SharedMemoryManager:
    def __init__(self, name, size=0, create=False, is_server=False):
        self.name = name
        self.is_server = is_server
        # 共享内存初始化
        try:
            if create:
                self.shm = posix_ipc.SharedMemory(name, posix_ipc.O_CREAT, size=size)
            else:
                # 客户端重试机制
                for _ in range(10):
                    try:
                        self.shm = posix_ipc.SharedMemory(name)
                        break
                    except posix_ipc.ExistentialError:
                        time.sleep(0.5)
                else:
                    raise TimeoutError(f"共享内存 {name} 连接超时")
                
            self.mem_map = mmap.mmap(self.shm.fd, size if size else self.shm.size)
            self.shm.close_fd()
        except Exception as e:
            self._safe_cleanup()
            raise e

        # 信号量初始化（服务端创建，客户端连接）
        self.sem_data_ready = self._init_semaphore(f"/{name}_ready", create)
        self.sem_data_done = self._init_semaphore(f"/{name}_done",create)
        print(f"信号量状态: ready={self.sem_data_ready.value}, done={self.sem_data_done.value}")

        atexit.register(self._safe_cleanup)

    @staticmethod
    def force_clean_semaphores(name):
        """强制清理残留信号量"""
        for sem_name in [f"/{name}_ready", f"/{name}_done"]:
            try:
                sem = posix_ipc.Semaphore(sem_name)
                sem.unlink()  # 彻底删除信号量
            except (posix_ipc.ExistentialError, PermissionError):
                pass

    def _init_semaphore(self, name, create):
        """初始化信号量"""
        for _ in range(10):
            try:
                return posix_ipc.Semaphore(
                    name, 
                    posix_ipc.O_CREAT if create else 0,
                    initial_value=0
                )
            except posix_ipc.ExistentialError:
                if not create: time.sleep(0.5)
                else: raise
        raise TimeoutError(f"信号量 {name} 初始化失败")

    def write_data(self, data):
        """写入数据并通知对方"""
        data_bytes = pickle.dumps(data)
        
        # 2. 写入数据长度（4 字节）
        data_len = len(data_bytes)
        len_bytes = struct.pack("I", data_len)
        self.mem_map[:4] = len_bytes

        # 3. 写入实际数据
        self.mem_map[4 : 4 + data_len] = data_bytes
        # 4. 刷新并通知
        self.mem_map.flush()
        self.sem_data_ready.release()
        print(f"信号量初始值: ready={self.sem_data_ready.value}, done={self.sem_data_done.value}")

    def read_data(self, timeout=30):
        """读取数据并等待通知"""
        self.sem_data_ready.acquire()
        
        # 1. 读取数据长度
        len_bytes = bytes(self.mem_map[:4])
        data_len = struct.unpack("I", len_bytes)[0]
        
        # 2. 读取实际数据
        data_bytes = bytes(self.mem_map[4 : 4 + data_len])
        
        # 3. 反序列化为 Python 对象
        data = pickle.loads(data_bytes)
        return data

    def notify_done(self):
        """通知处理完成"""
        self.sem_data_done.release()

    def wait_done(self, timeout=30):
        """等待处理完成"""
        self.sem_data_done.acquire()

    def _safe_cleanup(self):
        """安全清理资源"""
        try:
            if hasattr(self, 'mem_map'):
                self.mem_map.close()
            if hasattr(self, 'shm'):
                if self.is_server:  # 只有服务端负责销毁
                    posix_ipc.unlink_shared_memory(self.name)
            if hasattr(self, 'sem_data_ready'):
                if self.is_server:
                    self.sem_data_ready.unlink()
            if hasattr(self, 'sem_data_done'):
                if self.is_server:
                    self.sem_data_done.unlink()
        except:
            pass

    @staticmethod
    def cleanup_all():
        """清理所有可能的残留资源"""
        for id in range(8):
            for name in [f"image_data_{id}", f"result_data_{id}"]:
                try:
                    posix_ipc.unlink_shared_memory(name)
                except: pass
                try:
                    posix_ipc.unlink_semaphore(f"/{name}_ready")
                except: pass
                try:
                    posix_ipc.unlink_semaphore(f"/{name}_done")
                except: pass