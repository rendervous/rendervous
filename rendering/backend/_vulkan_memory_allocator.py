import typing
import weakref
import math
import sys
import os
from ._vulkan import *
import ctypes
import numpy as np
import torch
import threading
import traceback
# from cuda import cudart
# from cuda import cuda
from ._common import ViewTensor, Layout, lazy_constant, mutable_method, freezable_type

__SHOW_ALLOCATE_AND_FREES__ = False

__CUDA_DEVICE__ = torch.device('cuda:0')
__CPU_DEVICE__ = torch.device('cpu')
__TORCH_DEVICE__ = __CUDA_DEVICE__ if torch.cuda.is_available() else __CPU_DEVICE__


def str_memory_size(size):
    if size < 1024:
        return f"{size}B"
    if size < 1024*1024:
        return f"{size//1024}Kb"
    if size < 1024*1024*1024:
        return f"{size//1024//1024}Mb"
    return f"{size // 1024 ** 3}Gb"


class _Node:
    def __init__(self, value, prev, next):
        self.value = value
        self.prev = prev # if prev is None else weakref.ref(prev)
        self.next = next

    def remove(self):
        prev = self.prev
        next = self.next
        prev.next = next
        next.prev = prev
        # clear current node data
        self.prev = None
        self.next = None
        self.value = None

    def insert_after(self, value):
        node = _Node(value, self, self.next)
        next = self.next
        self.next = node
        if next is not None:
            next.prev = node
        return node

    def insert_before(self, value):
        node = _Node(value, self.prev, self)
        p = self.prev
        self.prev = node
        p.next = node
        return node

    def __repr__(self):
        return f"[{repr(self.value)}]"


class _Block:
    def __init__(self, offset, size, occupied, stack_trace):
        self.offset = offset
        self.size = size
        self.occupied = occupied
        self.stack_trace = stack_trace

    def __repr__(self):
        return f"[{self.offset}:{self.offset + self.size}]"
        # return f"{str_memory_size(self.size)} at {self.stack_trace}"


def _level_index_of(size):
    if size <= 1:
        return int(size)
    return int(math.log2(size) + 1)


def _align(x: int, alignment):
    return (x + alignment - 1) // alignment * alignment


class Allocator:

    def __init__(self, capacity):
        self.capacity = capacity
        self._start_node = _Node(None, None, None)
        self._end_node = self._start_node.insert_after(None)
        n_b = self._start_node.insert_after(_Block(0, capacity, False, "Start"))
        self._free_blocks = [set() for s in range(_level_index_of(capacity)+1)]  # for specific size gives a list of free blocks
        self._occupied_blocks = {}  # offset to node
        self._free_blocks[-1].add(n_b)
        self.locker = threading.Lock()
        self.used_memory = 0

    def not_core_path(self, line):
        return all(files not in line for files in ['_core', 'pydevd.py','pydev'])

    def malloc(self, size: int, alignment: int = 16) -> int:
        size = size + alignment - 1  # grant to find a suitable alignable offset
        start_index = _level_index_of(size) + 1
        with self.locker:
            for l in range(start_index, len(self._free_blocks)):
                if len(self._free_blocks[l]) > 0:  # Found a block
                    node = self._free_blocks[l].pop()
                    b : _Block = node.value
                    assert b.size >= size, "Wrong implementation. Block was find in a level doesnt correspond with the size"
                    new_free_block = _Block(b.offset+size, b.size-size, False, "Remain")
                    b.size = size  #shrink block to allocated size
                    b.occupied = True
                    # b.stack_trace = "->".join(l for l in traceback.format_stack() if self.not_core_path(l))
                    if new_free_block.size > 0: # left something free
                        new_free_node = node.insert_after(new_free_block)
                        self._free_blocks[_level_index_of(new_free_block.size)].add(new_free_node)
                    aligned_offset = _align(b.offset, alignment)
                    self._occupied_blocks[aligned_offset] = node
                    self.used_memory += size
                    return aligned_offset
        raise Exception("Out of Memory, can not fit the allocation size")

    def free(self, ptr: int):
        # with self.locker:
        assert ptr in self._occupied_blocks, "The memory address was not allocated in this allocator or was already free"
        node_to_free = self._occupied_blocks.pop(ptr)
        self.used_memory -= node_to_free.value.size

        node_to_free.value.occupied = False
        prev = node_to_free.prev
        if prev is not self._start_node and not prev.value.occupied:
            # merge with previous
            node_to_free.value.offset = prev.value.offset
            node_to_free.value.size += prev.value.size
            l = self._free_blocks[_level_index_of(prev.value.size)]
            assert prev in l, f"Node {prev} is not in list {l}"
            # if prev in l:
            l.remove(prev)
            prev.remove()
        next = node_to_free.next
        if next is not self._end_node and not next.value.occupied:
            node_to_free.value.size += next.value.size
            l = self._free_blocks[_level_index_of(next.value.size)]
            # if next in l:
            assert next in l, f"Node {next} is not in list {l}"
            l.remove(next)
            next.remove()
        self._free_blocks[_level_index_of(node_to_free.value.size)].add(node_to_free)

    def is_empty(self):
        # Has only one node as is not occupied
        return self._start_node.next.next is self._end_node and not self._start_node.next.value.occupied


class VulkanMemoryPage:
    def __init__(self, vk_device, capacity, memory_index: int, is_cpu: bool, is_gpu: bool):
        self.vk_device = vk_device
        self.capacity = capacity
        self.is_cpu = is_cpu
        self.is_gpu = is_gpu
        self.allocator = Allocator(capacity)
        self.vk_memory = None  # TODO: Create memory heap here and retrieve ptr
        self.memory_cpu_ptr = None
        self.memory_cuda_ptr = None
        self.memory_cuda_mem = None
        self.memory_vk_handle = None
        self.memory_device_ptr = None  # Memory managed by vulkan
        self.memory_as_tensor = { }
        prev = None
        if self.is_gpu:  # Prepare memory for exporting handle
            if os.name == 'nt':  # security for win32
                import win32security
                sec_desc = win32security.ConvertStringSecurityDescriptorToSecurityDescriptor(
                    "D:P(OA;;GARCSDWDWOCCDCLCSWLODTWPRPCRFA;;;WD)", 1)
                pdesc = ffi.from_buffer(sec_desc)
                prev = VkExportMemoryWin32HandleInfoKHR(
                    pNext=prev,
                    pAttributes=[(24, pdesc, 1)],
                    dwAccess=0x80000000 | 1,
                    name=None
                )
            prev = VkExportMemoryAllocateInfo(
                pNext=prev,
                handleTypes=
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
                if os.name == 'nt'
                else VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
            )
        prev = VkMemoryAllocateFlagsInfo(  # Allowing to get device address
            pNext=prev,
            flags=VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT  # Mandatory feature in rendezvous
        )
        alloc_info = VkMemoryAllocateInfo(
            pNext=prev,
            allocationSize=capacity,
            memoryTypeIndex=memory_index
        )
        self.vk_memory = vkAllocateMemory(self.vk_device, alloc_info, None)  # Allocate vulkan memory

        # Create a buffer with the full block to get mapped address (cpu) or gpu device address (gpu)
        external_info = VkExternalMemoryBufferCreateInfo(handleTypes=
                                            VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
                                            if os.name == 'nt'
                                            else VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
                                            )
        # temporal buffer to get the device address for the whole page
        full_block_info = VkBufferCreateInfo(pNext=external_info, flags=0, size=capacity,
                                                     usage=
                                                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                                     VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR
                                             )
        full_block = vkCreateBuffer(self.vk_device, full_block_info, None)
        vkBindBufferMemory(self.vk_device, full_block, self.vk_memory, 0)
        self.vkGetBufferDeviceAddressKHR = vkGetDeviceProcAddr(self.vk_device, "vkGetBufferDeviceAddressKHR")
        buffer_info = VkBufferDeviceAddressInfo(buffer=full_block)
        self.memory_device_ptr = self.vkGetBufferDeviceAddressKHR(self.vk_device, buffer_info)
        # vkDestroyBuffer(self.vk_device, full_block, None)
        self.full_block = full_block

        self.support_cuda = False

        # create a full tensor to map all memory in any case
        if is_cpu:
            # Permanent mapping for cpu based pages.
            self.memory_cpu_map = vkMapMemory_ptr(self.vk_device, self.vk_memory, 0, capacity, 0)
            self.memory_cpu_buffer = memoryview(ffi.buffer(self.memory_cpu_map, capacity))
            self.memory_cpu_ptr = int(ffi.cast('uint64_t', self.memory_cpu_map))
            self.memory_cuda_ptr = self.memory_cpu_ptr  # if cpu let cuda ptr to be the cpu ptr.

        if is_gpu:
            if os.name == 'nt':
                vkmemwin32info = VkMemoryGetWin32HandleInfoKHR(memory=self.vk_memory,
                                                               handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT)
                vkGetMemoryWin32 = vkGetDeviceProcAddr(self.vk_device, "vkGetMemoryWin32HandleKHR")
                self.memory_vk_handle = vkGetMemoryWin32(self.vk_device, vkmemwin32info)
                self.memory_vk_handle_win32 = int(ffi.cast('long', self.memory_vk_handle))
            else:
                vkmemfdinfo = VkMemoryGetFdInfoKHR(memory=self.vk_memory,
                                                   handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT)
                vkGetMemoryFdKHR = vkGetDeviceProcAddr(self.vk_device, "vkGetMemoryFdKHR")
                self.memory_vk_handle = vkGetMemoryFdKHR(self.vk_device, vkmemfdinfo)
            if torch.cuda.is_available():  # In a future to support other Non-nvidia general gpu implementations we need a switch here
                import cuda.cudart as cudart
                external_mem_hd = cudart.cudaExternalMemoryHandleDesc(0)
                external_mem_hd.size = capacity
                if os.name == 'nt':
                    external_mem_hd.handle.win32.handle = ffi.buffer(self.memory_vk_handle, 8)
                    external_mem_hd.type = cudart.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeOpaqueWin32
                else:
                    external_mem_hd.handle.fd = self.memory_vk_handle
                    external_mem_hd.type = cudart.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeOpaqueFd
                r, self.memory_cuda_mem = cudart.cudaImportExternalMemory(external_mem_hd)
                external_mem_buffer_desc = cudart.cudaExternalMemoryBufferDesc(0)
                external_mem_buffer_desc.offset = 0
                external_mem_buffer_desc.flags = 0
                external_mem_buffer_desc.size = capacity
                r, self.memory_cuda_ptr = cudart.cudaExternalMemoryGetMappedBuffer(self.memory_cuda_mem, external_mem_buffer_desc)
                self.support_cuda = True

        for t in [torch.int32, torch.float32, torch.uint8, torch.int64]:
            size = Layout.scalar_size(t)
            if is_cpu:
                self.memory_as_tensor[t] = ViewTensor.from_blob(self.memory_cpu_ptr, (capacity // size,), t,
                                                                __CPU_DEVICE__, owner=self).as_subclass(torch.Tensor)
            else:
                if self.support_cuda:
                    self.memory_as_tensor[t] = ViewTensor.from_blob(self.memory_cuda_ptr, (capacity // size,), t,
                                                                    __CUDA_DEVICE__, owner=self).as_subclass(torch.Tensor)
        if is_cpu or self.support_cuda:
            self.memory_as_tensor[float] = self.memory_as_tensor[torch.float32]
            self.memory_as_tensor[int] = self.memory_as_tensor[torch.int32]

    # def invalidate_cpu_map(self, offset, size):
    #     # pass
    #     assert self.is_cpu
    #     vkInvalidateMappedMemoryRanges(self.vk_device, 1, VkMappedMemoryRange(
    #         memory=self.vk_memory, offset=offset, size=size
    #     ))
    #
    # def flush_cpu_map(self, offset, size):
    #     assert self.is_cpu
    #     vkFlushMappedMemoryRanges(self.vk_device, 1, VkMappedMemoryRange(
    #         memory=self.vk_memory, offset=offset, size=size
    #     ))

    def as_tensor(self, dtype):
        return self.memory_as_tensor[dtype]

    def malloc(self, size: int, alignment: int = 16):
        offset = self.allocator.malloc(size, alignment)
        return offset

    def get_used_memory(self):
        return self.allocator.used_memory

    def free(self, offset: int):
        self.allocator.free(offset)

    @lazy_constant
    def device_ptr(self):
        # recompute only to debug
        # self.memory_device_ptr = self.vkGetBufferDeviceAddressKHR(self.vk_device, self.buffer_info)
        return self.memory_device_ptr

    @lazy_constant
    def cuda_ptr(self):
        return self.memory_cuda_ptr

    @lazy_constant
    def host_ptr(self):
        return self.memory_cpu_ptr

    def load_from_cpu_ptr(self, dst_offset: int, src_ptr: int, size: int):
        assert self.is_cpu
        ctypes.memmove(self.memory_cpu_ptr + dst_offset, src_ptr, size)

    def load_from_bytes(self, dst_offset: int, src: bytes):
        assert self.is_cpu
        size = len(bytes)
        self.memory_cpu_buffer[dst_offset:dst_offset + size] = src

    def destroy(self):
        self.memory_as_tensor = None
        if self.vk_memory is None:
            return
        vkDestroyBuffer(self.vk_device, self.full_block, None)
        self.full_block = None
        if self.is_cpu:
            vkUnmapMemory(self.vk_device, self.vk_memory)
        if self.is_gpu:
            if os.name == 'nt':
                if self.memory_vk_handle is not None and ffi.NULL != self.memory_vk_handle:
                    ctypes.windll.kernel32.CloseHandle(int(ffi.cast('long', self.memory_vk_handle)))
                    # win32api.CloseHandle(int(ffi.cast('long', vk_handle)))
            else:
                pass
                # import posix
                # posix.r
                # posix.close(vk_handle)
            # if torch.cuda.is_available():  # Change in future if other general gpu memory is used
            #     import cuda.cuda as cuda
            #     import cuda.cudart as cudart
            #     r = cudart.cudaFree(self.memory_cuda_ptr)
            #     r = cuda.cuDestroyExternalMemory(self.memory_cuda_mem)
        vkFreeMemory(self.vk_device, self.vk_memory, None)
        self.vk_memory = None
        self.memory_cpu_ptr = None
        self.memory_cuda_ptr = None
        self.memory_cuda_mem = None
        self.memory_vk_handle = None
        self.memory_device_ptr = None

    def __del__(self):
        self.destroy()


class VulkanMemory:
    def __init__(self, page_allocator: VulkanMemoryPage, offset, size):
        self.page_allocator = page_allocator
        self.offset = offset
        self.size = size
        self.debug_name = None
        # self.is_cpu = self.page_allocator.is_cpu
        # self.support_direct_tensor_map = self.is_cpu or self.page_allocator.support_cuda

    # def get_vulkan_page_buffer(self):
    #     return self.page_allocator.full_block
    #
    # def get_vulkan_page_buffer_info(self):
    #     return self.page_allocator.full_block_info
    #
    # def is_gpu(self):
    #     return self.page_allocator.is_gpu

    @lazy_constant
    def support_direct_tensor_map(self):
        return self.is_cpu or self.page_allocator.support_cuda

    @lazy_constant
    def is_cpu(self):
        return self.page_allocator.is_cpu

    @lazy_constant
    def device_ptr(self):
        return self.page_allocator.device_ptr + self.offset

    @lazy_constant
    def cuda_ptr(self):
        return self.page_allocator.cuda_ptr + self.offset

    def cuda_to_device_ptr(self, cuda_ptr):
        return cuda_ptr - self.page_allocator.cuda_ptr + self.page_allocator.device_ptr

    @lazy_constant
    def host_ptr(self):
        return self.page_allocator.host_ptr + self.offset

    @lazy_constant
    def vulkan_memory(self):
        return self.page_allocator.vk_memory

    @lazy_constant
    def vulkan_memory_offset(self):
        return self.offset

    def to_tensor(self, *shape: int, dtype: torch.dtype) -> ViewTensor:
        return ViewTensor.from_blob(self.cuda_ptr, shape, dtype, __TORCH_DEVICE__, owner=self)

    def as_tensor(self, dtype, offset, size):
        type_size = Layout.scalar_size(dtype)
        offset += self.offset
        # assert offset % type_size == 0, f'Bad alignment for {dtype}'
        # assert self.is_cpu or self.page_allocator.support_cuda, 'Invalid operation. Can not treat as direct tensor'
        element_start = offset // type_size
        element_count = size // type_size
        return self.page_allocator.as_tensor(dtype)[element_start:element_start + element_count]

    def as_bytes(self, offset, size):
        assert self.is_cpu
        offset += self.offset
        return self.page_allocator.memory_cpu_buffer[offset: offset + size]

    def load_from_cpu_ptr(self, dst_offset: int, src_ptr: int, size: int):
        assert self.is_cpu
        self.page_allocator.load_from_cpu_ptr(self.offset + dst_offset, src_ptr, size)

    def load_from_bytes(self, dst_offset: int, src: bytes):
        assert self.is_cpu
        size = len(bytes)
        self.page_allocator.load_from_bytes(self.offset + dst_offset, src)

    def _free(self):
        if self.page_allocator is None:
            return
        p = self.page_allocator
        self.page_allocator = None
        p.free(self.offset)
        self.offset = 0
        if __SHOW_ALLOCATE_AND_FREES__:
            # if self.debug_name is not None:
            print('freeing memory of '+(f'anonym {self.size}' if self.debug_name is None else self.debug_name))
        self.size = 0

    def __del__(self):
        self._free()


class VulkanAllocator:

    def __init__(self, vk_device, memory_index: int, memory_properties):
        self.pages : typing.List[VulkanMemoryPage] = []
        self.vk_device = vk_device
        self.is_cpu = bool(memory_properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        self.is_gpu = bool(memory_properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        self.is_gpu_dedicated = self.is_gpu and not self.is_cpu
        self.reserved_memory = 0
        self.memory_index = memory_index
        # self._allocated_objects = weakref.WeakSet()
        self._add_page(1, 1)

    def _add_page(self, size, alignment):
        # Here a new page is required
        # 64MB minimum for host memory or 1GB minimum on the GPU
        page_capacity = int(max(1 * 1024 ** 3, 2 ** int(math.log2(size + alignment) + 1)))
        print(f"[INFO] Creating page with {page_capacity // (1 << 20)}MB on {'CPU' if self.is_cpu else 'GPU'}")
        page = VulkanMemoryPage(self.vk_device, page_capacity, self.memory_index, self.is_cpu, self.is_gpu)
        self.reserved_memory += page_capacity
        self.pages.append(page)
        return page

    def allocate(self, size: int, alignment: int) -> VulkanMemory:
        for p in self.pages:
            try:
                offset = p.malloc(size, alignment)
                m = VulkanMemory(p, offset, size)
                # self._allocated_objects.add(m)
                return m
            except Exception as e:
                pass
        page = self._add_page(size, alignment)
        offset = page.malloc(size, alignment)
        m = VulkanMemory(page, offset, size)
        # self._allocated_objects.add(m)
        return m

    def clear_cache(self):
        """
        Free memory of empty pages
        """
        for i, p in enumerate(self.pages):
            if p.is_empty():
                p.destroy()
                self.pages[i] = None
        self.pages = [p for p in self.pages if p is not None]

    def destroy(self):
        if self.pages is None:
            return
        for p in self.pages:
            p.destroy()
        self.pages = None

    def get_used_memory(self):
        return sum(p.get_used_memory() for p in self.pages)


class VulkanMemoryManager:

    def __init__(self, vk_device, vk_physical_device):
        self.vk_device = vk_device
        self.mem_properties = vkGetPhysicalDeviceMemoryProperties(vk_physical_device)
        self.allocators = { } # map from memory index to VulkanAllocator
        self.debug_name = None
        # create buffer prototype for tensors
        prev = VkExternalMemoryBufferCreateInfo(
            handleTypes=VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT if os.name == 'nt' else VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
        )
        info = VkBufferCreateInfo(
            pNext=prev,
            size=1,
            usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
            ,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE
        )
        tensor_buffer_prototype = vkCreateBuffer(self.vk_device, info, None)
        mem_reqs = vkGetBufferMemoryRequirements(self.vk_device, tensor_buffer_prototype)
        vkDestroyBuffer(self.vk_device, tensor_buffer_prototype, None)
        self.tensor_gpu_index = self.__findMemoryType(mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        self.tensor_cpu_index = self.__findMemoryType(mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        if self.tensor_cpu_index not in self.allocators:
            self.allocators[self.tensor_cpu_index] = VulkanAllocator(self.vk_device, self.tensor_cpu_index, self.mem_properties.memoryTypes[self.tensor_cpu_index].propertyFlags)
        if self.tensor_gpu_index not in self.allocators:
            self.allocators[self.tensor_gpu_index] = VulkanAllocator(self.vk_device, self.tensor_gpu_index, self.mem_properties.memoryTypes[self.tensor_gpu_index].propertyFlags)

    def __findMemoryType(self, filter, mem_properties):
        selected_heap = -1
        covering_props = 0
        selected_type = -1
        for i, prop in enumerate(self.mem_properties.memoryTypes):
            if (filter & (1 << i)) and ((prop.propertyFlags & mem_properties) == mem_properties):
                if selected_heap == -1 or (prop.heapIndex == selected_heap and prop.propertyFlags > covering_props):
                    selected_type = i
                    selected_heap = prop.heapIndex
                    covering_props = prop.propertyFlags
        if selected_type != -1:
            return selected_type
        raise Exception("failed to find suitable memory type!")

    def clear_cache(self):
        for k, v in self.allocators.items():
            v.clear_cache()

    def set_debug_name(self, memory_name):
        self.debug_name = memory_name

    def peek_debug_name(self, mem):
        if self.debug_name is None:
            return None # 'Memory '+str(mem.size)
        d = self.debug_name
        self.debug_name = None
        return d

    def allocate_memory_for_tensor(self, size, memory_location):
        assert isinstance(size, int)
        if memory_location == 1:
            index = self.tensor_gpu_index
        else:
            index = self.tensor_cpu_index
        if index not in self.allocators:
            self.allocators[index] = VulkanAllocator(self.vk_device, index, self.mem_properties.memoryTypes[index].propertyFlags)
        mem = self.allocators[index].allocate(size, 256)
        mem.debug_name = self.peek_debug_name(mem)
        if __SHOW_ALLOCATE_AND_FREES__:
            #if mem.debug_name is not None:
            print(f"creating memory for tensor {mem.debug_name if mem.debug_name is not None else 'anonym '}{size}")
        return mem

    def allocate_memory_for_buffer(self, buffer, memory_location):
        mem_reqs = vkGetBufferMemoryRequirements(self.vk_device, buffer)
        index = self.__findMemoryType(mem_reqs.memoryTypeBits, memory_location)
        if index not in self.allocators:
            self.allocators[index] = VulkanAllocator(self.vk_device, index, self.mem_properties.memoryTypes[index].propertyFlags)
        mem = self.allocators[index].allocate(mem_reqs.size, mem_reqs.alignment)
        mem.debug_name = self.peek_debug_name(mem)
        if __SHOW_ALLOCATE_AND_FREES__:
            if mem.debug_name is not None:
                print(f'creating memory for {mem.debug_name}')
        return mem

    def allocate_memory_for_image(self, image, memory_location):
        mem_reqs = vkGetImageMemoryRequirements(self.vk_device, image)
        index = self.__findMemoryType(mem_reqs.memoryTypeBits, memory_location)
        if index not in self.allocators:
            print(f'[INFO] Creating allocator for {index}')
            self.allocators[index] = VulkanAllocator(self.vk_device, index, self.mem_properties.memoryTypes[index].propertyFlags)
        mem = self.allocators[index].allocate(mem_reqs.size, mem_reqs.alignment)
        mem.debug_name = self.peek_debug_name(mem)
        if __SHOW_ALLOCATE_AND_FREES__:
            if mem.debug_name is not None:
                print(f'creating memory for {mem.debug_name}')
        return mem

    def destroy(self):
        if self.allocators is None:
            return
        for k, a in self.allocators.items():
            a.destroy()
        self.allocators = None

    def get_used_cpu_memory(self):
        return self.allocators[self.tensor_cpu_index].get_used_memory()

    def get_used_gpu_memory(self):
        return self.allocators[self.tensor_gpu_index].get_used_memory()

    def torch_ptr_to_device_ptr(self, t: torch.Tensor) -> int:
        memory_index = self.tensor_cpu_index if t.device.type == 'cpu' else self.tensor_gpu_index
        t_ptr = t.data_ptr()
        p = self.allocators[memory_index].pages[0]
        return t_ptr - p.memory_cuda_ptr + p.memory_device_ptr