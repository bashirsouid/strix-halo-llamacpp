lines = [
    "# Dockerfile.therock\n",
    "# Builds llama.cpp with HIP/TheRock targeting gfx1151.\n",
    "# Base: kyuz0/vllm-therock-gfx1151:20251130-175119 (Fedora 43)\n",
    "\n",
    "FROM docker.io/kyuz0/vllm-therock-gfx1151:20251130-175119\n",
    "\n",
    "ENV PATH=\"/opt/venv/bin:${PATH}\"\n",
    "ENV HIP_PLATFORM=amd\n",
    "\n",
    "# Register ROCm lib dirs with the dynamic linker\n",
    "RUN ROCM_SDK=$(find /opt/venv -maxdepth 5 -name '_rocm_sdk_core' -type d 2>/dev/null | head -1) \\\n",
    "    && echo '/opt/rocm/lib' > /etc/ld.so.conf.d/rocm.conf \\\n",
    "    && echo \"${ROCM_SDK}/lib\" >> /etc/ld.so.conf.d/rocm.conf \\\n",
    "    && ldconfig\n",
    "\n",
    "RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp\n",
    "\n",
    "# Patch: disable UMA/APU detection in ggml-cuda.\n",
    "# ggml uses hipDeviceAttributeIntegrated to detect APU/UMA and then allocates\n",
    "# KV cache via hsa_amd_memory_pool_allocate (fine-grained memory). On gfx1151\n",
    "# with libhsa-runtime64.so.1.18.0 this segfaults at offset 0xac2de.\n",
    "# Replacing with hipDeviceAttributeEccEnabled (always 0 on consumer GPUs)\n",
    "# makes ggml treat the device as non-UMA and use plain hipMalloc instead.\n",
    "RUN grep -rl 'hipDeviceAttributeIntegrated' /opt/llama.cpp/ggml/src/ \\\n",
    "    | xargs sed -i 's/hipDeviceAttributeIntegrated/hipDeviceAttributeEccEnabled/g' \\\n",
    "    && echo 'Patched files:' \\\n",
    "    && grep -rl 'hipDeviceAttributeEccEnabled' /opt/llama.cpp/ggml/src/\n",
    "\n",
    "RUN cd /opt/llama.cpp && \\\n",
    "    cmake -B build -G Ninja \\\n",
    "      -DCMAKE_BUILD_TYPE=Release \\\n",
    "      -DGGML_HIP=ON \\\n",
    "      -DGGML_HIPBLAS=OFF \\\n",
    "      -DAMDGPU_TARGETS=\"gfx1151\" \\\n",
    "      -DCMAKE_HIP_ARCHITECTURES=\"gfx1151\" \\\n",
    "      -DCMAKE_C_COMPILER=gcc \\\n",
    "      -DCMAKE_CXX_COMPILER=g++ \\\n",
    "      -DCMAKE_PREFIX_PATH=/opt/rocm \\\n",
    "    && cmake --build build -j$(nproc) \\\n",
    "    && cmake --install build --prefix /usr \\\n",
    "    && ldconfig \\\n",
    "    && rm -rf /opt/llama.cpp/build\n",
    "\n",
    "EXPOSE 8000\n",
]

path = "Dockerfile.therock"
with open(path, "w") as f:
    f.writelines(lines)

content = open(path).read()
assert "\\\\" not in content
print(content)