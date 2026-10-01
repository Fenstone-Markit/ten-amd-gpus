"""patch_comm.py FILE: add the Fenstone RDNA all-reduce hook to vLLM's cuda_communicator.py (this image's version).
Counted edits; refuses unless every anchor matches exactly once. Opt-in at runtime: FENSTONE_RDNA_AR=1."""
import sys
p = sys.argv[1]; s = open(p).read(); n = 0
if "Fenstone: graph-only RDNA" in s:
    sys.exit("STOP: this file is already patched; start from the image's original cuda_communicator.py")
def rep(a, b):
    global s, n
    if s.count(a) != 1: sys.exit(f"STOP: anchor found {s.count(a)} times, not once: {a[:60]!r}")
    s = s.replace(a, b); n += 1
rep("import torch\nfrom torch.distributed import ProcessGroup\n", "import os\nimport torch\nfrom torch.distributed import ProcessGroup\n")
rep("        self.use_flashinfer_allreduce = use_flashinfer_allreduce\n",
    "        self.use_flashinfer_allreduce = use_flashinfer_allreduce\n"
    "        # Fenstone: graph-only RDNA HIP all-reduce (vLLM PR #57767 plus a TP=8 butterfly). Opt-in: FENSTONE_RDNA_AR=1.\n"
    "        self.rdna_ar_comm = None\n"
    "        if (\n"
    "            os.environ.get(\"FENSTONE_RDNA_AR\", \"0\") == \"1\"\n"
    "            and unique_name.split(\":\")[0] == \"tp\"\n"
    "            and self.world_size > 1\n"
    "            and self.cpu_group in torch.distributed.distributed_c10d._world.pg_map\n"
    "        ):\n"
    "            from vllm.distributed.device_communicators.rdna_custom_all_reduce import (\n"
    "                RdnaCustomAllreduce,\n"
    "            )\n"
    "\n"
    "            self.rdna_ar_comm = RdnaCustomAllreduce(self.cpu_group, self.device, enabled=True)\n")
rep("    def all_reduce(self, input_):\n",
    "    def all_reduce(self, input_):\n"
    "        rdna_ar_comm = getattr(self, \"rdna_ar_comm\", None)\n"
    "        if rdna_ar_comm is not None:\n"
    "            out = rdna_ar_comm.custom_all_reduce(input_)\n"
    "            if out is not None:\n"
    "                return out\n")
rep("    def destroy(self):\n",
    "    def destroy(self):\n"
    "        if getattr(self, \"rdna_ar_comm\", None) is not None:\n"
    "            self.rdna_ar_comm.close()\n"
    "            self.rdna_ar_comm = None\n")
open(p, "w").write(s); print(f"cuda_communicator.py patched: {n} edits")
