package ebpftrace

//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -target amd64 -tags linux -type cuda_tgid_stats cudaTrace ./bpf/cuda_trace.bpf.c -- -I./bpf -I/usr/include/x86_64-linux-gnu -O2 -Wall
