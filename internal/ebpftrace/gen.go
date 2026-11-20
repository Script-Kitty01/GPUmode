package ebpftrace

//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -target amd64 -tags linux -type cuda_tgid_stats cudaTrace ./bpf/cuda_trace.bpf.c -- -I./bpf -O2 -Wall
