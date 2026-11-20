// SPDX-License-Identifier: Dual BSD/GPL
#include "vmlinux_min.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

/*
 * Ignore inter-launch idle gaps > 500ms so pauses between HTTP requests or
 * benchmark iterations do not skew intra-burst GPU launch dispatch latency.
 */
#define MAX_INTRA_BURST_LAUNCH_GAP_NS (500ULL * 1000000ULL)

/* cudaMemcpyKind constants from CUDA Runtime API (fallback when libcuda is absent) */
#define CUDA_MEMCPY_HOST_TO_DEVICE 1ULL
#define CUDA_MEMCPY_DEVICE_TO_HOST 2ULL

struct cuda_tgid_stats {
	__u64 kernel_launches;
	__u64 last_launch_ns;
	__u64 launch_gap_sum_ns;
	__u64 launch_gap_count;
	__u64 max_launch_gap_ns;
	__u64 memcpy_h2d_calls;
	__u64 memcpy_h2d_bytes;
	__u64 memcpy_d2h_calls;
	__u64 memcpy_d2h_bytes;
	__u64 mem_alloc_calls;
	__u64 mem_alloc_bytes;
	__u64 sync_calls;
	__u64 sync_stall_sum_ns;
	__u64 max_sync_stall_ns;
	__u64 sched_switches;
	__u64 off_cpu_sum_ns;
	char comm[16];
};

struct {
	__uint(type, BPF_MAP_TYPE_LRU_HASH);
	__uint(max_entries, 1024);
	__type(key, __u32);
	__type(value, struct cuda_tgid_stats);
} tgid_stats SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 256);
	__type(key, __u32);
	__type(value, __u8);
} target_tgids SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_LRU_HASH);
	__uint(max_entries, 4096);
	__type(key, __u64);
	__type(value, __u64);
} tid_sync_enter SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_LRU_HASH);
	__uint(max_entries, 4096);
	__type(key, __u32);
	__type(value, __u64);
} tid_offcpu_enter SEC(".maps");

static __always_inline struct cuda_tgid_stats *get_or_init_current_stats(__u32 tgid)
{
	struct cuda_tgid_stats *st = bpf_map_lookup_elem(&tgid_stats, &tgid);
	if (st) {
		if (st->comm[0] == '\0')
			bpf_get_current_comm(&st->comm, sizeof(st->comm));
		return st;
	}

	struct cuda_tgid_stats zero = {};
	bpf_get_current_comm(&zero.comm, sizeof(zero.comm));
	bpf_map_update_elem(&tgid_stats, &tgid, &zero, BPF_NOEXIST);
	return bpf_map_lookup_elem(&tgid_stats, &tgid);
}

static __always_inline struct cuda_tgid_stats *get_or_init_task_stats(__u32 tgid, struct task_struct *task)
{
	struct cuda_tgid_stats *st = bpf_map_lookup_elem(&tgid_stats, &tgid);
	if (st)
		return st;

	struct cuda_tgid_stats zero = {};
	BPF_CORE_READ_STR_INTO(&zero.comm, task, comm);
	bpf_map_update_elem(&tgid_stats, &tgid, &zero, BPF_NOEXIST);
	return bpf_map_lookup_elem(&tgid_stats, &tgid);
}

SEC("uprobe/cuLaunchKernel")
int handle_cuda_launch(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 tgid = (__u32)(pid_tgid >> 32);
	if (tgid == 0)
		return 0;

	struct cuda_tgid_stats *st = get_or_init_current_stats(tgid);
	if (!st)
		return 0;

	__u64 now = bpf_ktime_get_ns();
	__u64 prev = st->last_launch_ns;
	st->last_launch_ns = now;
	__sync_fetch_and_add(&st->kernel_launches, 1);

	if (prev > 0 && now > prev) {
		__u64 delta = now - prev;
		if (delta <= MAX_INTRA_BURST_LAUNCH_GAP_NS) {
			__sync_fetch_and_add(&st->launch_gap_sum_ns, delta);
			__sync_fetch_and_add(&st->launch_gap_count, 1);
			if (delta > st->max_launch_gap_ns)
				st->max_launch_gap_ns = delta;
		}
	}
	return 0;
}

SEC("uprobe/cuMemcpyHtoD_v2")
int handle_memcpy_h2d(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 tgid = (__u32)(pid_tgid >> 32);
	if (tgid == 0)
		return 0;

	struct cuda_tgid_stats *st = get_or_init_current_stats(tgid);
	if (!st)
		return 0;

	__u64 bytes = (__u64)PT_REGS_PARM3(ctx);
	__sync_fetch_and_add(&st->memcpy_h2d_calls, 1);
	__sync_fetch_and_add(&st->memcpy_h2d_bytes, bytes);
	return 0;
}

SEC("uprobe/cuMemcpyDtoH_v2")
int handle_memcpy_d2h(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 tgid = (__u32)(pid_tgid >> 32);
	if (tgid == 0)
		return 0;

	struct cuda_tgid_stats *st = get_or_init_current_stats(tgid);
	if (!st)
		return 0;

	__u64 bytes = (__u64)PT_REGS_PARM3(ctx);
	__sync_fetch_and_add(&st->memcpy_d2h_calls, 1);
	__sync_fetch_and_add(&st->memcpy_d2h_bytes, bytes);
	return 0;
}

SEC("uprobe/cudaMemcpy")
int handle_cudart_memcpy(struct pt_regs *ctx)
{
	__u64 kind = (__u64)PT_REGS_PARM4(ctx);
	if (kind != CUDA_MEMCPY_HOST_TO_DEVICE && kind != CUDA_MEMCPY_DEVICE_TO_HOST)
		return 0;

	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 tgid = (__u32)(pid_tgid >> 32);
	if (tgid == 0)
		return 0;

	struct cuda_tgid_stats *st = get_or_init_current_stats(tgid);
	if (!st)
		return 0;

	__u64 bytes = (__u64)PT_REGS_PARM3(ctx);
	if (kind == CUDA_MEMCPY_HOST_TO_DEVICE) {
		__sync_fetch_and_add(&st->memcpy_h2d_calls, 1);
		__sync_fetch_and_add(&st->memcpy_h2d_bytes, bytes);
	} else {
		__sync_fetch_and_add(&st->memcpy_d2h_calls, 1);
		__sync_fetch_and_add(&st->memcpy_d2h_bytes, bytes);
	}
	return 0;
}

SEC("uprobe/cuMemAlloc_v2")
int handle_mem_alloc(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 tgid = (__u32)(pid_tgid >> 32);
	if (tgid == 0)
		return 0;

	struct cuda_tgid_stats *st = get_or_init_current_stats(tgid);
	if (!st)
		return 0;

	__u64 bytes = (__u64)PT_REGS_PARM2(ctx);
	__sync_fetch_and_add(&st->mem_alloc_calls, 1);
	__sync_fetch_and_add(&st->mem_alloc_bytes, bytes);
	return 0;
}

SEC("uprobe/cuStreamSynchronize")
int handle_sync_enter(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	if ((pid_tgid >> 32) == 0)
		return 0;

	__u64 now = bpf_ktime_get_ns();
	bpf_map_update_elem(&tid_sync_enter, &pid_tgid, &now, BPF_ANY);
	return 0;
}

SEC("uretprobe/cuStreamSynchronize")
int handle_sync_exit(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u64 *enter_ns = bpf_map_lookup_elem(&tid_sync_enter, &pid_tgid);
	if (!enter_ns)
		return 0;

	__u64 start = *enter_ns;
	bpf_map_delete_elem(&tid_sync_enter, &pid_tgid);

	__u32 tgid = (__u32)(pid_tgid >> 32);
	if (tgid == 0)
		return 0;

	struct cuda_tgid_stats *st = get_or_init_current_stats(tgid);
	if (!st)
		return 0;

	__u64 now = bpf_ktime_get_ns();
	__sync_fetch_and_add(&st->sync_calls, 1);
	if (now > start) {
		__u64 delta = now - start;
		__sync_fetch_and_add(&st->sync_stall_sum_ns, delta);
		if (delta > st->max_sync_stall_ns)
			st->max_sync_stall_ns = delta;
	}
	return 0;
}

SEC("raw_tracepoint/sched_switch")
int handle_sched_switch(struct bpf_raw_tracepoint_args *ctx)
{
	struct task_struct *prev = (struct task_struct *)ctx->args[1];
	struct task_struct *next = (struct task_struct *)ctx->args[2];
	if (!prev || !next)
		return 0;

	__u32 prev_pid = 0;
	__u32 prev_tgid = 0;
	__u32 next_pid = 0;
	__u32 next_tgid = 0;

	BPF_CORE_READ_INTO(&prev_pid, prev, pid);
	BPF_CORE_READ_INTO(&prev_tgid, prev, tgid);
	BPF_CORE_READ_INTO(&next_pid, next, pid);
	BPF_CORE_READ_INTO(&next_tgid, next, tgid);

	__u64 now = 0;

	if (prev_tgid != 0 && prev_pid != 0) {
		struct cuda_tgid_stats *prev_st = bpf_map_lookup_elem(&tgid_stats, &prev_tgid);
		if (!prev_st && bpf_map_lookup_elem(&target_tgids, &prev_tgid))
			prev_st = get_or_init_task_stats(prev_tgid, prev);

		if (prev_st) {
			now = bpf_ktime_get_ns();
			__sync_fetch_and_add(&prev_st->sched_switches, 1);
			bpf_map_update_elem(&tid_offcpu_enter, &prev_pid, &now, BPF_ANY);
		}
	}

	if (next_tgid != 0 && next_pid != 0) {
		__u64 *off_start = bpf_map_lookup_elem(&tid_offcpu_enter, &next_pid);
		if (off_start) {
			__u64 start = *off_start;
			bpf_map_delete_elem(&tid_offcpu_enter, &next_pid);

			struct cuda_tgid_stats *next_st = bpf_map_lookup_elem(&tgid_stats, &next_tgid);
			if (!next_st && bpf_map_lookup_elem(&target_tgids, &next_tgid))
				next_st = get_or_init_task_stats(next_tgid, next);

			if (next_st) {
				if (now == 0)
					now = bpf_ktime_get_ns();
				if (now > start)
					__sync_fetch_and_add(&next_st->off_cpu_sum_ns, now - start);
			}
		}
	}

	return 0;
}

char LICENSE[] SEC("license") = "Dual BSD/GPL";
