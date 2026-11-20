/* SPDX-License-Identifier: Dual BSD/GPL */
#ifndef __GPUMODE_VMLINUX_MIN_H__
#define __GPUMODE_VMLINUX_MIN_H__

#include <stdbool.h>
#include <linux/types.h>
#include <linux/bpf.h>

/*
 * Minimal x86_64 userspace/kernel pt_regs matching <bpf/bpf_tracing.h>
 * when __VMLINUX_H__ is not defined (System V AMD64 ABI registers).
 */
struct pt_regs {
	unsigned long r15;
	unsigned long r14;
	unsigned long r13;
	unsigned long r12;
	unsigned long rbp;
	unsigned long rbx;
	unsigned long r11;
	unsigned long r10;
	unsigned long r9;
	unsigned long r8;
	unsigned long rax;
	unsigned long rcx;
	unsigned long rdx;
	unsigned long rsi;
	unsigned long rdi;
	unsigned long orig_rax;
	unsigned long rip;
	unsigned long cs;
	unsigned long eflags;
	unsigned long rsp;
	unsigned long ss;
};

/*
 * CO-RE relocated subset of Linux struct task_struct used by sched_switch.
 */
struct task_struct {
	int pid;
	int tgid;
	char comm[16];
} __attribute__((preserve_access_index));

#endif /* __GPUMODE_VMLINUX_MIN_H__ */
