//go:build linux && cgo

package nvml

/*
#cgo LDFLAGS: -ldl

#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int gpumode_nvmlReturn_t;
typedef void* gpumode_nvmlDevice_t;

enum {
	GPUMODE_NVML_SUCCESS                 = 0,
	GPUMODE_NVML_ERROR_NOT_SUPPORTED     = 3,
	GPUMODE_NVML_ERROR_INSUFFICIENT_SIZE = 7,
	GPUMODE_NVML_ERROR_MEMORY            = 20
};

enum {
	GPUMODE_NVML_CLOCK_SM  = 1,
	GPUMODE_NVML_CLOCK_MEM = 2
};

enum {
	GPUMODE_NVML_TEMPERATURE_GPU = 0
};

enum {
	GPUMODE_NVML_PCIE_UTIL_TX_BYTES = 0,
	GPUMODE_NVML_PCIE_UTIL_RX_BYTES = 1
};

typedef struct {
	unsigned int gpu;
	unsigned int memory;
} gpumode_nvmlUtilization_t;

typedef struct {
	unsigned long long total;
	unsigned long long free;
	unsigned long long used;
} gpumode_nvmlMemory_t;

typedef struct {
	unsigned int pid;
	unsigned long long usedGpuMemory;
} gpumode_nvmlProcessInfo_v1_t;

typedef struct {
	unsigned int pid;
	unsigned long long usedGpuMemory;
	unsigned int gpuInstanceId;
	unsigned int computeInstanceId;
} gpumode_nvmlProcessInfo_v2_t;

typedef gpumode_nvmlReturn_t (*fn_nvmlInit_t)(void);
typedef gpumode_nvmlReturn_t (*fn_nvmlShutdown_t)(void);
typedef const char* (*fn_nvmlErrorString_t)(gpumode_nvmlReturn_t);
typedef gpumode_nvmlReturn_t (*fn_nvmlDeviceGetCount_t)(unsigned int*);
typedef gpumode_nvmlReturn_t (*fn_nvmlDeviceGetHandleByIndex_t)(unsigned int, gpumode_nvmlDevice_t*);
typedef gpumode_nvmlReturn_t (*fn_nvmlDeviceGetName_t)(gpumode_nvmlDevice_t, char*, unsigned int);
typedef gpumode_nvmlReturn_t (*fn_nvmlDeviceGetPowerUsage_t)(gpumode_nvmlDevice_t, unsigned int*);
typedef gpumode_nvmlReturn_t (*fn_nvmlDeviceGetPowerManagementLimit_t)(gpumode_nvmlDevice_t, unsigned int*);
typedef gpumode_nvmlReturn_t (*fn_nvmlDeviceGetClockInfo_t)(gpumode_nvmlDevice_t, int, unsigned int*);
typedef gpumode_nvmlReturn_t (*fn_nvmlDeviceGetMaxClockInfo_t)(gpumode_nvmlDevice_t, int, unsigned int*);
typedef gpumode_nvmlReturn_t (*fn_nvmlDeviceGetTemperature_t)(gpumode_nvmlDevice_t, int, unsigned int*);
typedef gpumode_nvmlReturn_t (*fn_nvmlDeviceGetUtilizationRates_t)(gpumode_nvmlDevice_t, gpumode_nvmlUtilization_t*);
typedef gpumode_nvmlReturn_t (*fn_nvmlDeviceGetMemoryInfo_t)(gpumode_nvmlDevice_t, gpumode_nvmlMemory_t*);
typedef gpumode_nvmlReturn_t (*fn_nvmlDeviceGetCurrentClocksEventReasons_t)(gpumode_nvmlDevice_t, unsigned long long*);
typedef gpumode_nvmlReturn_t (*fn_nvmlDeviceGetPcieThroughput_t)(gpumode_nvmlDevice_t, int, unsigned int*);
typedef gpumode_nvmlReturn_t (*fn_nvmlDeviceGetComputeRunningProcesses_v2_t)(gpumode_nvmlDevice_t, unsigned int*, gpumode_nvmlProcessInfo_v2_t*);
typedef gpumode_nvmlReturn_t (*fn_nvmlDeviceGetComputeRunningProcesses_v1_t)(gpumode_nvmlDevice_t, unsigned int*, gpumode_nvmlProcessInfo_v1_t*);

typedef struct {
	fn_nvmlInit_t                                init;
	fn_nvmlShutdown_t                            shutdown;
	fn_nvmlErrorString_t                         errorString;
	fn_nvmlDeviceGetCount_t                      getCount;
	fn_nvmlDeviceGetHandleByIndex_t              getHandleByIndex;
	fn_nvmlDeviceGetName_t                       getName;
	fn_nvmlDeviceGetPowerUsage_t                  getPowerUsage;
	fn_nvmlDeviceGetPowerManagementLimit_t       getPowerManagementLimit;
	fn_nvmlDeviceGetPowerManagementLimit_t       getEnforcedPowerLimit;
	fn_nvmlDeviceGetPowerManagementLimit_t       getPowerManagementDefaultLimit;
	fn_nvmlDeviceGetClockInfo_t                  getClockInfo;
	fn_nvmlDeviceGetMaxClockInfo_t               getMaxClockInfo;
	fn_nvmlDeviceGetTemperature_t                getTemperature;
	fn_nvmlDeviceGetUtilizationRates_t           getUtilizationRates;
	fn_nvmlDeviceGetMemoryInfo_t                 getMemoryInfo;
	fn_nvmlDeviceGetCurrentClocksEventReasons_t  getCurrentClocksEventReasons;
	fn_nvmlDeviceGetPcieThroughput_t             getPcieThroughput;
	fn_nvmlDeviceGetComputeRunningProcesses_v2_t getComputeRunningProcessesV2;
	fn_nvmlDeviceGetComputeRunningProcesses_v1_t getComputeRunningProcessesV1;
} gpumode_nvml_syms_t;

typedef struct {
	char               name[256];
	unsigned int       powerMW;
	unsigned int       tdpMW;
	unsigned int       smClockMHz;
	unsigned int       memClockMHz;
	unsigned int       maxSmClockMHz;
	unsigned int       maxMemClockMHz;
	unsigned int       tempC;
	unsigned int       gpuUtil;
	unsigned int       memUtil;
	unsigned long long vramUsedBytes;
	unsigned long long vramTotalBytes;
	unsigned long long throttleReasons;
	unsigned int       pcieTxKBps;
	unsigned int       pcieRxKBps;
} gpumode_nvml_raw_sample_t;

typedef struct {
	unsigned int       pid;
	unsigned long long usedGpuMemory;
} gpumode_nvml_raw_proc_t;

static void* gpumode_dlopen(const char* path, char* errBuf, size_t errCap) {
	if (errCap > 0) {
		errBuf[0] = '\0';
	}
	dlerror();
	void* h = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
	if (h == NULL && errCap > 0) {
		const char* err = dlerror();
		if (err != NULL) {
			strncpy(errBuf, err, errCap - 1);
			errBuf[errCap - 1] = '\0';
		} else {
			strncpy(errBuf, "dlopen returned NULL", errCap - 1);
			errBuf[errCap - 1] = '\0';
		}
	}
	return h;
}

static void* gpumode_resolve_sym(void* handle, const char* primary, const char* fallback) {
	dlerror();
	void* sym = dlsym(handle, primary);
	if (sym == NULL && fallback != NULL) {
		dlerror();
		sym = dlsym(handle, fallback);
	}
	return sym;
}

static int gpumode_load_syms(void* handle, gpumode_nvml_syms_t* syms, const char** missingOut) {
	memset(syms, 0, sizeof(*syms));
	void* s = NULL;

	s = gpumode_resolve_sym(handle, "nvmlInit_v2", "nvmlInit");
	if (!s) { *missingOut = "nvmlInit_v2/nvmlInit"; return -1; }
	memcpy(&syms->init, &s, sizeof(s));

	s = gpumode_resolve_sym(handle, "nvmlShutdown", NULL);
	if (!s) { *missingOut = "nvmlShutdown"; return -1; }
	memcpy(&syms->shutdown, &s, sizeof(s));

	s = gpumode_resolve_sym(handle, "nvmlErrorString", NULL);
	if (s) { memcpy(&syms->errorString, &s, sizeof(s)); }

	s = gpumode_resolve_sym(handle, "nvmlDeviceGetCount_v2", "nvmlDeviceGetCount");
	if (!s) { *missingOut = "nvmlDeviceGetCount_v2/nvmlDeviceGetCount"; return -1; }
	memcpy(&syms->getCount, &s, sizeof(s));

	s = gpumode_resolve_sym(handle, "nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetHandleByIndex");
	if (!s) { *missingOut = "nvmlDeviceGetHandleByIndex_v2/nvmlDeviceGetHandleByIndex"; return -1; }
	memcpy(&syms->getHandleByIndex, &s, sizeof(s));

	s = gpumode_resolve_sym(handle, "nvmlDeviceGetName", NULL);
	if (!s) { *missingOut = "nvmlDeviceGetName"; return -1; }
	memcpy(&syms->getName, &s, sizeof(s));

	s = gpumode_resolve_sym(handle, "nvmlDeviceGetPowerUsage", NULL);
	if (!s) { *missingOut = "nvmlDeviceGetPowerUsage"; return -1; }
	memcpy(&syms->getPowerUsage, &s, sizeof(s));

	s = gpumode_resolve_sym(handle, "nvmlDeviceGetPowerManagementLimit", NULL);
	if (s) { memcpy(&syms->getPowerManagementLimit, &s, sizeof(s)); }

	s = gpumode_resolve_sym(handle, "nvmlDeviceGetEnforcedPowerLimit", NULL);
	if (s) { memcpy(&syms->getEnforcedPowerLimit, &s, sizeof(s)); }

	s = gpumode_resolve_sym(handle, "nvmlDeviceGetPowerManagementDefaultLimit", NULL);
	if (s) { memcpy(&syms->getPowerManagementDefaultLimit, &s, sizeof(s)); }

	s = gpumode_resolve_sym(handle, "nvmlDeviceGetClockInfo", NULL);
	if (!s) { *missingOut = "nvmlDeviceGetClockInfo"; return -1; }
	memcpy(&syms->getClockInfo, &s, sizeof(s));

	s = gpumode_resolve_sym(handle, "nvmlDeviceGetMaxClockInfo", NULL);
	if (s) { memcpy(&syms->getMaxClockInfo, &s, sizeof(s)); }

	s = gpumode_resolve_sym(handle, "nvmlDeviceGetTemperature", NULL);
	if (!s) { *missingOut = "nvmlDeviceGetTemperature"; return -1; }
	memcpy(&syms->getTemperature, &s, sizeof(s));

	s = gpumode_resolve_sym(handle, "nvmlDeviceGetUtilizationRates", NULL);
	if (!s) { *missingOut = "nvmlDeviceGetUtilizationRates"; return -1; }
	memcpy(&syms->getUtilizationRates, &s, sizeof(s));

	s = gpumode_resolve_sym(handle, "nvmlDeviceGetMemoryInfo", NULL);
	if (!s) { *missingOut = "nvmlDeviceGetMemoryInfo"; return -1; }
	memcpy(&syms->getMemoryInfo, &s, sizeof(s));

	s = gpumode_resolve_sym(handle, "nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons");
	if (s) { memcpy(&syms->getCurrentClocksEventReasons, &s, sizeof(s)); }

	s = gpumode_resolve_sym(handle, "nvmlDeviceGetPcieThroughput", NULL);
	if (s) { memcpy(&syms->getPcieThroughput, &s, sizeof(s)); }

	s = gpumode_resolve_sym(handle, "nvmlDeviceGetComputeRunningProcesses_v2", NULL);
	if (s) { memcpy(&syms->getComputeRunningProcessesV2, &s, sizeof(s)); }

	s = gpumode_resolve_sym(handle, "nvmlDeviceGetComputeRunningProcesses", NULL);
	if (s) { memcpy(&syms->getComputeRunningProcessesV1, &s, sizeof(s)); }

	if (!syms->getComputeRunningProcessesV2 && !syms->getComputeRunningProcessesV1) {
		*missingOut = "nvmlDeviceGetComputeRunningProcesses_v2/nvmlDeviceGetComputeRunningProcesses";
		return -1;
	}

	return 0;
}

static gpumode_nvmlReturn_t gpumode_call_init(const gpumode_nvml_syms_t* syms) {
	return syms->init();
}

static gpumode_nvmlReturn_t gpumode_call_shutdown(const gpumode_nvml_syms_t* syms) {
	return syms->shutdown();
}

static const char* gpumode_call_error_string(const gpumode_nvml_syms_t* syms, gpumode_nvmlReturn_t rc) {
	if (syms->errorString != NULL) {
		return syms->errorString(rc);
	}
	return NULL;
}

static gpumode_nvmlReturn_t gpumode_call_device_count(const gpumode_nvml_syms_t* syms, unsigned int* countOut) {
	return syms->getCount(countOut);
}

static gpumode_nvmlReturn_t gpumode_sample_device(
	const gpumode_nvml_syms_t* syms,
	unsigned int gpuIndex,
	gpumode_nvml_raw_sample_t* out,
	const char** failedStep
) {
	memset(out, 0, sizeof(*out));
	gpumode_nvmlDevice_t dev = NULL;
	gpumode_nvmlReturn_t rc = syms->getHandleByIndex(gpuIndex, &dev);
	if (rc != GPUMODE_NVML_SUCCESS) {
		*failedStep = "nvmlDeviceGetHandleByIndex";
		return rc;
	}

	rc = syms->getName(dev, out->name, (unsigned int)(sizeof(out->name) - 1));
	if (rc != GPUMODE_NVML_SUCCESS) {
		*failedStep = "nvmlDeviceGetName";
		return rc;
	}
	out->name[sizeof(out->name) - 1] = '\0';

	unsigned int val = 0;
	rc = syms->getPowerUsage(dev, &val);
	if (rc == GPUMODE_NVML_SUCCESS) {
		out->powerMW = val;
	} else if (rc == GPUMODE_NVML_ERROR_NOT_SUPPORTED) {
		out->powerMW = 0;
	} else {
		*failedStep = "nvmlDeviceGetPowerUsage";
		return rc;
	}

	if (syms->getPowerManagementLimit != NULL && syms->getPowerManagementLimit(dev, &val) == GPUMODE_NVML_SUCCESS) {
		out->tdpMW = val;
	} else if (syms->getEnforcedPowerLimit != NULL && syms->getEnforcedPowerLimit(dev, &val) == GPUMODE_NVML_SUCCESS) {
		out->tdpMW = val;
	} else if (syms->getPowerManagementDefaultLimit != NULL && syms->getPowerManagementDefaultLimit(dev, &val) == GPUMODE_NVML_SUCCESS) {
		out->tdpMW = val;
	}

	val = 0;
	rc = syms->getClockInfo(dev, GPUMODE_NVML_CLOCK_SM, &val);
	if (rc == GPUMODE_NVML_SUCCESS) {
		out->smClockMHz = val;
	} else if (rc != GPUMODE_NVML_ERROR_NOT_SUPPORTED) {
		*failedStep = "nvmlDeviceGetClockInfo(SM)";
		return rc;
	}

	val = 0;
	rc = syms->getClockInfo(dev, GPUMODE_NVML_CLOCK_MEM, &val);
	if (rc == GPUMODE_NVML_SUCCESS) {
		out->memClockMHz = val;
	} else if (rc != GPUMODE_NVML_ERROR_NOT_SUPPORTED) {
		*failedStep = "nvmlDeviceGetClockInfo(MEM)";
		return rc;
	}

	if (syms->getMaxClockInfo != NULL) {
		val = 0;
		if (syms->getMaxClockInfo(dev, GPUMODE_NVML_CLOCK_SM, &val) == GPUMODE_NVML_SUCCESS) {
			out->maxSmClockMHz = val;
		}
		val = 0;
		if (syms->getMaxClockInfo(dev, GPUMODE_NVML_CLOCK_MEM, &val) == GPUMODE_NVML_SUCCESS) {
			out->maxMemClockMHz = val;
		}
	}

	val = 0;
	rc = syms->getTemperature(dev, GPUMODE_NVML_TEMPERATURE_GPU, &val);
	if (rc == GPUMODE_NVML_SUCCESS) {
		out->tempC = val;
	} else if (rc != GPUMODE_NVML_ERROR_NOT_SUPPORTED) {
		*failedStep = "nvmlDeviceGetTemperature";
		return rc;
	}

	gpumode_nvmlUtilization_t util;
	memset(&util, 0, sizeof(util));
	rc = syms->getUtilizationRates(dev, &util);
	if (rc == GPUMODE_NVML_SUCCESS) {
		out->gpuUtil = util.gpu;
		out->memUtil = util.memory;
	} else if (rc != GPUMODE_NVML_ERROR_NOT_SUPPORTED) {
		*failedStep = "nvmlDeviceGetUtilizationRates";
		return rc;
	}

	gpumode_nvmlMemory_t mem;
	memset(&mem, 0, sizeof(mem));
	rc = syms->getMemoryInfo(dev, &mem);
	if (rc != GPUMODE_NVML_SUCCESS) {
		*failedStep = "nvmlDeviceGetMemoryInfo";
		return rc;
	}
	out->vramUsedBytes = mem.used;
	out->vramTotalBytes = mem.total;

	if (syms->getCurrentClocksEventReasons != NULL) {
		unsigned long long mask = 0;
		if (syms->getCurrentClocksEventReasons(dev, &mask) == GPUMODE_NVML_SUCCESS) {
			out->throttleReasons = mask;
		}
	}
	if (syms->getPcieThroughput != NULL) {
		unsigned int tx = 0, rx = 0;
		if (syms->getPcieThroughput(dev, GPUMODE_NVML_PCIE_UTIL_TX_BYTES, &tx) == GPUMODE_NVML_SUCCESS) {
			out->pcieTxKBps = tx;
		}
		if (syms->getPcieThroughput(dev, GPUMODE_NVML_PCIE_UTIL_RX_BYTES, &rx) == GPUMODE_NVML_SUCCESS) {
			out->pcieRxKBps = rx;
		}
	}

	return GPUMODE_NVML_SUCCESS;
}

static gpumode_nvmlReturn_t gpumode_list_compute_procs(
	const gpumode_nvml_syms_t* syms,
	unsigned int gpuIndex,
	gpumode_nvml_raw_proc_t** procsOut,
	unsigned int* countOut,
	const char** failedStep
) {
	*procsOut = NULL;
	*countOut = 0;

	gpumode_nvmlDevice_t dev = NULL;
	gpumode_nvmlReturn_t rc = syms->getHandleByIndex(gpuIndex, &dev);
	if (rc != GPUMODE_NVML_SUCCESS) {
		*failedStep = "nvmlDeviceGetHandleByIndex";
		return rc;
	}

	if (syms->getComputeRunningProcessesV2 != NULL) {
		unsigned int count = 0;
		rc = syms->getComputeRunningProcessesV2(dev, &count, NULL);
		if (rc == GPUMODE_NVML_SUCCESS && count == 0) {
			return GPUMODE_NVML_SUCCESS;
		}
		for (int attempt = 0; attempt < 6 && (rc == GPUMODE_NVML_ERROR_INSUFFICIENT_SIZE || (rc == GPUMODE_NVML_SUCCESS && count > 0)); attempt++) {
			unsigned int cap = count + 8;
			gpumode_nvmlProcessInfo_v2_t* buf = (gpumode_nvmlProcessInfo_v2_t*)calloc(cap, sizeof(gpumode_nvmlProcessInfo_v2_t));
			if (buf == NULL) {
				*failedStep = "calloc";
				return GPUMODE_NVML_ERROR_MEMORY;
			}
			count = cap;
			rc = syms->getComputeRunningProcessesV2(dev, &count, buf);
			if (rc == GPUMODE_NVML_SUCCESS) {
				if (count == 0) {
					free(buf);
					return GPUMODE_NVML_SUCCESS;
				}
				if (count > cap) {
					count = cap;
				}
				gpumode_nvml_raw_proc_t* out = (gpumode_nvml_raw_proc_t*)calloc(count, sizeof(gpumode_nvml_raw_proc_t));
				if (out == NULL) {
					free(buf);
					*failedStep = "calloc";
					return GPUMODE_NVML_ERROR_MEMORY;
				}
				for (unsigned int i = 0; i < count; i++) {
					out[i].pid = buf[i].pid;
					out[i].usedGpuMemory = buf[i].usedGpuMemory;
				}
				free(buf);
				*procsOut = out;
				*countOut = count;
				return GPUMODE_NVML_SUCCESS;
			}
			free(buf);
		}
		*failedStep = "nvmlDeviceGetComputeRunningProcesses_v2";
		return rc;
	}

	unsigned int count = 0;
	rc = syms->getComputeRunningProcessesV1(dev, &count, NULL);
	if (rc == GPUMODE_NVML_SUCCESS && count == 0) {
		return GPUMODE_NVML_SUCCESS;
	}
	for (int attempt = 0; attempt < 6 && (rc == GPUMODE_NVML_ERROR_INSUFFICIENT_SIZE || (rc == GPUMODE_NVML_SUCCESS && count > 0)); attempt++) {
		unsigned int cap = count + 8;
		gpumode_nvmlProcessInfo_v1_t* buf = (gpumode_nvmlProcessInfo_v1_t*)calloc(cap, sizeof(gpumode_nvmlProcessInfo_v1_t));
		if (buf == NULL) {
			*failedStep = "calloc";
			return GPUMODE_NVML_ERROR_MEMORY;
		}
		count = cap;
		rc = syms->getComputeRunningProcessesV1(dev, &count, buf);
		if (rc == GPUMODE_NVML_SUCCESS) {
			if (count == 0) {
				free(buf);
				return GPUMODE_NVML_SUCCESS;
			}
			if (count > cap) {
				count = cap;
			}
			gpumode_nvml_raw_proc_t* out = (gpumode_nvml_raw_proc_t*)calloc(count, sizeof(gpumode_nvml_raw_proc_t));
			if (out == NULL) {
				free(buf);
				*failedStep = "calloc";
				return GPUMODE_NVML_ERROR_MEMORY;
			}
			for (unsigned int i = 0; i < count; i++) {
				out[i].pid = buf[i].pid;
				out[i].usedGpuMemory = buf[i].usedGpuMemory;
			}
			free(buf);
			*procsOut = out;
			*countOut = count;
			return GPUMODE_NVML_SUCCESS;
		}
		free(buf);
	}
	*failedStep = "nvmlDeviceGetComputeRunningProcesses";
	return rc;
}
*/
import "C"

import (
	"fmt"
	"os"
	"sync"
	"unsafe"
)

var defaultLibCandidates = []string{
	"libnvidia-ml.so.1",
	"libnvidia-ml.so",
	"/usr/lib/libnvidia-ml.so.1",
	"/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
	"/usr/lib64/libnvidia-ml.so.1",
}

// Client wraps a dynamically loaded libnvidia-ml handle with mutex-protected NVML calls.
type Client struct {
	mu      sync.Mutex
	handle  unsafe.Pointer
	syms    *C.gpumode_nvml_syms_t
	libPath string
	closed  bool
}

func openWithLibPath(libPath string) (*Client, error) {
	var candidates []string
	if libPath != "" {
		candidates = []string{libPath}
	} else {
		candidates = make([]string, 0, len(defaultLibCandidates)+1)
		if envPath := os.Getenv("GPUMODE_NVML_LIB"); envPath != "" {
			candidates = append(candidates, envPath)
		}
		candidates = append(candidates, defaultLibCandidates...)
	}

	var lastErr string
	for _, cand := range candidates {
		cPath := C.CString(cand)
		var errBuf [512]C.char
		handle := C.gpumode_dlopen(cPath, &errBuf[0], C.size_t(len(errBuf)))
		C.free(unsafe.Pointer(cPath))

		if handle == nil {
			lastErr = C.GoString(&errBuf[0])
			continue
		}

		symsPtr := (*C.gpumode_nvml_syms_t)(C.calloc(1, C.size_t(unsafe.Sizeof(C.gpumode_nvml_syms_t{}))))
		if symsPtr == nil {
			C.dlclose(handle)
			return nil, fmt.Errorf("%w: calloc failed for %q", ErrUnavailable, cand)
		}

		var missingSym *C.char
		if C.gpumode_load_syms(handle, symsPtr, &missingSym) != 0 {
			missingName := C.GoString(missingSym)
			C.free(unsafe.Pointer(symsPtr))
			C.dlclose(handle)
			return nil, fmt.Errorf("%w: missing symbol %s in %q", ErrUnavailable, missingName, cand)
		}

		if rc := C.gpumode_call_init(symsPtr); rc != C.GPUMODE_NVML_SUCCESS {
			errText := nvmlErrorText(symsPtr, rc)
			C.free(unsafe.Pointer(symsPtr))
			C.dlclose(handle)
			return nil, fmt.Errorf("%w: nvmlInit failed (%s) in %q", ErrUnavailable, errText, cand)
		}

		return &Client{
			handle:  handle,
			syms:    symsPtr,
			libPath: cand,
		}, nil
	}

	if libPath != "" {
		return nil, fmt.Errorf("%w: dlopen %q failed: %s", ErrUnavailable, libPath, lastErr)
	}
	return nil, fmt.Errorf("%w: %s", ErrUnavailable, lastErr)
}

func nvmlErrorText(syms *C.gpumode_nvml_syms_t, rc C.gpumode_nvmlReturn_t) string {
	if syms != nil {
		if cStr := C.gpumode_call_error_string(syms, rc); cStr != nil {
			return fmt.Sprintf("%s (code %d)", C.GoString(cStr), int(rc))
		}
	}
	return fmt.Sprintf("nvml error code %d", int(rc))
}

// Close shuts down NVML and releases the dynamic library handle. Close is idempotent and thread-safe.
func (c *Client) Close() error {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true

	var shutdownErr error
	if c.handle != nil {
		if c.syms != nil {
			if rc := C.gpumode_call_shutdown(c.syms); rc != C.GPUMODE_NVML_SUCCESS {
				shutdownErr = fmt.Errorf("nvml: nvmlShutdown failed: %s", nvmlErrorText(c.syms, rc))
			}
			C.free(unsafe.Pointer(c.syms))
			c.syms = nil
		}
		C.dlclose(c.handle)
		c.handle = nil
	}
	return shutdownErr
}

// LibPath returns the library path used when dlopen succeeded.
func (c *Client) LibPath() string {
	if c == nil {
		return ""
	}
	return c.libPath
}

// DeviceCount returns the number of visible NVIDIA GPUs.
func (c *Client) DeviceCount() (int, error) {
	if c == nil {
		return 0, errClientClosed
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return 0, errClientClosed
	}

	var count C.uint
	if rc := C.gpumode_call_device_count(c.syms, &count); rc != C.GPUMODE_NVML_SUCCESS {
		return 0, fmt.Errorf("nvml: nvmlDeviceGetCount failed: %s", nvmlErrorText(c.syms, rc))
	}
	return int(count), nil
}

// SampleDevice queries hardware telemetry for the GPU at gpuIndex.
func (c *Client) SampleDevice(gpuIndex int) (DeviceSample, error) {
	if c == nil {
		return DeviceSample{}, errClientClosed
	}
	if gpuIndex < 0 {
		return DeviceSample{}, fmt.Errorf("nvml: invalid gpu index %d", gpuIndex)
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return DeviceSample{}, errClientClosed
	}

	var raw C.gpumode_nvml_raw_sample_t
	var failedStep *C.char
	if rc := C.gpumode_sample_device(c.syms, C.uint(gpuIndex), &raw, &failedStep); rc != C.GPUMODE_NVML_SUCCESS {
		return DeviceSample{}, fmt.Errorf(
			"nvml: %s failed for gpu %d: %s",
			C.GoString(failedStep),
			gpuIndex,
			nvmlErrorText(c.syms, rc),
		)
	}

	mask := uint64(raw.throttleReasons)
	return DeviceSample{
		GPUName:                C.GoString(&raw.name[0]),
		PowerWatts:             float64(raw.powerMW) / 1000.0,
		SMClockMHz:             float64(raw.smClockMHz),
		MemClockMHz:            float64(raw.memClockMHz),
		TemperatureC:           float64(raw.tempC),
		GPUUtilPct:             float64(raw.gpuUtil),
		MemUtilPct:             float64(raw.memUtil),
		VRAMUsedMiB:            float64(raw.vramUsedBytes) / bytesPerMiB,
		VRAMTotalMiB:           float64(raw.vramTotalBytes) / bytesPerMiB,
		MaxSMClockMHz:          float64(raw.maxSmClockMHz),
		MaxMemClockMHz:         float64(raw.maxMemClockMHz),
		TDPWatts:               float64(raw.tdpMW) / 1000.0,
		ThrottleReasonsBitmask: mask,
		ThrottleReasons:        DecodeThrottleReasons(mask),
		PCIeTxKBps:             uint32(raw.pcieTxKBps),
		PCIeRxKBps:             uint32(raw.pcieRxKBps),
	}, nil
}

// ListComputeProcesses lists active compute processes running on the GPU at gpuIndex.
func (c *Client) ListComputeProcesses(gpuIndex int) ([]ComputeProcess, error) {
	if c == nil {
		return nil, errClientClosed
	}
	if gpuIndex < 0 {
		return nil, fmt.Errorf("nvml: invalid gpu index %d", gpuIndex)
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil, errClientClosed
	}

	var rawProcs *C.gpumode_nvml_raw_proc_t
	var count C.uint
	var failedStep *C.char
	rc := C.gpumode_list_compute_procs(c.syms, C.uint(gpuIndex), &rawProcs, &count, &failedStep)
	if rc != C.GPUMODE_NVML_SUCCESS {
		return nil, fmt.Errorf(
			"nvml: %s failed for gpu %d: %s",
			C.GoString(failedStep),
			gpuIndex,
			nvmlErrorText(c.syms, rc),
		)
	}
	if rawProcs != nil {
		defer C.free(unsafe.Pointer(rawProcs))
	}
	if count == 0 || rawProcs == nil {
		return []ComputeProcess{}, nil
	}

	entries := unsafe.Slice(rawProcs, int(count))
	procs := make([]ComputeProcess, len(entries))
	for i, entry := range entries {
		procs[i] = ComputeProcess{
			PID:         uint32(entry.pid),
			UsedVRAMMiB: usedVRAMBytesToMiB(uint64(entry.usedGpuMemory)),
		}
	}
	return procs, nil
}
