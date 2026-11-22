package nvml

import "errors"

// ErrUnavailable indicates that the NVML shared library could not be loaded
// or initialized on the current system.
var ErrUnavailable = errors.New("nvml library unavailable")

var errClientClosed = errors.New("nvml client closed")

const (
	nvmlValueNotAvailable = ^uint64(0)
	bytesPerMiB           = 1024.0 * 1024.0
)

// NVML clock throttle / event reason bitmasks (nvmlClocksEventReasons).
const (
	ThrottleReasonGpuIdle                   uint64 = 0x0000000000000001
	ThrottleReasonApplicationsClocksSetting uint64 = 0x0000000000000002
	ThrottleReasonSwPowerCap                uint64 = 0x0000000000000004
	ThrottleReasonHwSlowdown                uint64 = 0x0000000000000008
	ThrottleReasonSyncBoost                 uint64 = 0x0000000000000010
	ThrottleReasonSwThermalSlowdown         uint64 = 0x0000000000000020
	ThrottleReasonHwThermalSlowdown         uint64 = 0x0000000000000040
	ThrottleReasonHwPowerBrakeSlowdown      uint64 = 0x0000000000000080
	ThrottleReasonDisplayClockSetting       uint64 = 0x0000000000000100
)

// DecodeThrottleReasons converts an NVML clocks event/throttle bitmask into human-readable reason labels.
func DecodeThrottleReasons(mask uint64) []string {
	if mask == 0 || mask == nvmlValueNotAvailable {
		return nil
	}
	var reasons []string
	if mask&ThrottleReasonGpuIdle != 0 {
		reasons = append(reasons, "GpuIdle")
	}
	if mask&ThrottleReasonApplicationsClocksSetting != 0 {
		reasons = append(reasons, "ApplicationsClocksSetting")
	}
	if mask&ThrottleReasonSwPowerCap != 0 {
		reasons = append(reasons, "SwPowerCap")
	}
	if mask&ThrottleReasonHwSlowdown != 0 {
		reasons = append(reasons, "HwSlowdown")
	}
	if mask&ThrottleReasonSyncBoost != 0 {
		reasons = append(reasons, "SyncBoost")
	}
	if mask&ThrottleReasonSwThermalSlowdown != 0 {
		reasons = append(reasons, "SwThermalSlowdown")
	}
	if mask&ThrottleReasonHwThermalSlowdown != 0 {
		reasons = append(reasons, "HwThermalSlowdown")
	}
	if mask&ThrottleReasonHwPowerBrakeSlowdown != 0 {
		reasons = append(reasons, "HwPowerBrakeSlowdown")
	}
	if mask&ThrottleReasonDisplayClockSetting != 0 {
		reasons = append(reasons, "DisplayClockSetting")
	}
	return reasons
}

// DeviceSample holds a point-in-time hardware telemetry snapshot for a single GPU.
type DeviceSample struct {
	GPUName                string   `json:"gpuName"`
	PowerWatts             float64  `json:"powerWatts"`
	SMClockMHz             float64  `json:"smClockMhz"`
	MemClockMHz            float64  `json:"memClockMhz"`
	TemperatureC           float64  `json:"temperatureC"`
	GPUUtilPct             float64  `json:"gpuUtilPct"`
	MemUtilPct             float64  `json:"memUtilPct"`
	VRAMUsedMiB            float64  `json:"vramUsedMib"`
	VRAMTotalMiB           float64  `json:"vramTotalMib"`
	MaxSMClockMHz          float64  `json:"maxSmClockMhz"`
	MaxMemClockMHz         float64  `json:"maxMemClockMhz"`
	TDPWatts               float64  `json:"tdpWatts"`
	ThrottleReasonsBitmask uint64   `json:"throttleReasonsBitmask,omitempty"`
	ThrottleReasons        []string `json:"throttleReasons,omitempty"`
	PCIeTxKBps             uint32   `json:"pcieTxKbps,omitempty"`
	PCIeRxKBps             uint32   `json:"pcieRxKbps,omitempty"`
}

// ComputeProcess represents an active compute process holding a context on a GPU.
type ComputeProcess struct {
	PID         uint32  `json:"pid"`
	UsedVRAMMiB float64 `json:"usedVramMib"`
}

// Open loads and initializes NVML using the default library search order
// (checking GPUMODE_NVML_LIB first if non-empty, followed by standard Linux paths).
func Open() (*Client, error) {
	return OpenWithLibPath("")
}

// OpenWithLibPath loads and initializes NVML from libPath.
// When libPath is empty, default search paths are tried. When libPath is non-empty,
// only libPath is attempted without falling back to default paths.
func OpenWithLibPath(libPath string) (*Client, error) {
	return openWithLibPath(libPath)
}

func usedVRAMBytesToMiB(usedBytes uint64) float64 {
	if usedBytes == nvmlValueNotAvailable {
		return 0
	}
	return float64(usedBytes) / bytesPerMiB
}
