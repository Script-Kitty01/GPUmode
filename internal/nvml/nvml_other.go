//go:build !linux || !cgo

package nvml

import "fmt"

// Client is a stub NVML client on non-Linux or non-cgo builds.
type Client struct {
	libPath string
}

func openWithLibPath(libPath string) (*Client, error) {
	if libPath != "" {
		return nil, fmt.Errorf("%w: cannot load %q on unsupported platform or without cgo", ErrUnavailable, libPath)
	}
	return nil, fmt.Errorf("%w: unsupported platform or cgo disabled", ErrUnavailable)
}

// Close shuts down the NVML client.
func (c *Client) Close() error {
	return nil
}

// LibPath returns the resolved shared library path used to open NVML.
func (c *Client) LibPath() string {
	if c == nil {
		return ""
	}
	return c.libPath
}

// DeviceCount returns the number of visible NVIDIA GPUs.
func (c *Client) DeviceCount() (int, error) {
	return 0, ErrUnavailable
}

// SampleDevice queries hardware telemetry for the GPU at gpuIndex.
func (c *Client) SampleDevice(gpuIndex int) (DeviceSample, error) {
	return DeviceSample{}, ErrUnavailable
}

// ListComputeProcesses lists active compute processes running on the GPU at gpuIndex.
func (c *Client) ListComputeProcesses(gpuIndex int) ([]ComputeProcess, error) {
	return nil, ErrUnavailable
}
