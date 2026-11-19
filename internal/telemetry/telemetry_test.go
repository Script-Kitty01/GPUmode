package telemetry

import (
	"context"
	"math"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestParseNvidiaSMICSVLine(t *testing.T) {
	ts := time.Date(2026, 11, 18, 12, 0, 0, 0, time.UTC)

	t.Run("standard RTX 4050 Laptop GPU output", func(t *testing.T) {
		raw := "NVIDIA GeForce RTX 4050 Laptop GPU, 42.68, 2355, 8001, 58, 76, 44, 1842, 6141\n"
		s, err := ParseNvidiaSMICSVLine(raw, ts)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if s.GPUName != "NVIDIA GeForce RTX 4050 Laptop GPU" {
			t.Errorf("got GPUName %q, want %q", s.GPUName, "NVIDIA GeForce RTX 4050 Laptop GPU")
		}
		if math.Abs(s.PowerWatts-42.68) > 1e-6 {
			t.Errorf("got PowerWatts %v, want 42.68", s.PowerWatts)
		}
		if s.SMClockMHz != 2355 {
			t.Errorf("got SMClockMHz %v, want 2355", s.SMClockMHz)
		}
		if s.MemClockMHz != 8001 {
			t.Errorf("got MemClockMHz %v, want 8001", s.MemClockMHz)
		}
		if s.TemperatureC != 58 {
			t.Errorf("got TemperatureC %v, want 58", s.TemperatureC)
		}
		if s.GPUUtilPct != 76 {
			t.Errorf("got GPUUtilPct %v, want 76", s.GPUUtilPct)
		}
		if s.MemUtilPct != 44 {
			t.Errorf("got MemUtilPct %v, want 44", s.MemUtilPct)
		}
		if s.VRAMUsedMiB != 1842 || s.VRAMTotalMiB != 6141 {
			t.Errorf("got VRAM %v/%v, want 1842/6141", s.VRAMUsedMiB, s.VRAMTotalMiB)
		}
	})

	t.Run("handles [N/A] and [Not Supported] fields gracefully", func(t *testing.T) {
		raw := "NVIDIA GeForce RTX 4050 Laptop GPU, [N/A], 2100, [Not Supported], 51, [N/A], 12, 512, 6141"
		s, err := ParseNvidiaSMICSVLine(raw, ts)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if s.PowerWatts != 0 {
			t.Errorf("expected PowerWatts=0 for [N/A], got %v", s.PowerWatts)
		}
		if s.MemClockMHz != 0 {
			t.Errorf("expected MemClockMHz=0 for [Not Supported], got %v", s.MemClockMHz)
		}
		if s.GPUUtilPct != 0 {
			t.Errorf("expected GPUUtilPct=0 for [N/A], got %v", s.GPUUtilPct)
		}
		if s.SMClockMHz != 2100 {
			t.Errorf("expected SMClockMHz=2100, got %v", s.SMClockMHz)
		}
	})

	t.Run("rejects malformed or empty output", func(t *testing.T) {
		if _, err := ParseNvidiaSMICSVLine("", ts); err == nil {
			t.Errorf("expected error for empty input")
		}
		if _, err := ParseNvidiaSMICSVLine("NVIDIA GPU, 35.0, 2100", ts); err == nil {
			t.Errorf("expected error for insufficient CSV columns")
		}
	})
}

func TestTrapezoidalEnergyIntegration(t *testing.T) {
	m := NewMonitor("nvidia-smi", 100*time.Millisecond, 50)
	t0 := time.Date(2026, 11, 18, 12, 0, 0, 0, time.UTC)

	// Record linear ramp: t=0s (30W), t=1s (50W), t=2s (70W)
	// Integral from 0s to 2s = 0.5*(30+50)*1 + 0.5*(50+70)*1 = 40 + 60 = 100 Joules.
	// Avg power = 100J / 2s = 50W.
	samples := []Sample{
		{Timestamp: t0, GPUName: "RTX 4050", PowerWatts: 30, SMClockMHz: 2000, MemClockMHz: 8000, TemperatureC: 55, VRAMUsedMiB: 1200},
		{Timestamp: t0.Add(1 * time.Second), GPUName: "RTX 4050", PowerWatts: 50, SMClockMHz: 2200, MemClockMHz: 8000, TemperatureC: 60, VRAMUsedMiB: 1500},
		{Timestamp: t0.Add(2 * time.Second), GPUName: "RTX 4050", PowerWatts: 70, SMClockMHz: 2400, MemClockMHz: 8000, TemperatureC: 65, VRAMUsedMiB: 1800},
	}
	for _, s := range samples {
		m.RecordSample(s)
	}

	// 200 completed requests over 2.0s window, 10ms per request latency
	em := m.MeasureWindow(t0, t0.Add(2*time.Second), 200, 10.0)

	if math.Abs(em.DurationSec-2.0) > 1e-6 {
		t.Errorf("DurationSec = %v, want 2.0", em.DurationSec)
	}
	if math.Abs(em.TotalEnergyJoules-100.0) > 1e-6 {
		t.Errorf("TotalEnergyJoules = %v, want 100.0", em.TotalEnergyJoules)
	}
	if math.Abs(em.AvgPowerWatts-50.0) > 1e-6 {
		t.Errorf("AvgPowerWatts = %v, want 50.0", em.AvgPowerWatts)
	}
	if math.Abs(em.PeakPowerWatts-70.0) > 1e-6 {
		t.Errorf("PeakPowerWatts = %v, want 70.0", em.PeakPowerWatts)
	}
	if math.Abs(em.AvgSMClockMHz-2200.0) > 1e-6 {
		t.Errorf("AvgSMClockMHz = %v, want 2200.0", em.AvgSMClockMHz)
	}
	if math.Abs(em.PeakTempC-65.0) > 1e-6 {
		t.Errorf("PeakTempC = %v, want 65.0", em.PeakTempC)
	}
	if math.Abs(em.PeakVRAMMiB-1800.0) > 1e-6 {
		t.Errorf("PeakVRAMMiB = %v, want 1800.0", em.PeakVRAMMiB)
	}
	// JoulesPerRequest = 100J / 200 reqs = 0.5 J/req
	if math.Abs(em.JoulesPerRequest-0.5) > 1e-6 {
		t.Errorf("JoulesPerRequest = %v, want 0.5", em.JoulesPerRequest)
	}
	// JoulesPer1KRequests = 500 J
	if math.Abs(em.JoulesPer1KRequests-500.0) > 1e-6 {
		t.Errorf("JoulesPer1KRequests = %v, want 500.0", em.JoulesPer1KRequests)
	}
	// EDP = 0.5 J * 0.010 s = 0.005 J*s
	if math.Abs(em.EnergyDelayProduct-0.005) > 1e-6 {
		t.Errorf("EnergyDelayProduct = %v, want 0.005", em.EnergyDelayProduct)
	}
	// Perf/Watt = (200 req / 2s) / 50W = 100 RPS / 50W = 2.0 req/s/W
	if math.Abs(em.PerformancePerWatt-2.0) > 1e-6 {
		t.Errorf("PerformancePerWatt = %v, want 2.0", em.PerformancePerWatt)
	}
	if em.SampleCount != 3 {
		t.Errorf("SampleCount = %d, want 3", em.SampleCount)
	}

	// Short sub-poll microbenchmark window test (e.g. 20ms window between t0+1.5s and t0+1.52s)
	shortStart := t0.Add(1500 * time.Millisecond)
	shortEnd := t0.Add(1520 * time.Millisecond)
	shortEM := m.MeasureWindow(shortStart, shortEnd, 10, 2.0)
	// At t=1.5s, interpolated power is ~60W; over 0.02s energy is ~1.20J
	if shortEM.TotalEnergyJoules <= 0 || math.Abs(shortEM.AvgPowerWatts-60.2) > 0.5 {
		t.Errorf("short window interpolation failed: AvgPower=%v, Energy=%v", shortEM.AvgPowerWatts, shortEM.TotalEnergyJoules)
	}
}

func TestRingBufferConcurrencyAndPubSub(t *testing.T) {
	capacity := 10
	m := NewMonitor("nvidia-smi", 20*time.Millisecond, capacity)

	subCh, unsub := m.Subscribe()
	defer unsub()

	var wg sync.WaitGroup
	// Spawn concurrent writers and readers
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for i := 0; i < 15; i++ {
				m.RecordSample(Sample{
					Timestamp:  time.Now(),
					GPUName:    "RTX 4050",
					PowerWatts: float64(30 + worker + i),
					SMClockMHz: 2355,
				})
				_ = m.Latest()
				_ = m.History(5)
			}
		}(w)
	}
	wg.Wait()

	hist := m.History(0)
	if len(hist) != capacity {
		t.Fatalf("expected ring buffer length %d, got %d", capacity, len(hist))
	}

	limited := m.History(4)
	if len(limited) != 4 {
		t.Fatalf("expected limited history length 4, got %d", len(limited))
	}

	// Verify subscriber received streamed samples
	select {
	case s := <-subCh:
		if s.GPUName != "RTX 4050" {
			t.Errorf("unexpected sample from subscriber channel: %+v", s)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for sample on subscriber channel")
	}

	// Verify Start(ctx) polls custom sampler and exits cleanly when context is canceled
	ctx, cancel := context.WithCancel(context.Background())
	m.SetSampler(func(ctx context.Context) (Sample, error) {
		return Sample{
			Timestamp:  time.Now(),
			GPUName:    "Mock GPU",
			PowerWatts: 48.5,
			SMClockMHz: 2400,
		}, nil
	})
	m.Start(ctx)
	time.Sleep(50 * time.Millisecond)
	cancel()

	latest := m.Latest()
	if latest.GPUName != "Mock GPU" || latest.PowerWatts != 48.5 {
		t.Errorf("expected latest sample from background poller, got %+v", latest)
	}
}

func TestSimulateDVFSPolicies(t *testing.T) {
	m := NewMonitor("nvidia-smi", 150*time.Millisecond, 30)
	base := EnergyMeasurement{
		DurationSec:         1.0,
		TotalEnergyJoules:   50.0,
		AvgPowerWatts:       50.0,
		AvgSMClockMHz:       2400.0,
		JoulesPerRequest:    0.5,
		JoulesPer1KRequests: 500.0,
	}

	// Memory-bound workload (e.g. RMSNorm / Softmax with 0.85 memory-bound fraction)
	policies := m.SimulateDVFSPolicies(base, 10.0, 12.0, 0.85)
	if len(policies) < 3 {
		t.Fatalf("expected at least 3 DVFS candidate policies, got %d", len(policies))
	}

	baseline := policies[0]
	sweetSpot := policies[1] // -15% SM Clock
	eco := policies[2]       // -25% SM Clock

	if baseline.TargetSMClockMHz != 2400 {
		t.Errorf("baseline clock = %v, want 2400", baseline.TargetSMClockMHz)
	}
	if sweetSpot.TargetSMClockMHz != 2040 {
		t.Errorf("sweetSpot (-15%%) clock = %v, want 2040", sweetSpot.TargetSMClockMHz)
	}

	// Sweet spot (-15% clock) should save >20% power while losing <6% throughput on memory-bound workload
	if sweetSpot.PowerChangePct > -20.0 {
		t.Errorf("expected sweetSpot power reduction > 20%%, got %.2f%%", sweetSpot.PowerChangePct)
	}
	if sweetSpot.ThroughputChangePct < -6.5 {
		t.Errorf("expected memory-bound sweetSpot throughput change better than -6.5%%, got %.2f%%", sweetSpot.ThroughputChangePct)
	}
	if sweetSpot.JoulesPer1KRequests >= baseline.JoulesPer1KRequests {
		t.Errorf("expected sweetSpot Joules/1K (%v) < baseline (%v)", sweetSpot.JoulesPer1KRequests, baseline.JoulesPer1KRequests)
	}
	if !sweetSpot.MeetsLatencySLA {
		t.Errorf("expected sweetSpot to meet 12ms SLA, estimated latency = %.2fms", sweetSpot.EstimatedLatencyMS)
	}
	if !strings.Contains(sweetSpot.Note, "power.limit") {
		t.Errorf("expected DVFS note to explain mobile GPU power.limit / simulation, got %q", sweetSpot.Note)
	}
	if eco.EstimatedPowerWatts >= sweetSpot.EstimatedPowerWatts {
		t.Errorf("expected eco power (%v) < sweetSpot power (%v)", eco.EstimatedPowerWatts, sweetSpot.EstimatedPowerWatts)
	}
	if baseline.Provenance != "MEASURED" || sweetSpot.Provenance != "SIMULATED" {
		t.Errorf("expected baseline=MEASURED and sweetSpot=SIMULATED provenance, got %q and %q", baseline.Provenance, sweetSpot.Provenance)
	}
}

func TestAggregateWindowMeasurementsIdleSubtractionAndUncertainty(t *testing.T) {
	w1 := EnergyMeasurement{
		DurationSec:         1.0,
		TotalEnergyJoules:   50.0,
		AvgPowerWatts:       50.0,
		JoulesPerRequest:    0.5,
		JoulesPer1KRequests: 500.0,
		SampleCount:         3,
		Provenance:          "MEASURED",
	}
	w2 := EnergyMeasurement{
		DurationSec:         1.0,
		TotalEnergyJoules:   54.0,
		AvgPowerWatts:       54.0,
		JoulesPerRequest:    0.54,
		JoulesPer1KRequests: 540.0,
		SampleCount:         3,
		Provenance:          "MEASURED",
	}
	agg := AggregateWindowMeasurements([]EnergyMeasurement{w1, w2}, 10.0, 2000.0, 50*time.Millisecond)
	if agg.Provenance != "MEASURED" {
		t.Errorf("expected MEASURED provenance, got %s", agg.Provenance)
	}
	if agg.RepeatWindows != 2 {
		t.Errorf("expected RepeatWindows=2, got %d", agg.RepeatWindows)
	}
	if math.Abs(agg.IdlePowerWatts-10.0) > 1e-6 || math.Abs(agg.ActivePowerWatts-42.0) > 1e-6 {
		t.Errorf("expected Idle=10W and Active=42W, got Idle=%.2f Active=%.2f", agg.IdlePowerWatts, agg.ActivePowerWatts)
	}
	if agg.EnergyUncertaintyPct <= 0 || agg.EnergyUncertaintyJoules <= 0 {
		t.Errorf("expected positive energy uncertainty, got pct=%.2f J=%.2f", agg.EnergyUncertaintyPct, agg.EnergyUncertaintyJoules)
	}
}
