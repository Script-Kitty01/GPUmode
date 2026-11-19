package pareto

import (
	"testing"

	"gpumode/internal/config"
)

func TestParetoDominanceAndModeSelection(t *testing.T) {
	econ := config.DefaultEconomicsConfig()
	econ.LatencySLAMS = 25.0

	candidates := []Candidate{
		{
			ID:                     "base",
			Name:                   "Baseline Unfused",
			LatencyMS:              18.0,
			P95LatencyMS:           20.0,
			JoulesPer1KRequests:    2400.0,
			CostPer1MInferencesUSD: 8.50,
			PeakVRAMMiB:            1800.0,
			VerifiedCorrect:        true,
		},
		{
			ID:                     "fast_high_power",
			Name:                   "Fastest Fused float4 (512 threads)",
			LatencyMS:              9.2,
			P95LatencyMS:           10.1,
			JoulesPer1KRequests:    1350.0,
			CostPer1MInferencesUSD: 4.30,
			PeakVRAMMiB:            1100.0,
			VerifiedCorrect:        true,
		},
		{
			ID:                     "eco_compact",
			Name:                   "Energy-Optimal Fused float4 (256 threads)",
			LatencyMS:              10.1,
			P95LatencyMS:           11.0,
			JoulesPer1KRequests:    1120.0,
			CostPer1MInferencesUSD: 4.10,
			PeakVRAMMiB:            950.0,
			VerifiedCorrect:        true,
		},
	}

	res := Solve(candidates, econ)
	if len(res.Frontier) != 2 {
		t.Fatalf("expected 2 non-dominated Pareto candidates (base is dominated), got %d", len(res.Frontier))
	}
	if res.Performance.ID != "fast_high_power" {
		t.Fatalf("expected Performance mode winner fast_high_power, got %s", res.Performance.ID)
	}
	if res.EnergySaving.ID != "eco_compact" {
		t.Fatalf("expected EnergySaving mode winner eco_compact, got %s", res.EnergySaving.ID)
	}
	if res.CostSaving.ID != "eco_compact" {
		t.Fatalf("expected CostSaving mode winner eco_compact, got %s", res.CostSaving.ID)
	}
}
