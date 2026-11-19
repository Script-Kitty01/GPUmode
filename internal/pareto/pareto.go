package pareto

import (
	"math"
	"sort"

	"gpumode/internal/config"
)

// OperatingMode selects which Pareto-optimal point is recommended for production deployment.
type OperatingMode string

const (
	ModePerformance  OperatingMode = "PERFORMANCE"
	ModeEnergySaving OperatingMode = "ENERGY_SAVING"
	ModeBalanced     OperatingMode = "BALANCED"
)

// Candidate represents a verified kernel or execution configuration evaluated on multiple objectives.
type Candidate struct {
	ID                  string  `json:"id"`
	Name                string  `json:"name"`
	Variant             string  `json:"variant"`
	BlockSize           int     `json:"blockSize"`
	VecWidth            int     `json:"vecWidth"`
	Fused               bool    `json:"fused"`
	LatencyMS           float64 `json:"latencyMs"`
	P95LatencyMS        float64 `json:"p95LatencyMs"`
	ThroughputRPS       float64 `json:"throughputRps"`
	PowerWatts          float64 `json:"powerWatts"`
	JoulesPer1KRequests float64 `json:"joulesPer1kRequests"`
	PeakVRAMMiB         float64 `json:"peakVramMib"`
	MeetsLatencySLA     bool    `json:"meetsLatencySla"`
	VerifiedCorrect     bool    `json:"verifiedCorrect"`
	SanitizerClean      bool    `json:"sanitizerClean"`
	IsParetoOptimal     bool    `json:"isParetoOptimal"`
	ScalarizedScore     float64 `json:"scalarizedScore"` // Lower is better: J(θ) = αT + βE + γM
}

// ModeRecommendations maps each operating mode to the best verified Pareto-optimal candidate.
type ModeRecommendations struct {
	Performance  Candidate   `json:"performance"`
	EnergySaving Candidate   `json:"energySaving"`
	Balanced     Candidate   `json:"balanced"`
	Frontier     []Candidate `json:"frontier"`
	AllEvaluated []Candidate `json:"allEvaluated"`
}

// Dominates returns true if candidate a Pareto-dominates candidate b across
// (LatencyMS, JoulesPer1KRequests, PeakVRAMMiB),
// meaning a is <= b on all objectives and strictly < b on at least one objective.
func Dominates(a, b Candidate) bool {
	const eps = 1e-9
	noWorse := a.LatencyMS <= b.LatencyMS+eps &&
		a.JoulesPer1KRequests <= b.JoulesPer1KRequests+eps &&
		a.PeakVRAMMiB <= b.PeakVRAMMiB+eps

	if !noWorse {
		return false
	}

	strictlyBetter := a.LatencyMS < b.LatencyMS-eps ||
		a.JoulesPer1KRequests < b.JoulesPer1KRequests-eps ||
		a.PeakVRAMMiB < b.PeakVRAMMiB-eps

	return strictlyBetter
}

// Solve evaluates all candidates, filters for verified correctness & SLA compliance (falling back
// if none meet strict SLA), computes normalized multi-objective scores J(θ) = αT + βE + γM,
// extracts the non-dominated Pareto frontier, and selects the winner for each operating mode.
func Solve(candidates []Candidate, econ config.EconomicsConfig) ModeRecommendations {
	if len(candidates) == 0 {
		return ModeRecommendations{}
	}

	// Filter to verified candidates first
	valid := make([]Candidate, 0, len(candidates))
	for _, c := range candidates {
		if c.VerifiedCorrect {
			if econ.LatencySLAMS > 0 {
				p95 := c.P95LatencyMS
				if p95 <= 0 {
					p95 = c.LatencyMS
				}
				c.MeetsLatencySLA = p95 <= econ.LatencySLAMS
			} else {
				c.MeetsLatencySLA = true
			}
			valid = append(valid, c)
		}
	}
	if len(valid) == 0 {
		valid = append(valid, candidates...)
	}

	// Prefer SLA-compliant candidates for Pareto frontier if at least one exists
	slaPool := make([]Candidate, 0, len(valid))
	for _, c := range valid {
		if c.MeetsLatencySLA {
			slaPool = append(slaPool, c)
		}
	}
	if len(slaPool) == 0 {
		slaPool = valid
	}

	// Compute min/max for normalization of J(θ) = αT + βE + γM
	minT, maxT := slaPool[0].LatencyMS, slaPool[0].LatencyMS
	minE, maxE := slaPool[0].JoulesPer1KRequests, slaPool[0].JoulesPer1KRequests
	minM, maxM := slaPool[0].PeakVRAMMiB, slaPool[0].PeakVRAMMiB

	for _, c := range slaPool[1:] {
		minT = math.Min(minT, c.LatencyMS)
		maxT = math.Max(maxT, c.LatencyMS)
		minE = math.Min(minE, c.JoulesPer1KRequests)
		maxE = math.Max(maxE, c.JoulesPer1KRequests)
		minM = math.Min(minM, c.PeakVRAMMiB)
		maxM = math.Max(maxM, c.PeakVRAMMiB)
	}

	alpha := econ.AlphaCostWeight
	beta := econ.BetaEnergyWeight
	gamma := econ.GammaMemoryWeight
	if alpha+beta+gamma <= 0 {
		alpha, beta, gamma = 0.45, 0.35, 0.20
	}

	norm := func(v, lo, hi float64) float64 {
		if hi-lo <= 1e-12 {
			return 0.0
		}
		return (v - lo) / (hi - lo)
	}

	// Identify non-dominated Pareto set among slaPool
	paretoIDs := make(map[string]bool)
	frontier := make([]Candidate, 0, len(slaPool))

	for i := range slaPool {
		dominated := false
		for j := range slaPool {
			if i != j && Dominates(slaPool[j], slaPool[i]) {
				dominated = true
				break
			}
		}
		slaPool[i].ScalarizedScore = alpha*norm(slaPool[i].LatencyMS, minT, maxT) +
			beta*norm(slaPool[i].JoulesPer1KRequests, minE, maxE) +
			gamma*norm(slaPool[i].PeakVRAMMiB, minM, maxM)
		if !dominated {
			slaPool[i].IsParetoOptimal = true
			paretoIDs[slaPool[i].ID] = true
			frontier = append(frontier, slaPool[i])
		}
	}

	// Update allEvaluated flags & scores
	allEvaluated := make([]Candidate, len(valid))
	for i, c := range valid {
		c.IsParetoOptimal = paretoIDs[c.ID]
		c.ScalarizedScore = alpha*norm(c.LatencyMS, minT, maxT) +
			beta*norm(c.JoulesPer1KRequests, minE, maxE) +
			gamma*norm(c.PeakVRAMMiB, minM, maxM)
		allEvaluated[i] = c
	}

	sort.Slice(frontier, func(i, j int) bool {
		return frontier[i].LatencyMS < frontier[j].LatencyMS
	})

	pickFrom := frontier
	if len(pickFrom) == 0 {
		pickFrom = slaPool
	}

	perfWinner := pickFrom[0]
	energyWinner := pickFrom[0]
	balancedWinner := pickFrom[0]

	for _, c := range pickFrom[1:] {
		if c.LatencyMS < perfWinner.LatencyMS {
			perfWinner = c
		}
		if c.JoulesPer1KRequests < energyWinner.JoulesPer1KRequests {
			energyWinner = c
		}
		if c.ScalarizedScore < balancedWinner.ScalarizedScore {
			balancedWinner = c
		}
	}

	return ModeRecommendations{
		Performance:  perfWinner,
		EnergySaving: energyWinner,
		Balanced:     balancedWinner,
		Frontier:     frontier,
		AllEvaluated: allEvaluated,
	}
}
