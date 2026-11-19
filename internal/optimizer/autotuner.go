package optimizer

import (
	"context"
	"math"
	"time"

	"gpumode/internal/config"
	"gpumode/internal/harness"
	"gpumode/internal/pareto"
	"gpumode/internal/physics"
)

// TuningStrategy selects how the autotuner explores kernel parameters.
type TuningStrategy string

const (
	StrategyBayesian TuningStrategy = "BAYESIAN_UCB"
	StrategyGrid     TuningStrategy = "GRID_SEARCH"
)

// AutotuneTrial records one iteration of the autotuner and its surrogate acquisition score.
type AutotuneTrial struct {
	Iteration        int                     `json:"iteration"`
	CandidateID      string                  `json:"candidateId"`
	Name             string                  `json:"name"`
	BlockSize        int                     `json:"blockSize"`
	VecWidth         int                     `json:"vecWidth"`
	Fused            bool                    `json:"fused"`
	ObjectiveJ       float64                 `json:"objectiveJ"` // J(θ) = αT + βE + γM
	AcquisitionScore float64                 `json:"acquisitionScore"`
	Result           harness.BenchmarkResult `json:"result"`
}

// AutotuneSummary compares Bayesian UCB optimization vs Exhaustive Grid Search efficiency.
type AutotuneSummary struct {
	Strategy                 TuningStrategy          `json:"strategy"`
	BaselineResult           harness.BenchmarkResult `json:"baselineResult"`
	BestResult               harness.BenchmarkResult `json:"bestResult"`
	Trials                   []AutotuneTrial         `json:"trials"`
	ParetoCandidates         []pareto.Candidate      `json:"paretoCandidates"`
	ParetoRecommendations    pareto.ModeRecommendations `json:"paretoRecommendations"`
	BayesianEvaluations      int                     `json:"bayesianEvaluations"`
	GridTotalConfigurations  int                     `json:"gridTotalConfigurations"`
	TuningTimeSavedPct       float64                 `json:"tuningTimeSavedPct"`
	QualityPerTuningSec      float64                 `json:"qualityPerTuningSec"`
}

// Autotuner runs hardware-verified kernel optimization using either Bayesian UCB or Grid Search.
type Autotuner struct {
	Proposer KernelProposer
	Runner   *harness.Runner
}

func NewAutotuner(proposer KernelProposer, runner *harness.Runner) *Autotuner {
	if proposer == nil {
		proposer = NewDeterministicProposer()
	}
	return &Autotuner{
		Proposer: proposer,
		Runner:   runner,
	}
}

// Run executes baseline measurement + candidate autotuning on the GPU and computes Pareto-optimal winners.
func (a *Autotuner) Run(
	ctx context.Context,
	opCategory string,
	batchSize, hiddenDim, iterations int,
	strategy TuningStrategy,
	econ config.EconomicsConfig,
) AutotuneSummary {
	tStart := time.Now()
	baseSpec, candSpecs := a.Proposer.ProposeCandidates(opCategory, batchSize, hiddenDim, iterations)

	// Evaluate baseline first
	baseResults := a.Runner.EvaluateBatch(ctx, []harness.KernelSpec{baseSpec})
	var baseRes harness.BenchmarkResult
	if len(baseResults) > 0 {
		baseRes = baseResults[0]
	}

	var evaluated []harness.BenchmarkResult
	var trials []AutotuneTrial

	gridTotal := len(candSpecs)
	if strategy == StrategyBayesian && len(candSpecs) > 3 {
		// Bayesian Optimization with RBF Gaussian Process Surrogate + Upper Confidence Bound (LCB for minimization)
		// Step 1: Seed surrogate with 2 boundary points (smallest scalar vs mid float4)
		seedIndices := []int{0, len(candSpecs) - 2}
		visited := map[int]bool{}
		seedSpecs := make([]harness.KernelSpec, 0, len(seedIndices))
		for _, idx := range seedIndices {
			visited[idx] = true
			seedSpecs = append(seedSpecs, candSpecs[idx])
		}
		seedRes := a.Runner.EvaluateBatch(ctx, seedSpecs)
		for i, r := range seedRes {
			obj := computeObjectiveJ(r, baseRes, econ)
			evaluated = append(evaluated, r)
			trials = append(trials, AutotuneTrial{
				Iteration:        i + 1,
				CandidateID:      r.Spec.ID,
				Name:             r.Spec.Name,
				BlockSize:        r.Spec.BlockSize,
				VecWidth:         r.Spec.VecWidth,
				Fused:            r.Spec.Fused,
				ObjectiveJ:       obj,
				AcquisitionScore: obj,
				Result:           r,
			})
		}

		// Step 2: Acquisition loop for remaining budget (select top 2 via GP surrogate LCB)
		budget := 2
		for step := 0; step < budget; step++ {
			bestNextIdx := -1
			bestAcq := math.MaxFloat64
			for idx, cand := range candSpecs {
				if visited[idx] {
					continue
				}
				mean, std := gpPredict(cand, trials)
				lcb := mean - 0.35*std // Lower Confidence Bound acquisition for minimization
				if lcb < bestAcq {
					bestAcq = lcb
					bestNextIdx = idx
				}
			}
			if bestNextIdx < 0 {
				break
			}
			visited[bestNextIdx] = true
			nextResList := a.Runner.EvaluateBatch(ctx, []harness.KernelSpec{candSpecs[bestNextIdx]})
			if len(nextResList) > 0 {
				r := nextResList[0]
				obj := computeObjectiveJ(r, baseRes, econ)
				evaluated = append(evaluated, r)
				trials = append(trials, AutotuneTrial{
					Iteration:        len(trials) + 1,
					CandidateID:      r.Spec.ID,
					Name:             r.Spec.Name,
					BlockSize:        r.Spec.BlockSize,
					VecWidth:         r.Spec.VecWidth,
					Fused:            r.Spec.Fused,
					ObjectiveJ:       obj,
					AcquisitionScore: bestAcq,
					Result:           r,
				})
			}
		}
	} else {
		// Full Grid Search
		allRes := a.Runner.EvaluateBatch(ctx, candSpecs)
		for i, r := range allRes {
			obj := computeObjectiveJ(r, baseRes, econ)
			evaluated = append(evaluated, r)
			trials = append(trials, AutotuneTrial{
				Iteration:        i + 1,
				CandidateID:      r.Spec.ID,
				Name:             r.Spec.Name,
				BlockSize:        r.Spec.BlockSize,
				VecWidth:         r.Spec.VecWidth,
				Fused:            r.Spec.Fused,
				ObjectiveJ:       obj,
				AcquisitionScore: obj,
				Result:           r,
			})
		}
	}

	// Build Pareto candidate list (including baseline + all evaluated candidates)
	paretoList := make([]pareto.Candidate, 0, len(evaluated)+1)
	paretoList = append(paretoList, toParetoCandidate(baseRes, econ))

	bestRes := baseRes
	bestObj := 1.0

	for _, t := range trials {
		r := t.Result
		paretoList = append(paretoList, toParetoCandidate(r, econ))
		if r.CompiledOK && r.ParityPassed && r.EdgeCasePassed && r.SanitizerPassed {
			if t.ObjectiveJ < bestObj || !bestRes.CompiledOK {
				bestObj = t.ObjectiveJ
				bestRes = r
			}
		}
	}

	paretoRecs := pareto.Solve(paretoList, econ)
	elapsedSec := math.Max(0.05, time.Since(tStart).Seconds())
	speedup := 1.0
	if bestRes.MedianLatencyMS > 0 && baseRes.MedianLatencyMS > 0 {
		speedup = baseRes.MedianLatencyMS / bestRes.MedianLatencyMS
	}

	timeSavedPct := 0.0
	if gridTotal > 0 && len(trials) < gridTotal {
		timeSavedPct = (float64(gridTotal-len(trials)) / float64(gridTotal)) * 100.0
	}

	return AutotuneSummary{
		Strategy:                strategy,
		BaselineResult:          baseRes,
		BestResult:              bestRes,
		Trials:                  trials,
		ParetoCandidates:        paretoList,
		ParetoRecommendations:   paretoRecs,
		BayesianEvaluations:     len(trials),
		GridTotalConfigurations: gridTotal,
		TuningTimeSavedPct:      timeSavedPct,
		QualityPerTuningSec:     speedup / elapsedSec,
	}
}

func computeObjectiveJ(r, base harness.BenchmarkResult, econ config.EconomicsConfig) float64 {
	if !r.CompiledOK || !r.ParityPassed || !r.EdgeCasePassed {
		return 10.0 // Penalty for invalid kernel
	}
	baseT := math.Max(1e-4, base.MedianLatencyMS)
	baseE := math.Max(1e-4, base.Energy.JoulesPer1KRequests)
	baseM := math.Max(1.0, base.PeakVRAMMiB)

	tNorm := r.MedianLatencyMS / baseT
	eNorm := r.Energy.JoulesPer1KRequests / baseE
	mNorm := r.PeakVRAMMiB / baseM

	alpha := econ.AlphaCostWeight
	beta := econ.BetaEnergyWeight
	gamma := econ.GammaMemoryWeight
	if alpha+beta+gamma <= 0 {
		alpha, beta, gamma = 0.45, 0.35, 0.20
	}
	return alpha*tNorm + beta*eNorm + gamma*mNorm
}

// gpPredict computes an RBF Gaussian kernel surrogate mean and uncertainty (std) over feature vector (log2(block), vecWidth).
func gpPredict(cand harness.KernelSpec, observed []AutotuneTrial) (mean, std float64) {
	if len(observed) == 0 {
		return 1.0, 1.0
	}
	x1 := math.Log2(float64(cand.BlockSize))
	x2 := float64(cand.VecWidth)

	var weightSum, valSum, maxSim float64
	for _, obs := range observed {
		o1 := math.Log2(float64(obs.BlockSize))
		o2 := float64(obs.VecWidth)
		distSq := (x1-o1)*(x1-o1) + 0.5*(x2-o2)*(x2-o2)
		k := math.Exp(-distSq / 1.5)
		weightSum += k
		valSum += k * obs.ObjectiveJ
		if k > maxSim {
			maxSim = k
		}
	}
	if weightSum <= 1e-9 {
		mean = 0.8
	} else {
		mean = valSum / weightSum
	}
	// Prior favoriser for vectorized float4 on memory-bound kernels
	if cand.VecWidth == 4 {
		mean -= 0.08
	}
	std = math.Sqrt(math.Max(0.01, 1.0-maxSim*maxSim))
	return mean, std
}

func toParetoCandidate(r harness.BenchmarkResult, econ config.EconomicsConfig) pareto.Candidate {
	wm := physics.WorkloadMetrics{
		LatencyMS:          r.MedianLatencyMS,
		P95LatencyMS:       r.P95LatencyMS,
		ThroughputRPS:      r.ThroughputRPS,
		AvgPowerWatts:      r.Energy.AvgPowerWatts,
		EnergyJoulesPerReq: r.Energy.JoulesPerRequest,
		PeakVRAMMiB:        r.PeakVRAMMiB,
	}
	kpi := physics.ComputeBusinessKPIs(wm, econ)
	verified := r.CompiledOK && r.ParityPassed && r.EdgeCasePassed && r.SanitizerPassed
	return pareto.Candidate{
		ID:                  r.Spec.ID,
		Name:                r.Spec.Name,
		Variant:             r.Spec.Variant,
		BlockSize:           r.Spec.BlockSize,
		VecWidth:            r.Spec.VecWidth,
		Fused:               r.Spec.Fused,
		LatencyMS:           r.MedianLatencyMS,
		P95LatencyMS:        r.P95LatencyMS,
		ThroughputRPS:       r.ThroughputRPS,
		PowerWatts:          r.Energy.AvgPowerWatts,
		JoulesPer1KRequests: kpi.JoulesPer1KInferences,
		PeakVRAMMiB:         r.PeakVRAMMiB,
		MeetsLatencySLA:     kpi.MeetsLatencySLA,
		VerifiedCorrect:     verified,
		SanitizerClean:      r.SanitizerPassed,
	}
}
