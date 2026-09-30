# Labelling stage on a synthetic signal-only truth stream, written as the
# CSV an LDC release provides for its blind year (columns t, X, Y, Z);
# included by runtests.jl.

@testset "Labeling stage (synthetic truth stream)" begin
    fs = 0.2
    n = 60_000
    centre = 30_000
    k = 0:(n-1)
    # A Gaussian-enveloped 5 mHz burst in the A channel at a matched-filter
    # SNR of 60 against the analytic Sangria TDI PSD; X = -A/√2, Z = A/√2
    # and Y = 0 recombine to that A and to a vanishing E
    psd = f -> ldc_tdi_psd(f; channel = :A, model = "sangria")
    envelope = exp.(-0.5 .* ((k .- (centre - 1)) ./ 400) .^ 2)
    s = scale_to_snr(envelope .* cos.(2π * 5e-3 .* k ./ fs), fs, 60.0; psd = psd)
    @test isapprox(matched_filter_snr(s, fs; psd = psd), 60.0; rtol = 1e-6)
    mktempdir() do dir
        truth = joinpath(dir, "truth.csv")
        CSV.write(
            truth,
            DataFrame(t = 5.0 .* k, X = -s ./ sqrt(2), Y = zeros(n), Z = s ./ sqrt(2)),
        )
        base = Dict{String,Any}(
            "paths" => Dict{String,Any}(
                "inputs" => joinpath(dir, "inputs"),
                "models" => joinpath(dir, "models"),
                "plots" => joinpath(dir, "plots"),
                "results" => joinpath(dir, "results"),
            ),
            "ldc" => Dict{String,Any}(
                "label_span" => "fixed",
                "label_before_sec" => 20_000.0,
                "label_after_sec" => 2_000.0,
                "label_window_size" => 1000,
                "label_step" => 10,
                "output_prefix" => "synthetic",
            ),
        )

        # Fixed spans: one merger located from the SNR peak (no catalogue), at
        # the largest |A| of the burst, labelled 4000 samples before and 400
        # after
        fixed = label_truth_stream(base; truth_csv = truth)
        @test nrow(fixed.events) == 1
        merger = fixed.events.merger_index[1]
        @test abs(merger - centre) <= 20
        @test fixed.events.merger_time_s[1] == 5.0 * (merger - 1)
        labels = CSV.read(fixed.label_path, DataFrame)
        @test nrow(labels) == n
        positive = findall(==(1), labels.Label)
        @test first(positive) == merger - 4000 && last(positive) == merger + 400
        @test fixed.n_label_runs == 1
        @test isapprox(fixed.positive_fraction, 4401 / n)
        @test fixed.events.label_start_index[1] == merger - 4000
        @test fixed.events.label_peak_snr[1] > 8
        # The column is written in single precision
        @test all(
            isapprox.(labels.SNR[positive], fixed.events.label_peak_snr[1]; rtol = 1e-6),
        )
        @test all(isfile, (fixed.events_path, fixed.spans_path, fixed.snapshot_path))
        # The signal onset lies inside the span, before the merger
        onset = fixed.events.signal_start_index[1]
        @test merger - 4000 < onset < merger

        # Detectable spans: the union of the windows reaching the threshold,
        # one run around the burst and nothing at the record ends
        detectable = deepcopy(base)
        detectable["ldc"]["label_span"] = "detectable"
        detectable["ldc"]["output_prefix"] = "synthetic_detectable"
        spans = label_truth_stream(detectable; truth_csv = truth)
        labels_d = CSV.read(spans.label_path, DataFrame).Label
        @test labels_d[centre] == 1 && labels_d[1] == 0 && labels_d[end] == 0
        @test spans.n_label_runs == 1
        @test !("label_start_index" in DataFrames.names(spans.events))
        # The fixed-span onset is where the detectable span of the same scan starts
        @test onset == findfirst(==(1), labels_d)

        # Two mergers 5000 samples apart with overlapping spans: the later
        # event's onset is sought past the earlier span (which covers the first
        # burst to 3σ), and equals the onset of the burst on its own
        second = centre + 5000
        s2 = s .+ circshift(s, 5000)
        truth2 = joinpath(dir, "truth2.csv")
        CSV.write(
            truth2,
            DataFrame(t = 5.0 .* k, X = -s2 ./ sqrt(2), Y = zeros(n), Z = s2 ./ sqrt(2)),
        )
        pair = deepcopy(base)
        pair["ldc"]["output_prefix"] = "synthetic_pair"
        pair["ldc"]["peak_min_separation_sec"] = 20_000.0
        pair["ldc"]["label_after_sec"] = 6_000.0
        two = label_truth_stream(pair; truth_csv = truth2)
        @test nrow(two.events) == 2
        @test abs(two.events.merger_index[2] - second) <= 20
        @test two.events.label_start_index[2] < two.events.label_end_index[1]
        @test two.events.signal_start_index[2] > two.events.label_end_index[1]
        lead(i) = two.events.signal_start_index[i] - two.events.merger_index[i]
        @test abs(lead(1) - (onset - merger)) <= 30
        @test abs(lead(2) - (onset - merger)) <= 30

        # Arguments and malformed truth
        @test_throws ArgumentError label_truth_stream(
            base;
            truth_csv = truth,
            h5_file = truth,
        )
        @test_throws ArgumentError label_truth_stream(base)
        bad = joinpath(dir, "bad.csv")
        CSV.write(bad, DataFrame(t = [0.0, 5.0], X = [0.0, 0.0], Y = [0.0, 0.0]))
        @test_throws ArgumentError MilliHertzQML.MilliHertzBase.read_truth_csv(bad)
        CSV.write(bad, DataFrame(t = [5.0, 5.0], X = zeros(2), Y = zeros(2), Z = zeros(2)))
        @test_throws ArgumentError MilliHertzQML.MilliHertzBase.read_truth_csv(bad)
        @test_throws ArgumentError MilliHertzQML.MilliHertzBase.read_truth_csv(
            joinpath(dir, "absent.csv"),
        )
    end
end
