# Whitening PSD of a TDI record by its kind: the Robson–Cornish–Liu model,
# the analytic LDC TDI PSD, a Welch estimate, or none; and the PSD recorded
# in a feature sidecar.

"""
    whitening_psd_from_sidecar(sidecar_path) -> Union{Nothing, Function}

The whitening PSD recorded in a feature sidecar written by the
pre-processor: the Robson–Cornish–Liu model, the LDC analytic model, the
persisted Welch table (`<stem>_psd.csv` beside the features), or `nothing`
for `psd = "none"`.
"""
function whitening_psd_from_sidecar(sidecar_path::AbstractString)
    isfile(sidecar_path) || throw(ArgumentError("feature sidecar not found: $sidecar_path"))
    features = get(TOML.parsefile(sidecar_path), "features", Dict{String,Any}())
    mode = cfgget(features, "psd", "model"; type = String)
    if mode == "model"
        years = cfgget(features, "observation_years", 1.0; type = Float64)
        return f -> lisa_noise_psd(f; observation_years = years)
    elseif mode == "ldc"
        model = cfgget(features, "ldc_model", "sangria"; type = String)
        tdi2 = cfgget(features, "ldc_tdi2", false; type = Bool)
        years = cfgget(features, "ldc_observation_years", 0.0; type = Float64)
        return f -> ldc_tdi_psd(
            f;
            channel = :A,
            model = model,
            tdi2 = tdi2,
            observation_years = years,
        )
    elseif mode == "welch"
        stem = replace(sidecar_path, r"_features\.toml$" => "")
        table_path = stem * "_psd.csv"
        isfile(table_path) || throw(
            ArgumentError("Welch PSD table not found beside the sidecar: $table_path"),
        )
        table = CSV.read(table_path, DataFrame)
        return interpolated_psd(
            Vector{Float64}(table.frequency_hz),
            Vector{Float64}(table.psd),
        )
    elseif mode == "none"
        return nothing
    end
    throw(
        ArgumentError("sidecar psd = $(repr(mode)); expected model, ldc, welch, or none."),
    )
end

# --- Replay ------------------------------------------------------------

"""
    whitening_psd(settings, A, fs) -> (psd, description, table)

Callable one-sided PSD whitening the record `A` sampled at `fs` [Hz] for
the mode `settings.psd` of the `[preprocessing]` section
([`preprocessing_settings`](@ref)): `"model"` (Robson–Cornish–Liu strain
sensitivity with the confusion fit of `observation_years`, for
sky-averaged simulator products), `"channel"` (that sensitivity times the
sky-averaged response ``R(f)``, the Michelson-channel PSD of the
simulator's constellation-response products), `"ldc"` (analytic A-channel TDI PSD of the `ldc` package in
fractional-frequency units — `ldc_model`, `ldc_tdi2`,
`ldc_observation_years` — for LDC products), `"welch"` (median-averaged
estimate from the record itself over segments of `welch_segment_length`
samples, smoothed in log-frequency by `psd_smoothing_dex` dex when that is
positive; [`smooth_psd`](@ref)), or `"none"` (no whitening; `psd` is `nothing`). `description` is
the human-readable account persisted in the sidecar; `table` is the
estimated PSD as a `DataFrame` (`frequency_hz`, `psd`) for `"welch"` and
`nothing` otherwise.
"""
function whitening_psd(settings::NamedTuple, A::AbstractVector{<:Real}, fs::Real)
    mode = settings.psd
    if mode == "model"
        model_years = settings.observation_years
        psd_model = f -> lisa_noise_psd(f; observation_years = model_years)
        return psd_model,
        "Robson–Cornish–Liu 2019 strain sensitivity, confusion fit $model_years yr",
        nothing
    elseif mode == "channel"
        model_years = settings.observation_years
        psd_channel =
            f ->
                sky_averaged_response(f) *
                lisa_noise_psd(f; observation_years = model_years)
        return psd_channel,
        "Robson–Cornish–Liu 2019 Michelson-channel PSD (sensitivity times the sky-averaged response), confusion fit $model_years yr",
        nothing
    elseif mode == "ldc"
        model = settings.ldc_model
        tdi2 = settings.ldc_tdi2
        ldc_years = settings.ldc_observation_years
        psd_ldc =
            f -> ldc_tdi_psd(
                f;
                channel = :A,
                model = model,
                tdi2 = tdi2,
                observation_years = ldc_years,
            )
        psd_ldc(1e-3)   # validates the model name before the record is processed
        return psd_ldc,
        "LDC analytic A-channel PSD, model $model, TDI $(tdi2 ? 2 : 1.5), confusion $ldc_years yr",
        nothing
    elseif mode == "welch"
        segment = settings.welch_segment_length
        segment <= length(A) || throw(
            ArgumentError(
                "welch_segment_length = $segment exceeds the record length $(length(A)).",
            ),
        )
        freqs, table = welch_psd(A, fs; segment_length = segment, average = :median)
        smoothing = settings.psd_smoothing_dex
        description = "median Welch estimate of the record, segment $segment samples"
        if smoothing > 0
            table = smooth_psd(freqs, table, smoothing)
            description *= ", smoothed by $smoothing dex in log-frequency"
        end
        return interpolated_psd(freqs, table),
        description,
        DataFrame(frequency_hz = freqs, psd = table)
    elseif mode == "none"
        return nothing, "none", nothing
    end
    throw(
        ArgumentError("psd = $(repr(mode)); expected model, channel, ldc, welch, or none."),
    )
end
