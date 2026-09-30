# Pre-processing stage: an HDF5 TDI product is
# reduced to the orthogonal A channel, high-passed, whitened, and cut into
# sliding windows whose spectral features (and, with a point-wise label
# file, window labels) are persisted as CSV beside a TOML sidecar. The
# sidecar carries a hash of every parameter that determines the product,
# so a repeated invocation with unchanged inputs reuses the files.

"""
    tdi_sample_count(path, group) -> Int

Number of samples of the TDI object at `group` of the HDF5 file `path`,
read from the dataset extents without loading the record: the length of
the `t` dataset of a group of plain datasets, or the row count of a
compound dataset (the two layouts accepted by [`read_tdi`](@ref)).
"""
function tdi_sample_count(path::AbstractString, group::AbstractString)
    isfile(path) || throw(ArgumentError("HDF5 file not found: $path"))
    return h5open(path, "r") do file
        haskey(file, group) ||
            throw(ArgumentError("$path holds no object at $(repr(group))."))
        obj = file[group]
        if obj isa HDF5.Group
            length(hdf5_dataset(obj, "t"))
        elseif obj isa HDF5.Dataset
            length(obj)
        else
            throw(
                ArgumentError("$(repr(group)) of $path is neither a group nor a dataset."),
            )
        end
    end
end

"""
    preprocessing_parameters(settings, h5_path, tdi_group, label_path) -> Dict{String, Any}

Every parameter that determines the pre-processed product: the source
file (root-relative path, size, modification time) and TDI group, the
window geometry, the whitening mode with the parameters of that mode, the
analysis bands, the record high-pass, the feature set, and — when
`label_path` is non-empty — the label file (path, size, modification
time). Its digest ([`parameter_digest`](@ref)) keys the reuse of an
existing product.
"""
function preprocessing_parameters(
    settings::NamedTuple,
    h5_path::AbstractString,
    tdi_group::AbstractString,
    label_path::AbstractString,
)
    parameters = Dict{String,Any}(
        "source" => provenance_path(h5_path),
        "source_size" => filesize(h5_path),
        "source_mtime" => mtime(h5_path),
        "tdi_group" => String(tdi_group),
        "window_size" => settings.window_size,
        "step_size" => settings.step_size,
        "psd" => settings.psd,
        "low_band_hz" => collect(settings.low_band_hz),
        "high_band_hz" => collect(settings.high_band_hz),
        "band_edges_hz" => collect(settings.band_edges_hz),
        "highpass_cutoff_hz" => settings.highpass_cutoff_hz,
        "highpass_order" => settings.highpass_order,
        "edge_margin" => settings.edge_margin,
        "feature_set" => String(settings.feature_set),
    )
    if settings.psd == "model" || settings.psd == "channel"
        parameters["observation_years"] = settings.observation_years
    elseif settings.psd == "ldc"
        parameters["ldc_model"] = settings.ldc_model
        parameters["ldc_tdi2"] = settings.ldc_tdi2
        parameters["ldc_observation_years"] = settings.ldc_observation_years
    elseif settings.psd == "welch"
        parameters["welch_segment_length"] = settings.welch_segment_length
        # Recorded only when set, so that products made before the key
        # existed keep their identity
        settings.psd_smoothing_dex > 0 &&
            (parameters["psd_smoothing_dex"] = settings.psd_smoothing_dex)
    end
    if !isempty(label_path)
        parameters["label_file"] = provenance_path(label_path)
        parameters["label_file_size"] = filesize(label_path)
        parameters["label_file_mtime"] = mtime(label_path)
    end
    return parameters
end

"""
    parameter_digest(parameters) -> String

Sixteen-digit hexadecimal digest of a parameter dictionary: the hash of
its key-sorted TOML rendering, so that the digest depends on the values
only and is reproducible across processes.
"""
function parameter_digest(parameters::AbstractDict)
    io = IOBuffer()
    TOML.print(io, parameters; sorted = true)
    return string(hash(String(take!(io))); base = 16, pad = 16)
end

"""
    reusable_features(sidecar_path, digest) -> Union{Nothing, Dict{String, Any}}

The `features` section of the sidecar at `sidecar_path` when it records
`parameter_hash == digest`, otherwise `nothing`.
"""
function reusable_features(sidecar_path::AbstractString, digest::AbstractString)
    isfile(sidecar_path) || return nothing
    features = get(TOML.parsefile(sidecar_path), "features", Dict{String,Any}())
    get(features, "parameter_hash", nothing) == digest || return nothing
    return features
end

"""
    preprocess_record(config; h5_file = nothing, tdi_group = nothing, label_file = "",
                      output_prefix = nothing, force = false) -> NamedTuple

Pre-processing stage: the TDI product `h5_file` (group `tdi_group`) is read
([`read_tdi`](@ref)), combined into the orthogonal A channel
([`tdi_to_aet`](@ref)), high-passed below the analysis bands
([`highpass_record`](@ref)), whitened by the configured PSD
([`whitening_psd`](@ref), [`whiten_record`](@ref)), and cut into sliding
windows whose features ([`window_features`](@ref)) are written to
`<inputs>/<output_prefix>_features.csv` with a TOML sidecar
`<output_prefix>_features.toml` holding the window geometry, the feature
and whitening description, the parameter hash, and the provenance
sections of [`write_toml`](@ref). A point-wise `label_file` (columns
`Label`, optionally `SNR`, one row per sample) yields the window labels
`<output_prefix>_labels.csv` ([`window_labels`](@ref)); the `"welch"`
mode also persists the estimated PSD as `<output_prefix>_psd.csv`.

Every parameter comes from the `[preprocessing]`, `[paths]`, and
`[resources]` sections of `config` ([`preprocessing_settings`](@ref),
[`pipeline_paths`](@ref), [`resource_settings`](@ref)); the keyword
arguments replace only the external inputs. A memory pre-flight
([`check_memory`](@ref)) precedes the read. When the feature table and
its sidecar exist and the sidecar's `parameter_hash` equals the digest of
the requested parameters ([`preprocessing_parameters`](@ref)), the product
is reused and nothing is recomputed unless `force`; otherwise existing
files are backed up before being replaced.

Returns `(features_path, labels_path, sidecar_path, psd_path, n_windows,
geometry, skipped)`: `labels_path` and `psd_path` are `nothing` when not
produced, `geometry` is `(window_size, step_size, sample_rate)`, and
`skipped` is `true` when the existing product was reused.
"""
function preprocess_record(
    config::AbstractDict;
    h5_file::Union{Nothing,AbstractString} = nothing,
    tdi_group::Union{Nothing,AbstractString} = nothing,
    label_file::AbstractString = "",
    output_prefix::Union{Nothing,AbstractString} = nothing,
    force::Bool = false,
)
    @timeit TIMER "preprocessing" begin
        settings = preprocessing_settings(config)
        resources = resource_settings(config)
        h5_path = resolvepath(override(h5_file, settings.h5_file))
        group = String(override(tdi_group, settings.tdi_group))
        prefix = String(override(output_prefix, settings.output_prefix))
        label_path = isempty(label_file) ? "" : resolvepath(label_file)
        has_labels = !isempty(label_path)
        isfile(h5_path) || throw(ArgumentError("HDF5 file not found: $h5_path"))
        has_labels &&
            !isfile(label_path) &&
            throw(ArgumentError("label file not found: $label_path"))

        out_dir = pipeline_paths(config).inputs
        features_path = joinpath(out_dir, "$(prefix)_features.csv")
        labels_path = joinpath(out_dir, "$(prefix)_labels.csv")
        psd_path = joinpath(out_dir, "$(prefix)_psd.csv")
        sidecar_path = replace(features_path, r"\.csv$" => ".toml")
        # Never overwrite an input: an output prefix equal to the stem of the
        # telemetry file would land the window labels on the point-wise labels.
        inputs = has_labels ? (h5_path, label_path) : (h5_path,)
        for out in (features_path, labels_path, psd_path, sidecar_path), inp in inputs
            abspath(out) == abspath(inp) && throw(
                ArgumentError(
                    "output $out coincides with input $inp; choose another output prefix.",
                ),
            )
        end

        parameters = preprocessing_parameters(settings, h5_path, group, label_path)
        digest = parameter_digest(parameters)
        writes_psd = settings.psd == "welch"
        existing = force ? nothing : reusable_features(sidecar_path, digest)
        reusable =
            existing !== nothing &&
            isfile(features_path) &&
            (!has_labels || isfile(labels_path)) &&
            (!writes_psd || isfile(psd_path))

        if reusable
            @info "pre-processed product reused; pass force = true to recompute" features_path parameter_hash =
                digest
            geometry = (
                window_size = cfgget(existing, "window_size", 0; type = Int, min = 2),
                step_size = cfgget(existing, "step_size", 0; type = Int, min = 1),
                sample_rate = cfgget(
                    existing,
                    "sample_rate",
                    0.0;
                    type = Float64,
                    min = 1e-9,
                ),
            )
            n_windows = cfgget(existing, "n_windows", 0; type = Int, min = 1)
        else
            n_points = tdi_sample_count(h5_path, group)
            check_memory(
                record_memory_estimate_gib(n_points),
                resources;
                stage = "preprocessing",
            )

            @info "reading the TDI product" h5_path group
            tdi = read_tdi(h5_path; group = group)
            fs = 1 / tdi.dt
            if settings.sample_rate !== nothing &&
               !isapprox(settings.sample_rate, fs; rtol = 1e-9)
                throw(
                    ArgumentError(
                        "sample_rate = $(settings.sample_rate) Hz disagrees with the file's " *
                        "$fs Hz (dt = $(tdi.dt) s).",
                    ),
                )
            end
            n_points = length(tdi.t)
            n_windows = window_count(n_points, settings.window_size, settings.step_size)
            @info "record" n_points sample_rate_hz = fs window_size = settings.window_size step_size =
                settings.step_size n_windows

            # Orthogonal A channel; zero-phase high-pass below the analysis
            # bands, since the steep low-frequency noise would otherwise leak
            # into every window through the taper; whitening of the whole
            # record so that the tapered periodogram of every window is
            # unbiased.
            A, _, _ = tdi_to_aet(tdi.X, tdi.Y, tdi.Z)
            if settings.highpass_cutoff_hz > 0
                A = highpass_record(
                    A,
                    fs;
                    cutoff = settings.highpass_cutoff_hz,
                    order = settings.highpass_order,
                )
            end
            psd, psd_description, psd_table = whitening_psd(settings, A, fs)
            @info "conditioning" highpass_cutoff_hz = settings.highpass_cutoff_hz highpass_order =
                settings.highpass_order whitening = psd_description feature_set =
                settings.feature_set
            psd !== nothing && (A = whiten_record(A, fs; psd = psd))

            raw_labels = Int[]
            raw_snrs = Float32[]
            if has_labels
                label_table = CSV.read(label_path, DataFrame)
                nrow(label_table) == n_points || throw(
                    DimensionMismatch(
                        "$(nrow(label_table)) labels for $n_points telemetry samples.",
                    ),
                )
                "Label" in DataFrames.names(label_table) ||
                    throw(ArgumentError("$label_path lacks the column Label."))
                raw_labels = Int.(label_table[:, :Label])
                raw_snrs =
                    "SNR" in DataFrames.names(label_table) ?
                    Float32.(label_table[:, :SNR]) : zeros(Float32, n_points)
            end

            features = window_features(
                A,
                fs;
                window_size = settings.window_size,
                step_size = settings.step_size,
                low_band = settings.low_band_hz,
                high_band = settings.high_band_hz,
                band_edges = settings.band_edges_hz,
                feature_set = settings.feature_set,
            )
            names = feature_names(
                settings.feature_set;
                n_bands = length(settings.band_edges_hz) - 1,
            )

            # Edge margin: the circular high-pass and whitening filters ring
            # over a stretch of the record at each end (the whitening
            # filter's impulse response is long when the PSD carries sharp
            # features such as the TDI null), so the first and last windows
            # are not validly conditioned and are dropped from the product.
            n_record = n_windows
            margin = edge_margin_windows(settings)
            2 * margin < n_record || throw(
                ArgumentError(
                    "edge_margin = $(settings.edge_margin) window lengths drops " *
                    "$(2 * margin) windows, but the record holds only $n_record.",
                ),
            )
            kept = (margin+1):(n_record-margin)
            n_windows = length(kept)
            margin > 0 &&
                @info "edge margin" edge_margin = settings.edge_margin dropped_each_end =
                    margin first_window = first(kept) n_windows

            write_csv(features_path, DataFrame(features[kept, :], names))
            psd_table !== nothing && write_csv(psd_path, psd_table)
            write_toml(
                sidecar_path,
                Dict{String,Any}(
                    "features" => Dict{String,Any}(
                        "source" => provenance_path(h5_path),
                        "tdi_group" => group,
                        "window_size" => settings.window_size,
                        "step_size" => settings.step_size,
                        "sample_rate" => fs,
                        "feature_set" => String(settings.feature_set),
                        "feature_names" => String.(names),
                        "psd" => settings.psd,
                        "psd_description" => psd_description,
                        "psd_smoothing_dex" => settings.psd_smoothing_dex,
                        "low_band_hz" => collect(settings.low_band_hz),
                        "high_band_hz" => collect(settings.high_band_hz),
                        "band_edges_hz" => collect(settings.band_edges_hz),
                        "highpass_cutoff_hz" => settings.highpass_cutoff_hz,
                        "highpass_order" => settings.highpass_order,
                        "edge_margin" => settings.edge_margin,
                        "edge_margin_windows" => margin,
                        "first_window" => first(kept),
                        "n_windows_record" => n_record,
                        "n_windows" => n_windows,
                        "parameter_hash" => digest,
                    ),
                );
                tag = true,
            )
            if has_labels
                labels, snrs = window_labels(
                    raw_labels,
                    raw_snrs;
                    window_size = settings.window_size,
                    step_size = settings.step_size,
                )
                write_csv(labels_path, DataFrame(Label = labels[kept], SNR = snrs[kept]))
            end
            geometry = (
                window_size = settings.window_size,
                step_size = settings.step_size,
                sample_rate = fs,
            )
            @info "pre-processed product written" features_path n_windows
        end

        (
            features_path = features_path,
            labels_path = has_labels ? labels_path : nothing,
            sidecar_path = sidecar_path,
            psd_path = writes_psd ? psd_path : nothing,
            n_windows = n_windows,
            geometry = geometry,
            skipped = reusable,
        )
    end
end
