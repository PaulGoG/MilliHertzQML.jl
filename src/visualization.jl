# Figure interface of the classifier and its studies (implemented by the
# CairoMakie extension).

"""
    animate_training_history(history, path; framerate = 5, hold_frames = 10,
                             size = figure_size(2), px_per_unit = 2) -> String

Animated counterpart of [`figure_training_history`](@ref): the two stacked
panels sharing the epoch axis on a canvas of `size` [pt], revealed one
epoch per frame at fixed axis limits, held for `hold_frames` frames at the
end. The epoch of least
validation loss — the checkpoint whose weights the run saves — is marked in
both panels once the sweep reaches it. `history` holds the vectors
`epochs`, `train_loss`, `val_loss`, and `val_acc`; `path` must name a GIF,
the only file written. Returns `path`. Requires CairoMakie to be loaded.
"""
function animate_training_history end

"""
    figure_training_history(history) -> Figure

Two stacked panels sharing the epoch axis: training and validation loss
(solid and dashed), and validation accuracy. `history` holds the vectors
`epochs`, `train_loss`, `val_loss`, `val_acc`. Requires CairoMakie.
"""
function figure_training_history end

"""
    figure_loss_survival(p_loss, scored_fraction, events_detected, n_events;
                         stretch_batches, model = true) -> Figure

Two stacked panels against the permanent per-batch loss probability of a
delivery channel [%], on a logarithmic axis: the windows a replay could
score as a fraction of the lossless mission, and the coalescences it still
detected of `n_events`.

A window is scored only when its whole conditioning stretch has been
delivered, so survival requires `stretch_batches` consecutive batches to
arrive. `model` adds the independent-batch estimate
`(1 - p)^stretch_batches` and marks `1 / stretch_batches`, the loss rate
at which the mean spacing of losses equals the stretch. Requires
CairoMakie.
"""
function figure_loss_survival end

"""
    figure_seed_spread(seeds, thresholds, far_per_30d, events_detected, n_events;
                       baseline_seed = nothing, target_far = nothing) -> Figure

Two stacked panels over the initialisation seeds of repeated training
runs: the decision threshold each run fitted, and the false-alarm rate it
then delivered on the blind record, annotated with the events each run
recovered of `n_events`. `baseline_seed` marks the selected run and `target_far` the rate the threshold criterion asked for. Requires
CairoMakie.
"""
function figure_seed_spread end

"""
    figure_gap_study(levels, families, scored_fraction, events_detected, n_events,
                     false_alarms_per_30d; reference_far = nothing) -> Figure

Three panels side by side over the levels of a delivery-gap study, one row
per level with the level names on the vertical axis in the given order
(first level on top) and a dashed separator between consecutive families:
the windows a replay scored as a fraction of the reference mission, the
coalescences it detected of `n_events`, and its false-alarm episodes per
30 days. A `NaN` rate (a replay that scored nothing) leaves its row empty
in the third panel; `reference_far` draws the reference's rate there as a
labelled guide. Requires CairoMakie.
"""
function figure_gap_study end

"""
    figure_grid_seeds(names, validation_episodes, blind_far_per_30d, events_detected,
                      n_events; seeds, selected = nothing, target_far = nothing) -> Figure

Two panels side by side over the configurations of a model grid trained
at several initialisation seeds, one row per configuration (`names`, the
first on top) and one column of the matrices per seed (`seeds`): the
false-alarm episodes of every run on its validation block, the selection
statistic, and its false-alarm rate per 30 days on the blind record, on a
logarithmic axis when every rate is positive. Seeds are told apart by
marker shape; a run that missed a blind event of `n_events` is drawn open;
a vertical bar marks the median of every configuration. `selected` shades
the row of the selected configuration and `target_far` draws the requested
rate. Entries that are not finite (a run without results) are left out.
Requires CairoMakie.
"""
function figure_grid_seeds end
