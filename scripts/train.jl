# scripts/train.jl — dispatcher of the training stage: command line,
# terminal dashboard, file logger, and the training-history figure. The
# stage itself is `train_classifier` in the package.

include(joinpath(@__DIR__, "common.jl"))

using ArgParse, Logging, LoggingExtras, Printf
using UnicodePlots: lineplot, lineplot!
using CairoMakie: CairoMakie

function parse_commandline()
    s = ArgParseSettings(
        description = "Train the MilliHertzQML variational quantum classifier",
    )
    @add_arg_table s begin
        "config"
        help = "Path to the configuration file"
        required = false
        default = joinpath(project_root(), "config.toml")
        "--run-id"
        help = "Run identifier of the outputs (default: generated)"
        default = ""
        "--test-mode"
        help = "Sample and epoch caps from [training] for a fast validation run"
        action = :store_true
    end
    return parse_args(s)
end

"""
    format_duration(seconds) -> String

`seconds` as `HH:MM:SS`.
"""
function format_duration(seconds::Real)
    h = floor(Int, seconds / 3600)
    m = floor(Int, (seconds % 3600) / 60)
    s = floor(Int, seconds % 60)
    return @sprintf("%02d:%02d:%02d", h, m, s)
end

"""
    update_dashboard(run_id, test_mode, epoch, total_epochs, lr, history, elapsed)

Redraw the in-terminal training dashboard: epoch counter, learning rate,
elapsed and estimated remaining time, and the loss and validation-accuracy
histories as UnicodePlots line plots.
"""
function update_dashboard(
    run_id::AbstractString,
    test_mode::Bool,
    epoch::Integer,
    total_epochs::Integer,
    lr::Real,
    history::NamedTuple,
    elapsed::Real,
)
    rule = repeat("=", 80)
    print("\033[2J")
    print("\033[H")
    println(rule)
    println(
        "  MilliHertzQML TRAINING DASHBOARD | Run ID: $run_id | Mode: $(test_mode ? "TEST" : "FULL")",
    )
    println(rule)
    avg_per_epoch = elapsed / epoch
    @printf(
        "  Epoch: %3d/%3d | LR: %.5f | Elapsed: %s | ETA: %s\n",
        epoch,
        total_epochs,
        lr,
        format_duration(elapsed),
        format_duration((total_epochs - epoch) * avg_per_epoch)
    )
    println(repeat("-", 80))
    if length(history.train_loss) > 1
        p_loss = lineplot(
            history.epochs,
            history.train_loss,
            title = "Loss",
            name = "Train",
            color = :blue,
            width = 60,
            height = 10,
        )
        lineplot!(
            p_loss,
            history.epochs,
            history.val_loss,
            name = "Validation",
            color = :red,
        )
        println(p_loss)
        p_acc = lineplot(
            history.epochs,
            history.val_acc,
            title = "Validation accuracy",
            color = :green,
            width = 60,
            height = 10,
            ylim = (0, 1),
        )
        println(p_acc)
    else
        println("\n  [Waiting for more data to plot...]\n")
    end
    @printf(
        "  Current -> Train loss: %.4f | Val loss: %.4f | Val acc: %.4f\n",
        history.train_loss[end],
        history.val_loss[end],
        history.val_acc[end]
    )
    println(rule)
    return nothing
end

"""
    plot_training_history(history, stem, run_id)

Training and validation loss, and validation accuracy, per epoch, exported
at `stem` as PDF and PNG with a provenance sidecar.
"""
function plot_training_history(
    history::NamedTuple,
    stem::AbstractString,
    run_id::AbstractString,
)
    return save_figure(figure_training_history(history), stem; run_id = run_id)
end

function main()
    args = parse_commandline()
    config = load_config(args["config"])
    run_id = isempty(args["run-id"]) ? new_run_id() : args["run-id"]
    test_mode = args["test-mode"]
    paths = pipeline_paths(config)
    run_dir = joinpath(paths.models, "run_$run_id")
    plot_dir = joinpath(paths.plots, "run_$run_id")
    mkpath(run_dir)
    mkpath(plot_dir)

    # Every log record is teed to <run_dir>/training.log for the duration
    # of the stage; the stream is closed once the stage returns.
    log_stream = open(joinpath(run_dir, "training.log"), "w")
    console_logger = global_logger()
    global_logger(TeeLogger(console_logger, FileLogger(log_stream; always_flush = true)))
    on_epoch =
        (epoch, max_epochs, lr, history, elapsed) ->
            update_dashboard(run_id, test_mode, epoch, max_epochs, lr, history, elapsed)
    result = try
        train_classifier(config; run_id = run_id, test_mode = test_mode, on_epoch = on_epoch)
    finally
        global_logger(console_logger)
        flush(log_stream)
        close(log_stream)
    end

    plot_training_history(result.history, joinpath(plot_dir, "training_metrics"), run_id)
    println("Training results and logs saved to $(result.run_dir); figures in $plot_dir")
    report_timing()
    return nothing
end

main()
