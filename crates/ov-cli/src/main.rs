//! `openvoice` — the open-voice CLI. This binary is the composition root: it
//! is the only place that names concrete adapters and wires them into the
//! engine (in `auto` preference order).

mod cli;
mod compose;

use clap::Parser;

fn main() -> anyhow::Result<()> {
    let args = cli::Cli::parse();
    let filter = if args.verbose { "debug" } else { "warn" };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(filter)),
        )
        .with_writer(std::io::stderr)
        .init();

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(cli::run(args))
}
