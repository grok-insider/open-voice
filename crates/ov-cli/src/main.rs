//! `openvoice` — the open-voice CLI. This binary is the composition root: it
//! is the only place that names concrete adapters and wires them into the
//! engine (in `auto` preference order).

mod cli;
mod compose;

use clap::Parser;

/// Restore default SIGPIPE behavior so `openvoice ... | head` exits quietly
/// instead of panicking with "failed printing to stdout: Broken pipe".
#[cfg(unix)]
fn reset_sigpipe() {
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

#[cfg(not(unix))]
fn reset_sigpipe() {}

fn main() -> anyhow::Result<()> {
    reset_sigpipe();
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
