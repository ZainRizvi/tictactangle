//! Exact game-theoretic solver for Tic Tac Two.
//!
//! Subcommands:
//!   cargo run --release -- count           Print the exact state-space math.
//!   cargo run --release -- verify          Run correctness self-tests.
//!   cargo run --release -- solve [out.bin] Solve everything; persist + report.
//!   cargo run --release -- query <board25> <cr> <cc> <turn> [table.bin]
//!                                          Exact value + optimal moves.
//!
//! See README.md for the state-space derivation and results.

mod canon;
mod check;
mod index;
mod solve;
mod verify;

use canon::{center_decode, Canon};
use index::{Indexer, BLOCKS, CENTERS, SQUARES};
use solve::{Solver, LOSS, UNKNOWN, WIN};
use std::io::Write;
use tictactwo_engine::{Move, State};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");
    match cmd {
        "count" => cmd_count(),
        "verify" => verify::run(),
        "check" => check::run(args.get(2).map(|s| s.as_str()).unwrap_or("table.bin")),
        "solve" => cmd_solve(args.get(2).map(|s| s.as_str()).unwrap_or("table.bin")),
        "query" => cmd_query(&args[2..]),
        _ => {
            eprintln!(
                "usage:\n  solver count\n  solver verify\n  solver check [table.bin]\n  \
                 solver solve [out.bin]\n  \
                 solver query <board25> <cr> <cc> <turn> [table.bin]"
            );
        }
    }
}

/// Print the exact state-space breakdown and contrast with the naive estimate.
fn cmd_count() {
    let binom = index::Binom::new();
    let ix = Indexer::new();
    println!("Tic Tac Two exact reachable-state count (turn-symmetry normalized,\nanti-loop ruleset with the last-slide/ban dimension)\n");
    println!(
        "{:>8}  {:>16}  {:>4}  {:>18}",
        "(a,b)", "configs", "cbF", "states"
    );
    let mut total_cfg = 0u64;
    let mut total = 0u64;
    for (i, &(a, b)) in BLOCKS.iter().enumerate() {
        let ca = binom.c(SQUARES, a as usize);
        let cb = binom.c(SQUARES - a as usize, b as usize);
        let cfg = ca * cb;
        let cbf = ix.cb_size(i); // 9 (no slide) or 89 (slide-legal: center*ls*ban)
        total_cfg += cfg;
        total += cfg * cbf;
        println!(
            "{:>8}  {:>16}  {:>4}  {:>18}",
            format!("({},{})", a, b),
            cfg,
            cbf,
            cfg * cbf
        );
    }
    let plain_total = total_cfg * CENTERS;
    println!("\n  configs (no center): {}", total_cfg);
    println!("  x9 grid centers (pre-anti-loop):  {}", plain_total);
    println!("  with anti-loop (center*ls*ban):   {}", total);
    println!(
        "\n  The naive estimate (raw board colorings, no grid, no alternation) was\n  \
         119,999,650. Turn-symmetry + alternation gives exactly {} configs\n  \
         (13 count blocks: alternation allows gaps up to 2, e.g. (2,4), not just\n  \
         |a-b|<=1). x9 grid centers = {} for the pre-anti-loop rules. The\n  \
         anti-loop rule adds a last-slide/ban dimension (cbF=89 for slide-legal\n  \
         blocks instead of 9), giving {} total states.",
        total_cfg, plain_total, total
    );

    assert_eq!(ix.total(), total, "indexer total must match combinatorics");
    println!("\n  indexer total() = {}  (consistency check OK)", ix.total());
}

/// Full solve: build table, run fixpoint, report, persist.
fn cmd_solve(out_path: &str) {
    let t_start = std::time::Instant::now();
    println!("Building value table...");
    let solver = Solver::new();
    println!(
        "  {} states, {:.2} GB (1 byte each)",
        solver.total,
        solver.total as f64 / 1e9
    );

    println!("Solving (forward fixpoint labeling)...");
    solver.solve(|pass, w, l, dt, detail| {
        println!(
            "  pass {:>3}: +{:>12} WIN  +{:>12} LOSS   ({:.1?}){}",
            pass, w, l, dt, detail
        );
    });

    let (w, l, d) = solver.tally();
    let total = solver.total;
    println!("\nState value distribution:");
    println!("  WIN  : {:>13}  ({:.3}%)", w, 100.0 * w as f64 / total as f64);
    println!("  LOSS : {:>13}  ({:.3}%)", l, 100.0 * l as f64 / total as f64);
    println!("  DRAW : {:>13}  ({:.3}%)", d, 100.0 * d as f64 / total as f64);

    // Value of the initial position: empty board, center (2,2), turn 1.
    let init = State {
        board: [0; 25],
        cr: 2,
        cc: 2,
        turn: 1,
        ls_r: -1,
        ls_c: -1,
        bn_r: -1,
        bn_c: -1,
    };
    let ic = Canon::from_state(&init).unwrap();
    let iv = solver.value_of(&ic);
    println!("\nInitial position value: {}", label_str(iv));

    let (_v, best) = solve::optimal_moves(&init, &solver);
    println!("Optimal first moves ({} equivalent):", best.len());
    for m in &best {
        println!("  {}", move_str(m));
    }

    // Persist.
    print!("\nWriting table to {} ... ", out_path);
    std::io::stdout().flush().ok();
    match solver.save(out_path) {
        Ok(bytes) => println!("{} bytes", bytes),
        Err(e) => println!("FAILED: {}", e),
    }

    println!("\nTotal wall time: {:.1?}", t_start.elapsed());
}

/// Query a single position given in the tools/ data format.
fn cmd_query(args: &[String]) {
    if args.len() < 4 {
        eprintln!("query <board25chars> <cr> <cc> <turn> [table.bin]");
        return;
    }
    let board_str = &args[0];
    let cr: i8 = args[1].parse().expect("cr");
    let cc: i8 = args[2].parse().expect("cc");
    let turn: u8 = args[3].parse().expect("turn");
    let table = args.get(4).map(|s| s.as_str()).unwrap_or("table.bin");
    // Optional anti-loop slide state: ls_r ls_c bn_r bn_c (each 1..3, or -1/0
    // = none). Position 5..9. Defaults to none.
    let sf = |i: usize| args.get(i).and_then(|s| s.parse::<i8>().ok()).unwrap_or(-1);
    let norm = |v: i8| if v == 0 { -1 } else { v };
    let (ls_r, ls_c, bn_r, bn_c) = (norm(sf(5)), norm(sf(6)), norm(sf(7)), norm(sf(8)));

    let mut board = [0u8; 25];
    let bytes = board_str.as_bytes();
    assert_eq!(bytes.len(), 25, "board must be 25 chars of 0/1/2");
    for i in 0..25 {
        board[i] = bytes[i] - b'0';
    }
    let s = State {
        board,
        cr,
        cc,
        turn,
        ls_r,
        ls_c,
        bn_r,
        bn_c,
    };

    let solver = Solver::load(table).unwrap_or_else(|e| {
        eprintln!("could not load {}: {} — solve first", table, e);
        std::process::exit(1);
    });

    let c = Canon::from_state(&s).expect("position violates alternation counts");
    let v = solver.value_of(&c);
    println!("Position value ({}): {}", "side to move", label_str(v));
    let (_v, best) = solve::optimal_moves(&s, &solver);
    println!("Optimal moves ({}):", best.len());
    for m in &best {
        println!("  {}", move_str(m));
    }
}

fn label_str(v: u8) -> &'static str {
    match v {
        WIN => "WIN (side to move forces a win)",
        LOSS => "LOSS (side to move loses under optimal play)",
        _ => "DRAW",
    }
}

/// Human-readable rendering of a packed engine move.
fn move_str(m: &Move) -> String {
    let kind = m >> 16;
    let a = (m >> 8) & 0xFF;
    let b = m & 0xFF;
    match kind {
        0 => format!("place at ({},{})", a / 5, a % 5),
        1 => format!("slide grid dr={} dc={}", a as i32 - 1, b as i32 - 1),
        _ => format!("move ({},{}) -> ({},{})", a / 5, a % 5, b / 5, b % 5),
    }
}
