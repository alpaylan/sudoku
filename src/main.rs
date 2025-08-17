use sudoku::Sudoku;

fn main() {
    tracing_subscriber::fmt()
        .with_line_number(true)
        .with_file(true)
        .without_time()
        .with_level(true)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let mode = std::env::var("MODE").unwrap_or_else(|_| "generate".to_string());
    tracing::info!("Running in mode: {}", mode);
    match mode.as_str() {
        "generate" => {
            let engine = std::env::var("ENGINE").unwrap_or_else(|_| "z3".to_string());
            tracing::info!("creating a new sudoku using engine: {}", engine);
            let mut sudoku = match engine.as_str() {
                "z3" => Sudoku::create_via_z3(),
                "rng" => {
                    let mut rng = rand::rng();
                    Sudoku::create(&mut rng)
                }
                _ => {
                    eprintln!("Unknown engine: {}", engine);
                    std::process::exit(1);
                }
            };
            tracing::info!("Sudoku created successfully");
            tracing::debug!("\n{}", sudoku);
            tracing::info!("Adding holes to sudoku");
            sudoku.add_holes(10);

            // Save the Sudoku grid to a file
            let path =
                std::env::var("OUTPUT_FILE").unwrap_or_else(|_| "sudoku_grid.txt".to_string());

            sudoku
                .save_to_file(&path)
                .expect("Failed to save Sudoku grid");
        }
        "solve" => {
            let path =
                std::env::var("INPUT_FILE").unwrap_or_else(|_| "sudoku_grid.txt".to_string());
            let mut sudoku = Sudoku::load_from_file(&path).expect("Failed to load Sudoku grid");
            tracing::info!("Sudoku loaded successfully");
            println!("{}", sudoku);
            // Solve the Sudoku puzzle
            sudoku.fill();
            tracing::info!("Sudoku solved successfully");
            println!("{}", sudoku);
        }
        _ => {
            eprintln!("Unknown mode: {}", mode);
            std::process::exit(1);
        }
    }
}
