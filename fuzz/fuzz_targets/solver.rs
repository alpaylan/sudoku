#![no_main]

use libfuzzer_sys::fuzz_target;
use sudoku::Point;
use sudoku::Sudoku;

fuzz_target!(|data: &[u8]| {
    // Interpret each input triple as a sudoku point (row, col, value)
    if data.len() % 3 != 0 {
        return;
    }

    let mut fills = Vec::new();
    for chunk in data.chunks(3) {
        if let [row, col, value] = chunk {
            if (0..=8).contains(row as &u8)
                && (0..=8).contains(col as &u8)
                && (1..=9).contains(value as &u8)
                && !fills
                    .iter()
                    .any(|(p, _)| *p == Point::new(*row as i32, *col as i32))
            {
                let point = Point::new(*row as i32, *col as i32);
                fills.push((point, *value));
            } else {
                return;
            }
        }
    }

    // Load the Sudoku grid from file
    println!(
        "fills: {}",
        fills
            .iter()
            .map(|(p, v)| format!("{}:{}", p, v))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let sudoku = Sudoku::load_from_file("grid1.txt").expect("Failed to load Sudoku grid");
    sudoku.check(fills);
});
