use std::{
    fmt::Display,
    io::{Read as _, Write as _, stdout},
};

use crossterm::{
    cursor::MoveTo,
    execute, queue,
    terminal::{Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
};
use rand::{
    Rng,
    seq::{IndexedRandom, SliceRandom as _},
};
use z3::{
    Config, Context, SatResult, Solver,
    ast::{Ast, Bool, Int},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sudoku {
    grid: [[u8; 9]; 9],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Point {
    row: i32,
    col: i32,
}

impl Display for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({}, {})", self.row, self.col)
    }
}

impl std::ops::Sub for Point {
    type Output = i32;

    fn sub(self, other: Self) -> Self::Output {
        (self.row - other.row) * 9 + (self.col - other.col)
    }
}

impl Point {
    pub fn new(row: i32, col: i32) -> Self {
        Self { row, col }
    }

    fn next(&self) -> Option<Self> {
        let mut next_row = self.row;
        let mut next_col = self.col + 1;
        if next_col == 9 {
            next_col = 0;
            next_row += 1;
        }
        if next_row == 9 {
            None
        } else {
            Some(Self::new(next_row, next_col))
        }
    }

    fn prev(&self) -> Option<Self> {
        let mut prev_row = self.row;
        let mut prev_col = self.col;
        if prev_col == 0 {
            prev_col = 8;
            if prev_row == 0 {
                return None; // No previous point exists
            }
            prev_row -= 1;
        } else {
            prev_col -= 1;
        }
        Some(Self::new(prev_row, prev_col))
    }

    fn valid(&self) -> bool {
        self.row >= 0 && self.col >= 0 && self.row < 9 && self.col < 9
    }
}

const UNINIT: u8 = 0;

fn redraw_board<S: std::fmt::Display>(sudoku: &S) -> Result<(), std::io::Error> {
    let mut out = stdout();
    // Move to top-left, clear everything, then print the board, all on the SAME handle
    queue!(out, MoveTo(0, 0), Clear(ClearType::All))?;
    write!(out, "{sudoku}")?;
    out.flush()?;
    Ok(())
}

impl Sudoku {
    pub fn create(rng: &mut impl Rng) -> Self {
        let grid = [[UNINIT; 9]; 9];
        let mut sudoku = Self { grid };
        let mut p = Point::new(0, 0);

        let mut backtracking_depth = 0;
        let mut backtracking_start_point = None;

        let mut out = stdout();
        execute!(out, EnterAlternateScreen).unwrap();

        while p.valid() {
            // clear the previous print
            redraw_board(&sudoku).expect("Failed to redraw the board");
            // Fill P with a random number that fits the Sudoku rules
            let candidates: Vec<u8> = sudoku.candidates(p);
            if let Some(&value) = candidates.choose(rng) {
                sudoku.set(p, value);
                // Reset backtracking depth if we successfully set a value
                if let Some(start_point) = backtracking_start_point {
                    if p == start_point {
                        backtracking_start_point = None;
                        backtracking_depth = 0;
                    }
                } else {
                    backtracking_depth -= 1;
                }
            } else {
                // If no candidates are available, backtrack, but backtracking should
                // remember the current point and keep increasing backtracking depth
                // until it is able to pass the point
                // wait for user input to see the backtracking
                if backtracking_start_point.is_none() {
                    backtracking_start_point = Some(p);
                    backtracking_depth = 0;
                }

                backtracking_depth += 1;
                for _ in 0..backtracking_depth {
                    sudoku.set(p, UNINIT);
                    p = p.prev().unwrap_or(Point::new(0, 0)); // Go back to the previous point
                }
                continue;
            }

            p = match p.next() {
                Some(next) => next,
                None => break, // No more points to fill
            };
        }

        execute!(out, LeaveAlternateScreen).unwrap();

        sudoku
    }

    pub fn create_via_z3() -> Self {
        let grid = [[UNINIT; 9]; 9];
        let mut sudoku = Self { grid };
        sudoku.fill();
        sudoku
    }

    pub fn fill(&mut self) {
        let solution = self.solve();
        assert!(
            solution.get_model().is_some(),
            "Z3 solver failed to find a solution"
        );

        let model = solution
            .get_model()
            .expect("Failed to get model from Z3 solver");

        for row in 0..9 {
            for col in 0..9 {
                let cell = Int::new_const(solution.get_context(), format!("cell_{}_{}", row, col));
                self.grid[row][col] =
                    model.get_const_interp(&cell).unwrap().as_i64().unwrap() as u8;
            }
        }
    }

    pub fn solve(&self) -> Solver {
        let cfg = Config::new();
        let ctx = Context::new(&cfg);
        let solver = Solver::new(&ctx);
        let mut grid = vec![];
        for row in 0..9 {
            let mut row_vec = vec![];
            for col in 0..9 {
                let cell = Int::new_const(&ctx, format!("cell_{}_{}", row, col));
                if self.grid[row][col] != UNINIT {
                    // If the cell is already filled, assert its value
                    solver.assert(&cell._eq(&Int::from_i64(&ctx, self.grid[row][col] as i64)));
                }
                row_vec.push(cell);
            }
            grid.push(row_vec);
        }

        // Each cell must be between 1 and 9
        for row in 0..9 {
            for col in 0..9 {
                let cell = &grid[row][col];
                solver.assert(&cell.ge(&Int::from_i64(&ctx, 1)));
                solver.assert(&cell.le(&Int::from_i64(&ctx, 9)));
            }
        }

        // Each row must have unique values
        for row in 0..9 {
            let mut seen = vec![];
            for col in 0..9 {
                let cell = &grid[row][col];
                seen.push(cell);
            }
            solver.assert(&z3::ast::Int::distinct(&ctx, &seen));
        }
        // Each column must have unique values
        for col in 0..9 {
            let mut seen = vec![];
            for row in 0..9 {
                let cell = &grid[row][col];
                seen.push(cell);
            }
            solver.assert(&z3::ast::Int::distinct(&ctx, &seen));
        }
        // Each 3x3 subgrid must have unique values
        for block_row in 0..3 {
            for block_col in 0..3 {
                let mut seen = vec![];
                for row in 0..3 {
                    for col in 0..3 {
                        let cell = &grid[block_row * 3 + row][block_col * 3 + col];
                        seen.push(cell);
                    }
                }
                solver.assert(&z3::ast::Int::distinct(&ctx, &seen));
            }
        }

        // Check if the constraints are satisfiable
        tracing::debug!("Checking Sudoku constraints with Z3 solver");
        // Get the model if the constraints are satisfiable
        if solver.check() == SatResult::Sat {
            tracing::debug!("Sudoku constraints are satisfiable");
        }
        solver
    }

    pub fn count_solutions(&self) -> u8 {
        assert!(
            self.grid.iter().any(|row| row.contains(&UNINIT)),
            "Sudoku must have at least one uninitialized cell to count solutions"
        );

        let mut c = 0;
        let solution = self.solve();
        loop {
            if solution.check() != SatResult::Sat {
                break; // No more solutions
            }
            let model = solution
                .get_model()
                .expect("Failed to get model from Z3 solver");

            c += 1;
            // get a list of all the assignments to UNINIT fields
            let mut holes = vec![];
            for i in 0..81 {
                let row = i / 9;
                let col = i % 9;
                if self.grid[row][col] == UNINIT {
                    holes.push((
                        Point::new(row as i32, col as i32),
                        model
                            .get_const_interp(&Int::new_const(
                                solution.get_context(),
                                format!("cell_{}_{}", row, col),
                            ))
                            .unwrap()
                            .as_i64()
                            .unwrap() as u8,
                    ));
                }
            }

            // assert that it is false that this set of instantiations is correct all together
            let assignments = holes.into_iter().map(|(p, v)| {
                let cell =
                    Int::new_const(solution.get_context(), format!("cell_{}_{}", p.row, p.col));
                cell._eq(&Int::from_i64(solution.get_context(), v as i64))
            });
            solution.assert(&Bool::not(&Bool::and(
                solution.get_context(),
                &assignments.collect::<Vec<_>>(),
            )));
        }
        c
    }

    pub fn add_holes(&mut self, count: usize) {
        let mut rng = rand::rng();
        let mut indices: Vec<usize> = (0..81).collect();
        indices.shuffle(&mut rng);
        for (i, index) in indices.iter().take(count).enumerate() {
            tracing::debug!("Adding hole {} at index: {}", i, index);
            let row = index / 9;
            let col = index % 9;
            let tmp = self.grid[row][col];
            self.grid[row][col] = UNINIT; // Set the cell to uninitialized
            // check if you can still solve the sudoku
            tracing::debug!(
                "Checking if Sudoku is still solvable after adding hole at ({}, {})",
                row,
                col
            );
            let c = self.count_solutions();
            match c {
                0 => unreachable!(
                    "No solutions found after removing one element, this means constraint solving fails, and possibly a bug"
                ),
                1 => tracing::info!(
                    "Sudoku is still solvable after removing one element, {} solution found",
                    c
                ),
                _ => {
                    tracing::warn!(
                        "Sudoku now has {} solutions, adding the removed element back",
                        c
                    );
                    self.grid[row][col] = tmp; // Restore the cell
                }
            }
        }
    }

    fn set(&mut self, point: Point, value: u8) {
        if point.valid() && (1..=10).contains(&value) {
            self.grid[point.row as usize][point.col as usize] = value;
        } else {
            panic!(
                "Invalid point or value: Point({:?}), Value({})",
                point, value
            );
        }
    }

    fn valid(&self) -> bool {
        // A row cannot contain duplicates
        for row in 0..9 {
            let mut seen = [false; 10];
            for col in 0..9 {
                let value = self.grid[row][col];
                if value == UNINIT {
                    return false;
                }

                if seen[value as usize] {
                    return false; // Duplicate found in row
                }

                seen[value as usize] = true;
            }
        }
        // A column cannot contain duplicates
        for col in 0..9 {
            let mut seen = [false; 10];
            for row in 0..9 {
                let value = self.grid[row][col];
                if value == UNINIT {
                    return false;
                }
                if seen[value as usize] {
                    return false; // Duplicate found in column
                }
                seen[value as usize] = true;
            }
        }

        // A 3x3 subgrid cannot contain duplicates
        for block_row in 0..3 {
            for block_col in 0..3 {
                let mut seen = [false; 10];
                for row in 0..3 {
                    for col in 0..3 {
                        let value = self.grid[block_row * 3 + row][block_col * 3 + col];
                        if value == UNINIT {
                            return false;
                        }
                        if seen[value as usize] {
                            return false; // Duplicate found in subgrid
                        }
                        seen[value as usize] = true;
                    }
                }
            }
        }

        true
    }

    fn candidates(&self, p: Point) -> Vec<u8> {
        if !p.valid() {
            return vec![];
        }

        let mut candidates: Vec<u8> = (1..=9).collect();
        // Remove numbers already present in the same row
        for col in 0..9 {
            let value = self.grid[p.row as usize][col];
            if value != UNINIT {
                candidates.retain(|&x| x != value);
            }
        }
        // Remove numbers already present in the same column
        for row in 0..9 {
            let value = self.grid[row][p.col as usize];
            if value != UNINIT {
                candidates.retain(|&x| x != value);
            }
        }
        // Remove numbers already present in the same 3x3 subgrid
        let block_row = p.row / 3;
        let block_col = p.col / 3;
        for row in 0..3 {
            for col in 0..3 {
                let value =
                    self.grid[(block_row * 3 + row) as usize][(block_col * 3 + col) as usize];
                if value != UNINIT {
                    candidates.retain(|&x| x != value);
                }
            }
        }

        candidates
    }

    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        for i in 0..9 {
            if i % 3 == 0 && i != 0 {
                writeln!(f)?;
            }
            for j in 0..9 {
                write!(f, "{} ", self.grid[i][j])?;
                if j % 3 == 2 {
                    write!(f, " ")?;
                }
            }
            writeln!(f)?;
        }

        Ok(())
    }

    pub fn load_from_file(path: &str) -> std::io::Result<Self> {
        let mut str = String::new();
        std::fs::File::open(path)?.read_to_string(&mut str)?;
        str.retain(|c| c.is_ascii_digit());
        assert_eq!(
            str.len(),
            81,
            "Invalid Sudoku grid format in file:\n{}",
            str
        );
        let mut grid = [[UNINIT; 9]; 9];
        for (i, c) in str.chars().enumerate() {
            let value = c.to_digit(10).unwrap() as u8;
            grid[i / 9][i % 9] = value;
        }
        Ok(Self { grid })
    }
}

impl Sudoku {
    pub fn check(&self, solution: Vec<(Point, u8)>) -> bool {
        // Go through each fill and set the value in the grid
        let mut sudoku = self.clone();
        for (point, value) in solution.iter() {
            if point.valid() && *value >= 1 && *value <= 9 {
                sudoku.set(*point, *value);
            } else {
                return false; // Invalid fill
            }
        }
        // Check if the Sudoku is valid after all solution
        if sudoku.valid() {
            panic!(
                "Sudoku is valid after solution: {:?} and filled with {:?}",
                sudoku, solution
            );
        }
        false
    }
}

impl Display for Sudoku {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // create a 3x3 grid representation where each cell is 3x3
        writeln!(f, "{}", "-".repeat(30))?;
        for row in 0..9 {
            if row % 3 == 0 && row != 0 {
                writeln!(f, "{}", "-".repeat(30))?;
            }
            for col in 0..9 {
                if col % 3 == 0 {
                    write!(f, "|")?;
                }
                if self.grid[row][col] == UNINIT {
                    write!(f, "   ")?;
                } else {
                    write!(f, " {} ", self.grid[row][col])?;
                }
            }
            writeln!(f, "|")?;
        }
        writeln!(f, "{}", "-".repeat(30))?;
        Ok(())
    }
}
