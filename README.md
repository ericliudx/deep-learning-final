# Minesweeper Neural Network Agent - Brief Writeup

## Overview
Project to train neural network agents to play Minesweeper without explicit logic programming. The network uses only visible board state as input and outputs the next cell to open.

---

## Tasks
1. **Traditional Boards**
   - Easy (9x9, 10 mines), Intermediate (16x16, 40 mines), Expert (30x16, 99 mines).
   - Design NN agent to select next cell until game completion.
   - Compare vs logic bot on win rate, survival steps, mines triggered.

2. **Variable Mine Counts**
   - 30x30 board, mines = 0%–30%.
   - Train NN to adapt to varying densities.
   - Compare vs logic bot on performance curves.

3. **Variable Board Sizes**
   - Single NN for K×K boards (K≥5) with ~20% mines.
   - Compare vs logic bot for win rate, survival steps, mines triggered.

4. **Bonus: Board Generation**
   - Explore generative models to create boards NN plays well on.

---

## Key Points for Writeup
- Input representation, output mapping to actions.
- Model architectures used (e.g., CNN, attention).
- Data generation strategy (self-play, logic bot mimicry).
- Training methods, overfitting prevention.
- Comparison metrics and results summary.
- Experiments on sequential vs static game state, and attention use.

---

## Observations
- The bot performed reasonably on simpler boards.
- The bot was **unable to effectively play the more complex boards** (larger sizes or variable mine densities), struggling to generalize beyond simpler cases.
