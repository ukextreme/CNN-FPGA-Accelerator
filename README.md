# CNN FPGA Accelerator — LeNet-5 Inference in Hardware

A LeNet-5 convolutional neural network running as synthesisable RTL on FPGA, taking
MNIST digits in and classifications out. The project widened the datapath from 16 to
32 bits for accuracy, then rebuilt both convolution layers as **weight-stationary
systolic arrays** — cutting inference from **36,924 to 17,299 clocks per frame (2.13×)**
while producing **bit-identical** predictions to the original.

**B.Tech Project · Department of Electrical Engineering, IIT Kharagpur**
Supervised by **Prof. Debdeep Mukhopadhyay**

---

## Results

| | Baseline | Systolic | Speedup |
|---|---:|---:|---|
| conv1 | 19,600 clocks | ~1,060 clocks | **18.5×** |
| conv2 | 2,500 clocks | ~1,260 clocks | **2.0×** |
| **Whole network** | **36,924 clocks/frame** | **17,299 clocks/frame** | **2.13×** |
| conv1 multipliers | 6 | 150 | |
| conv2 multipliers | 96 | 400 | |

- **550 MACs** total across both arrays (5×5 PEs × 6 output channels for conv1,
  5×5 × 16 for conv2).
- **98.67% MNIST accuracy**, bit-exact against the baseline across **842 frames**.
- Clock counts are **simulation figures**, not post-route results — synthesis has not
  been re-run against the systolic design.

---

## Why systolic arrays

The baseline convolution walks the loop nest `(batch, row, col, ky, kx)` and issues
**one multiply per clock**. Every source pixel is re-read from SRAM once per kernel
tap, so a 5×5 kernel fetches each pixel roughly **25 times over**. The multiplier sits
idle; the memory port is the bottleneck.

A systolic array unrolls the kernel taps into **hardware** rather than into **time**:
one processing element per tap, all firing every clock. A line buffer reads each pixel
**exactly once** and holds the K−1 rows still needed for the sliding window.

> Compute scales with the number of PEs. Memory traffic does not. That is the whole point.

---

## How it works

### Weight stationary

Each PE latches one kernel tap at the start of a frame and holds it for the entire
pass. The weight ROM is read K×K times per layer instead of once per multiply.

### The 2:1 delay ratio

This is the subtle part. Data is delayed **twice** per PE; partial sums **once**. With
`x(t)` the pixel entering PE0 at clock `t`:

```
p_out(j,t) = p_out(j-1,t-1) + w[j]*x(t-1-2j)
           = SUM_m w[m] * x(t-(1+j)-m)
```

Every term shares one window **only because the partial sum marches at half the speed
of the data**. A 1:1 ratio — the classic mistake — makes each PE add a different tap of
the *same* sample, which is silently wrong.

A consequence: `w[0]` meets the **newest** sample, so horizontal taps load **reversed**
(PE `j` holds `kx = K-1-j`). Vertical taps are not reversed — systolic row `r` holds
`ky = r`.

### Line buffers

The line buffer covers **only the vertical dimension** — the horizontal K comes free
from the pixel marching PE to PE. So it is K−1 row delays, not a K×K register file. On
Xilinx these row delays map to **SRL16/SRL32 LUT shift registers** rather than block RAM.

### Folding conv2's six input channels

A 5×5 array holds one input channel's taps at a time, and interleaving channels
cycle-by-cycle would break the chain — it assumes consecutive clocks carry consecutive
columns of one stream. Instead the 14×14 plane is streamed **once per channel**,
weights reloaded each pass, with partial sums held in a **100 × 16 × 32-bit** store
between passes. This folds the array 6× rather than replicating it.

---

## Verification

The baseline forms `bias + SUM(25 products)` in a 32-bit register with **no saturation**.
The systolic version forms the same sum in a different order. Two's-complement addition
is associative **even through overflow**, so the 32-bit total is identical bit for bit,
and the same slice `[30:15]` is taken.

So the acceptance criterion is **exact equality**, not "close enough" — any mismatch is
a bug, not a rounding artefact.

```
run_unit.do      290 checks on pe / systolic_row / linebuf
                 against an independent golden model
sim.bat 40       40/40 predictions identical to the baseline
```

`score.py` grades a run against the real MNIST test labels and writes `predictions.txt`
— the file to diff between the baseline and systolic builds. That diff must be empty.

---

## Repository layout

```
16bit/                        16-bit datapath — the main working design
  lenet.v                     top level: layer sequencing, ROM/SRAM wiring
  cnn.v                       conv / acc / mac primitives (baseline path)
  pe.v                        processing element + systolic_row (K-PE chain)
  linebuf.v                   K-1 row delays producing K vertical taps
  systolic_conv.v             conv1 — single input channel, one pass
  systolic_conv2.v            conv2 — six input channels, folded passes
  global.v                    all `define parameters and build switches
  lenet_roms.v                weight and bias ROMs
  bhvsrams.v, bhv_*.v         behavioural SRAM models
  lenet_tb.v                  full-network testbench
  tb_systolic.v               block-level unit tests
  run.do, run_unit.do, sim.bat   ModelSim scripts
  png_to_yuv.py               builds test_900f.yuv stimulus from test_900.png
  score.py                    grades result.log against MNIST labels
  report.py                   generates a simulation report from ModelSim logs
  README_systolic.md          detailed design notes on the systolic rebuild
  tensorflow/                 LeNet training notebooks + MNIST data

32 bit file changes/          32-bit datapath widening
  lenetv2.v, globalv2.v, lenet_romsv2.v, bhvsramsv2.v
  LeNet-Lab-Solutionv2.ipynb

BTP CNN Basics.pdf            background notes
OneNote.pdf                   working notes
nexys4ddr_rm.pdf              Nexys4 DDR board reference manual
```

### Build switches

The systolic path is selected in `global.v`:

```verilog
`define SYSTOLIC_CONV1
`define SYSTOLIC_CONV2
```

Comment either out to fall back to the baseline `iterator` + `conv/acc/mac` path for
that layer. **Both builds must produce identical output** — that is the regression test.

Datapath parameters also live in `global.v` (`WD = 15`, `WDP = 16`, `WIGHT_SHIFT = 15`,
per-layer kernel sizes and channel counts).

---

## Running it

Requires **ModelSim** (ships with Intel Quartus Prime Lite) and Python 3 with `numpy`.

```bash
sim.bat                      # all 900 frames -> result.log, then scored
sim.bat 20                   # first 20 frames (fast check)
sim.bat gui                  # open the ModelSim GUI

vsim -c -do run_unit.do      # block-level unit tests
python score.py result.log   # grade a log against MNIST labels
```

`sim.bat` generates `test_900f.yuv` from `test_900.png` via `png_to_yuv.py` if it is
missing. Frame *k* is MNIST test image *k* — `export_png.ipynb` tiled the grid as
`X_test[30*i + j]` and `png_to_yuv.py` reads the tiles back in the same order, so the
mapping is the identity.

---

## Two traps, for anyone extending this

1. **The pixel index needs two cycles of delay, not one.** The address register and the
   SRAM read are each a clock.
2. **`lenet.v` gates the weight *and bias* ROMs with the same enable as the source
   SRAM** (`cena_src` for conv1, `cena_relu1_buf` for conv2). A layer engine must assert
   it during weight load too, or every PE latches zero.

---

## Not done yet

DFX (partial reconfiguration), loop tiling, FINN integration, pruning. Synthesis has
not been re-run against the systolic design — all timing figures here are simulation
clock counts.

---

## Credits

- **Uday Keshav** ([@ukextreme](https://github.com/ukextreme)) — 32-bit datapath
  widening, systolic array and line buffer design for conv1/conv2, conv2 channel
  folding, unit test infrastructure, verification and reporting tooling.
- **Aayush Srivastava** ([@AayushSrivastava0307](https://github.com/AayushSrivastava0307))
  — baseline LeNet-5 RTL and project setup.

Developed at IIT Kharagpur under Prof. Debdeep Mukhopadhyay. Full commit history is
preserved in this repository; the collaboration originated at
[AayushSrivastava0307/BTP](https://github.com/AayushSrivastava0307/BTP).

## Built with

Verilog RTL · ModelSim · Intel Quartus Prime · TensorFlow (training) · Python · Nexys4 DDR
